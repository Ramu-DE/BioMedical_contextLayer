"""Neurosymbolic pipeline. Task 6.4, Requirements 9.1-9.4.

Order of operations is enforced here, in code:

    assemble context -> fetch facts -> LLM proposes -> rules evaluate -> gate

The LLM never sees the rule verdict, and cannot argue with it. The rule engine
never sees the LLM's text, and cannot be influenced by it. They meet only at the
governance gate, which gives the symbolic side precedence.

That separation is the whole claim: pattern recognition and rule following in
one answer, with the interaction between them visible in the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from context_layer.assembler.package import ContextAssembler
from context_layer.governance.gate import GovernanceGate
from context_layer.rules.engine import evaluate as evaluate_rules
from context_layer.rules.graph_facts import FACT_KEYS, GraphFactBuilder
from context_layer.rules.loader import PACKS_DIR, all_rules, load_packs
from context_layer.rules.models import FactSet
from context_layer.types import ContextPackage, GovernedResponse, RuleVerdict

if TYPE_CHECKING:
    from config import Config

SYSTEM_PROMPT = """\
You answer questions about an enterprise pharmaceutical knowledge graph using \
ONLY the context provided below.

Rules for your answer:
- Use only facts present in the context. Do not add outside knowledge.
- Cite the element id in square brackets after each claim, e.g. [graph:Drug:X:0].
- If the context does not answer the question, say so plainly instead of guessing.
- Be concise: three sentences or fewer unless the question demands more.
- Do not give individual treatment advice. Describe evidence, not prescriptions.

You are the pattern-recognition half of a neurosymbolic system. A separate \
deterministic rule engine evaluates governance policy independently and may \
override you. Do not speculate about policy compliance or consent status; state \
only what the context supports.
"""

# Entity ids that may appear in a question, e.g. PAT015, D001, CT010.
ID_PATTERN = re.compile(r"\b(PAT\d+|D\d{3}|DIS\d+|CT\d+|AE\d+|B\d{3}|G\d{3}|P\d{3}|POL[_A-Z0-9]*)\b")

# Relevance gating for rule scope.
#
# Facts must be *in scope for the question*, otherwise unrelated rules fire and
# block everything. An earlier version scanned data-asset types on every
# question, which meant a data-quality policy about sales tables blocked a
# question about PD-1 inhibitors. Rules only bind to entities the question or
# its retrieved context actually touches.
#
# Data-asset types are included only when the question is about data assets.
DATA_ASSET_TYPES = ("Table", "ARD", "Entity")
DATA_ASSET_TERMS = frozenset(
    {
        "table", "tables", "hub", "layer", "ard", "dataset", "datasets", "mart",
        "load", "loads", "incremental", "job", "jobs", "pipeline", "pipelines",
        "schema", "column", "columns", "lineage", "certified", "certify",
        "quality", "dq", "governance", "compliant", "compliance", "policy",
        "policies", "warehouse", "ingestion", "ingest", "batch", "catalog",
    }
)

CLINICAL_TYPES = ("Patient", "Drug", "ClinicalTrial", "AdverseEvent", "Biomarker")


@dataclass(frozen=True)
class Trace:
    """Everything that happened, for demos and audit."""

    package: ContextPackage
    facts: FactSet
    verdict: RuleVerdict
    response: GovernedResponse
    decision_id: str | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        prior = len(self.package.by_kind("prior_decision"))
        return (
            f"context={len(self.package.elements)} elements "
            f"({self.package.token_count} tok, {prior} prior decision(s)), "
            f"facts={len(self.facts)}, rules_fired={len(self.verdict.fired)}, "
            f"blocked={self.verdict.blocked}, "
            f"grounding={self.response.grounding_confidence}"
        )

    def timing_summary(self) -> str:
        return ", ".join(f"{k}={v:.0f}ms" for k, v in self.timings.items())


class ContextLayerAgent:
    """The full neurosymbolic path for one question."""

    def __init__(
        self,
        cfg: Config,
        llm=None,
        assembler=None,
        fact_builder=None,
        store=None,
        write_back: bool = True,
    ) -> None:
        self.cfg = cfg
        self.rules = all_rules(load_packs(PACKS_DIR))
        self.gate = GovernanceGate(cfg)
        self.write_back = write_back
        self.fact_builder = (
            fact_builder if fact_builder is not None else GraphFactBuilder(cfg)
        )

        # Runtime store: enables continuous incorporation (R7.1) and the
        # decision write-back loop (R6.1-6.2). Constructed before the assembler
        # so the RuntimeRetriever can read prior decisions.
        self.store = store
        if self.store is None and write_back:
            try:
                from context_layer.clients.embeddings import BedrockEmbeddings
                from context_layer.runtime.store import RuntimeStore

                self.store = RuntimeStore(cfg, embedder=BedrockEmbeddings(cfg))
                self.store.ensure_schema()
            except Exception as e:  # noqa: BLE001
                print(f"[agent] runtime store unavailable: {type(e).__name__}: {e}")
                self.store = None

        if assembler is not None:
            self.assembler = assembler
        else:
            from context_layer.assembler.retrievers import RuntimeRetriever

            self.assembler = ContextAssembler(
                cfg, runtime=RuntimeRetriever(cfg, store=self.store)
            )
        self._llm = llm

    # ── neural half ───────────────────────────────────────────────────────────

    def _lazy_llm(self):
        if self._llm is None:
            from langchain_aws import ChatBedrockConverse

            kwargs = {
                "model": self.cfg.bedrock_model_id,
                "region_name": self.cfg.aws_region,
                "temperature": 0,
                "max_tokens": 600,
            }
            if self.cfg.aws_profile:
                import boto3

                session = boto3.Session(
                    profile_name=self.cfg.aws_profile, region_name=self.cfg.aws_region
                )
                kwargs["client"] = session.client("bedrock-runtime")
            self._llm = ChatBedrockConverse(**kwargs)
        return self._llm

    @staticmethod
    def _text(content) -> str:
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    def propose(self, package: ContextPackage, on_token=None) -> str | None:
        """Ask the model for an answer grounded in the package.

        When ``on_token`` is supplied the response is streamed and each text
        delta is handed over as it arrives, so a UI can show the neural half
        forming. Streaming reaches first token in roughly a quarter of the time
        a blocking call takes, which is the difference between a live demo and a
        spinner.

        Falls back to a blocking invoke if streaming is unavailable, so the
        answer never depends on the transport.
        """
        if package.is_empty:
            return None
        prompt = (
            f"{SYSTEM_PROMPT}\n\n=== CONTEXT ===\n"
            f"{ContextAssembler.render(package)}\n\n"
            f"=== QUESTION ===\n{package.question}"
        )
        llm = self._lazy_llm()

        if on_token is not None:
            try:
                parts: list[str] = []
                for chunk in llm.stream(prompt):
                    delta = self._text(chunk.content)
                    if not delta:
                        continue
                    parts.append(delta)
                    on_token(delta, "".join(parts))
                return "".join(parts).strip() or None
            except Exception as e:  # noqa: BLE001
                print(f"[agent] streaming failed, falling back: {type(e).__name__}: {e}")

        try:
            result = llm.invoke(prompt)
        except Exception as e:  # noqa: BLE001
            print(f"[agent] LLM unavailable: {type(e).__name__}: {e}")
            return None
        return self._text(result.content).strip() or None

    # ── symbolic half ─────────────────────────────────────────────────────────

    @staticmethod
    def question_is_about_data_assets(question: str) -> bool:
        """Whether data-asset governance rules are in scope for this question."""
        words = {
            w.strip(".,?!:;()'\"").casefold() for w in (question or "").split()
        }
        return bool(words & DATA_ASSET_TERMS)

    def gather_facts(self, question: str, package: ContextPackage) -> FactSet:
        """Facts scoped to what the question and its context actually touch.

        Scoping is the relevance mechanism for the symbolic layer. A fact that
        is not in scope cannot trigger a rule, which is what stops an unrelated
        policy from blocking an unrelated question.
        """
        facts = []

        # 1. Entity ids named explicitly in the question.
        wanted: dict[str, list[str]] = {}
        for token in ID_PATTERN.findall(question):
            label = _label_for_id(token)
            if label:
                wanted.setdefault(label, []).append(token)
        if wanted:
            facts += list(self.fact_builder.for_entities(wanted).facts)

        # 2. Entities named in plain language: "HER2", "Chronic Myeloid Leukemia".
        #    Without this, only ID-shaped mentions ever enter scope, so natural
        #    phrasing could never bind a rule or light the traversal graph.
        try:
            by_name = self.fact_builder.resolve_by_name(question)
        except Exception as e:  # noqa: BLE001
            print(f"[agent] name resolution failed: {type(e).__name__}: {e}")
            by_name = {}
        for label, ids in by_name.items():
            wanted.setdefault(label, []).extend(
                i for i in ids if i not in wanted.get(label, [])
            )
        if by_name:
            facts += list(self.fact_builder.for_entities(by_name).facts)

        # 3. Entities reachable from CURIEs the retriever surfaced.
        if package.curies:
            facts += list(self.fact_builder.for_curies(list(package.curies)).facts)

        # 4. Entities named in retrieved graph elements, resolved by id.
        for element in package.by_kind("graph_node"):
            for token in ID_PATTERN.findall(element.content):
                label = _label_for_id(token)
                if label:
                    wanted.setdefault(label, []).append(token)
        if wanted:
            facts += list(self.fact_builder.for_entities(wanted).facts)

        # 5. Data-asset governance only when the question concerns data assets.
        if self.question_is_about_data_assets(question):
            for label in DATA_ASSET_TYPES:
                facts += list(self.fact_builder.for_type(label, limit=50).facts)

        unique = {(f.entity_type, f.entity_id): f for f in facts}
        return FactSet(
            facts=tuple(sorted(unique.values(), key=lambda f: (f.entity_type, f.entity_id))),
            source_note="scoped to question and retrieved context",
        )

    # ── full path ─────────────────────────────────────────────────────────────

    def run(self, question: str, on_stage=None, on_token=None,
            on_retriever=None) -> Trace:
        """Execute the full path.

        ``on_stage(name, payload)`` is called as each stage completes, so a UI
        can render progressively rather than blocking for the whole run. The
        callback is advisory: an exception in it must never affect the answer.
        """
        import time

        timings: dict[str, float] = {}

        def emit(name: str, payload: dict) -> None:
            if on_stage is None:
                return
            try:
                on_stage(name, payload)
            except Exception as e:  # noqa: BLE001
                print(f"[agent] stage callback failed for {name}: {type(e).__name__}: {e}")

        emit("start", {"question": question})

        t0 = time.perf_counter()
        # Assemblers are pluggable; not every implementation reports per-source
        # progress. Degrade to a plain call rather than requiring the kwarg.
        if on_retriever is not None:
            try:
                package = self.assembler.assemble(question, on_retriever=on_retriever)
            except TypeError:
                package = self.assembler.assemble(question)
        else:
            package = self.assembler.assemble(question)
        timings["assemble"] = (time.perf_counter() - t0) * 1000
        emit(
            "assemble",
            {
                "package": package,
                "ms": timings["assemble"],
                "elements": len(package.elements),
                "tokens": package.token_count,
                "degradations": list(package.degradations),
                "priors": len(package.by_kind("prior_decision")),
            },
        )

        t0 = time.perf_counter()
        facts = self.gather_facts(question, package)
        timings["facts"] = (time.perf_counter() - t0) * 1000
        emit(
            "facts",
            {
                "facts": facts,
                "ms": timings["facts"],
                "count": len(facts),
                "types": list(facts.types),
                "in_scope": self.question_is_about_data_assets(question),
            },
        )

        t0 = time.perf_counter()
        first_token_ms: list[float] = []

        def token_hook(delta: str, accumulated: str) -> None:
            if not first_token_ms:
                first_token_ms.append((time.perf_counter() - t0) * 1000)
            if on_token is not None:
                try:
                    on_token(delta, accumulated)
                except Exception as e:  # noqa: BLE001
                    print(f"[agent] token callback failed: {type(e).__name__}: {e}")

        proposal = self.propose(               # neural
            package, on_token=token_hook if on_token is not None else None
        )
        timings["llm"] = (time.perf_counter() - t0) * 1000
        emit(
            "neural",
            {
                "proposal": proposal,
                "ms": timings["llm"],
                "first_token_ms": first_token_ms[0] if first_token_ms else None,
                "streamed": bool(first_token_ms),
            },
        )

        t0 = time.perf_counter()
        verdict = evaluate_rules(facts, self.rules)   # symbolic, independent
        timings["rules"] = (time.perf_counter() - t0) * 1000
        emit(
            "symbolic",
            {
                "verdict": verdict,
                "ms": timings["rules"],
                "fired": [f.rule_id for f in verdict.fired],
                "blocked": verdict.blocked,
                "evaluated": len(self.rules),
                # The traversal view needs facts and verdict together.
                "facts_ref": facts,
            },
        )

        response = self.gate.evaluate(question, package, verdict, proposal)
        emit(
            "gate",
            {
                "response": response,
                "branch": self._gate_branch(response, verdict),
            },
        )

        decision_id = self._record(question, package, response, timings)
        emit("record", {"decision_id": decision_id})

        return Trace(
            package=package,
            facts=facts,
            verdict=verdict,
            response=response,
            decision_id=decision_id,
            timings=timings,
        )

    @staticmethod
    def _gate_branch(response, verdict) -> str:
        """Which governance branch produced this response."""
        if response.refused and "scope" in (response.refusal_reason or ""):
            return "clinical_advice_guard"
        if verdict.blocked:
            return "symbolic_block_override"
        if response.refused:
            return "grounding_threshold"
        return "released"

    # ── write-back: continuous incorporation, R6.1-6.2, R7.1, R7.4 ────────────

    def _record(self, question, package, response, timings) -> str | None:
        """Append the decision and its actions. Graph write only — no training."""
        if self.store is None:
            return None
        try:
            from context_layer.runtime.store import (
                action as make_action,
                decision_from_response,
            )

            decision = decision_from_response(
                question, response, grounded_in=package.curies
            )
            self.store.append_decision(decision)
        except Exception as e:  # noqa: BLE001
            print(f"[agent] decision write-back failed: {type(e).__name__}: {e}")
            return None

        # Actions are secondary: a failure here must not discard the decision id,
        # otherwise the caller cannot attach an outcome to a decision that was
        # in fact persisted.
        for tool, ms in timings.items():
            try:
                self.store.append_action(
                    make_action(
                        decision.decision_id,
                        tool_name=tool,
                        arguments={"question_chars": len(question)},
                        duration_ms=ms,
                    )
                )
            except Exception as e:  # noqa: BLE001
                print(f"[agent] action write-back failed for {tool}: {type(e).__name__}: {e}")
        return decision.decision_id

    def record_outcome(self, decision_id: str, outcome_type: str, detail: str = "") -> None:
        """Record a decision-quality outcome. R6.3.

        Decision-quality, not clinical: whether the answer was accepted, whether
        the rule held, whether retrieval sufficed.
        """
        if self.store is None or not decision_id:
            return
        from context_layer.runtime.store import outcome as make_outcome

        self.store.append_outcome(make_outcome(decision_id, outcome_type, detail))

    def close(self) -> None:
        self.assembler.close()
        self.fact_builder.close()
        if self.store is not None:
            self.store.close()

    def __enter__(self) -> ContextLayerAgent:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _label_for_id(token: str) -> str | None:
    prefixes = {
        "PAT": "Patient",
        "DIS": "Disease",
        "CT": "ClinicalTrial",
        "AE": "AdverseEvent",
        "POL": "DataGovernancePolicy",
    }
    for prefix, label in prefixes.items():
        if token.startswith(prefix):
            return label if label in FACT_KEYS else None
    if re.fullmatch(r"D\d{3}", token):
        return "Drug"
    if re.fullmatch(r"B\d{3}", token):
        return "Biomarker"
    if re.fullmatch(r"G\d{3}", token):
        return "Gene"
    if re.fullmatch(r"P\d{3}", token):
        return "Protein"
    return None
