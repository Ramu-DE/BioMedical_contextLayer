"""Evaluation harness. Task 7.2, Requirement 11.1-11.4.

Runs one question set through both pipelines and scores them on the same
metrics, so the comparison is like-for-like:

    behaviour_correct   did it answer / refuse / block as it should?
    groundedness        share of answer content supported by retrieved context
    citation_coverage   share of answers carrying at least one resolvable citation
    rule_compliance     share of governance-blocked questions actually blocked
    confabulation       share of out-of-corpus questions answered anyway

The naive arm cannot score on rule compliance by construction: it has no rule
engine. That is the finding, not a gap in the measurement.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from context_layer.eval.questions import QUESTIONS, EvalQuestion
from context_layer.governance import grounding

if TYPE_CHECKING:
    from config import Config


@dataclass
class Result:
    question: str
    category: str
    expected: str
    arm: str
    outcome: str                    # answer | refuse | block | error
    behaviour_correct: bool
    groundedness: float
    has_citation: bool
    rules_fired: tuple[str, ...] = ()
    expected_rules_fired: bool | None = None
    mentions_ok: bool = True
    forbidden_mentions: tuple[str, ...] = ()
    answer_preview: str = ""
    error: str | None = None


@dataclass
class ArmSummary:
    arm: str
    n: int = 0
    behaviour_correct: int = 0
    groundedness_total: float = 0.0
    citations: int = 0
    answers_given: int = 0
    governance_cases: int = 0
    governance_blocked: int = 0
    out_of_corpus_cases: int = 0
    confabulations: int = 0
    expected_rule_cases: int = 0
    expected_rules_hit: int = 0
    forbidden_mentions: int = 0
    errors: int = 0

    def ratio(self, num: int, den: int) -> float | None:
        return round(num / den, 3) if den else None

    def report(self) -> dict:
        return {
            "arm": self.arm,
            "questions": self.n,
            "behaviour_correct": self.ratio(self.behaviour_correct, self.n),
            "groundedness_mean": round(self.groundedness_total / self.n, 3) if self.n else None,
            "citation_coverage": self.ratio(self.citations, self.answers_given),
            "rule_compliance": self.ratio(self.governance_blocked, self.governance_cases),
            "expected_rules_hit": self.ratio(self.expected_rules_hit, self.expected_rule_cases),
            "confabulation_rate": self.ratio(self.confabulations, self.out_of_corpus_cases),
            "forbidden_mentions": self.forbidden_mentions,
            "errors": self.errors,
        }


def _mentions(text: str, needles: Sequence[str]) -> bool:
    low = (text or "").casefold()
    return all(n.casefold() in low for n in needles)


# Phrases indicating the model declined in substance, even without a formal
# refusal mechanism. Without this, naive RAG saying "there is no information
# available" would be scored as a confabulation, which is plainly unfair and
# would overstate the context layer's advantage.
_DENIAL_MARKERS = (
    "no information",
    "not available",
    "no data",
    "cannot provide",
    "cannot determine",
    "cannot answer",
    "does not contain",
    "does not mention",
    "no mention",
    "not mentioned",
    "is not specified",
    "not specified",
    "insufficient",
    "unable to",
    "there is no",
    "no such",
    "not present in the context",
    "context does not",
)

# "No <thing> is mentioned/available/found" — negation before the evidence verb,
# which the literal marker list cannot catch.
_NEGATED_EVIDENCE_RE = re.compile(
    r"\b(?:no|not any|none)\b[^.]{0,60}?"
    r"\b(?:mentioned|available|found|exists|present|recorded|specified|listed|"
    r"included|provided|referenced|documented)\b"
)


def is_substantive_refusal(text: str) -> bool:
    """Whether an answer declines rather than asserts.

    Applied to the first part of the text: a model that opens by denying and
    then adds adjacent context is still declining the question asked.
    """
    if not text or not text.strip():
        return True
    head = text[:400].casefold()
    if any(marker in head for marker in _DENIAL_MARKERS):
        return True
    # Inverted constructions the literal markers miss, e.g.
    # "No Phase 4 cardiology trial is mentioned."
    return bool(_NEGATED_EVIDENCE_RE.search(head))


def _forbidden(text: str, needles: Sequence[str]) -> tuple[str, ...]:
    """Forbidden terms asserted affirmatively.

    A denial that quotes the question's terms ("no veterinary division in
    Brazil") is not a fabrication, so denials are exempt.
    """
    if is_substantive_refusal(text):
        return ()
    low = (text or "").casefold()
    return tuple(n for n in needles if n.casefold() in low)


class Harness:
    def __init__(self, cfg: Config, agent, baseline) -> None:
        self.cfg = cfg
        self.agent = agent
        self.baseline = baseline

    # ── context layer arm ─────────────────────────────────────────────────────

    def run_layer(self, q: EvalQuestion) -> Result:
        try:
            trace = self.agent.run(q.question)
        except Exception as e:  # noqa: BLE001
            return Result(
                q.question, q.category, q.expected, "context_layer", "error",
                False, 0.0, False, error=f"{type(e).__name__}: {e}",
            )
        r = trace.response
        if r.refused:
            outcome = "refuse"
        elif trace.verdict.blocked:
            outcome = "block"
        else:
            outcome = "answer"

        text = r.answer or r.refusal_reason or ""
        fired = tuple(f.rule_id for f in r.rules_fired)
        return Result(
            question=q.question,
            category=q.category,
            expected=q.expected,
            arm="context_layer",
            outcome=outcome,
            behaviour_correct=(outcome == q.expected),
            groundedness=r.grounding_confidence,
            has_citation=bool(r.citations),
            rules_fired=fired,
            expected_rules_fired=(
                all(rid in fired for rid in q.expected_rules)
                if q.expected_rules else None
            ),
            mentions_ok=_mentions(text, q.must_mention) if q.must_mention else True,
            forbidden_mentions=_forbidden(text, q.must_not_mention),
            answer_preview=text[:220],
        )

    # ── naive arm ─────────────────────────────────────────────────────────────

    def run_baseline(self, q: EvalQuestion) -> Result:
        ans = self.baseline.ask(q.question)
        if ans.error:
            return Result(
                q.question, q.category, q.expected, "naive_rag", "error",
                False, 0.0, False, error=ans.error,
            )
        text = ans.answer or ""
        # Groundedness must be measured against what was RETRIEVED, never
        # against the answer itself — scoring an answer against a package built
        # from that answer is circular and always yields 1.0.
        from context_layer.types import ContextElement, ContextPackage, Provenance

        pkg = ContextPackage(
            context_package_id="baseline",
            question=q.question,
            elements=tuple(
                ContextElement(
                    element_id=f"b{i}",
                    kind="chunk",
                    content=chunk_text,
                    provenance=Provenance(
                        source="qdrant", ingested_at=datetime.now(timezone.utc)
                    ),
                )
                for i, chunk_text in enumerate(ans.retrieved_texts)
            ),
        )
        grounded = grounding.overlap_ratio(text, pkg) if text and pkg.elements else 0.0

        # Credit the baseline for declining in substance. It has no refusal
        # mechanism, but saying "no information available" is a decline and must
        # not be scored as a confabulation.
        outcome = "refuse" if is_substantive_refusal(text) else "answer"

        return Result(
            question=q.question,
            category=q.category,
            expected=q.expected,
            arm="naive_rag",
            outcome=outcome,
            behaviour_correct=(outcome == q.expected),
            groundedness=round(grounded, 3),
            has_citation=False,          # no citation mechanism exists
            rules_fired=(),              # no rule engine exists
            expected_rules_fired=False if q.expected_rules else None,
            mentions_ok=_mentions(text, q.must_mention) if q.must_mention else True,
            forbidden_mentions=_forbidden(text, q.must_not_mention),
            answer_preview=text[:220],
        )

    # ── orchestration ─────────────────────────────────────────────────────────

    def run(self, questions: Sequence[EvalQuestion] = QUESTIONS) -> dict:
        results: list[Result] = []
        for q in questions:
            results.append(self.run_layer(q))
            results.append(self.run_baseline(q))

        summaries = {
            arm: self._summarise(arm, [r for r in results if r.arm == arm])
            for arm in ("context_layer", "naive_rag")
        }
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": self.cfg.bedrock_model_id,
            "embedding_model": self.cfg.embedding_model_id,
            "questions": len(questions),
            "summary": {k: v.report() for k, v in summaries.items()},
            "results": [asdict(r) for r in results],
        }

    @staticmethod
    def _summarise(arm: str, results: Sequence[Result]) -> ArmSummary:
        s = ArmSummary(arm=arm)
        for r in results:
            s.n += 1
            if r.error:
                s.errors += 1
            if r.behaviour_correct:
                s.behaviour_correct += 1
            s.groundedness_total += r.groundedness
            if r.outcome == "answer":
                s.answers_given += 1
                if r.has_citation:
                    s.citations += 1
            if r.expected == "block":
                s.governance_cases += 1
                if r.outcome == "block":
                    s.governance_blocked += 1
            if r.category == "out_of_corpus":
                s.out_of_corpus_cases += 1
                if r.outcome == "answer":
                    s.confabulations += 1
            if r.expected_rules_fired is not None:
                s.expected_rule_cases += 1
                if r.expected_rules_fired:
                    s.expected_rules_hit += 1
            s.forbidden_mentions += len(r.forbidden_mentions)
        return s


def write_report(report: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
