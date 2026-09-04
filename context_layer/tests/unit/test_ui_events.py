"""Pipeline stage events, which the real-time UI depends on.

The callback is advisory: it must never be able to break an answer. That is the
property most worth testing, because a UI bug should not become a governance bug.
"""

from __future__ import annotations

import pytest

from context_layer.agent.pipeline import ContextLayerAgent
from context_layer.eval.adversarial import SCENARIOS
from context_layer.rules.models import Fact, FactSet
from context_layer.types import (
    ContextElement,
    ContextPackage,
    FiredRule,
    GovernedResponse,
    Provenance,
    RuleVerdict,
    utcnow,
)

EXPECTED_ORDER = ["start", "assemble", "facts", "neural", "symbolic", "gate", "record"]


class StubAssembler:
    def __init__(self, package, report_sources=True):
        self.package = package
        self.report_sources = report_sources

    def assemble(self, question, k=6, on_retriever=None):
        if on_retriever is not None:
            on_retriever("vector", 2, 12.0, [])
            on_retriever("graph", 1, 8.0, [])
            on_retriever("runtime", 0, 0.0, ["runtime_retriever_unavailable: none"])
        return self.package

    def close(self):
        pass


class LegacyAssembler:
    """An assembler predating per-source reporting. Must still work."""

    def __init__(self, package):
        self.package = package

    def assemble(self, question, k=6):
        return self.package

    def close(self):
        pass


class StubFactBuilder:
    def for_entities(self, wanted):
        return FactSet(
            facts=tuple(
                Fact(entity_id=i, entity_type=label)
                for label, ids in wanted.items() for i in ids
            )
        )

    def for_type(self, label, limit=500):
        return FactSet()

    def for_curies(self, curies, limit=200):
        return FactSet()

    def close(self):
        pass


class StubLLM:
    def __init__(self, text="Pembrolizumab inhibits PD-1 [e1]."):
        self.text = text

    def invoke(self, prompt):
        class R:
            content = self.text
        return R()


def package(question="Can I include patient PAT015?"):
    """Non-empty: propose() short-circuits on an empty package, so an empty
    fixture would silently skip the streaming path entirely."""
    return ContextPackage(
        context_package_id="ctx_ui",
        question=question,
        elements=(
            ContextElement(
                element_id="e1",
                kind="chunk",
                content="Pembrolizumab is a PD-1 inhibitor treating melanoma.",
                provenance=Provenance(source="test", ingested_at=utcnow()),
                score=0.9,
                curies=("RXNORM:1657993",),
            ),
        ),
    )


def agent(cfg, llm=None, assembler=None):
    return ContextLayerAgent(
        cfg,
        llm=llm or StubLLM(),
        assembler=assembler or StubAssembler(package()),
        fact_builder=StubFactBuilder(),
        store=None,
        write_back=False,
    )


class StreamingLLM:
    """Emits deltas so token streaming can be tested without Bedrock."""

    def __init__(self, parts=("Pembro", "lizumab ", "inhibits PD-1 [e1].")):
        self.parts = parts

    def stream(self, prompt):
        for part in self.parts:
            class C:
                content = part
            yield C()

    def invoke(self, prompt):
        class R:
            content = "".join(self.parts)
        return R()


# ── stage ordering ────────────────────────────────────────────────────────────


def test_stages_emit_in_order(cfg):
    seen = []
    agent(cfg).run("Can I include patient PAT015?", on_stage=lambda n, p: seen.append(n))
    assert seen == EXPECTED_ORDER


def test_run_without_callback_still_works(cfg):
    trace = agent(cfg).run("Can I include patient PAT015?")
    assert trace.response is not None


def test_each_stage_carries_the_payload_the_ui_needs(cfg):
    payloads = {}
    agent(cfg).run("q", on_stage=lambda n, p: payloads.__setitem__(n, p))
    assert {"package", "ms", "elements", "tokens", "degradations", "priors"} <= set(
        payloads["assemble"]
    )
    assert {"facts", "ms", "count", "types", "in_scope"} <= set(payloads["facts"])
    assert {"proposal", "ms"} <= set(payloads["neural"])
    assert {"verdict", "ms", "fired", "blocked", "evaluated"} <= set(payloads["symbolic"])
    assert {"response", "branch"} <= set(payloads["gate"])
    assert "decision_id" in payloads["record"]


def test_symbolic_stage_reports_rules_evaluated(cfg):
    payloads = {}
    a = agent(cfg)
    a.run("q", on_stage=lambda n, p: payloads.__setitem__(n, p))
    assert payloads["symbolic"]["evaluated"] == len(a.rules) >= 13


# ── callback isolation: a UI fault must not break an answer ────────────────────


def test_exploding_callback_does_not_break_the_run(cfg, capsys):
    def boom(name, payload):
        raise RuntimeError("UI exploded")

    trace = agent(cfg).run("Can I include patient PAT015?", on_stage=boom)
    assert trace.response is not None
    assert "stage callback failed" in capsys.readouterr().out


def test_exploding_callback_does_not_change_the_verdict(cfg):
    clean = agent(cfg).run("Can I include patient PAT015?")
    noisy = agent(cfg).run(
        "Can I include patient PAT015?",
        on_stage=lambda n, p: (_ for _ in ()).throw(ValueError("boom")),
    )
    assert clean.verdict.blocked == noisy.verdict.blocked
    assert clean.response.symbolic_verdict == noisy.response.symbolic_verdict


# ── gate branch labelling, used by the UI ─────────────────────────────────────


def test_branch_released():
    resp = GovernedResponse(
        context_package_id="c", symbolic_verdict="no rules fired",
        grounding_confidence=0.9, answer="ok",
    )
    assert ContextLayerAgent._gate_branch(resp, RuleVerdict()) == "released"


def test_branch_symbolic_block():
    verdict = RuleVerdict(fired=(FiredRule("r", "block", "why"),))
    resp = GovernedResponse(
        context_package_id="c", symbolic_verdict="r(block)",
        grounding_confidence=0.6, answer="Blocked by policy. why",
    )
    assert ContextLayerAgent._gate_branch(resp, verdict) == "symbolic_block_override"


def test_branch_clinical_guard():
    resp = GovernedResponse.refusal(
        context_package_id="c",
        reason="This asks for individual treatment guidance, outside this system's scope.",
    )
    assert ContextLayerAgent._gate_branch(resp, RuleVerdict()) == "clinical_advice_guard"


def test_branch_grounding_threshold():
    resp = GovernedResponse.refusal(
        context_package_id="c", reason="Insufficient grounding (0.20 < 0.60): nothing found."
    )
    assert ContextLayerAgent._gate_branch(resp, RuleVerdict()) == "grounding_threshold"


def test_every_branch_has_a_ui_label():
    from context_layer.ui.app import BRANCH_LABEL

    for branch in (
        "released", "symbolic_block_override", "clinical_advice_guard",
        "grounding_threshold",
    ):
        assert branch in BRANCH_LABEL
        label, colour = BRANCH_LABEL[branch]
        assert label and colour.startswith("#")


def test_ui_stage_list_matches_pipeline_events():
    from context_layer.ui.app import STAGES

    ui_keys = [key for key, _, _, _ in STAGES]
    assert ui_keys == [s for s in EXPECTED_ORDER if s != "start"]


def test_stage_labels_name_the_two_halves_explicitly():
    """The neural/symbolic split is the core claim; the labels must say so."""
    from context_layer.ui.app import STAGES

    subtitles = {key: sub for key, _, sub, _ in STAGES}
    assert subtitles["neural"] == "pattern recognition"
    assert subtitles["symbolic"] == "rule following"
    assert subtitles["record"] == "context"


def test_every_stage_has_a_title_and_subtitle():
    from context_layer.ui.app import STAGES

    for key, title, subtitle, colour in STAGES:
        assert title and subtitle and colour.startswith("#"), key


def test_ui_exposes_every_adversarial_scenario():
    from context_layer.ui.app import SCENARIOS as ui_scenarios

    assert set(ui_scenarios) == set(SCENARIOS)


# ── per-retriever events, used by the UI's retrieval panel ────────────────────


def test_retriever_events_are_reported(cfg):
    seen = []
    agent(cfg).run("q", on_retriever=lambda n, c, ms, d: seen.append((n, c)))
    assert [n for n, _ in seen] == ["vector", "graph", "runtime"]


def test_legacy_assembler_without_the_kwarg_still_runs(cfg):
    """A pluggable assembler predating on_retriever must not break the pipeline."""
    a = agent(cfg, assembler=LegacyAssembler(package()))
    trace = a.run("q", on_retriever=lambda *args: None)
    assert trace.response is not None


# ── token streaming ───────────────────────────────────────────────────────────


def test_tokens_stream_and_accumulate(cfg):
    deltas, finals = [], []

    def on_token(delta, accumulated):
        deltas.append(delta)
        finals.append(accumulated)

    a = agent(cfg, llm=StreamingLLM())
    trace = a.run("q", on_token=on_token)
    assert len(deltas) == 3
    assert finals[-1] == "Pembrolizumab inhibits PD-1 [e1]."
    assert trace.response is not None


def test_first_token_latency_is_measured(cfg):
    payloads = {}
    agent(cfg, llm=StreamingLLM()).run(
        "q", on_stage=lambda n, p: payloads.__setitem__(n, p), on_token=lambda d, a: None
    )
    assert payloads["neural"]["streamed"] is True
    assert payloads["neural"]["first_token_ms"] is not None


def test_no_streaming_without_a_token_callback(cfg):
    payloads = {}
    agent(cfg, llm=StreamingLLM()).run(
        "q", on_stage=lambda n, p: payloads.__setitem__(n, p)
    )
    assert payloads["neural"]["streamed"] is False


def test_exploding_token_callback_does_not_break_the_run(cfg, capsys):
    def boom(delta, accumulated):
        raise RuntimeError("UI exploded mid-stream")

    trace = agent(cfg, llm=StreamingLLM()).run("q", on_token=boom)
    assert trace.response is not None
    assert "token callback failed" in capsys.readouterr().out


def test_streaming_does_not_change_the_verdict(cfg):
    plain = agent(cfg, llm=StreamingLLM()).run("Can I include patient PAT015?")
    streamed = agent(cfg, llm=StreamingLLM()).run(
        "Can I include patient PAT015?", on_token=lambda d, a: None
    )
    assert plain.verdict.blocked == streamed.verdict.blocked


# ── the injection dropdown must explain itself ────────────────────────────────


def test_every_scenario_has_help_text():
    """An unlabelled dropdown of hostile scenarios is worse than none."""
    from context_layer.ui.app import SCENARIO_HELP

    expected = set(SCENARIOS) | {"none"}
    assert set(SCENARIO_HELP) == expected, (
        f"missing help: {expected - set(SCENARIO_HELP)}"
    )
    for key, (title, detail) in SCENARIO_HELP.items():
        assert title and len(detail) > 40, f"{key} help is too thin"


def test_help_states_what_should_happen():
    """Help should say the expected outcome, not just describe the attack."""
    from context_layer.ui.app import SCENARIO_HELP

    outcomes = " ".join(d for _, d in SCENARIO_HELP.values()).lower()
    assert "should still block" in outcomes or "should not" in outcomes
