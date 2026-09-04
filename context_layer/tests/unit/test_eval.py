"""Evaluation harness correctness. Requirement 11.

The harness measures the system it is testing, so its own defects matter. Two
real bugs are covered here: circular groundedness (scoring an answer against a
package built from that answer) and mis-scoring a substantive denial as a
confabulation, both of which flattered the context layer.
"""

from __future__ import annotations

import pytest

from context_layer.eval.adversarial import SCENARIOS
from context_layer.eval.harness import (
    ArmSummary,
    Result,
    Harness,
    is_substantive_refusal,
)
from context_layer.eval.questions import QUESTIONS, by_category
from context_layer.governance import grounding
from context_layer.types import ContextElement, ContextPackage, Provenance, utcnow

PROV = Provenance(source="t", ingested_at=utcnow())


def pkg(*contents, question="q"):
    return ContextPackage(
        context_package_id="c",
        question=question,
        elements=tuple(
            ContextElement(element_id=f"e{i}", kind="chunk", content=c, provenance=PROV)
            for i, c in enumerate(contents)
        ),
    )


# ── question set integrity ────────────────────────────────────────────────────


def test_question_set_covers_every_expected_behaviour():
    assert {q.expected for q in QUESTIONS} == {"answer", "refuse", "block"}


def test_question_set_has_multiple_categories():
    cats = by_category()
    assert len(cats) >= 5
    assert "out_of_corpus" in cats
    assert "consent_governance" in cats


def test_block_questions_declare_the_rule_they_expect():
    for q in QUESTIONS:
        if q.expected == "block":
            assert q.expected_rules, f"{q.question!r} expects a block but names no rule"


def test_questions_are_unique():
    texts = [q.question for q in QUESTIONS]
    assert len(texts) == len(set(texts))


# ── substantive refusal detection ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Based on the context provided, there is no information available.",
        "The context does not contain data about that.",
        "I cannot determine the answer from the supplied context.",
        "No Phase 4 cardiology trial is mentioned.",
        "Insufficient grounding to answer.",
        "",
        "   ",
    ],
)
def test_denials_are_recognised_as_refusals(text):
    assert is_substantive_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Pembrolizumab is a PD-1 inhibitor treating melanoma.",
        "The market share of the veterinary division in Brazil is 14 percent.",
        "Three hub tables are fully compliant with the DQ policy.",
    ],
)
def test_assertions_are_not_refusals(text):
    assert not is_substantive_refusal(text)


def test_denial_quoting_forbidden_terms_is_not_a_fabrication():
    """The bug: 'no veterinary division in Brazil' was scored as inventing both."""
    from context_layer.eval.harness import _forbidden

    denial = "There is no information about a veterinary division in Brazil."
    assert _forbidden(denial, ("Brazil", "veterinary")) == ()


def test_affirmative_use_of_forbidden_terms_is_flagged():
    from context_layer.eval.harness import _forbidden

    claim = "Our veterinary division holds 14 percent share in Brazil."
    assert set(_forbidden(claim, ("Brazil", "veterinary"))) == {"Brazil", "veterinary"}


# ── groundedness must not be circular ─────────────────────────────────────────


def test_groundedness_against_own_answer_would_be_meaningless():
    """Documents why the baseline arm must score against retrieved context."""
    answer = "Pembrolizumab is a PD-1 inhibitor treating melanoma."
    circular = grounding.overlap_ratio(answer, pkg(answer))
    assert circular == 1.0, "self-comparison saturates, hence the earlier bug"


def test_groundedness_against_real_context_discriminates():
    answer = "Pembrolizumab is a PD-1 inhibitor treating melanoma."
    supported = grounding.overlap_ratio(answer, pkg("Pembrolizumab inhibits PD-1 in melanoma."))
    unsupported = grounding.overlap_ratio(answer, pkg("Quarterly parking allocation trends."))
    assert supported > unsupported
    assert unsupported < 0.3


def test_groundedness_is_zero_without_context():
    assert grounding.overlap_ratio("anything", pkg()) == 0.0


# ── summary arithmetic ────────────────────────────────────────────────────────


def result(**kw):
    base = dict(
        question="q", category="grounded_retrieval", expected="answer",
        arm="context_layer", outcome="answer", behaviour_correct=True,
        groundedness=0.8, has_citation=True,
    )
    base.update(kw)
    return Result(**base)


def test_summary_counts_behaviour_and_citations():
    s = Harness._summarise("context_layer", [result(), result(behaviour_correct=False)])
    r = s.report()
    assert r["questions"] == 2
    assert r["behaviour_correct"] == 0.5
    assert r["citation_coverage"] == 1.0


def test_rule_compliance_only_counts_block_cases():
    rows = [
        result(expected="block", outcome="block"),
        result(expected="block", outcome="answer", behaviour_correct=False),
        result(expected="answer", outcome="answer"),
    ]
    r = Harness._summarise("context_layer", rows).report()
    assert r["rule_compliance"] == 0.5


def test_confabulation_only_counts_out_of_corpus():
    rows = [
        result(category="out_of_corpus", expected="refuse", outcome="answer"),
        result(category="out_of_corpus", expected="refuse", outcome="refuse"),
        result(category="grounded_retrieval", outcome="answer"),
    ]
    r = Harness._summarise("naive_rag", rows).report()
    assert r["confabulation_rate"] == 0.5


def test_ratios_are_none_when_denominator_is_zero():
    r = Harness._summarise("naive_rag", [result()]).report()
    assert r["rule_compliance"] is None       # no block cases present
    assert r["confabulation_rate"] is None    # no out_of_corpus cases


def test_errors_are_counted():
    r = Harness._summarise("x", [result(error="boom", behaviour_correct=False)]).report()
    assert r["errors"] == 1


# ── adversarial generators ────────────────────────────────────────────────────


def test_every_scenario_injects_and_labels_itself():
    base = pkg("Real evidence about PD-1 inhibitors.")
    for name, generator in SCENARIOS.items():
        poisoned = generator(base)
        assert len(poisoned.elements) > len(base.elements), name
        assert any(f"adversarial:{name}" == d for d in poisoned.degradations), name


def test_scenarios_preserve_the_original_evidence():
    base = pkg("Real evidence about PD-1 inhibitors.")
    original = {e.content for e in base.elements}
    for name, generator in SCENARIOS.items():
        contents = {e.content for e in generator(base).elements}
        assert original <= contents, f"{name} discarded real evidence"


def test_scenarios_preserve_the_question():
    base = pkg("evidence", question="the original question")
    for generator in SCENARIOS.values():
        assert generator(base).question == "the original question"


def test_poisoned_packages_get_a_fresh_id():
    base = pkg("evidence")
    for generator in SCENARIOS.values():
        assert generator(base).context_package_id != base.context_package_id


def test_signal_drowning_adds_substantial_noise():
    base = pkg("evidence")
    assert len(SCENARIOS["signal_drowning"](base).elements) >= 12


def test_prompt_injection_content_is_present_but_inert():
    """The injection text must reach the package: the point is that rules ignore it."""
    poisoned = SCENARIOS["prompt_injection"](pkg("evidence"))
    assert any("SYSTEM OVERRIDE" in e.content for e in poisoned.elements)
