"""Governance: grounding, guards, and gate precedence.

Requirements 9.2, 9.3, 10.1-10.4. The precedence tests are the important ones:
they prove a deterministic rule can veto fluent model output.
"""

from __future__ import annotations

import pytest

from context_layer.governance import grounding
from context_layer.governance.gate import GovernanceGate, check_clinical_advice
from context_layer.types import (
    ContextElement,
    ContextPackage,
    FiredRule,
    Provenance,
    RuleVerdict,
    utcnow,
)

PROV = Provenance(source="neo4j:Drug", ingested_at=utcnow(), confidence=1.0)


def pkg(elements=(), degradations=(), truncated=()):
    return ContextPackage(
        context_package_id="ctx_test",
        question="q",
        elements=tuple(elements),
        degradations=tuple(degradations),
        truncated=tuple(truncated),
    )


def el(eid="e1", content="Pembrolizumab is a PD-1 inhibitor treating melanoma.",
       kind="chunk", curies=("RXNORM:1657993",)):
    return ContextElement(
        element_id=eid, kind=kind, content=content, provenance=PROV,
        score=0.8, curies=curies,
    )


# ── citations, R10.4 ──────────────────────────────────────────────────────────


def test_extracts_valid_citation_markers():
    p = pkg([el("e1"), el("e2", "Nivolumab also inhibits PD-1.")])
    cites = grounding.extract_citations("Answer [e1] and more [e2].", p)
    assert len(cites) == 2


def test_ignores_unknown_citation_markers():
    assert grounding.extract_citations("Answer [nope].", pkg([el("e1")])) == ()


def test_detects_fabricated_citations():
    assert grounding.fabricated_citations("see [ghost]", pkg([el("e1")])) == ("ghost",)


def test_duplicate_markers_cite_once():
    cites = grounding.extract_citations("[e1] and again [e1]", pkg([el("e1")]))
    assert len(cites) == 1


# ── grounding score, R10.2 ────────────────────────────────────────────────────


def test_empty_package_scores_zero():
    assert grounding.score("anything", pkg()) == 0.0


def test_empty_answer_scores_zero():
    assert grounding.score("", pkg([el()])) == 0.0


def test_well_cited_overlapping_answer_scores_high():
    p = pkg([el()])
    s = grounding.score(
        "Pembrolizumab is a PD-1 inhibitor treating melanoma [e1].", p
    )
    assert s > 0.6


def test_uncited_unrelated_answer_scores_low():
    s = grounding.score("Quarterly revenue in Brazil rose sharply.", pkg([el()]))
    assert s < 0.3


def test_fabricated_citation_is_penalised():
    p = pkg([el()])
    honest = grounding.score("Pembrolizumab inhibits PD-1 [e1].", p)
    faked = grounding.score("Pembrolizumab inhibits PD-1 [e1] [ghost].", p)
    assert faked < honest


def test_degradation_lowers_confidence():
    p_ok = pkg([el()])
    p_bad = pkg([el()], degradations=("graph_retriever_unavailable: x",))
    answer = "Pembrolizumab is a PD-1 inhibitor treating melanoma [e1]."
    assert grounding.score(answer, p_bad) < grounding.score(answer, p_ok)


def test_runtime_degradation_is_not_penalised():
    """The decision store is expected to be absent early on."""
    p_ok = pkg([el()])
    p_rt = pkg([el()], degradations=("runtime_retriever_unavailable: none",))
    answer = "Pembrolizumab is a PD-1 inhibitor treating melanoma [e1]."
    assert grounding.score(answer, p_rt) == grounding.score(answer, p_ok)


def test_missing_context_note_names_the_gap():
    note = grounding.missing_context_note(pkg())
    assert "no context was retrieved" in note
    note2 = grounding.missing_context_note(pkg([el(kind="chunk")]))
    assert "no graph evidence" in note2


def test_missing_context_note_reports_truncation():
    note = grounding.missing_context_note(pkg([el()], truncated=("dropped1",)))
    assert "dropped by token budget" in note


# ── clinical advice guard, R10.3 ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "What dose of pembrolizumab should I give my patient?",
        "Should I take nivolumab for my melanoma?",
        "Is it safe for me to start this drug?",
        "How much should the patient take daily?",
        "What dosage should I use for him?",
    ],
)
def test_guard_blocks_individual_treatment_requests(question):
    assert check_clinical_advice(question).triggered


@pytest.mark.parametrize(
    "question",
    [
        "Which drugs target PD-1 and what diseases do they treat?",
        "What is the mechanism of pembrolizumab?",
        "Which trials studied NSCLC and what adverse events were reported?",
        "What policy applies to hub layer tables?",
        "Which biomarker predicts response to immunotherapy?",
        "What was the enrollment in trial CT001?",
    ],
)
def test_guard_allows_research_questions(question):
    """False positives here would make the system useless for its actual job."""
    assert not check_clinical_advice(question).triggered


def test_guard_can_be_disabled():
    q = "What dose should I give my patient?"
    assert check_clinical_advice(q, enabled=True).triggered
    assert not check_clinical_advice(q, enabled=False).triggered


# ── gate precedence, R9.2 ─────────────────────────────────────────────────────


BLOCK = RuleVerdict(fired=(FiredRule("r.block", "block", "Consent withdrawn.",
                                     ("PAT015",), "bio:Policy/POL001"),))
WARN = RuleVerdict(fired=(FiredRule("r.warn", "warn", "Severe adverse event present.",),))
CLEAN = RuleVerdict()

GOOD_ANSWER = "Pembrolizumab is a PD-1 inhibitor treating melanoma [e1]."


def test_advice_guard_outranks_everything(cfg):
    r = GovernanceGate(cfg).evaluate(
        "What dose should I give my patient?", pkg([el()]), BLOCK, GOOD_ANSWER
    )
    assert r.refused and "clinical decision support" in r.refusal_reason


def test_block_verdict_overrides_a_confident_answer(cfg):
    """The core neurosymbolic property."""
    r = GovernanceGate(cfg).evaluate("q", pkg([el()]), BLOCK, GOOD_ANSWER)
    assert not r.refused
    assert r.answer.startswith("Blocked by policy")
    assert "Consent withdrawn" in r.answer
    # R9.3: the overridden proposal stays visible
    assert r.neural_proposal == GOOD_ANSWER
    assert r.answer != r.neural_proposal
    assert r.rules_fired[0].policy_iri == "bio:Policy/POL001"


def test_low_grounding_refuses_rather_than_guessing(cfg):
    r = GovernanceGate(cfg).evaluate(
        "q", pkg([el()]), CLEAN, "Revenue in Brazil grew by 12 percent."
    )
    assert r.refused
    assert "Insufficient grounding" in r.refusal_reason


def test_absent_proposal_refuses(cfg):
    r = GovernanceGate(cfg).evaluate("q", pkg([el()]), CLEAN, None)
    assert r.refused


def test_grounded_answer_is_released_with_citations(cfg):
    r = GovernanceGate(cfg).evaluate("q", pkg([el()]), CLEAN, GOOD_ANSWER)
    assert not r.refused
    assert r.answer == GOOD_ANSWER
    assert r.citations
    assert r.symbolic_verdict == "no rules fired"


def test_warnings_are_appended_not_blocking(cfg):
    r = GovernanceGate(cfg).evaluate("q", pkg([el()]), WARN, GOOD_ANSWER)
    assert not r.refused
    assert "Advisories:" in r.answer
    assert "Severe adverse event" in r.answer


def test_response_always_carries_the_package_id(cfg):
    for verdict, proposal in ((BLOCK, GOOD_ANSWER), (CLEAN, None), (WARN, GOOD_ANSWER)):
        r = GovernanceGate(cfg).evaluate("q", pkg([el()]), verdict, proposal)
        assert r.context_package_id == "ctx_test"


def test_degradations_propagate_to_the_response(cfg):
    p = pkg([el()], degradations=("graph_retriever_unavailable: x",))
    r = GovernanceGate(cfg).evaluate("q", p, CLEAN, GOOD_ANSWER)
    assert "graph_retriever_unavailable: x" in r.degradations
