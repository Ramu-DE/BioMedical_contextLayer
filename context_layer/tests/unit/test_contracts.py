"""Contract invariants from the requirements spec.

These tests do not need credentials: they verify that the types themselves
refuse to represent an invalid state.
"""

from __future__ import annotations

import pytest

from context_layer.types import (
    Citation,
    ContextPackage,
    Decision,
    EntityMention,
    FiredRule,
    GovernedResponse,
    RuleVerdict,
    utcnow,
)


# ── R4.4: unlinked mentions are explicit, not guessed ─────────────────────────


def test_unlinked_mention_has_no_curie():
    m = EntityMention(
        chunk_id="c1", surface_form="unknown compound", curie=None,
        confidence=0.31, start=0, end=17,
    )
    assert not m.is_linked
    assert m.curie is None


def test_linked_mention_reports_linked():
    m = EntityMention(
        chunk_id="c1", surface_form="aspirin", curie="CHEBI:15365",
        confidence=0.94, start=0, end=7,
    )
    assert m.is_linked


# ── R10.4: a citation must resolve to something ───────────────────────────────


def test_citation_requires_a_resolvable_target():
    with pytest.raises(ValueError, match="chunk_id or a curie"):
        Citation(source="somewhere")


def test_citation_accepts_curie_only():
    assert Citation(curie="CHEBI:15365").curie == "CHEBI:15365"


# ── R3.2-3.4: rule verdicts are explicit about blocking ───────────────────────


def test_empty_verdict_does_not_block():
    v = RuleVerdict()
    assert not v.blocked
    assert v.block_rationale is None
    assert v.summary() == "no rules fired"


def test_block_severity_blocks_and_exposes_rationale():
    v = RuleVerdict(
        fired=(
            FiredRule("r.warn", "warn", "be careful"),
            FiredRule("r.block", "block", "contraindicated in renal impairment"),
        )
    )
    assert v.blocked
    assert v.block_rationale == "contraindicated in renal impairment"
    assert "r.block(block)" in v.summary()


def test_warn_only_verdict_does_not_block():
    v = RuleVerdict(fired=(FiredRule("r.warn", "warn", "monitor closely"),))
    assert not v.blocked


# ── R8: package accounting ────────────────────────────────────────────────────


def test_package_reports_token_count_and_curies(package):
    assert package.token_count > 0
    assert set(package.curies) == {"CHEBI:15365", "HGNC:9604"}
    assert len(package.by_kind("chunk")) == 1
    assert len(package.by_kind("graph_node")) == 1
    assert not package.is_empty


def test_empty_package_is_empty():
    p = ContextPackage(context_package_id="ctx_0", question="q")
    assert p.is_empty
    assert p.token_count == 0
    assert p.curies == ()


def test_package_records_degradation(element):
    p = ContextPackage(
        context_package_id="ctx_1",
        question="q",
        elements=(element(),),
        degradations=("graph_retriever_unavailable",),
    )
    assert "graph_retriever_unavailable" in p.degradations


# ── R10.1-10.2: the response contract cannot lie ──────────────────────────────


def test_refusal_must_not_carry_an_answer():
    with pytest.raises(ValueError, match="must not carry an answer"):
        GovernedResponse(
            context_package_id="ctx_1",
            symbolic_verdict="no rules fired",
            grounding_confidence=0.2,
            answer="here is a guess",
            refused=True,
            refusal_reason="insufficient grounding",
        )


def test_refusal_must_state_a_reason():
    with pytest.raises(ValueError, match="must state a reason"):
        GovernedResponse(
            context_package_id="ctx_1",
            symbolic_verdict="no rules fired",
            grounding_confidence=0.2,
            refused=True,
        )


def test_non_refusal_must_carry_an_answer():
    with pytest.raises(ValueError, match="must carry an answer"):
        GovernedResponse(
            context_package_id="ctx_1",
            symbolic_verdict="no rules fired",
            grounding_confidence=0.9,
        )


def test_refusal_helper_builds_a_valid_refusal():
    r = GovernedResponse.refusal(
        context_package_id="ctx_1",
        reason="grounding 0.41 below threshold 0.6; no graph evidence for COX-1",
    )
    assert r.refused and r.answer is None
    assert "below threshold" in r.refusal_reason


def test_response_exposes_both_neural_and_symbolic(package):
    r = GovernedResponse(
        context_package_id=package.context_package_id,
        symbolic_verdict="r.block(block)",
        grounding_confidence=0.88,
        answer="Blocked: contraindicated in renal impairment.",
        neural_proposal="Aspirin is generally well tolerated.",
        rules_fired=(FiredRule("r.block", "block", "contraindicated"),),
        citations=(Citation(curie="CHEBI:15365", source="ChEBI"),),
    )
    payload = r.to_dict()
    # R9.3: both halves visible, so the override is auditable
    assert payload["neural_proposal"] != payload["answer"]
    assert payload["symbolic_verdict"] == "r.block(block)"
    assert len(payload["rules_fired"]) == 1
    assert isinstance(payload["citations"], list)


# ── R6.1: decisions hash rather than store raw answers ────────────────────────


def test_decision_answer_hash_is_stable_and_short():
    h1 = Decision.hash_answer("same text")
    h2 = Decision.hash_answer("same text")
    assert h1 == h2 and len(h1) == 16
    assert Decision.hash_answer(None) == Decision.hash_answer("")


def test_decision_is_frozen():
    d = Decision(
        decision_id="d1", question="q", context_package_id="ctx_1",
        rules_fired=(), answer_hash="abc", grounding_confidence=0.9,
        decided_at=utcnow(),
    )
    # R6.4: append-only — a decision record cannot be edited in place
    with pytest.raises(Exception):
        d.grounding_confidence = 0.1  # type: ignore[misc]
