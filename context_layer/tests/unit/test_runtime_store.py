"""Runtime context store. Requirements 5.x, 6.x, 7.x."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from context_layer.runtime.store import (
    RuntimeStore,
    _to_dt,
    action,
    decision_from_response,
    outcome,
)
from context_layer.types import (
    Citation,
    Decision,
    Event,
    FiredRule,
    GovernedResponse,
    new_id,
    utcnow,
)


class FakeSession:
    def __init__(self, recorder):
        self.recorder = recorder

    def run(self, cypher, **params):
        self.recorder.append((cypher, params))
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def session(self, database=None):
        return FakeSession(self.calls)

    def close(self):
        self.closed = True


@pytest.fixture
def store(cfg):
    driver = FakeDriver()
    s = RuntimeStore(cfg, driver=driver)
    return s, driver


# ── R6.4: append-only is structural ───────────────────────────────────────────


def test_store_exposes_no_mutating_verbs(store):
    s, _ = store
    public = [m for m in dir(s) if not m.startswith("_")]
    forbidden = [
        m for m in public
        if any(x in m.lower() for x in ("update", "delete", "remove", "drop"))
    ]
    assert forbidden == [], f"append-only store exposes mutators: {forbidden}"


def test_supersede_only_sets_valid_to(store):
    """Closing a validity window must not rewrite content."""
    s, driver = store
    s.supersede_decision("dec_1")
    cypher = driver.calls[-1][0]
    assert "SET d.valid_to" in cypher
    assert "DELETE" not in cypher.upper()
    for prop in ("question", "grounding_confidence", "answer_hash", "rules_fired"):
        assert f"d.{prop} =" not in cypher


# ── R5.3: namespace isolation ─────────────────────────────────────────────────


def test_labels_are_prefixed(store):
    s, _ = store
    assert s.label("Decision").startswith(s.prefix)
    assert s.decision_label == f"{s.prefix}Decision"


def test_every_write_uses_the_prefixed_label(store):
    s, driver = store
    s.append_decision(_decision())
    s.append_event(_event())
    for cypher, _ in driver.calls:
        if "CREATE (" in cypher:
            assert s.prefix in cypher, f"unprefixed write: {cypher[:80]}"


# ── Neo4j property typing: maps must be serialised ────────────────────────────


def _decision(**kw):
    base = dict(
        decision_id=new_id("dec"),
        question="Which drugs target PD-1?",
        context_package_id="ctx_1",
        rules_fired=("r.a",),
        answer_hash="abc123",
        grounding_confidence=0.72,
        decided_at=utcnow(),
        grounded_in=("RXNORM:1657993",),
    )
    base.update(kw)
    return Decision(**base)


def _event(**kw):
    now = utcnow()
    base = dict(
        event_id=new_id("evt"),
        event_type="dq_check_failed",
        occurred_at=now - timedelta(minutes=5),
        ingested_at=now,
        payload={"table": "hub.sales_fact", "failures": 3},
        about_curies=("RXNORM:1657993",),
    )
    base.update(kw)
    return Event(**base)


def test_action_arguments_are_json_not_a_map(store):
    """Neo4j rejects Map properties; this bug silently lost every action."""
    s, driver = store
    s.append_action(action("dec_1", "assemble", {"question_chars": 56}, 12.5))
    params = driver.calls[-1][1]
    assert isinstance(params["arguments"], str)
    assert json.loads(params["arguments"]) == {"question_chars": 56}


def test_event_payload_is_json_not_a_map(store):
    s, driver = store
    s.append_event(_event())
    params = driver.calls[-1][1]
    assert isinstance(params["payload"], str)
    assert json.loads(params["payload"])["failures"] == 3


def test_no_write_passes_a_dict_as_a_property(store):
    """Guards the whole class of CypherTypeError we hit."""
    s, driver = store
    s.append_decision(_decision())
    s.append_event(_event())
    s.append_action(action("d", "t", {"a": 1}, 1.0))
    s.append_outcome(outcome("d", "answer_accepted", "ok"))
    for cypher, params in driver.calls:
        for key, value in params.items():
            if key in ("specs",):  # list-of-map query args are fine
                continue
            assert not isinstance(value, dict), f"{key} passed as dict in {cypher[:60]}"


# ── R5.4: bi-temporal fields both recorded ────────────────────────────────────


def test_event_records_both_timestamps(store):
    s, driver = store
    e = _event()
    s.append_event(e)
    params = driver.calls[-1][1]
    assert params["occurred_at"] != params["ingested_at"]
    assert "occurred_at" in driver.calls[-1][0]
    assert "ingested_at" in driver.calls[-1][0]


def test_decision_defaults_valid_from_to_decided_at(store):
    s, driver = store
    d = _decision()
    s.append_decision(d)
    params = driver.calls[-1][1]
    assert params["valid_from"] == d.decided_at.isoformat()


# ── R7.3: as_of window semantics ──────────────────────────────────────────────


def test_as_of_filters_on_both_window_bounds(store):
    s, driver = store
    s.as_of(utcnow())
    cypher = driver.calls[-1][0]
    assert "d.valid_from <= datetime($ts)" in cypher
    assert "d.valid_to IS NULL OR d.valid_to > datetime($ts)" in cypher


def test_recent_decisions_excludes_superseded(store):
    s, driver = store
    s.recent_decisions("a question", k=3)
    assert "d.valid_to IS NULL" in driver.calls[-1][0]


def test_recent_decisions_falls_back_to_recency_without_embedder(store):
    """Must still close the loop when Bedrock is unavailable."""
    s, driver = store
    assert s._embedder is None
    s.recent_decisions("q", k=2)
    cypher = driver.calls[-1][0]
    assert "ORDER BY d.decided_at DESC" in cypher
    assert "db.index.vector.queryNodes" not in cypher


# ── constructors ──────────────────────────────────────────────────────────────


def test_decision_from_response_captures_rules_and_hash():
    resp = GovernedResponse(
        context_package_id="ctx_9",
        symbolic_verdict="r.block(block)",
        grounding_confidence=0.61,
        answer="Blocked by policy.",
        rules_fired=(FiredRule("consent.withdrawn_patient_data", "block", "why"),),
        citations=(Citation(curie="RXNORM:1"),),
    )
    d = decision_from_response("q?", resp, grounded_in=("RXNORM:1",))
    assert d.rules_fired == ("consent.withdrawn_patient_data",)
    assert d.context_package_id == "ctx_9"
    assert d.grounding_confidence == 0.61
    assert d.answer_hash == Decision.hash_answer("Blocked by policy.")
    assert d.grounded_in == ("RXNORM:1",)


def test_decision_from_refusal_hashes_empty_answer():
    resp = GovernedResponse.refusal(
        context_package_id="ctx_1", reason="insufficient grounding"
    )
    d = decision_from_response("q", resp)
    assert d.answer_hash == Decision.hash_answer(None)


def test_action_and_outcome_get_unique_ids():
    a1, a2 = action("d", "t", {}, 1.0), action("d", "t", {}, 1.0)
    assert a1.action_id != a2.action_id
    o1, o2 = outcome("d", "x"), outcome("d", "x")
    assert o1.outcome_id != o2.outcome_id


# ── datetime normalisation ────────────────────────────────────────────────────


def test_to_dt_handles_naive_datetime():
    from datetime import datetime

    assert _to_dt(datetime(2026, 1, 1, 12, 0)).tzinfo is not None


def test_to_dt_handles_iso_string_with_z():
    assert _to_dt("2026-01-01T12:00:00Z").year == 2026


def test_to_dt_falls_back_for_garbage():
    assert _to_dt("not a date").tzinfo is not None


def test_store_close_is_scoped_to_owned_drivers(cfg):
    driver = FakeDriver()
    with RuntimeStore(cfg, driver=driver):
        pass
    assert not driver.closed, "must not close a driver it does not own"
