"""Rule engine behaviour. Requirements 3.1-3.6.

No credentials needed: the engine is pure, so every test constructs its own
FactSet. That is the point of the design.
"""

from __future__ import annotations

import pytest

from context_layer.rules.engine import evaluate, explain, matching_facts
from context_layer.rules.loader import PACKS_DIR, all_rules, load_pack, load_packs
from context_layer.rules.models import (
    Condition,
    Fact,
    FactSet,
    Rule,
    RulePackError,
)


def fact(entity_type="Patient", entity_id="PAT001", props=None, relations=None):
    return Fact(
        entity_id=entity_id,
        entity_type=entity_type,
        properties=props or {},
        relations=relations or {},
    )


def rule(rule_id="r.test", severity="block", **cond):
    cond.setdefault("entity", "Patient")
    return Rule(
        rule_id=rule_id,
        description="test rule",
        severity=severity,
        rationale="because the test says so.",
        condition=Condition(**cond),
    )


# ── R3.3: never invents a rule ────────────────────────────────────────────────


def test_no_rules_means_no_verdict():
    v = evaluate(FactSet(facts=(fact(),)), [])
    assert v.fired == () and not v.blocked


def test_no_matching_fact_means_no_fire():
    fs = FactSet(facts=(fact(props={"consent_status": "Active"}),))
    v = evaluate(fs, [rule(where={"consent_status": "Withdrawn"})])
    assert v.fired == ()


# ── R3.2, R3.4: fires with traceable trigger, blocks on block ─────────────────


def test_fires_and_reports_triggering_entity():
    fs = FactSet(facts=(fact(props={"consent_status": "Withdrawn", "name": "P1"}),))
    v = evaluate(fs, [rule(where={"consent_status": "Withdrawn"})])
    assert len(v.fired) == 1
    assert v.blocked
    assert v.fired[0].triggered_by == ("PAT001",)
    assert "P1" in v.fired[0].rationale


def test_warn_does_not_block():
    fs = FactSet(facts=(fact(props={"consent_status": "Withdrawn"}),))
    v = evaluate(fs, [rule(severity="warn", where={"consent_status": "Withdrawn"})])
    assert v.fired and not v.blocked


def test_blocks_sort_before_warnings():
    fs = FactSet(facts=(fact(props={"x": 1}),))
    v = evaluate(
        fs,
        [
            rule("r.warn", "warn", where={"x": 1}),
            rule("r.inform", "inform", where={"x": 1}),
            rule("r.block", "block", where={"x": 1}),
        ],
    )
    assert [f.severity for f in v.fired] == ["block", "warn", "inform"]


# ── R3.5: determinism ─────────────────────────────────────────────────────────


def test_identical_input_yields_identical_verdict():
    fs = FactSet(
        facts=(
            fact("Patient", "PAT001", {"consent_status": "Withdrawn"}),
            fact("Patient", "PAT002", {"consent_status": "Active"}),
        )
    )
    rules = [rule(where={"consent_status": "Withdrawn"})]
    a, b = evaluate(fs, rules), evaluate(fs, rules)
    assert a == b


def test_fact_order_does_not_change_the_verdict():
    f1 = fact("Patient", "PAT001", {"consent_status": "Withdrawn"})
    f2 = fact("Patient", "PAT002", {"consent_status": "Withdrawn"})
    rules = [rule(where={"consent_status": "Withdrawn"})]
    v1 = evaluate(FactSet(facts=(f1, f2)), rules)
    v2 = evaluate(FactSet(facts=(f2, f1)), rules)
    assert {t for f in v1.fired for t in f.triggered_by} == {
        t for f in v2.fired for t in f.triggered_by
    }


# ── Operators ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec,value,expected",
    [
        ({"equals": "Withdrawn"}, "withdrawn", True),   # case-insensitive
        ({"equals": "Withdrawn"}, "Active", False),
        ({"not_equals": "Approved"}, "Investigational", True),
        ({"in": ["a", "b"]}, "B", True),
        ({"not_in": ["Completed"]}, "Recruiting", True),
        ({"gt": 5}, 6, True),
        ({"gt": 5}, 5, False),
        ({"gte": 5}, 5, True),
        ({"lt": 0.95}, 0.9, True),
        ({"lt": 0.95}, 0.99, False),
        ({"lte": 100}, 100, True),
        ({"exists": False}, None, True),
        ({"exists": False}, "", True),
        ({"exists": True}, "value", True),
        ({"exists": True}, None, False),
        ({"contains": "Diabetes"}, "Hypertension;Type 2 Diabetes", True),
        ({"contains": "Asthma"}, "Hypertension", False),
    ],
)
def test_operator_semantics(spec, value, expected):
    fs = FactSet(facts=(fact(props={"p": value} if value is not None else {}),))
    v = evaluate(fs, [rule(where={"p": spec})])
    assert bool(v.fired) is expected


def test_numeric_comparison_on_string_values():
    """Neo4j sometimes returns numbers as strings; comparison must still work."""
    fs = FactSet(facts=(fact(props={"dq_pass_rate": "0.80"}),))
    assert evaluate(fs, [rule(where={"dq_pass_rate": {"lt": 0.95}})]).fired


def test_non_numeric_value_fails_numeric_comparison_safely():
    fs = FactSet(facts=(fact(props={"dq_pass_rate": "unknown"}),))
    assert not evaluate(fs, [rule(where={"dq_pass_rate": {"lt": 0.95}})]).fired


# ── Relations ─────────────────────────────────────────────────────────────────


def test_without_relation_fires_when_link_absent():
    fs = FactSet(facts=(fact("Table", "db.hub.t1", {"layer": "hub"}),))
    r = rule("g.dq", entity="Table", where={"layer": "hub"}, without_relation="HAS_DQ_RULE")
    assert evaluate(fs, [r]).fired


def test_without_relation_silent_when_link_present():
    fs = FactSet(
        facts=(
            fact("Table", "db.hub.t1", {"layer": "hub"}, {"HAS_DQ_RULE": ("DQ1",)}),
        )
    )
    r = rule("g.dq", entity="Table", where={"layer": "hub"}, without_relation="HAS_DQ_RULE")
    assert not evaluate(fs, [r]).fired


def test_with_relation_respects_min_count():
    fs = FactSet(facts=(fact("Drug", "D1", {}, {"TREATS": ("DIS1",)}),))
    one = rule("d.one", entity="Drug", with_relation="TREATS", min_relation_count=1)
    two = rule("d.two", entity="Drug", with_relation="TREATS", min_relation_count=2)
    assert evaluate(fs, [one]).fired
    assert not evaluate(fs, [two]).fired


def test_entity_type_must_match():
    fs = FactSet(facts=(fact("Drug", "D1", {"consent_status": "Withdrawn"}),))
    assert not evaluate(fs, [rule(entity="Patient", where={"consent_status": "Withdrawn"})]).fired


# ── FactSet helpers ───────────────────────────────────────────────────────────


def test_factset_indexing():
    fs = FactSet(facts=(fact("Patient", "P1"), fact("Drug", "D1")))
    assert len(fs) == 2
    assert set(fs.types) == {"Patient", "Drug"}
    assert len(fs.of_type("Patient")) == 1
    assert fs.by_id("D1").entity_type == "Drug"
    assert fs.by_id("nope") is None


def test_fact_label_prefers_human_names():
    assert fact(props={"name": "Pembrolizumab"}).label() == "Pembrolizumab"
    assert fact(entity_id="PAT007", props={}).label() == "PAT007"


# ── Shipped packs ─────────────────────────────────────────────────────────────


def test_shipped_packs_load_and_validate():
    packs = load_packs(PACKS_DIR)
    assert len(packs) >= 3
    rules = all_rules(packs)
    assert len(rules) >= 12
    for r in rules:
        assert r.severity in ("block", "warn", "inform")
        assert r.rationale and r.description
        assert r.condition.entity


def test_shipped_packs_have_unique_ids():
    ids = [r.rule_id for r in all_rules(load_packs(PACKS_DIR))]
    assert len(ids) == len(set(ids))


def test_blocking_rules_cite_a_policy_where_one_exists():
    """Every block rule should be traceable to a governance artifact."""
    blocks = [r for r in all_rules(load_packs(PACKS_DIR)) if r.severity == "block"]
    assert blocks
    cited = [r for r in blocks if r.policy_iri]
    assert len(cited) >= 3, "blocking rules should cite the policy they enforce"


# ── Loader validation, R3.1 ───────────────────────────────────────────────────


def write(tmp_path, text, name="p.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_loader_rejects_missing_required_keys(tmp_path):
    p = write(tmp_path, "name: x\nrules:\n  - id: a\n    severity: block\n")
    with pytest.raises(RulePackError, match="missing required key"):
        load_pack(p)


def test_loader_rejects_bad_severity(tmp_path):
    p = write(
        tmp_path,
        "name: x\nrules:\n  - id: a\n    description: d\n    severity: fatal\n"
        "    rationale: r\n    when: {entity: Patient}\n",
    )
    with pytest.raises(RulePackError, match="severity"):
        load_pack(p)


def test_loader_rejects_unknown_operator(tmp_path):
    p = write(
        tmp_path,
        "name: x\nrules:\n  - id: a\n    description: d\n    severity: warn\n"
        "    rationale: r\n    when:\n      entity: Patient\n"
        "      where:\n        p:\n          regex: '.*'\n",
    )
    with pytest.raises(RulePackError, match="unknown operator"):
        load_pack(p)


def test_loader_rejects_unknown_when_key(tmp_path):
    p = write(
        tmp_path,
        "name: x\nrules:\n  - id: a\n    description: d\n    severity: warn\n"
        "    rationale: r\n    when: {entity: Patient, whatever: 1}\n",
    )
    with pytest.raises(RulePackError, match="unknown keys in 'when'"):
        load_pack(p)


def test_loader_rejects_duplicate_ids(tmp_path):
    body = (
        "    description: d\n    severity: warn\n    rationale: r\n"
        "    when: {entity: Patient}\n"
    )
    p = write(tmp_path, f"name: x\nrules:\n  - id: dup\n{body}  - id: dup\n{body}")
    with pytest.raises(RulePackError, match="duplicate rule id"):
        load_pack(p)


def test_loader_rejects_empty_rules(tmp_path):
    with pytest.raises(RulePackError, match="non-empty 'rules'"):
        load_pack(write(tmp_path, "name: x\nrules: []\n"))


def test_loader_rejects_invalid_yaml(tmp_path):
    with pytest.raises(RulePackError, match="invalid YAML"):
        load_pack(write(tmp_path, "name: x\nrules:\n  - id: [unclosed\n"))


def test_loader_rejects_missing_file(tmp_path):
    with pytest.raises(RulePackError, match="not found"):
        load_pack(tmp_path / "nope.yaml")


def test_condition_cannot_require_and_forbid_same_relation():
    with pytest.raises(RulePackError, match="cannot require and forbid"):
        Condition(entity="Patient", with_relation="X", without_relation="Y")


# ── Purity, R3.6 ──────────────────────────────────────────────────────────────


def test_engine_performs_no_network_io(monkeypatch):
    """Break sockets entirely; evaluation must still succeed."""
    import socket

    def forbidden(*a, **k):
        raise AssertionError("rule engine attempted network I/O")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    fs = FactSet(facts=(fact(props={"consent_status": "Withdrawn"}),))
    v = evaluate(fs, all_rules(load_packs(PACKS_DIR)))
    assert v.blocked


def test_explain_lists_every_rule():
    fs = FactSet(facts=(fact(props={"consent_status": "Withdrawn"}),))
    rules = all_rules(load_packs(PACKS_DIR))
    text = explain(fs, rules)
    assert "FIRED" in text
    for r in rules:
        assert r.rule_id in text


def test_matching_facts_is_deterministic():
    fs = FactSet(
        facts=tuple(
            fact("Patient", f"PAT{i:03d}", {"consent_status": "Withdrawn"})
            for i in range(5)
        )
    )
    r = rule(where={"consent_status": "Withdrawn"})
    assert [f.entity_id for f in matching_facts(fs, r)] == [
        f"PAT{i:03d}" for i in range(5)
    ]
