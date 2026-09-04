"""Rule pack sanity: guard against rules that can never fire.

A rule that never fires is indistinguishable from a rule that is broken. The
``governance.ard_below_dq_threshold`` rule originally compared a percentage
value (97.8) against a fraction threshold (0.95) and was therefore dead. These
tests make that class of mistake fail loudly.
"""

from __future__ import annotations

import pytest

from context_layer.rules.engine import evaluate
from context_layer.rules.loader import PACKS_DIR, all_rules, load_packs
from context_layer.rules.models import Fact, FactSet

# Scales observed in the live graph, used to build realistic probe facts.
LIVE_SCALES = {
    "dq_pass_rate": 100.0,      # percent, range 97.8 .. 100.0
    "enrollment": 1000,         # participants
    "row_count_millions": 10.0,
}


@pytest.fixture(scope="module")
def rules():
    return all_rules(load_packs(PACKS_DIR))


def test_every_rule_can_fire_for_some_fact(rules):
    """Construct a fact designed to trip each rule. None may be unreachable."""
    unreachable = []
    for r in rules:
        cond = r.condition
        props: dict = {}
        for prop, spec in cond.where.items():
            if not isinstance(spec, dict):
                props[prop] = spec
                continue
            for op, expected in spec.items():
                if op == "equals":
                    props[prop] = expected
                elif op == "not_equals":
                    props[prop] = f"other-than-{expected}"
                elif op == "in":
                    props[prop] = expected[0]
                elif op == "not_in":
                    props[prop] = "value-not-in-list"
                elif op == "contains":
                    props[prop] = f"prefix {expected} suffix"
                elif op == "gt":
                    props[prop] = float(expected) + 1
                elif op == "gte":
                    props[prop] = float(expected)
                elif op == "lt":
                    props[prop] = float(expected) - 1
                elif op == "lte":
                    props[prop] = float(expected)
                elif op == "exists":
                    if expected:
                        props[prop] = "present"
                    else:
                        props.pop(prop, None)
        relations: dict = {}
        if cond.with_relation:
            count = cond.min_relation_count or 1
            relations[cond.with_relation] = tuple(f"t{i}" for i in range(count))
        probe = Fact(
            entity_id="PROBE",
            entity_type=cond.entity,
            properties=props,
            relations=relations,
        )
        if not evaluate(FactSet(facts=(probe,)), [r]).fired:
            unreachable.append(r.rule_id)
    assert not unreachable, f"rules that can never fire: {unreachable}"


@pytest.mark.parametrize("prop,scale", LIVE_SCALES.items())
def test_numeric_thresholds_match_the_live_scale(rules, prop, scale):
    """A threshold far below the observed scale signals a unit mismatch."""
    problems = []
    for r in rules:
        spec = r.condition.where.get(prop)
        if not isinstance(spec, dict):
            continue
        for op, value in spec.items():
            if op not in ("gt", "gte", "lt", "lte"):
                continue
            # A percent-scale property compared against <1 is the classic bug.
            if scale > 1 and float(value) <= 1:
                problems.append(
                    f"{r.rule_id}: {prop} {op} {value} but live scale peaks at {scale}"
                )
    assert not problems, "; ".join(problems)


def test_dq_threshold_fires_on_the_lowest_live_value(rules):
    """97.8 is the lowest observed dq_pass_rate; the rule must catch it."""
    rule = next(r for r in rules if r.rule_id == "governance.ard_below_dq_threshold")
    low = Fact(
        entity_id="ARD_LOW",
        entity_type="ARD",
        properties={"name": "Customer Master", "dq_pass_rate": 97.8, "certified": True},
    )
    high = Fact(
        entity_id="ARD_HIGH",
        entity_type="ARD",
        properties={"name": "Finance Monthly Revenue", "dq_pass_rate": 100.0, "certified": True},
    )
    assert evaluate(FactSet(facts=(low,)), [rule]).fired
    assert not evaluate(FactSet(facts=(high,)), [rule]).fired


def test_every_rule_references_a_known_entity_type(rules):
    """Rules must target types the fact builder can actually fetch."""
    from context_layer.rules.graph_facts import FACT_KEYS

    unknown = sorted(
        {r.condition.entity for r in rules if r.condition.entity not in FACT_KEYS}
    )
    assert not unknown, f"rules target types the fact builder cannot fetch: {unknown}"


def test_every_tracked_relation_used_by_rules_is_fetched(rules):
    """A rule asserting a relation the builder never loads would be dead."""
    from context_layer.rules.graph_facts import TRACKED_RELATIONS

    used = {
        rel
        for r in rules
        for rel in (r.condition.with_relation, r.condition.without_relation)
        if rel
    }
    missing = sorted(used - set(TRACKED_RELATIONS))
    assert not missing, f"relations used by rules but not fetched: {missing}"
