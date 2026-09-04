"""Deterministic rule engine. Task 4.2, Requirements 3.2-3.6.

This is the symbolic half of the neurosymbolic system. It is a pure function:
same FactSet plus same rules always yields the same verdict, with no network
call and no LLM involvement. That property is what lets a rule override
generated text and still be defensible.

The engine never infers a rule it was not given (Requirement 3.3).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from context_layer.rules.models import Condition, Fact, FactSet, Rule
from context_layer.types import FiredRule, RuleVerdict

# Severity ordering for stable, meaningful output: blocks first.
_SEVERITY_ORDER = {"block": 0, "warn": 1, "inform": 2}


# ── Predicate evaluation ──────────────────────────────────────────────────────


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _norm(value: Any) -> Any:
    """Case-insensitive comparison for strings; identity otherwise."""
    return value.casefold() if isinstance(value, str) else value


def _match_operator(actual: Any, op: str, expected: Any) -> bool:
    if op == "exists":
        present = actual is not None and actual != ""
        return present is bool(expected)
    if op == "equals":
        return _norm(actual) == _norm(expected)
    if op == "not_equals":
        return _norm(actual) != _norm(expected)
    if op == "in":
        return _norm(actual) in {_norm(v) for v in expected}
    if op == "not_in":
        return _norm(actual) not in {_norm(v) for v in expected}
    if op == "contains":
        if actual is None:
            return False
        if isinstance(actual, (list, tuple)):
            return _norm(expected) in {_norm(v) for v in actual}
        return _norm(expected) in _norm(str(actual))
    a, b = _as_number(actual), _as_number(expected)
    if a is None or b is None:
        return False
    if op == "gt":
        return a > b
    if op == "gte":
        return a >= b
    if op == "lt":
        return a < b
    if op == "lte":
        return a <= b
    return False


def _match_where(fact: Fact, where: Mapping[str, Any]) -> bool:
    for prop, spec in where.items():
        actual = fact.get(prop)
        if isinstance(spec, dict):
            if not all(_match_operator(actual, op, exp) for op, exp in spec.items()):
                return False
        elif _norm(actual) != _norm(spec):
            return False
    return True


def _match_condition(fact: Fact, cond: Condition) -> bool:
    if fact.entity_type != cond.entity:
        return False
    if not _match_where(fact, cond.where):
        return False
    if cond.with_relation is not None:
        minimum = cond.min_relation_count or 1
        if fact.relation_count(cond.with_relation) < minimum:
            return False
    if cond.without_relation is not None:
        if fact.relation_count(cond.without_relation) > 0:
            return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────


def matching_facts(facts: FactSet, rule: Rule) -> tuple[Fact, ...]:
    """Facts that satisfy a rule's condition. Deterministic order."""
    return tuple(
        f for f in facts.of_type(rule.condition.entity)
        if _match_condition(f, rule.condition)
    )


def evaluate(facts: FactSet, rules: Sequence[Rule]) -> RuleVerdict:
    """Evaluate rules against facts. Pure: no network, no LLM, no clock.

    A rule fires when at least one fact satisfies its condition. The entities
    that triggered it are reported so the verdict is traceable to data.
    """
    fired: list[FiredRule] = []
    for rule in rules:
        hits = matching_facts(facts, rule)
        if not hits:
            continue
        triggered = tuple(f.entity_id for f in hits)
        labels = ", ".join(f.label() for f in hits[:3])
        if len(hits) > 3:
            labels += f", and {len(hits) - 3} more"
        fired.append(
            FiredRule(
                rule_id=rule.rule_id,
                severity=rule.severity,
                rationale=f"{rule.rationale} Triggered by: {labels}.",
                triggered_by=triggered,
                policy_iri=rule.policy_iri,
            )
        )
    fired.sort(key=lambda r: (_SEVERITY_ORDER.get(r.severity, 9), r.rule_id))
    return RuleVerdict(fired=tuple(fired))


def explain(facts: FactSet, rules: Iterable[Rule]) -> str:
    """Human-readable evaluation trace, for debugging and demos."""
    lines = [f"facts: {len(facts)} across {len(facts.types)} types"]
    for rule in rules:
        hits = matching_facts(facts, rule)
        mark = "FIRED" if hits else "  ok "
        lines.append(
            f"  [{mark}] {rule.rule_id} ({rule.severity}) "
            f"-> {len(hits)} match(es)"
        )
    return "\n".join(lines)
