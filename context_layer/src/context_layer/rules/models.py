"""Fact and rule models for the symbolic layer. Task 4.1.

The engine is a pure function over a FactSet. Graph I/O happens in
``graph_facts.py`` and produces the FactSet; the engine never touches a
network, which is what makes verdicts reproducible and auditable
(Requirements 3.5, 3.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Severity = Literal["block", "warn", "inform"]
SEVERITIES: tuple[Severity, ...] = ("block", "warn", "inform")


# ── Facts ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Fact:
    """One entity, as the rule engine sees it.

    ``relations`` maps a relationship type to the entity ids it reaches, which
    lets rules assert presence or absence of a link without another query.
    """

    entity_id: str
    entity_type: str
    properties: Mapping[str, Any] = field(default_factory=dict)
    curies: tuple[str, ...] = ()
    relations: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def relation_count(self, rel_type: str) -> int:
        return len(self.relations.get(rel_type, ()))

    def label(self) -> str:
        """Human-readable identifier for rationale text."""
        for key in ("name", "title", "table_name", "fqn", "symbol"):
            if self.properties.get(key):
                return str(self.properties[key])
        return self.entity_id


@dataclass(frozen=True)
class FactSet:
    """Everything the engine is allowed to reason over."""

    facts: tuple[Fact, ...] = ()
    source_note: str = ""

    def of_type(self, entity_type: str) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.entity_type == entity_type)

    def by_id(self, entity_id: str) -> Fact | None:
        for f in self.facts:
            if f.entity_id == entity_id:
                return f
        return None

    @property
    def types(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for f in self.facts:
            seen.setdefault(f.entity_type, None)
        return tuple(seen)

    def __len__(self) -> int:
        return len(self.facts)


# ── Rules ─────────────────────────────────────────────────────────────────────


class RulePackError(ValueError):
    """A rule pack is malformed. Raised at load time, never at request time."""


@dataclass(frozen=True)
class Condition:
    """Selection criteria applied to one entity type.

    ``where`` maps a property to either a literal (equality) or an operator
    mapping, e.g. ``{gt: 5}``, ``{in: [a, b]}``, ``{exists: false}``.
    """

    entity: str
    where: Mapping[str, Any] = field(default_factory=dict)
    with_relation: str | None = None
    without_relation: str | None = None
    min_relation_count: int | None = None

    def __post_init__(self) -> None:
        if not self.entity:
            raise RulePackError("condition requires an 'entity' type")
        if self.with_relation and self.without_relation:
            raise RulePackError(
                "condition cannot require and forbid a relation at once"
            )


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    severity: Severity
    rationale: str
    condition: Condition
    policy_iri: str | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise RulePackError("rule requires an 'id'")
        if not self.description:
            raise RulePackError(f"rule {self.rule_id} requires a 'description'")
        if self.severity not in SEVERITIES:
            raise RulePackError(
                f"rule {self.rule_id} severity {self.severity!r} "
                f"must be one of {SEVERITIES}"
            )
        if not self.rationale:
            raise RulePackError(f"rule {self.rule_id} requires a 'rationale'")


@dataclass(frozen=True)
class RulePack:
    name: str
    rules: tuple[Rule, ...]
    source_path: str = ""

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for r in self.rules:
            if r.rule_id in seen:
                raise RulePackError(f"duplicate rule id {r.rule_id!r} in {self.name}")
            seen.add(r.rule_id)

    def __len__(self) -> int:
        return len(self.rules)
