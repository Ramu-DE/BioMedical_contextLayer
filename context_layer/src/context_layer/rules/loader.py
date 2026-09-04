"""Rule pack loading and validation. Task 4.1, Requirement 3.1.

Packs are declarative YAML. Validation happens here, at load time, so a
malformed pack fails on startup rather than mid-request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from context_layer.rules.models import (
    Condition,
    Rule,
    RulePack,
    RulePackError,
)

PACKS_DIR = Path(__file__).parent / "packs"

REQUIRED_RULE_KEYS = {"id", "description", "severity", "rationale", "when"}
ALLOWED_RULE_KEYS = REQUIRED_RULE_KEYS | {"policy_iri", "references"}
ALLOWED_WHEN_KEYS = {
    "entity",
    "where",
    "with_relation",
    "without_relation",
    "min_relation_count",
}
ALLOWED_OPERATORS = {
    "equals",
    "not_equals",
    "in",
    "not_in",
    "gt",
    "gte",
    "lt",
    "lte",
    "exists",
    "contains",
}


def _parse_condition(raw: Any, rule_id: str) -> Condition:
    if not isinstance(raw, dict):
        raise RulePackError(f"rule {rule_id}: 'when' must be a mapping")
    unknown = set(raw) - ALLOWED_WHEN_KEYS
    if unknown:
        raise RulePackError(
            f"rule {rule_id}: unknown keys in 'when': {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_WHEN_KEYS)}"
        )
    where = raw.get("where") or {}
    if not isinstance(where, dict):
        raise RulePackError(f"rule {rule_id}: 'where' must be a mapping")
    for prop, spec in where.items():
        if isinstance(spec, dict):
            bad = set(spec) - ALLOWED_OPERATORS
            if bad:
                raise RulePackError(
                    f"rule {rule_id}: unknown operator(s) {sorted(bad)} on "
                    f"property {prop!r}. Allowed: {sorted(ALLOWED_OPERATORS)}"
                )
    return Condition(
        entity=raw.get("entity", ""),
        where=where,
        with_relation=raw.get("with_relation"),
        without_relation=raw.get("without_relation"),
        min_relation_count=raw.get("min_relation_count"),
    )


def _parse_rule(raw: Any, pack_name: str) -> Rule:
    if not isinstance(raw, dict):
        raise RulePackError(f"{pack_name}: each rule must be a mapping, got {type(raw)}")
    rule_id = raw.get("id", "<missing id>")
    missing = REQUIRED_RULE_KEYS - set(raw)
    if missing:
        raise RulePackError(f"rule {rule_id}: missing required key(s) {sorted(missing)}")
    unknown = set(raw) - ALLOWED_RULE_KEYS
    if unknown:
        raise RulePackError(f"rule {rule_id}: unknown key(s) {sorted(unknown)}")
    refs = raw.get("references") or []
    if isinstance(refs, str):
        refs = [refs]
    return Rule(
        rule_id=str(raw["id"]),
        description=str(raw["description"]).strip(),
        severity=str(raw["severity"]),
        rationale=" ".join(str(raw["rationale"]).split()),
        condition=_parse_condition(raw["when"], rule_id),
        policy_iri=raw.get("policy_iri"),
        references=tuple(str(r) for r in refs),
    )


def load_pack(path: str | Path) -> RulePack:
    """Load and validate a single pack. Raises RulePackError on any problem."""
    path = Path(path)
    if not path.is_file():
        raise RulePackError(f"rule pack not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise RulePackError(f"{path.name}: invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise RulePackError(f"{path.name}: top level must be a mapping")
    name = raw.get("name") or path.stem
    rules_raw = raw.get("rules")
    if not isinstance(rules_raw, list) or not rules_raw:
        raise RulePackError(f"{path.name}: requires a non-empty 'rules' list")
    return RulePack(
        name=str(name),
        rules=tuple(_parse_rule(r, str(name)) for r in rules_raw),
        source_path=str(path),
    )


def load_packs(directory: str | Path = PACKS_DIR) -> tuple[RulePack, ...]:
    """Load every .yaml pack in a directory, sorted for deterministic order."""
    directory = Path(directory)
    if not directory.is_dir():
        raise RulePackError(f"rule pack directory not found: {directory}")
    paths = sorted(
        p for p in directory.iterdir() if p.suffix in (".yaml", ".yml")
    )
    if not paths:
        raise RulePackError(f"no rule packs found in {directory}")
    packs = tuple(load_pack(p) for p in paths)
    _assert_unique_ids(packs)
    return packs


def _assert_unique_ids(packs: Iterable[RulePack]) -> None:
    seen: dict[str, str] = {}
    for pack in packs:
        for rule in pack.rules:
            if rule.rule_id in seen:
                raise RulePackError(
                    f"duplicate rule id {rule.rule_id!r} in {pack.name} "
                    f"and {seen[rule.rule_id]}"
                )
            seen[rule.rule_id] = pack.name


def all_rules(packs: Iterable[RulePack]) -> tuple[Rule, ...]:
    return tuple(r for p in packs for r in p.rules)
