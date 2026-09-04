#!/usr/bin/env python3
"""Evaluate the rule packs against the live graph. Task 4.3 verification.

Fetches facts from Neo4j, then evaluates purely. Demonstrates that the symbolic
layer detects real violations in real data, not contrived examples.

Usage:
    .venv/bin/python scripts/check_rules.py                 # scan everything
    .venv/bin/python scripts/check_rules.py --patient PAT015
    .venv/bin/python scripts/check_rules.py --explain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from context_layer.rules.engine import evaluate, explain  # noqa: E402
from context_layer.rules.graph_facts import GraphFactBuilder  # noqa: E402
from context_layer.rules.loader import PACKS_DIR, all_rules, load_packs  # noqa: E402

SCAN_TYPES = (
    "Patient",
    "Drug",
    "ClinicalTrial",
    "AdverseEvent",
    "Table",
    "ARD",
    "Entity",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", help="scope evaluation to one patient id")
    ap.add_argument("--drug", help="scope evaluation to one drug id")
    ap.add_argument("--explain", action="store_true", help="show per-rule trace")
    args = ap.parse_args()

    cfg = config_module.load()
    packs = load_packs(PACKS_DIR)
    rules = all_rules(packs)
    print(f"loaded {len(packs)} packs, {len(rules)} rules")
    for p in packs:
        print(f"  {len(p):>2} rules  {p.name}")

    with GraphFactBuilder(cfg) as builder:
        if args.patient or args.drug:
            wanted: dict[str, list[str]] = {}
            if args.patient:
                wanted["Patient"] = [args.patient]
            if args.drug:
                wanted["Drug"] = [args.drug]
            facts = builder.for_entities(wanted)
            scope = f"scoped to {wanted}"
        else:
            facts = builder.for_types(SCAN_TYPES)
            scope = f"full scan of {len(SCAN_TYPES)} types"

    print(f"\nfacts: {len(facts)} ({scope})")
    for t in facts.types:
        print(f"  {len(facts.of_type(t)):>4}  {t}")

    verdict = evaluate(facts, rules)

    print(f"\n{'BLOCKED' if verdict.blocked else 'allowed'} — {verdict.summary()}")
    if verdict.fired:
        print()
        for f in verdict.fired:
            marker = {"block": "BLOCK", "warn": " WARN", "inform": " INFO"}[f.severity]
            print(f"[{marker}] {f.rule_id}")
            print(f"         {f.rationale}")
            if f.policy_iri:
                print(f"         policy: {f.policy_iri}")
            print(f"         entities: {list(f.triggered_by[:6])}"
                  f"{' ...' if len(f.triggered_by) > 6 else ''}")
            print()

    if args.explain:
        print("─── trace ───")
        print(explain(facts, rules))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
