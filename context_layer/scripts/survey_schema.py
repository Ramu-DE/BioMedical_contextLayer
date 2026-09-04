#!/usr/bin/env python3
"""Survey the live graph's schema: labels, their properties, and edge patterns.

Tells us what the context layer actually has to work with, rather than what we
assumed. Output drives rule authoring (Phase 4) and the entity linker's target
vocabulary (Phase 2).

Usage:
    .venv/bin/python scripts/survey_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


def main() -> int:
    cfg = config_module.load()
    driver = GraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_user, str(cfg.neo4j_password))
    )
    db = cfg.neo4j_kg_database
    try:
        with driver.session(database=db) as s:
            print("═══ label → properties ═══")
            rows = s.run(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                WITH label, keys(n) AS ks
                UNWIND ks AS k
                RETURN label, collect(DISTINCT k) AS props, count(*) AS n
                ORDER BY label
                """
            )
            for r in rows:
                print(f"\n{r['label']}")
                print(f"  props: {sorted(set(r['props']))}")

            print("\n═══ edge patterns ═══")
            rows = s.run(
                """
                MATCH (a)-[r]->(b)
                RETURN labels(a)[0] AS src, type(r) AS rel,
                       labels(b)[0] AS dst, count(*) AS n
                ORDER BY n DESC, rel
                """
            )
            for r in rows:
                print(f"  ({r['src']})-[:{r['rel']}]->({r['dst']})  x{r['n']}")

            print("\n═══ governance: Policy nodes ═══")
            for r in s.run("MATCH (p:Policy) RETURN properties(p) AS p LIMIT 10"):
                print(f"  {r['p']}")

            print("\n═══ rules: DQRule / RuleType ═══")
            for r in s.run("MATCH (n:DQRule) RETURN properties(n) AS p LIMIT 10"):
                print(f"  DQRule  {r['p']}")
            for r in s.run("MATCH (n:RuleType) RETURN properties(n) AS p LIMIT 10"):
                print(f"  RuleType {r['p']}")

            print("\n═══ clinical vocabularies (entity-linking targets) ═══")
            for label in ("SNOMED", "RxNorm", "MedDRA", "OMOP"):
                rows = s.run(
                    f"MATCH (n:{label}) RETURN properties(n) AS p LIMIT 4"
                )
                print(f"\n  {label}:")
                for r in rows:
                    print(f"    {r['p']}")

            print("\n═══ therapeutic areas ═══")
            for r in s.run(
                "MATCH (n:TherapeuticArea) RETURN properties(n) AS p LIMIT 10"
            ):
                print(f"  {r['p']}")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
