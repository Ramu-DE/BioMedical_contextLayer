#!/usr/bin/env python3
"""Derive the three relationships the published Graph Engineering model names
but the loaded dataset does not contain.

Every edge created here carries ``derivation`` and ``confidence`` so a derived
assertion is never mistaken for a curated one. That distinction matters more than
the coverage: an inferred edge presented as fact is exactly the failure mode the
whole layer exists to prevent.

    evaluated_in      Drug -> ClinicalTrial   inverse_of INVESTIGATES        1.0
    is_biomarker_for  Gene -> Biomarker       gene symbol in biomarker name  0.9
    indicates         Biomarker -> Disease    disease name in significance   0.9
                                              curated abbreviation map       0.8

Usage:
    .venv/bin/python scripts/derive_relationships.py [--dry-run] [--purge]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402

# Abbreviations used in Biomarker.clinical_significance that no substring match
# will ever catch. Curated deliberately rather than fuzzy-matched, so each one is
# reviewable.
ABBREVIATION_MAP: dict[str, str] = {
    "CML monitoring": "Chronic Myeloid Leukemia",
    "Alzheimer's diagnostic marker": "Alzheimer's Disease",
    "Diabetes control marker": "Type 2 Diabetes",
    "Diabetes screening": "Type 2 Diabetes",
}

DERIVED = "derived"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--purge", action="store_true", help="remove derived edges")
    args = ap.parse_args()

    cfg = config_module.load()
    driver = GraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_user, str(cfg.neo4j_password))
    )
    now = datetime.now(timezone.utc).isoformat()
    db = cfg.neo4j_kg_database

    def run(cypher: str, **params):
        if args.dry_run:
            # Rewrite the write into a count so --dry-run reports coverage.
            return []
        with driver.session(database=db) as s:
            return list(s.run(cypher, **params))

    def count(cypher: str, **params) -> int:
        with driver.session(database=db) as s:
            rec = s.run(cypher, **params).single()
            return rec[0] if rec else 0

    try:
        if args.purge:
            for rel in ("EVALUATED_IN", "IS_BIOMARKER_FOR", "INDICATES"):
                n = count(
                    f"MATCH ()-[r:{rel}]->() WHERE r.derivation IS NOT NULL "
                    "WITH r LIMIT 10000 DELETE r RETURN count(r)"
                )
                print(f"  deleted {n} derived :{rel}")
            return 0

        print("═══ 1. evaluated_in  (Drug -> ClinicalTrial) ═══")
        candidates = count(
            "MATCH (t:ClinicalTrial)-[:INVESTIGATES]->(d:Drug) RETURN count(*)"
        )
        print(f"  candidates from INVESTIGATES inverse: {candidates}")
        run(
            """
            MATCH (t:ClinicalTrial)-[i:INVESTIGATES]->(d:Drug)
            MERGE (d)-[r:EVALUATED_IN]->(t)
            SET r.derivation = 'inverse_of:INVESTIGATES',
                r.confidence = 1.0,
                r.dataset = $ds,
                r.derived_at = $now,
                r.arm_type = i.arm_type,
                r.dosage = i.dosage
            """,
            ds=DERIVED,
            now=now,
        )

        print("\n═══ 2. is_biomarker_for  (Gene -> Biomarker) ═══")
        rows = []
        with driver.session(database=db) as s:
            rows = list(
                s.run(
                    """
                    MATCH (g:Gene), (b:Biomarker)
                    WHERE toLower(b.name) CONTAINS toLower(g.symbol)
                    RETURN g.symbol AS gene, b.name AS biomarker
                    ORDER BY gene
                    """
                )
            )
        for r in rows:
            print(f"  {r['gene']:<10} -> {r['biomarker']}")
        run(
            """
            MATCH (g:Gene), (b:Biomarker)
            WHERE toLower(b.name) CONTAINS toLower(g.symbol)
            MERGE (g)-[r:IS_BIOMARKER_FOR]->(b)
            SET r.derivation = 'gene_symbol_in_biomarker_name',
                r.evidence = g.symbol,
                r.confidence = 0.9,
                r.dataset = $ds,
                r.derived_at = $now
            """,
            ds=DERIVED,
            now=now,
        )

        print("\n═══ 3. indicates  (Biomarker -> Disease) ═══")
        with driver.session(database=db) as s:
            exact = list(
                s.run(
                    """
                    MATCH (b:Biomarker), (x:Disease)
                    WHERE b.clinical_significance IS NOT NULL
                      AND toLower(b.clinical_significance) CONTAINS toLower(x.name)
                    RETURN b.name AS b, x.name AS d
                    """
                )
            )
        for r in exact:
            print(f"  [name match]  {r['b']:<24} -> {r['d']}")
        run(
            """
            MATCH (b:Biomarker), (x:Disease)
            WHERE b.clinical_significance IS NOT NULL
              AND toLower(b.clinical_significance) CONTAINS toLower(x.name)
            MERGE (b)-[r:INDICATES]->(x)
            SET r.derivation = 'disease_name_in_clinical_significance',
                r.evidence = b.clinical_significance,
                r.confidence = 0.9,
                r.dataset = $ds,
                r.derived_at = $now
            """,
            ds=DERIVED,
            now=now,
        )

        pairs = [
            {"sig": sig, "disease": disease}
            for sig, disease in ABBREVIATION_MAP.items()
        ]
        with driver.session(database=db) as s:
            curated = list(
                s.run(
                    """
                    UNWIND $pairs AS p
                    MATCH (b:Biomarker {clinical_significance: p.sig})
                    MATCH (x:Disease {name: p.disease})
                    RETURN b.name AS b, x.name AS d, p.sig AS sig
                    """,
                    pairs=pairs,
                )
            )
        for r in curated:
            print(f"  [curated]     {r['b']:<24} -> {r['d']}   ({r['sig']!r})")
        run(
            """
            UNWIND $pairs AS p
            MATCH (b:Biomarker {clinical_significance: p.sig})
            MATCH (x:Disease {name: p.disease})
            MERGE (b)-[r:INDICATES]->(x)
            SET r.derivation = 'curated_abbreviation_map',
                r.evidence = p.sig,
                r.confidence = 0.8,
                r.dataset = $ds,
                r.derived_at = $now
            """,
            pairs=pairs,
            ds=DERIVED,
            now=now,
        )

        if args.dry_run:
            print("\n--dry-run: nothing written")
            return 0

        print("\n═══ result ═══")
        for rel, src, dst in (
            ("EVALUATED_IN", "Drug", "ClinicalTrial"),
            ("IS_BIOMARKER_FOR", "Gene", "Biomarker"),
            ("INDICATES", "Biomarker", "Disease"),
        ):
            n = count(f"MATCH (:{src})-[r:{rel}]->(:{dst}) RETURN count(r)")
            derived = count(
                f"MATCH (:{src})-[r:{rel}]->(:{dst}) "
                "WHERE r.derivation IS NOT NULL RETURN count(r)"
            )
            print(f"  {rel:<18} {n:>3} edges ({derived} derived)")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
