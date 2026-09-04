#!/usr/bin/env python3
"""Load the ontology and validate live graph nodes against SHACL. Tasks 1.2-1.5."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config as config_module  # noqa: E402
from context_layer.knowledge.ontology_loader import load_modules  # noqa: E402
from context_layer.knowledge.provenance import DcatCatalog, IriMinter  # noqa: E402
from context_layer.knowledge.shacl_gate import ShaclGate  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402

SAMPLE = {"Drug": "drug_id", "Disease": "disease_id", "Patient": "patient_id",
          "ClinicalTrial": "trial_id", "AdverseEvent": "event_id",
          "Biomarker": "biomarker_id", "Gene": "gene_id", "Protein": "protein_id"}

def main() -> int:
    cfg = config_module.load()
    bundle = load_modules()
    print(bundle.report())

    gate = ShaclGate()
    print(f"\nSHACL: {len(gate.shape_files)} shape files, "
          f"{len(gate.covered_classes)} target classes")
    print(f"  covered: {list(gate.covered_classes)}")
    mis = gate.miscased_targets()
    if mis:
        print(f"  UPSTREAM DEFECT: shape targets mis-cased vs ontology: "
              f"{[f'{a} should be {b}' for a, b in mis]}")
        print(f"  (handled by case-insensitive resolution)")

    minter = IriMinter()
    print(f"\nIRI minting (deterministic):")
    for _ in range(2):
        print(f"  Drug/D001 -> {minter.mint('Drug', 'D001')}")

    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, str(cfg.neo4j_password)))
    catalog = DcatCatalog(minter)
    print("\n═══ validating live nodes against SHACL ═══")
    total = conforming = 0
    try:
        with driver.session(database=cfg.neo4j_kg_database) as s:
            for label, key in SAMPLE.items():
                if not gate.has_shape(label):
                    print(f"  {label}: no shape declared, skipped")
                    continue
                rows = list(s.run(
                    f"MATCH (n:`{label}` {{dataset:'biomed'}}) RETURN properties(n) AS p LIMIT 25"))
                ok = bad = 0
                first_error = ""
                for r in rows:
                    props = dict(r["p"])
                    iri = minter.mint(label, str(props.get(key, "unknown")))
                    report = gate.validate(label, props, iri)
                    total += 1
                    if report.conforms:
                        ok += 1; conforming += 1
                    else:
                        bad += 1
                        if not first_error:
                            first_error = report.summary()[:150]
                print(f"  {label:<15} {ok:>3} conform, {bad:>3} violate"
                      + (f"  e.g. {first_error}" if first_error else ""))
                catalog.record(f"{label}-nodes", f"{label} nodes", "neo4j:biomed", len(rows))
    finally:
        driver.close()

    print(f"\ntotal: {conforming}/{total} nodes conform "
          f"({conforming/total:.0%})" if total else "no nodes validated")
    out = "dcat_catalog.ttl"
    catalog.serialize(out)
    print(f"DCAT: {len(catalog.records)} dataset records -> {out} "
          f"({len(catalog.to_rdf())} triples)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
