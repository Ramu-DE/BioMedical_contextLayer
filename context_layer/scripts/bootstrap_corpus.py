#!/usr/bin/env python3
"""Bootstrap the governance text corpus into the vector store. Task 2.3.

Pulls real text from the live Neo4j graph (Policy, ARD, RuleType, RxNorm),
embeds it with Bedrock Titan, and upserts to Qdrant with full provenance.

Unlike the pre-existing idp_chunks collection, the stored ``text`` payload is
the actual source text — VectorRecord rejects stringified vectors outright.

Usage:
    .venv/bin/python scripts/bootstrap_corpus.py [--dry-run]
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
from context_layer.clients.embeddings import BedrockEmbeddings  # noqa: E402
from context_layer.clients.vector_store import (  # noqa: E402
    VectorRecord,
    build_vector_store,
)
from neo4j import GraphDatabase  # noqa: E402

# Each entry: (cypher, source label, id field, text builder)
CORPUS_QUERIES = [
    (
        "MATCH (p:Policy) RETURN p.policy_id AS id, p.name AS name, "
        "p.description AS description, p.scope AS scope",
        "neo4j:Policy",
        lambda r: f"Policy {r['name']}: {r['description']} (scope: {r['scope']})",
    ),
    (
        "MATCH (a:ARD) WHERE a.description IS NOT NULL "
        "RETURN a.ard_id AS id, a.name AS name, a.description AS description, "
        "a.grain AS grain, a.owner AS owner, a.certified AS certified",
        "neo4j:ARD",
        lambda r: (
            f"Analytics-ready dataset {r['name']}: {r['description']} "
            f"Grain: {r['grain']}. Owner: {r['owner']}. Certified: {r['certified']}."
        ),
    ),
    (
        "MATCH (t:RuleType) RETURN toString(t.rule_type_id) AS id, t.name AS name, "
        "t.description AS description, t.dimension AS dimension",
        "neo4j:RuleType",
        lambda r: (
            f"Data quality rule type {r['name']}: {r['description']} "
            f"Quality dimension: {r['dimension']}."
        ),
    ),
    (
        "MATCH (x:RxNorm) WHERE x.indication IS NOT NULL "
        "RETURN x.cui AS id, x.name AS name, x.indication AS indication, x.type AS type",
        "neo4j:RxNorm",
        lambda r: f"Medication {r['name']} is indicated for: {r['indication']}.",
    ),
]


def collect(driver, database: str) -> list[dict]:
    items: list[dict] = []
    with driver.session(database=database) as s:
        for cypher, source, build in CORPUS_QUERIES:
            for rec in s.run(cypher):
                r = dict(rec)
                items.append(
                    {
                        "source_id": r["id"],
                        "source": source,
                        "name": r.get("name", ""),
                        "text": build(r).strip(),
                    }
                )
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="collect but do not write")
    args = ap.parse_args()

    cfg = config_module.load()
    problems = cfg.check()
    if problems:
        print("config not ready:")
        for p in problems:
            print(f"  - {p}")
        return 1

    driver = GraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_user, str(cfg.neo4j_password))
    )
    try:
        items = collect(driver, cfg.neo4j_kg_database)
    finally:
        driver.close()

    print(f"collected {len(items)} text items from the graph")
    by_source: dict[str, int] = {}
    for it in items:
        by_source[it["source"]] = by_source.get(it["source"], 0) + 1
    for src, n in sorted(by_source.items()):
        print(f"  {n:>3}  {src}")

    if not items:
        print("nothing to index")
        return 1

    print(f"\nsample: {items[0]['text'][:110]}...")
    if args.dry_run:
        print("\n--dry-run: stopping before embedding")
        return 0

    embedder = BedrockEmbeddings(cfg)
    store = build_vector_store(cfg)
    store.ensure_collection(cfg.embedding_dim)
    print(f"\ncollection {cfg.qdrant_collection!r} ready (dim={cfg.embedding_dim})")

    now = datetime.now(timezone.utc).isoformat()
    records = []
    for it in items:
        vec = embedder.embed_one(it["text"])
        records.append(
            VectorRecord(
                text=it["text"],
                vector=vec,
                payload={
                    "source": it["source"],
                    "source_id": it["source_id"],
                    "name": it["name"],
                    "ingested_at": now,
                    "embedding_model": cfg.embedding_model_id,
                    "confidence": 1.0,
                },
            )
        )
    print(f"embedded {len(records)} items with {cfg.embedding_model_id}")

    written = store.upsert(records)
    print(f"upserted {written} points; collection now holds {store.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
