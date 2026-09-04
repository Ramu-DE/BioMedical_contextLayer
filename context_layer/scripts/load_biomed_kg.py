#!/usr/bin/env python3
"""Load the biomedical dataset into Neo4j. Tasks 0.4 and 1.6.

Idempotent: every write is a MERGE on the natural key, so re-running converges
rather than duplicating. Every node carries provenance (Requirement 2.2) and
``dataset='biomed'`` so it stays distinguishable from the pre-existing
enterprise governance subgraph.

Also builds the CURIE crosswalk from external_mappings.csv, which is what the
entity linker resolves against (Requirement 4.1, 4.3).

Usage:
    .venv/bin/python scripts/load_biomed_kg.py --repo /tmp/triage/BioMedical_...
    .venv/bin/python scripts/load_biomed_kg.py --dry-run
    .venv/bin/python scripts/load_biomed_kg.py --purge   # remove dataset=biomed
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from context_layer.knowledge.biomed_manifest import (  # noqa: E402
    CURIE_PREFIX,
    DERIVED_RELS,
    MAPPING_ENTITY_LABEL,
    NODES,
    RELS,
)
from neo4j import GraphDatabase  # noqa: E402

DATASET = "biomed"
DEFAULT_REPO = Path(
    "/tmp/triage/BioMedical_KnowledgeGraph_ontology_MCP/neo4j-neptune-mcp-platform"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [
            {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


def coerce(value: str):
    """CSV is all strings; recover numbers so range rules can compare them."""
    if value is None or value == "":
        return None
    low = value.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


class Loader:
    def __init__(self, driver, database: str, now: str, dry_run: bool) -> None:
        self.driver = driver
        self.database = database
        self.now = now
        self.dry_run = dry_run
        self.stats: dict[str, int] = {}

    def run(self, cypher: str, **params):
        if self.dry_run:
            return []
        with self.driver.session(database=self.database) as s:
            return list(s.run(cypher, **params))

    # ── constraints ───────────────────────────────────────────────────────────

    def ensure_constraints(self) -> None:
        for spec in NODES:
            self.run(
                f"CREATE CONSTRAINT ctx_{spec.label.lower()}_key IF NOT EXISTS "
                f"FOR (n:`{spec.label}`) REQUIRE n.`{spec.key}` IS UNIQUE"
            )
        self.run(
            "CREATE CONSTRAINT ctx_externalconcept_curie IF NOT EXISTS "
            "FOR (n:ExternalConcept) REQUIRE n.curie IS UNIQUE"
        )

    # ── nodes ─────────────────────────────────────────────────────────────────

    def load_nodes(self, repo: Path) -> None:
        node_dir = repo / "sampledata" / "nodes"
        for spec in NODES:
            path = node_dir / f"{spec.csv}.csv"
            if not path.is_file():
                print(f"  MISSING {path.name}")
                continue
            rows = read_csv(path)
            payload = []
            for r in rows:
                if not r.get(spec.key):
                    continue
                props = {k: coerce(v) for k, v in r.items() if k != spec.key}
                props = {k: v for k, v in props.items() if v is not None}
                payload.append({"key": r[spec.key], "props": props})
            self.run(
                f"""
                UNWIND $rows AS row
                MERGE (n:`{spec.label}` {{`{spec.key}`: row.key}})
                SET n += row.props,
                    n.dataset = $dataset,
                    n.source = $source,
                    n.ingested_at = $now,
                    n.confidence = 1.0
                """,
                rows=payload,
                dataset=DATASET,
                source=f"github:ramu-de/BioMedical_KnowledgeGraph_ontology_MCP/{spec.csv}.csv",
                now=self.now,
            )
            self.stats[f"node:{spec.label}"] = len(payload)
            print(f"  {len(payload):>4}  :{spec.label}")

    # ── relationships ─────────────────────────────────────────────────────────

    def _load_rel(self, rows, spec, prop_cols) -> tuple[int, int]:
        """Returns (attempted, created). A gap means the keys do not match."""
        payload = []
        for r in rows:
            src, dst = r.get(spec.src_col), r.get(spec.dst_col)
            if not src or not dst:
                continue
            props = {c: coerce(r[c]) for c in prop_cols if r.get(c)}
            payload.append({"src": src, "dst": dst, "props": props})
        if not payload:
            return 0, 0
        res = self.run(
            f"""
            UNWIND $rows AS row
            MATCH (a:`{spec.src_label}` {{`{spec.src_col}`: row.src}})
            MATCH (b:`{spec.dst_label}` {{`{spec.target_key}`: row.dst}})
            MERGE (a)-[r:`{spec.rel_type}`]->(b)
            SET r += row.props, r.dataset = $dataset, r.ingested_at = $now
            RETURN count(r) AS created
            """,
            rows=payload,
            dataset=DATASET,
            now=self.now,
        )
        created = res[0]["created"] if res else 0
        return len(payload), created

    @staticmethod
    def _report(spec, attempted: int, created: int, tag: str = "") -> None:
        """Surface key mismatches instead of letting zero-edge loads look fine."""
        suffix = f"  {tag}" if tag else ""
        line = (
            f"  {created:>4}  ({spec.src_label})-[:{spec.rel_type}]->"
            f"({spec.dst_label}){suffix}"
        )
        if attempted and created == 0:
            line += f"  !! 0 of {attempted} matched — key mismatch on {spec.target_key}"
        elif created < attempted:
            line += f"  ({attempted - created} of {attempted} unmatched)"
        print(line)

    def load_rels(self, repo: Path) -> None:
        rel_dir = repo / "relationships"
        for spec in RELS:
            path = rel_dir / f"{spec.csv}.csv"
            if not path.is_file():
                print(f"  MISSING {path.name}")
                continue
            rows = read_csv(path)
            prop_cols = [
                c for c in (rows[0] if rows else {}) if c not in (spec.src_col, spec.dst_col)
            ]
            attempted, created = self._load_rel(rows, spec, prop_cols)
            key = f"rel:{spec.rel_type}"
            self.stats[key] = self.stats.get(key, 0) + created
            self._report(spec, attempted, created)

    def load_derived_rels(self, repo: Path) -> None:
        """Edges implied by a foreign key inside a node CSV."""
        node_dir = repo / "sampledata" / "nodes"
        for spec in DERIVED_RELS:
            path = node_dir / f"{spec.csv}.csv"
            if not path.is_file():
                continue
            rows = read_csv(path)
            if not rows or spec.dst_col not in rows[0]:
                print(f"  SKIP {spec.csv}: no column {spec.dst_col!r}")
                continue
            attempted, created = self._load_rel(rows, spec, prop_cols=[])
            key = f"rel:{spec.rel_type}"
            self.stats[key] = self.stats.get(key, 0) + created
            self._report(spec, attempted, created, "[derived]")

    def load_polymorphic(self, repo: Path) -> None:
        rel_dir = repo / "relationships"

        rows = read_csv(rel_dir / "node_belongs_to_cluster.csv")
        by_type: dict[str, list[dict]] = {}
        for r in rows:
            by_type.setdefault(r["node_type"], []).append(r)
        for node_type, group in by_type.items():
            label = MAPPING_ENTITY_LABEL.get(node_type, node_type)
            key = next(
                (s.key for s in NODES if s.label == label), None
            )
            if key is None:
                print(f"  SKIP cluster members of unknown type {node_type}")
                continue
            self.run(
                f"""
                UNWIND $rows AS row
                MATCH (a:`{label}` {{`{key}`: row.node_id}})
                MATCH (c:Cluster {{cluster_id: row.cluster_id}})
                MERGE (a)-[r:BELONGS_TO_CLUSTER]->(c)
                SET r.membership_score = toFloat(row.membership_score),
                    r.dataset = $dataset, r.ingested_at = $now
                """,
                rows=group,
                dataset=DATASET,
                now=self.now,
            )
            print(f"  {len(group):>4}  ({label})-[:BELONGS_TO_CLUSTER]->(Cluster)")

        rows = read_csv(rel_dir / "policy_governs_entity.csv")
        by_type = {}
        for r in rows:
            by_type.setdefault(r["entity_type"], []).append(r)
        for entity_type, group in by_type.items():
            label = MAPPING_ENTITY_LABEL.get(entity_type, entity_type)
            key = next((s.key for s in NODES if s.label == label), None)
            if key is None:
                print(f"  SKIP policy targets of unknown type {entity_type}")
                continue
            self.run(
                f"""
                UNWIND $rows AS row
                MATCH (p:DataGovernancePolicy {{policy_id: row.policy_id}})
                MATCH (e:`{label}` {{`{key}`: row.entity_id}})
                MERGE (p)-[r:GOVERNS_ENTITY]->(e)
                SET r.enforcement_level = row.enforcement_level,
                    r.dataset = $dataset, r.ingested_at = $now
                """,
                rows=group,
                dataset=DATASET,
                now=self.now,
            )
            print(f"  {len(group):>4}  (DataGovernancePolicy)-[:GOVERNS_ENTITY]->({label})")

    # ── CURIE crosswalk ───────────────────────────────────────────────────────

    def load_crosswalk(self, repo: Path) -> None:
        path = repo / "sampledata" / "nodes" / "external_mappings.csv"
        rows = read_csv(path)
        grouped: dict[str, list[dict]] = {}
        unknown_std: set[str] = set()
        for r in rows:
            std = r["standard"]
            prefix = CURIE_PREFIX.get(std)
            if prefix is None:
                unknown_std.add(std)
                prefix = std.upper().replace(" ", "")
            label = MAPPING_ENTITY_LABEL.get(r["entity_type"])
            if label is None:
                continue
            grouped.setdefault(label, []).append(
                {
                    "entity_id": r["entity_id"],
                    "curie": f"{prefix}:{r['external_id']}",
                    "standard": std,
                    "external_id": r["external_id"],
                    "confidence": float(r.get("confidence") or 1.0),
                }
            )
        total = 0
        for label, group in grouped.items():
            key = next((s.key for s in NODES if s.label == label), None)
            if key is None:
                continue
            self.run(
                f"""
                UNWIND $rows AS row
                MATCH (e:`{label}` {{`{key}`: row.entity_id}})
                MERGE (c:ExternalConcept {{curie: row.curie}})
                SET c.standard = row.standard,
                    c.external_id = row.external_id,
                    c.dataset = $dataset,
                    c.ingested_at = $now
                MERGE (e)-[m:MAPS_TO]->(c)
                SET m.confidence = row.confidence,
                    m.dataset = $dataset, m.ingested_at = $now
                """,
                rows=group,
                dataset=DATASET,
                now=self.now,
            )
            total += len(group)
            print(f"  {len(group):>4}  ({label})-[:MAPS_TO]->(ExternalConcept)")
        if unknown_std:
            print(f"  note: unmapped vocabularies, prefix derived: {sorted(unknown_std)}")
        self.stats["crosswalk"] = total

    # ── purge ─────────────────────────────────────────────────────────────────

    def purge(self) -> None:
        res = self.run(
            "MATCH (n) WHERE n.dataset = $dataset "
            "WITH n LIMIT 50000 DETACH DELETE n RETURN count(n) AS c",
            dataset=DATASET,
        )
        deleted = res[0]["c"] if res else 0
        print(f"deleted {deleted} nodes where dataset='{DATASET}'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--purge", action="store_true", help="remove dataset=biomed nodes")
    args = ap.parse_args()

    cfg = config_module.load()
    problems = [p for p in cfg.check() if "QDRANT" not in p]
    if problems:
        print("config not ready:")
        for p in problems:
            print(f"  - {p}")
        return 1

    if not args.purge and not (args.repo / "sampledata" / "nodes").is_dir():
        print(f"dataset not found under {args.repo}")
        print("clone it first:")
        print("  git clone --depth 1 https://github.com/ramu-de/"
              "BioMedical_KnowledgeGraph_ontology_MCP.git /tmp/triage/BioMedical_KnowledgeGraph_ontology_MCP")
        return 1

    driver = GraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_user, str(cfg.neo4j_password))
    )
    now = datetime.now(timezone.utc).isoformat()
    loader = Loader(driver, cfg.neo4j_kg_database, now, args.dry_run)
    try:
        if args.purge:
            loader.purge()
            return 0

        before = loader.run("MATCH (n) RETURN count(n) AS c")
        print(f"graph before: {before[0]['c'] if before else '?'} nodes\n")

        print("═══ constraints ═══")
        loader.ensure_constraints()
        print("  done\n")
        print("═══ nodes ═══")
        loader.load_nodes(args.repo)
        print("\n═══ relationships ═══")
        loader.load_rels(args.repo)
        print("\n═══ derived relationships ═══")
        loader.load_derived_rels(args.repo)
        print("\n═══ polymorphic relationships ═══")
        loader.load_polymorphic(args.repo)
        print("\n═══ CURIE crosswalk ═══")
        loader.load_crosswalk(args.repo)

        if not args.dry_run:
            after = loader.run(
                "MATCH (n) RETURN count(n) AS total, "
                "count(CASE WHEN n.dataset = 'biomed' THEN 1 END) AS biomed"
            )[0]
            rels = loader.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
            print(
                f"\ngraph after: {after['total']} nodes "
                f"({after['biomed']} biomed), {rels} relationships"
            )
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
