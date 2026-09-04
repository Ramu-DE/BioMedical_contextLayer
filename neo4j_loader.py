"""
Neo4j Aura Loader
=================

Loads eval results into the existing Neo4j Aura graph alongside
the context_layer nodes (Ctx_Event, Ctx_Decision, etc.).

Uses MERGE to avoid duplicates — safe to run multiple times.

Usage:
  python3 neo4j_loader.py               # load from eval_context.db
  python3 neo4j_loader.py --clean       # remove only Eval nodes, then reload
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from neo4j import GraphDatabase

def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

load_env()

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")


def get_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def clean_eval_nodes(session):
    """Remove only eval-related nodes, preserving the existing graph."""
    eval_labels = ["Eval_TestCase", "Eval_Run", "Eval_Verdict",
                   "Eval_Category", "Eval_FailureMode", "Eval_Proof"]
    for label in eval_labels:
        result = session.run(f"MATCH (n:{label}) DETACH DELETE n RETURN count(n) AS cnt")
        cnt = result.single()["cnt"]
        if cnt > 0:
            print(f"  Removed {cnt} {label} nodes.")


def load_from_sqlite(session):
    db_path = Path(__file__).parent / "eval_context.db"
    if not db_path.exists():
        print("ERROR: eval_context.db not found. Run run_eval_pipeline.py first.")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    # ── Test Cases ────────────────────────────────────────────
    rows = conn.execute("SELECT * FROM test_cases").fetchall()
    for r in rows:
        session.run("""
            MERGE (t:Eval_TestCase {test_id: $tid})
            SET t.category = $cat, t.prompt = $prompt,
                t.output = $output, t.ground_truth = $gt,
                t.is_correct = $ic
        """, tid=r[0], cat=r[1], prompt=r[2], output=r[3], gt=r[4], ic=bool(r[5]))
    print(f"  Loaded {len(rows)} Eval_TestCase nodes.")

    # ── Categories ────────────────────────────────────────────
    cats = conn.execute("SELECT DISTINCT category FROM test_cases").fetchall()
    for (cat,) in cats:
        session.run("""
            MERGE (c:Eval_Category {name: $cat})
            WITH c
            MATCH (t:Eval_TestCase {category: $cat})
            MERGE (t)-[:BELONGS_TO]->(c)
        """, cat=cat)
    print(f"  Loaded {len(cats)} Eval_Category nodes.")

    # ── Eval Runs ─────────────────────────────────────────────
    rows = conn.execute("SELECT * FROM eval_runs").fetchall()
    for r in rows:
        session.run("""
            MERGE (e:Eval_Run {run_id: $rid})
            SET e.timestamp = $ts, e.model = $model, e.config = $cfg
        """, rid=r[0], ts=r[1], model=r[2], cfg=r[3])
    print(f"  Loaded {len(rows)} Eval_Run nodes.")

    # ── Verdicts + Relationships ──────────────────────────────
    rows = conn.execute("SELECT * FROM verdicts").fetchall()
    for r in rows:
        session.run("""
            MERGE (v:Eval_Verdict {verdict_id: $vid})
            SET v.method = $method, v.result = $result,
                v.correct = $correct, v.explanation = $expl,
                v.latency_ms = $lat
            WITH v
            MATCH (t:Eval_TestCase {test_id: $tid})
            MERGE (t)-[:RECEIVED_VERDICT]->(v)
            WITH v
            MATCH (e:Eval_Run {run_id: $rid})
            MERGE (e)-[:CONTAINS_VERDICT]->(v)
        """, vid=r[0], rid=r[1], tid=r[2], method=r[3], result=r[4],
            correct=bool(r[5]), expl=r[6][:500] if r[6] else "", lat=r[7])
    print(f"  Loaded {len(rows)} Eval_Verdict nodes with relationships.")

    # ── Failure Modes ─────────────────────────────────────────
    session.run("""
        MATCH (v:Eval_Verdict)
        WHERE v.correct = false AND v.result = 'PASS'
        MERGE (f:Eval_FailureMode {name: 'FalsePass:' + v.method})
        MERGE (v)-[:EXHIBITS]->(f)
    """)
    session.run("""
        MATCH (v:Eval_Verdict)
        WHERE v.correct = false AND v.result <> 'PASS'
        MERGE (f:Eval_FailureMode {name: 'FalseNegative:' + v.method})
        MERGE (v)-[:EXHIBITS]->(f)
    """)
    print("  Created Eval_FailureMode nodes.")

    # ── Proof Summary Node ────────────────────────────────────
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN correct = 0 AND result = 'PASS' THEN 1 ELSE 0 END) as false_passes
        FROM verdicts WHERE method LIKE 'llm_judge%'
    """).fetchone()
    total_judgments, false_passes = stats
    fp_rate = round(false_passes / total_judgments * 100, 1) if total_judgments else 0

    session.run("""
        MERGE (p:Eval_Proof {name: 'LLM_Judge_Worse_Than_No_Evals'})
        SET p.thesis = 'A permissive LLM judge creates false confidence that masks real failures',
            p.total_judgments = $total,
            p.false_passes = $fp,
            p.false_pass_rate_pct = $rate,
            p.conclusion = 'LLM judge removes more protection than it adds'
        WITH p
        MATCH (r:Eval_Run)
        MERGE (p)-[:EVIDENCED_BY]->(r)
    """, total=total_judgments, fp=false_passes, rate=fp_rate)
    print(f"  Created Eval_Proof summary (false pass rate: {fp_rate}%).")

    # ── Link to existing context layer if present ─────────────
    result = session.run("""
        MATCH (ctx:Ctx_Event) RETURN count(ctx) AS cnt
    """).single()["cnt"]
    if result > 0:
        session.run("""
            MATCH (p:Eval_Proof {name: 'LLM_Judge_Worse_Than_No_Evals'})
            MERGE (ctx:Ctx_Event {name: 'Eval_LLM_Judge_Proof', type: 'evaluation'})
            SET ctx.description = 'Proved LLM-as-Judge worse than no evals: ' + toString(p.false_pass_rate_pct) + '% false pass rate'
            MERGE (ctx)-[:TRIGGERED]->(p)
        """)
        print(f"  Linked to existing context layer ({result} Ctx_Event nodes found).")

    conn.close()


def print_summary(session):
    print()
    print("  ── Neo4j Graph Summary ──")
    result = session.run("""
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label STARTS WITH 'Eval_')
        RETURN labels(n)[0] AS label, count(n) AS cnt
        ORDER BY cnt DESC
    """)
    for record in result:
        print(f"    {record['label']:<25s} {record['cnt']:>4d} nodes")

    result = session.run("""
        MATCH ()-[r]->()
        WHERE type(r) IN ['RECEIVED_VERDICT','CONTAINS_VERDICT','BELONGS_TO','EXHIBITS','EVIDENCED_BY','TRIGGERED']
        RETURN type(r) AS rel, count(r) AS cnt
        ORDER BY cnt DESC
    """)
    for record in result:
        print(f"    {record['rel']:<25s} {record['cnt']:>4d} edges")

    total = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
    print(f"\n  Total graph: {total} nodes (existing + eval)")


def main():
    print("=" * 60)
    print("  Loading Eval Results → Neo4j Aura")
    print("=" * 60)
    print(f"  URI: {NEO4J_URI}")
    print()

    driver = get_driver()
    print("  Connected.")

    with driver.session(database=NEO4J_DATABASE) as session:
        if "--clean" in sys.argv:
            print("  Cleaning previous eval nodes...")
            clean_eval_nodes(session)

        print("  Loading eval data...")
        load_from_sqlite(session)
        print_summary(session)

    driver.close()
    print()
    print("  Done! Explore in Neo4j Browser:")
    print("    MATCH (p:Eval_Proof)-[*1..3]-(n) RETURN p, n")
    print("    MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict) WHERE v.correct = false RETURN t, v")
    print()


if __name__ == "__main__":
    main()
