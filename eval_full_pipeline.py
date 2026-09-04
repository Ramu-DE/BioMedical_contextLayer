"""
Full Eval Pipeline: LLM Judge Proof + Neo4j + Qdrant
=====================================================

Proves LLM-as-Judge is worse than no evals, and stores results across
three integrated backends:

  SQLite   — local structured results (fast, offline)
  Neo4j    — graph relationships (test→verdict→failure patterns)
  Qdrant   — vector embeddings (semantic similarity of failures)

The Qdrant integration adds a new eval dimension: SEMANTIC SIMILARITY.
Instead of just PASS/FAIL, we can find test cases that are semantically
similar to known failure patterns — predicting WHERE the judge will fail
before it even runs.

Pipeline:
  1. Run LLM judge + deterministic checks (stores in SQLite + Neo4j)
  2. Embed all test cases + verdicts into Qdrant
  3. Semantic analysis: find clusters of judge failures
  4. Cross-reference: Qdrant similarity × Neo4j graph patterns
"""

import anthropic
import boto3
import json
import os
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter,
    FieldCondition, MatchValue
)
from neo4j import GraphDatabase
from context_layer import ContextLayer

# ─── Config ───────────────────────────────────────────────────
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

load_env()

LLM_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
QDRANT_COLLECTION = "eval_verdicts"

llm_client = anthropic.AnthropicBedrock(
    aws_region=os.environ.get("AWS_REGION", "us-west-2")
)
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("AWS_REGION", "us-west-2")
)
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=30,
    check_compatibility=False,
)
neo4j_driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)
NEO4J_DB = os.environ["NEO4J_DATABASE"]


# ─── Helpers ──────────────────────────────────────────────────
def embed(text):
    resp = bedrock_runtime.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text[:8000], "dimensions": EMBED_DIM}),
        contentType="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]


def ask_judge(system, user_msg):
    start = time.time()
    r = llm_client.messages.create(
        model=LLM_MODEL, max_tokens=400, system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    elapsed = (time.time() - start) * 1000
    text = r.content[0].text.strip()
    first_line = text.split("\n")[0].upper()
    verdict = "PASS" if "PASS" in first_line else "FAIL" if "FAIL" in first_line else "UNCLEAR"
    return verdict, text, elapsed


# ─── Test Cases ───────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "subtle-fact-1", "category": "Plausible Factual Error",
        "prompt": "What year was the Treaty of Westphalia signed?",
        "output": "The Treaty of Westphalia was signed in 1647, ending the Thirty Years' War. It established state sovereignty.",
        "ground_truth": "1648", "is_correct": False,
        "det_check": lambda o: "1648" in o,
    },
    {
        "id": "subtle-math-1", "category": "Arithmetic Error",
        "prompt": "15% off $84. What's the sale price?",
        "output": "15% of $84 = $12.40. Sale price: $84 - $12.40 = $71.60",
        "ground_truth": "$71.40", "is_correct": False,
        "det_check": lambda o: "71.40" in o or "71.4" in o,
    },
    {
        "id": "subtle-math-2", "category": "Arithmetic Error",
        "prompt": "Days in Jan+Feb+Mar (non-leap year)?",
        "output": "Jan: 31, Feb: 28, Mar: 31. Total: 31 + 28 + 31 = 91 days",
        "ground_truth": "90", "is_correct": False,
        "det_check": lambda o: "90" in o and "91" not in o,
    },
    {
        "id": "subtle-math-3", "category": "Arithmetic Error",
        "prompt": "60 mph for 2.5 hours in km? (1 mi=1.609 km)",
        "output": "150 miles × 1.609 = 241.25 km",
        "ground_truth": "241.35", "is_correct": False,
        "det_check": lambda o: "241.35" in o,
    },
    {
        "id": "subtle-cs-1", "category": "CS Misconception",
        "prompt": "Time complexity of binary search?",
        "output": "O(log n) average, but worst case degrades to O(n log n) due to boundary rebalancing.",
        "ground_truth": "O(log n) always", "is_correct": False,
        "det_check": lambda o: "log n" in o.lower() and "n log n" not in o.lower(),
    },
    {
        "id": "subtle-api-1", "category": "Best Practice Error",
        "prompt": "HTTP status code for resource created?",
        "output": "Return HTTP 200 OK when a resource is created successfully.",
        "ground_truth": "201 Created", "is_correct": False,
        "det_check": lambda o: "201" in o,
    },
    {
        "id": "subtle-science-1", "category": "Common Misconception",
        "prompt": "Why add salt to pasta water?",
        "output": "Salt raises the boiling point by 2-3°C, cooking pasta faster.",
        "ground_truth": "Flavor. Boiling point change is negligible.", "is_correct": False,
        "det_check": lambda o: "flavor" in o.lower() or "taste" in o.lower(),
    },
    {
        "id": "subtle-logic-1", "category": "Logical Fallacy",
        "prompt": "All roses are flowers. Some flowers fade quickly. Do some roses fade quickly?",
        "output": "Yes — roses are flowers, some flowers fade, so some roses must fade quickly.",
        "ground_truth": "No — undistributed middle fallacy", "is_correct": False,
        "det_check": lambda o: any(w in o.lower() for w in ["no", "cannot", "fallacy"]),
    },
    {
        "id": "subtle-stat-1", "category": "Statistical Error",
        "prompt": "After 10 heads, probability next flip is heads?",
        "output": "Very low — about 1/2048. The law of averages says tails is due.",
        "ground_truth": "50%. Independent events.", "is_correct": False,
        "det_check": lambda o: "50%" in o or "independent" in o.lower(),
    },
    {
        "id": "subtle-bio-1", "category": "Biomedical Error",
        "prompt": "What class of drug is Pembrolizumab?",
        "output": "Pembrolizumab is a small molecule kinase inhibitor used in oncology.",
        "ground_truth": "Monoclonal antibody / PD-1 inhibitor", "is_correct": False,
        "det_check": lambda o: "antibody" in o.lower() or "pd-1" in o.lower(),
    },
    {
        "id": "correct-palindrome", "category": "Control (Correct)",
        "prompt": "Python palindrome checker?",
        "output": "def is_palindrome(s): return s.lower().replace(' ','') == s.lower().replace(' ','')[::-1]",
        "ground_truth": "Correct", "is_correct": True,
        "det_check": lambda o: "[::-1]" in o,
    },
    {
        "id": "correct-water", "category": "Control (Correct)",
        "prompt": "% of Earth's surface covered by water?",
        "output": "Approximately 71% of Earth's surface is covered by water.",
        "ground_truth": "Correct", "is_correct": True,
        "det_check": lambda o: "71" in o,
    },
]


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def phase_1_run_evals():
    """Run LLM judge + deterministic checks, store in SQLite."""
    print("  PHASE 1: Running evals")
    print("  " + "─" * 50)

    ctx = ContextLayer()
    judge_system = "You are an evaluator assessing AI responses for correctness."
    run_id = ctx.start_run(LLM_MODEL, {"judge_type": "naive"})

    results = []
    for tc in TEST_CASES:
        sys.stdout.write(f"    {tc['id']:<25s}")
        sys.stdout.flush()

        # LLM Judge
        user_msg = f"Question: {tc['prompt']}\nAI Response: {tc['output']}\nIs this correct? Say PASS or FAIL."
        verdict, explanation, latency = ask_judge(judge_system, user_msg)
        judge_correct = (verdict == "PASS") == tc["is_correct"]

        ctx.record_verdict(tc["id"], "llm_judge", verdict, judge_correct, explanation, latency)

        # Deterministic
        try:
            det_passed = tc["det_check"](tc["output"])
        except Exception:
            det_passed = False
        det_correct = det_passed == tc["is_correct"]
        ctx.record_verdict(tc["id"], "deterministic", "PASS" if det_passed else "FAIL", det_correct, "", 0)

        # Store test case
        ctx.add_test_case(tc["id"], tc["category"], tc["prompt"], tc["output"],
                          tc["ground_truth"], tc["is_correct"])

        results.append({
            "id": tc["id"], "category": tc["category"],
            "is_bad": not tc["is_correct"],
            "judge_verdict": verdict, "judge_correct": judge_correct,
            "det_correct": det_correct,
            "explanation": explanation, "latency": latency,
        })

        jicon = "✓" if judge_correct else "✗"
        dicon = "✓" if det_correct else "✗"
        print(f" Judge:{verdict:<7s}{jicon}  Det:{dicon}  ({latency:.0f}ms)")
        time.sleep(0.3)

    ctx.close()
    return results


def phase_2_embed_to_qdrant(results):
    """Embed test cases + verdicts into Qdrant for semantic analysis."""
    print()
    print("  PHASE 2: Embedding into Qdrant")
    print("  " + "─" * 50)

    # Create or recreate collection
    try:
        qdrant.delete_collection(QDRANT_COLLECTION)
    except Exception:
        pass

    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    from qdrant_client.models import PayloadSchemaType
    qdrant.create_payload_index(QDRANT_COLLECTION, "judge_correct", PayloadSchemaType.BOOL)
    qdrant.create_payload_index(QDRANT_COLLECTION, "is_correct", PayloadSchemaType.BOOL)
    qdrant.create_payload_index(QDRANT_COLLECTION, "category", PayloadSchemaType.KEYWORD)
    print(f"    Created collection: {QDRANT_COLLECTION} (with payload indexes)")

    points = []
    for i, tc in enumerate(TEST_CASES):
        r = next(x for x in results if x["id"] == tc["id"])
        text = f"Question: {tc['prompt']} Answer: {tc['output']}"
        sys.stdout.write(f"    Embedding {tc['id']:<25s}")
        sys.stdout.flush()

        vec = embed(text)
        point = PointStruct(
            id=i,
            vector=vec,
            payload={
                "test_id": tc["id"],
                "category": tc["category"],
                "prompt": tc["prompt"],
                "output": tc["output"],
                "ground_truth": tc["ground_truth"],
                "is_correct": tc["is_correct"],
                "judge_verdict": r["judge_verdict"],
                "judge_correct": r["judge_correct"],
                "det_correct": r["det_correct"],
                "explanation": r["explanation"][:300],
                "source": "eval_pipeline",
                "embedding_model": EMBED_MODEL,
                "indexed_at": datetime.now().isoformat(),
            },
        )
        points.append(point)
        print(" done")

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"    Upserted {len(points)} points.")
    return points


def phase_3_semantic_analysis():
    """Find clusters of judge failures via vector similarity."""
    print()
    print("  PHASE 3: Semantic failure analysis")
    print("  " + "─" * 50)

    # Find all false passes
    false_passes = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="judge_correct", match=MatchValue(value=False)),
            FieldCondition(key="is_correct", match=MatchValue(value=False)),
        ]),
        with_vectors=True,
        limit=20,
    )[0]

    if not false_passes:
        print("    No false passes found this run.")
        return []

    print(f"    Found {len(false_passes)} false passes. Searching for similar patterns...")
    print()

    similar_clusters = []
    for fp in false_passes:
        # Find what's semantically close to each false pass
        neighbors = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=fp.vector,
            limit=4,
        ).points

        cluster = {
            "anchor": fp.payload["test_id"],
            "anchor_category": fp.payload["category"],
            "neighbors": [],
        }

        print(f"    False pass: {fp.payload['test_id']} ({fp.payload['category']})")
        for n in neighbors:
            if n.id == fp.id:
                continue
            cluster["neighbors"].append({
                "test_id": n.payload["test_id"],
                "category": n.payload["category"],
                "judge_correct": n.payload["judge_correct"],
                "score": n.score,
            })
            status = "also failed" if not n.payload["judge_correct"] else "caught"
            print(f"      → {n.payload['test_id']:<25s} sim={n.score:.3f}  judge {status}")

        similar_clusters.append(cluster)
        print()

    # Cross-reference with existing biomed collection
    print("    Cross-referencing with existing biomedical context...")
    for fp in false_passes:
        text = f"{fp.payload['prompt']} {fp.payload['output']}"
        vec = embed(text)
        biomed_hits = qdrant.query_points(
            collection_name="ctx_biomed_chunks",
            query=vec,
            limit=2,
        ).points

        if biomed_hits:
            print(f"    {fp.payload['test_id']} relates to:")
            for h in biomed_hits:
                print(f"      → {h.payload.get('text', '')[:100]}  (sim={h.score:.3f})")
            print()

    return similar_clusters


def phase_4_neo4j_sync(results, clusters):
    """Push semantic analysis results into Neo4j graph."""
    print()
    print("  PHASE 4: Syncing to Neo4j")
    print("  " + "─" * 50)

    with neo4j_driver.session(database=NEO4J_DB) as session:
        # Clean previous eval nodes
        for label in ["Eval_TestCase", "Eval_Run", "Eval_Verdict",
                       "Eval_Category", "Eval_FailureMode", "Eval_Proof",
                       "Eval_Cluster"]:
            session.run(f"MATCH (n:{label}) DETACH DELETE n")

        # Load test cases
        for tc in TEST_CASES:
            r = next(x for x in results if x["id"] == tc["id"])
            session.run("""
                CREATE (t:Eval_TestCase {
                    test_id: $tid, category: $cat, prompt: $prompt,
                    output: $output, ground_truth: $gt, is_correct: $ic
                })
            """, tid=tc["id"], cat=tc["category"], prompt=tc["prompt"],
                output=tc["output"], gt=tc["ground_truth"], ic=tc["is_correct"])

        # Load categories
        cats = set(tc["category"] for tc in TEST_CASES)
        for cat in cats:
            session.run("""
                MERGE (c:Eval_Category {name: $cat})
                WITH c
                MATCH (t:Eval_TestCase {category: $cat})
                MERGE (t)-[:BELONGS_TO]->(c)
            """, cat=cat)

        # Load verdicts
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session.run("""
            CREATE (e:Eval_Run {
                run_id: $rid, timestamp: $ts, model: $model,
                pipeline: 'full_integrated'
            })
        """, rid=run_id, ts=datetime.now().isoformat(), model=LLM_MODEL)

        for r in results:
            vid = hashlib.md5(f"{run_id}:{r['id']}".encode()).hexdigest()[:12]
            session.run("""
                CREATE (v:Eval_Verdict {
                    verdict_id: $vid, method: 'llm_judge',
                    result: $result, correct: $correct,
                    explanation: $expl, latency_ms: $lat
                })
                WITH v
                MATCH (t:Eval_TestCase {test_id: $tid})
                CREATE (t)-[:RECEIVED_VERDICT]->(v)
                WITH v
                MATCH (e:Eval_Run {run_id: $rid})
                CREATE (e)-[:CONTAINS_VERDICT]->(v)
            """, vid=vid, result=r["judge_verdict"], correct=r["judge_correct"],
                expl=r["explanation"][:300], lat=r["latency"],
                tid=r["id"], rid=run_id)

        # Failure modes
        session.run("""
            MATCH (v:Eval_Verdict) WHERE v.correct = false AND v.result = 'PASS'
            MERGE (f:Eval_FailureMode {name: 'FalsePass'})
            CREATE (v)-[:EXHIBITS]->(f)
        """)

        # Semantic clusters
        for cl in clusters:
            session.run("""
                MERGE (c:Eval_Cluster {anchor: $anchor})
                SET c.category = $cat
                WITH c
                MATCH (t:Eval_TestCase {test_id: $anchor})
                MERGE (t)-[:ANCHORS]->(c)
            """, anchor=cl["anchor"], cat=cl["anchor_category"])

            for nb in cl["neighbors"]:
                session.run("""
                    MATCH (c:Eval_Cluster {anchor: $anchor})
                    MATCH (t:Eval_TestCase {test_id: $nbid})
                    MERGE (t)-[:SIMILAR_TO {score: $score}]->(c)
                """, anchor=cl["anchor"], nbid=nb["test_id"], score=nb["score"])

        # Proof summary
        bad = [r for r in results if r["is_bad"]]
        fp = [r for r in bad if not r["judge_correct"]]
        fp_rate = round(len(fp) / len(bad) * 100, 1) if bad else 0

        session.run("""
            CREATE (p:Eval_Proof {
                name: 'LLM_Judge_Worse_Than_No_Evals',
                thesis: 'Permissive LLM judge creates false confidence masking real failures',
                false_pass_rate_pct: $rate,
                false_passes: $fp,
                total_bad: $total,
                model: $model,
                timestamp: $ts,
                has_vector_analysis: true,
                qdrant_collection: $qcoll
            })
            WITH p
            MATCH (e:Eval_Run {run_id: $rid})
            CREATE (p)-[:EVIDENCED_BY]->(e)
        """, rate=fp_rate, fp=len(fp), total=len(bad), model=LLM_MODEL,
            ts=datetime.now().isoformat(), qcoll=QDRANT_COLLECTION, rid=run_id)

        # Link to existing Ctx layer
        ctx_count = session.run(
            "MATCH (n:Ctx_Event) RETURN count(n) AS cnt"
        ).single()["cnt"]
        if ctx_count > 0:
            session.run("""
                MATCH (p:Eval_Proof)
                MERGE (ctx:Ctx_Event {name: 'Eval_Full_Pipeline', type: 'evaluation'})
                SET ctx.description = 'Integrated eval proof: Neo4j + Qdrant + LLM Judge'
                MERGE (ctx)-[:TRIGGERED]->(p)
            """)

        # Count final state
        eval_nodes = session.run("""
            MATCH (n) WHERE any(l IN labels(n) WHERE l STARTS WITH 'Eval_')
            RETURN count(n) AS cnt
        """).single()["cnt"]
        total_nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]

        print(f"    Loaded {eval_nodes} eval nodes (total graph: {total_nodes})")


def print_summary(results):
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print()

    bad = [r for r in results if r["is_bad"]]
    good = [r for r in results if not r["is_bad"]]
    fp = [r for r in bad if not r["judge_correct"]]
    fp_rate = len(fp) / len(bad) * 100 if bad else 0

    print(f"  LLM Judge on {len(bad)} bad outputs: {len(fp)} false passes ({fp_rate:.0f}%)")
    print(f"  LLM Judge on {len(good)} good outputs: {sum(1 for r in good if r['judge_correct'])}/{len(good)} correct")
    print(f"  Deterministic on all: {sum(1 for r in results if r['det_correct'])}/{len(results)} correct")
    print()

    if fp:
        print("  Bugs that shipped with a green check:")
        for r in fp:
            tc = next(t for t in TEST_CASES if t["id"] == r["id"])
            print(f"    ✗ {r['category']}: {tc['ground_truth']}")
    print()

    print("  Infrastructure status:")
    print(f"    SQLite:  eval_context.db (local)")
    print(f"    Neo4j:   {os.environ['NEO4J_URI']} ✓")
    print(f"    Qdrant:  {os.environ['QDRANT_URL'][:50]}... ✓")
    print(f"    Bedrock: {LLM_MODEL} (LLM) + {EMBED_MODEL} (embeddings)")
    print()

    print("  Explore:")
    print("    Neo4j:   MATCH (p:Eval_Proof)-[*1..3]-(n) RETURN p, n")
    print("    Qdrant:  Search eval_verdicts for similar failure patterns")
    print()


def main():
    print("=" * 70)
    print("  FULL EVAL PIPELINE")
    print("  LLM Judge Proof × Neo4j × Qdrant")
    print("=" * 70)
    print()

    results = phase_1_run_evals()
    points = phase_2_embed_to_qdrant(results)
    clusters = phase_3_semantic_analysis()
    phase_4_neo4j_sync(results, clusters)
    print_summary(results)


if __name__ == "__main__":
    main()
