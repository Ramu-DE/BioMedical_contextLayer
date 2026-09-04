#!/usr/bin/env python3
"""
Qdrant Proof Queries: LLM-as-Judge Is Worse Than No Evals

Run: python3 proof_queries_qdrant.py
Requires: qdrant-client
"""

import os, json
from dotenv import load_dotenv
load_dotenv()
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

SEPARATOR = "=" * 70


def query_1_all_collections():
    """Show all collections — eval_verdicts lives alongside biomed/gov vectors"""
    print(f"\n{SEPARATOR}")
    print("QUERY 1: All Qdrant Collections (eval layer alongside existing data)")
    print(SEPARATOR)
    collections = client.get_collections().collections
    for c in collections:
        info = client.get_collection(c.name)
        print(f"  {c.name:25s} → {info.points_count:>6} points, dim={info.config.params.vectors.size}")


def query_2_eval_verdicts_overview():
    """All eval verdict vectors — the embedded proof"""
    print(f"\n{SEPARATOR}")
    print("QUERY 2: All Eval Verdict Vectors")
    print(SEPARATOR)
    results = client.scroll(
        collection_name="eval_verdicts",
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    points = results[0]
    print(f"  Total points: {len(points)}")
    for p in points:
        pl = p.payload
        marker = "✗ FALSE PASS" if not pl.get("judge_correct") and pl.get("judge_verdict") == "PASS" else ""
        print(f"  [{p.id}] test={pl.get('test_id','?'):20s} judge={pl.get('judge_verdict','?'):5s} correct={pl.get('is_correct','')}  judge_correct={pl.get('judge_correct','')}  {marker}")


def query_3_false_passes_only():
    """Filter: only false passes — judge said PASS on bad outputs"""
    print(f"\n{SEPARATOR}")
    print("QUERY 3: FALSE PASSES ONLY (judge approved bad outputs)")
    print(SEPARATOR)
    results = client.scroll(
        collection_name="eval_verdicts",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="judge_correct", match=MatchValue(value=False)),
                FieldCondition(key="is_correct", match=MatchValue(value=False)),
            ]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    points = results[0]
    print(f"  False passes found: {len(points)}")
    for p in points:
        pl = p.payload
        print(f"\n  TEST: {pl.get('test_id')}")
        print(f"  Category: {pl.get('category')}")
        print(f"  Judge said: {pl.get('judge_verdict')} (WRONG — output was incorrect)")
        print(f"  Explanation: {pl.get('judge_explanation', 'n/a')[:120]}...")


def query_4_correct_passes():
    """Filter: correct verdicts — where the judge got it right"""
    print(f"\n{SEPARATOR}")
    print("QUERY 4: CORRECT VERDICTS (judge got these right)")
    print(SEPARATOR)
    results = client.scroll(
        collection_name="eval_verdicts",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="judge_correct", match=MatchValue(value=True)),
            ]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    points = results[0]
    print(f"  Correct verdicts: {len(points)}")
    for p in points:
        pl = p.payload
        print(f"  [{pl.get('test_id'):20s}] verdict={pl.get('judge_verdict'):5s} actually_correct={pl.get('is_correct')}")


def query_5_semantic_similarity():
    """Find the most similar false-pass verdicts using vector search"""
    print(f"\n{SEPARATOR}")
    print("QUERY 5: SEMANTIC FAILURE CLUSTERS (nearest neighbors)")
    print(SEPARATOR)
    all_points = client.scroll(
        collection_name="eval_verdicts",
        scroll_filter=Filter(
            must=[FieldCondition(key="judge_correct", match=MatchValue(value=False))]
        ),
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    if all_points[0]:
        anchor = all_points[0][0]
        print(f"  Anchor: {anchor.payload.get('test_id')} (a false pass)")
        results = client.query_points(
            collection_name="eval_verdicts",
            query=anchor.vector,
            limit=5,
            with_payload=True,
        )
        print(f"  Nearest neighbors by semantic similarity:")
        for r in results.points:
            pl = r.payload
            print(f"    score={r.score:.4f}  test={pl.get('test_id'):20s}  judge_correct={pl.get('judge_correct')}  category={pl.get('category')}")
    else:
        print("  No false-pass points found for similarity search")


def query_6_category_breakdown():
    """Group verdicts by category — see which error types fool the judge"""
    print(f"\n{SEPARATOR}")
    print("QUERY 6: CATEGORY BREAKDOWN — Which error types fool the judge?")
    print(SEPARATOR)
    results = client.scroll(
        collection_name="eval_verdicts",
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    points = results[0]
    categories = {}
    for p in points:
        cat = p.payload.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "false_passes": 0, "correct": 0}
        categories[cat]["total"] += 1
        if p.payload.get("judge_correct"):
            categories[cat]["correct"] += 1
        elif not p.payload.get("is_correct") and p.payload.get("judge_verdict") == "PASS":
            categories[cat]["false_passes"] += 1

    for cat, stats in sorted(categories.items(), key=lambda x: -x[1]["false_passes"]):
        fp_rate = (stats["false_passes"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {cat:30s}  total={stats['total']:2d}  false_passes={stats['false_passes']:2d}  rate={fp_rate:.0f}%")


def query_7_proof_summary():
    """Calculate the proof from raw vector data — independent of Neo4j"""
    print(f"\n{SEPARATOR}")
    print("QUERY 7: PROOF SUMMARY — Calculated from Qdrant vectors alone")
    print(SEPARATOR)
    results = client.scroll(
        collection_name="eval_verdicts",
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    points = results[0]
    total = len(points)
    bad_outputs = [p for p in points if not p.payload.get("is_correct")]
    false_passes = [p for p in bad_outputs if p.payload.get("judge_verdict") == "PASS"]

    print(f"  Total eval verdicts:     {total}")
    print(f"  Known-bad outputs:       {len(bad_outputs)}")
    print(f"  Judge said PASS on bad:  {len(false_passes)}")
    if bad_outputs:
        rate = len(false_passes) / len(bad_outputs) * 100
        print(f"  FALSE PASS RATE:         {rate:.0f}%")
    print()
    print("  CONCLUSION: With no evals, you know you have no safety net.")
    print("  With an LLM judge, you THINK you have a safety net — but")
    print(f"  {len(false_passes)} of {len(bad_outputs)} known-bad outputs got approved.")
    print("  False confidence is worse than no confidence.")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  PROOF: LLM-as-Judge Is Worse Than No Evals                    ║")
    print("║  Qdrant Vector Queries                                         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    query_1_all_collections()
    query_2_eval_verdicts_overview()
    query_3_false_passes_only()
    query_4_correct_passes()
    query_5_semantic_similarity()
    query_6_category_breakdown()
    query_7_proof_summary()
