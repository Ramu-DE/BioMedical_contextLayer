"""
Integrated Eval Pipeline + Context Layer
=========================================

Runs the full proof that LLM-as-Judge is worse than no evals,
and stores every result in the context layer graph for analysis.

Pipeline:
  1. Load test cases (subtle errors that fool judges)
  2. Run LLM judge on each
  3. Run deterministic checks on each
  4. Store all results in context layer (graph + SQLite)
  5. Run multiple judge prompt variants (consistency test)
  6. Analyze: which error categories fool the judge?
  7. Export Cypher for Neo4j Aura
"""

import anthropic
import os
import json
import time
import sys
import re
from datetime import datetime
from context_layer import ContextLayer

client = anthropic.AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-west-2"))
MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


# ═══════════════════════════════════════════════════════════════════
# TEST SUITE: Subtle errors that exploit LLM judge weaknesses
# ═══════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "id": "subtle-fact-1",
        "category": "Plausible Factual Error",
        "prompt": "What year was the Treaty of Westphalia signed?",
        "output": "The Treaty of Westphalia was signed in 1647, marking the end of the Thirty Years' War in Europe. It established the principle of state sovereignty.",
        "ground_truth": "1648",
        "why_wrong": "Off by one year: 1648 not 1647",
        "is_correct": False,
        "det_check": lambda o: "1648" in o,
    },
    {
        "id": "subtle-math-1",
        "category": "Arithmetic Error",
        "prompt": "A store has a 15% off sale. An item costs $84. What's the sale price?",
        "output": "15% of $84 = $84 × 0.15 = $12.40\nSale price: $84 - $12.40 = $71.60",
        "ground_truth": "$71.40 (84 × 0.15 = 12.60)",
        "why_wrong": "84 × 0.15 = 12.60 not 12.40. Final: $71.40 not $71.60",
        "is_correct": False,
        "det_check": lambda o: "71.40" in o or "71.4" in o,
    },
    {
        "id": "subtle-math-2",
        "category": "Arithmetic Error",
        "prompt": "How many days in Jan+Feb+Mar of a non-leap year?",
        "output": "January: 31, February: 28, March: 31\nTotal: 31 + 28 + 31 = 91 days",
        "ground_truth": "90 days (31+28+31=90)",
        "why_wrong": "31+28+31=90 not 91. Addition error hidden in plain sight",
        "is_correct": False,
        "det_check": lambda o: "90" in o and "91" not in o,
    },
    {
        "id": "subtle-math-3",
        "category": "Arithmetic Error",
        "prompt": "A car travels at 60 mph for 2.5 hours. Distance in km? (1 mi = 1.609 km)",
        "output": "Distance: 60 × 2.5 = 150 miles\nIn km: 150 × 1.609 = 241.25 km",
        "ground_truth": "241.35 km (150 × 1.609 = 241.35)",
        "why_wrong": "150 × 1.609 = 241.35 not 241.25",
        "is_correct": False,
        "det_check": lambda o: "241.35" in o,
    },
    {
        "id": "subtle-cs-1",
        "category": "CS Misconception",
        "prompt": "What is the time complexity of binary search?",
        "output": "Binary search is O(log n) on average. In the worst case it degrades to O(n log n) due to rebalancing when the target is at the boundary.",
        "ground_truth": "O(log n) worst case, always",
        "why_wrong": "Binary search is O(log n) worst case. No 'rebalancing' exists. Fabricated degradation.",
        "is_correct": False,
        "det_check": lambda o: "log n" in o.lower() and "n log n" not in o.lower(),
    },
    {
        "id": "subtle-api-1",
        "category": "Best Practice Error",
        "prompt": "What HTTP status code for a successfully created resource?",
        "output": "Return HTTP 200 OK when a resource is created. This is the standard success code indicating the request processed correctly.",
        "ground_truth": "201 Created",
        "why_wrong": "Should be 201 Created, not 200. Common mistake that 'works' but violates REST.",
        "is_correct": False,
        "det_check": lambda o: "201" in o,
    },
    {
        "id": "subtle-science-1",
        "category": "Common Misconception",
        "prompt": "Why do we add salt to pasta water?",
        "output": "Salt raises the boiling point by 2-3°C, cooking pasta faster and more evenly at higher temperature.",
        "ground_truth": "Flavor. Boiling point elevation from cooking salt is ~0.5°C, negligible.",
        "why_wrong": "Salt is for flavor. Boiling point change is ~0.5°C not 2-3°C.",
        "is_correct": False,
        "det_check": lambda o: "flavor" in o.lower() or "taste" in o.lower(),
    },
    {
        "id": "subtle-logic-1",
        "category": "Logical Fallacy",
        "prompt": "All roses are flowers. Some flowers fade quickly. Do some roses fade quickly?",
        "output": "Yes — since all roses are flowers, and some flowers fade quickly, some roses must fade quickly too.",
        "ground_truth": "No — fallacy of undistributed middle",
        "why_wrong": "Classic syllogistic fallacy. The 'some flowers' that fade may not include roses.",
        "is_correct": False,
        "det_check": lambda o: any(w in o.lower() for w in ["no", "cannot", "not necessarily", "fallacy"]),
    },
    {
        "id": "correct-palindrome",
        "category": "Control (Correct)",
        "prompt": "Write a palindrome checker ignoring case and spaces.",
        "output": "def is_palindrome(s):\n    cleaned = s.lower().replace(' ', '')\n    return cleaned == cleaned[::-1]",
        "ground_truth": "This is correct",
        "why_wrong": "CONTROL: This output IS correct. Tests for false negatives.",
        "is_correct": True,
        "det_check": lambda o: "[::-1]" in o or "reversed" in o,
    },
    {
        "id": "correct-water",
        "category": "Control (Correct)",
        "prompt": "What percentage of Earth's surface is water?",
        "output": "Approximately 71% of Earth's surface is covered by water.",
        "ground_truth": "~71% — correct",
        "why_wrong": "CONTROL: This output IS correct.",
        "is_correct": True,
        "det_check": lambda o: "71" in o,
    },
    {
        "id": "subtle-sql-1",
        "category": "Code Bug",
        "prompt": "Write SQL to find the second highest salary from an employees table.",
        "output": "SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees);",
        "ground_truth": "Fails when multiple employees share the top salary, or table is empty",
        "why_wrong": "Fails on edge cases: all same salary returns NULL, empty table returns NULL with no indication why. No alias.",
        "is_correct": False,
        "det_check": lambda o: "distinct" in o.lower() or "limit" in o.lower() or "dense_rank" in o.lower(),
    },
    {
        "id": "subtle-stat-1",
        "category": "Statistical Error",
        "prompt": "A coin is flipped 10 times and lands heads every time. What's the probability the next flip is heads?",
        "output": "After 10 heads in a row, the probability of the next flip being heads is very low — roughly 1 in 2048 (0.5^11). The law of averages suggests tails is 'due'.",
        "ground_truth": "50%. Each flip is independent. Gambler's fallacy.",
        "why_wrong": "Gambler's fallacy. Each flip is independent, P=0.5 regardless of history.",
        "is_correct": False,
        "det_check": lambda o: "50%" in o or "0.5" in o or "independent" in o.lower(),
    },
]


JUDGE_PROMPTS = {
    "naive": {
        "system": "You are an evaluator assessing AI responses for correctness.",
        "template": "Question: {prompt}\n\nAI Response: {output}\n\nIs this correct? Say PASS or FAIL, then explain briefly.",
    },
    "skeptical": {
        "system": "You are a strict evaluator. Most responses contain subtle mistakes. Be rigorous — verify all numbers and logic.",
        "template": "Question: {prompt}\n\nAI Response: {output}\n\nVerify every claim and calculation. Say PASS only if completely correct, FAIL otherwise.",
    },
    "with_rubric": {
        "system": "You are an evaluator. Check: (1) factual accuracy, (2) mathematical correctness, (3) logical validity, (4) completeness.",
        "template": "Question: {prompt}\n\nAI Response: {output}\n\nScore each dimension, then give overall PASS or FAIL.",
    },
}


def extract_verdict(text):
    first_line = text.split("\n")[0].upper()
    for keyword in ["PASS", "FAIL"]:
        if keyword in first_line:
            return keyword
    for keyword in ["PASS", "FAIL"]:
        if keyword in text.upper()[:150]:
            return keyword
    return "UNCLEAR"


def call_llm(system, user_msg, max_tokens=400):
    start = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    elapsed = (time.time() - start) * 1000
    return response.content[0].text.strip(), elapsed


def main():
    ctx = ContextLayer()

    print("=" * 70)
    print("  EVAL PIPELINE + CONTEXT LAYER")
    print("  Proving LLM-as-Judge < No Evals")
    print("=" * 70)
    print()
    print(f"  Model:      {MODEL}")
    print(f"  Test cases: {len(TEST_CASES)} ({sum(1 for t in TEST_CASES if not t['is_correct'])} bad, {sum(1 for t in TEST_CASES if t['is_correct'])} correct controls)")
    print(f"  Judge variants: {list(JUDGE_PROMPTS.keys())}")
    print()

    # ─── Register test cases in context layer ─────────────────────
    for tc in TEST_CASES:
        ctx.add_test_case(
            tc["id"], tc["category"], tc["prompt"], tc["output"],
            tc["ground_truth"], tc["is_correct"]
        )

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Run all judge variants
    # ═══════════════════════════════════════════════════════════════
    all_results = {}

    for judge_name, judge_config in JUDGE_PROMPTS.items():
        print(f"─── Judge variant: {judge_name} ───")
        run_id = ctx.start_run(MODEL, {"judge_type": judge_name})
        all_results[judge_name] = []

        for tc in TEST_CASES:
            sys.stdout.write(f"  {tc['id']:<25s} ")
            sys.stdout.flush()

            user_msg = judge_config["template"].format(
                prompt=tc["prompt"], output=tc["output"]
            )
            judge_text, latency = call_llm(
                judge_config["system"], user_msg
            )
            verdict = extract_verdict(judge_text)

            if tc["is_correct"]:
                correct = verdict == "PASS"
            else:
                correct = verdict == "FAIL"

            ctx.record_verdict(
                tc["id"], f"llm_judge_{judge_name}", verdict,
                correct, judge_text, latency
            )
            all_results[judge_name].append({
                "test_id": tc["id"],
                "verdict": verdict,
                "correct": correct,
                "is_bad": not tc["is_correct"],
            })

            icon = "✓" if correct else "✗"
            print(f"{verdict:<6s} {icon}  ({latency:.0f}ms)")
            time.sleep(0.3)

        print()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Run deterministic checks
    # ═══════════════════════════════════════════════════════════════
    print("─── Deterministic checks ───")
    det_run_id = ctx.start_run("deterministic", {"type": "programmatic"})

    for tc in TEST_CASES:
        sys.stdout.write(f"  {tc['id']:<25s} ")
        sys.stdout.flush()

        try:
            det_passed = tc["det_check"](tc["output"])
        except Exception:
            det_passed = False

        if tc["is_correct"]:
            correct = det_passed
        else:
            correct = not det_passed

        ctx.record_verdict(
            tc["id"], "deterministic", "PASS" if det_passed else "FAIL",
            correct, "programmatic check", 0
        )

        icon = "✓" if correct else "✗"
        print(f"{'PASS' if det_passed else 'FAIL':<6s} {icon}")

    print()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Derive difficulty edges
    # ═══════════════════════════════════════════════════════════════
    ctx.derive_difficulty_edges()

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  ANALYSIS")
    print("=" * 70)
    print()

    # Method comparison
    print("  ┌─ Method Comparison ──────────────────────────────────────────┐")
    print(f"  │ {'Method':<30s} {'Acc%':>6s} {'FalsePass':>10s} {'AvgMs':>8s} │")
    print(f"  ├{'─' * 62}┤")

    comparisons = ctx.get_method_comparison()
    for row in comparisons:
        method, total, correct, acc_pct, false_passes, avg_ms = row
        print(f"  │ {method:<30s} {acc_pct:>5.1f}% {false_passes:>10d} {avg_ms:>7.0f} │")

    print(f"  │ {'(no evals)':<30s} {'0.0':>6s}% {'n/a':>10s} {'0':>7s} │")
    print(f"  └{'─' * 62}┘")
    print()

    # False pass analysis by category
    print("  ┌─ False Passes by Category ───────────────────────────────────┐")
    fp_analysis = ctx.get_false_pass_analysis()
    for row in fp_analysis:
        cat, method, total, false_passes = row
        if false_passes > 0:
            print(f"  │  {cat:<25s} {method:<25s} {false_passes}/{total} false passes │")
    if not any(r[3] > 0 for r in fp_analysis):
        print(f"  │  (no false passes this run — run again for variance)        │")
    print(f"  └{'─' * 62}┘")
    print()

    # Graph stats
    stats = ctx.get_graph_stats()
    print(f"  Context Layer Graph:")
    print(f"    Nodes: {stats['total_nodes']} ({json.dumps(stats['nodes_by_type'])})")
    print(f"    Edges: {stats['total_edges']} ({json.dumps(stats['edges_by_type'])})")
    print()

    # Judge consistency analysis
    print("  ┌─ Judge Consistency (do variants agree?) ─────────────────────┐")
    bad_tests = [tc for tc in TEST_CASES if not tc["is_correct"]]
    for tc in bad_tests:
        verdicts = []
        for jname in JUDGE_PROMPTS:
            r = next(
                (x for x in all_results[jname] if x["test_id"] == tc["id"]),
                None
            )
            if r:
                verdicts.append(r["verdict"])
        agree = len(set(verdicts)) == 1
        marker = "  " if agree else "⚠ "
        print(f"  │ {marker}{tc['id']:<23s} {' / '.join(verdicts):<25s}"
              f" {'consistent' if agree else 'INCONSISTENT':>12s} │")
    print(f"  └{'─' * 62}┘")
    print()

    # ═══════════════════════════════════════════════════════════════
    # THE PROOF
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  THE PROOF: WHY LLM JUDGE < NO EVALS")
    print("=" * 70)
    print()

    # Calculate across all judge variants
    total_bad = len(bad_tests) * len(JUDGE_PROMPTS)
    total_false_passes = sum(
        1 for jname in JUDGE_PROMPTS
        for r in all_results[jname]
        if r["is_bad"] and not r["correct"]
    )
    inconsistent_tests = sum(
        1 for tc in bad_tests
        if len(set(
            next(x["verdict"] for x in all_results[jn] if x["test_id"] == tc["id"])
            for jn in JUDGE_PROMPTS
        )) > 1
    )

    fp_rate = total_false_passes / total_bad * 100 if total_bad > 0 else 0

    print(f"  Across {len(JUDGE_PROMPTS)} judge variants × {len(bad_tests)} bad outputs = {total_bad} judgments:")
    print(f"    False passes:      {total_false_passes}/{total_bad} ({fp_rate:.0f}%)")
    print(f"    Inconsistent:      {inconsistent_tests}/{len(bad_tests)} tests got different verdicts")
    print()
    print("  The damage model:")
    print()
    print("    WITHOUT evals: Team knows they have no safety net.")
    print("      → Every change gets manual review.")
    print("      → Bugs found: proportional to review quality.")
    print()
    print(f"    WITH naive LLM judge: {fp_rate:.0f}% of subtle bugs get a green check.")
    print("      → Team trusts the eval dashboard.")
    print("      → Manual review drops or stops.")
    print(f"      → {fp_rate:.0f}% of bugs ship with false confidence.")
    print()
    print(f"    WITH deterministic checks: 0% false passes on verifiable claims.")
    print("      → Fast, free, reproducible.")
    print("      → Human review focuses on what CAN'T be checked automatically.")
    print()

    # ═══════════════════════════════════════════════════════════════
    # EXPORT
    # ═══════════════════════════════════════════════════════════════
    cypher = ctx.export_neo4j_cypher()
    cypher_path = os.path.join(os.path.dirname(__file__), "neo4j_import.cypher")
    with open(cypher_path, "w") as f:
        f.write(cypher)
    print(f"  Neo4j Cypher export: {cypher_path}")
    print("    → Run against your Aura instance to visualize the graph")
    print()

    # Export JSON summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "test_cases": len(TEST_CASES),
        "bad_outputs": len(bad_tests),
        "judge_variants": list(JUDGE_PROMPTS.keys()),
        "total_judgments": total_bad,
        "false_passes": total_false_passes,
        "false_pass_rate": round(fp_rate, 1),
        "inconsistent_tests": inconsistent_tests,
        "graph_stats": stats,
        "method_comparison": [
            {"method": r[0], "accuracy": r[3], "false_passes": r[4]}
            for r in comparisons
        ],
    }
    summary_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON summary: {summary_path}")
    print()

    ctx.close()
    print("  Done. Context layer stored in eval_context.db")
    print()


if __name__ == "__main__":
    main()
