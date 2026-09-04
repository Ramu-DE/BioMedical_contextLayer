#!/usr/bin/env python3
"""
PoC-Relevant Eval: LLM Judge vs Deterministic on Pharma/Biomedical Domain

Uses REAL data from our Neo4j knowledge graph:
- Drug mechanisms (Pembrolizumab = PD-1 inhibitor, NOT PD-L1)
- Patient consent (PAT015 = Withdrawn, all others Active)
- Clinical trial adverse events (CT001 reports pneumonitis = Severe)
- Biomarker significance (PD-L1 = immunotherapy response)
- Drug interactions and classifications

Tests whether the LLM judge catches subtle domain errors that matter
for patient safety and clinical governance.
"""

import json, time, os, sys
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ── Bedrock client ──
from anthropic import AnthropicBedrock
bedrock = AnthropicBedrock(aws_region="us-west-2")
MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ── Neo4j ──
from neo4j import GraphDatabase
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USERNAME")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD")
NEO4J_DB = os.environ.get("NEO4J_DATABASE")
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ── Qdrant ──
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=30, check_compatibility=False)

# ── Embeddings ──
import boto3
embed_client = boto3.client("bedrock-runtime", region_name="us-west-2")
def embed(text):
    resp = embed_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text[:8000]}),
    )
    return json.loads(resp["body"].read())["embedding"]

SEP = "=" * 72

# ═══════════════════════════════════════════════════════════════════
# TEST CASES — All from our actual PoC domain
# Each has: prompt, bad output (subtle error), ground truth, category,
# and a deterministic check that catches it
# ═══════════════════════════════════════════════════════════════════

TEST_CASES = [
    # 1. Drug mechanism confusion — PD-1 vs PD-L1 (from our KG)
    {
        "id": "pharma-mech-1",
        "category": "Drug Mechanism",
        "prompt": "What is the mechanism of action of Pembrolizumab?",
        "output": "Pembrolizumab is a PD-L1 inhibitor that blocks the PD-L1 protein on tumor cells, preventing immune evasion.",
        "ground_truth": "PD-1 inhibitor",
        "is_correct": False,
        "deterministic_check": lambda o: "PD-1 inhibitor" in o and "PD-L1 inhibitor" not in o,
        "why_subtle": "PD-1 and PD-L1 are closely related but different targets. Pembrolizumab blocks PD-1 on T-cells, not PD-L1 on tumor cells.",
    },
    # 2. Drug mechanism — Atezolizumab IS PD-L1 (control — correct)
    {
        "id": "pharma-mech-2",
        "category": "Drug Mechanism",
        "prompt": "What is the mechanism of action of Atezolizumab?",
        "output": "Atezolizumab is a PD-L1 inhibitor that blocks PD-L1 on tumor cells.",
        "ground_truth": "PD-L1 inhibitor",
        "is_correct": True,
        "deterministic_check": lambda o: "PD-L1 inhibitor" in o,
        "why_subtle": "Control case — this is correct per our KG.",
    },
    # 3. Patient consent — PAT015 is Withdrawn, not Active
    {
        "id": "consent-1",
        "category": "Patient Governance",
        "prompt": "Can we include PAT015 in the trial outcome analysis?",
        "output": "Yes, PAT015 has Active consent status and can be included in the analysis alongside other trial participants.",
        "ground_truth": "No — PAT015 consent is Withdrawn",
        "is_correct": False,
        "deterministic_check": lambda o: "withdrawn" in o.lower() or "cannot" in o.lower() or "no" in o.lower().split(".")[0],
        "why_subtle": "PAT015 is the ONLY patient with Withdrawn consent. Including them violates governance. All other patients are Active.",
    },
    # 4. Patient consent — PAT001 IS Active (control — correct)
    {
        "id": "consent-2",
        "category": "Patient Governance",
        "prompt": "Can we include PAT001 in the trial outcome analysis?",
        "output": "Yes, PAT001 has Active consent and can be included in the analysis.",
        "ground_truth": "Yes — PAT001 consent is Active",
        "is_correct": True,
        "deterministic_check": lambda o: "active" in o.lower() and ("yes" in o.lower() or "can be included" in o.lower()),
        "why_subtle": "Control case — PAT001 is Active.",
    },
    # 5. Adverse event severity — Pneumonitis is Severe, not Moderate
    {
        "id": "ae-severity-1",
        "category": "Adverse Event",
        "prompt": "What is the severity of immune-related pneumonitis reported in CT001?",
        "output": "Immune-related pneumonitis in CT001 is classified as Moderate severity, requiring monitoring but not treatment discontinuation.",
        "ground_truth": "Severe",
        "is_correct": False,
        "deterministic_check": lambda o: "severe" in o.lower() and "moderate" not in o.lower(),
        "why_subtle": "Downgrading Severe to Moderate in an adverse event report is a patient safety issue. CT001 reports pneumonitis as Severe.",
    },
    # 6. Clinical trial phase — CT010 is Phase 2 Active, not Phase 3
    {
        "id": "trial-phase-1",
        "category": "Clinical Trial",
        "prompt": "What phase is clinical trial CT010 and what is its status?",
        "output": "CT010 is a Phase 3 clinical trial that has been completed.",
        "ground_truth": "Phase 2, Active",
        "is_correct": False,
        "deterministic_check": lambda o: "phase 2" in o.lower() and "active" in o.lower(),
        "why_subtle": "CT010 is Phase 2 Active — the only active trial. Reporting it as Phase 3 Completed misrepresents regulatory status.",
    },
    # 7. Biomarker — PD-L1 predicts immunotherapy, not chemotherapy
    {
        "id": "biomarker-1",
        "category": "Biomarker",
        "prompt": "What does PD-L1 Expression predict in terms of treatment response?",
        "output": "High PD-L1 expression predicts better response to platinum-based chemotherapy regimens.",
        "ground_truth": "Predicts immunotherapy response",
        "is_correct": False,
        "deterministic_check": lambda o: "immunotherapy" in o.lower() and "chemotherapy" not in o.lower(),
        "why_subtle": "PD-L1 is an immunotherapy biomarker. Linking it to chemotherapy could lead to wrong treatment decisions.",
    },
    # 8. Drug class — Trastuzumab targets HER2, not EGFR
    {
        "id": "pharma-target-1",
        "category": "Drug Mechanism",
        "prompt": "What does Trastuzumab target?",
        "output": "Trastuzumab is an EGFR inhibitor used in the treatment of HER2-positive breast cancer.",
        "ground_truth": "HER2 inhibitor",
        "is_correct": False,
        "deterministic_check": lambda o: "her2" in o.lower() and "egfr" not in o.lower().replace("her2", ""),
        "why_subtle": "HER2 and EGFR are both receptor tyrosine kinases but different targets. Trastuzumab binds HER2, not EGFR.",
    },
    # 9. Patient disease — PAT006 has Alzheimer's, not Parkinson's
    {
        "id": "patient-disease-1",
        "category": "Patient Data",
        "prompt": "What disease does patient PAT006 have?",
        "output": "PAT006 has been diagnosed with Parkinson's Disease and is being treated with dopamine agonists.",
        "ground_truth": "Alzheimer's Disease",
        "is_correct": False,
        "deterministic_check": lambda o: "alzheimer" in o.lower() and "parkinson" not in o.lower(),
        "why_subtle": "Confusing Alzheimer's with Parkinson's — both neurodegenerative but completely different treatments and prognosis.",
    },
    # 10. Drug mechanism — Metformin is AMPK activator (control)
    {
        "id": "pharma-mech-3",
        "category": "Drug Mechanism",
        "prompt": "What is the mechanism of Metformin?",
        "output": "Metformin works as an AMPK activator, improving insulin sensitivity and reducing hepatic glucose production.",
        "ground_truth": "AMPK activator",
        "is_correct": True,
        "deterministic_check": lambda o: "ampk" in o.lower(),
        "why_subtle": "Control — correct per our KG.",
    },
    # 11. Governance — Including withdrawn patient in aggregate stats
    {
        "id": "governance-agg-1",
        "category": "Patient Governance",
        "prompt": "How many patients with Active consent can be included in the renal cell carcinoma analysis?",
        "output": "There are 2 patients with renal cell carcinoma (PAT007 and PAT015) who can be included in the analysis.",
        "ground_truth": "1 — only PAT007 (PAT015 consent is Withdrawn)",
        "is_correct": False,
        "deterministic_check": lambda o: ("1" in o or "one" in o.lower()) and "pat015" not in o.lower().replace("withdrawn", "").replace("cannot", "").replace("excluded", ""),
        "why_subtle": "PAT015 has Withdrawn consent. Including them in the count violates data governance, even though they have the right disease.",
    },
    # 12. Trial safety — CT004 reports ARIA as Severe, not Mild
    {
        "id": "ae-severity-2",
        "category": "Adverse Event",
        "prompt": "What are the adverse events reported in trial CT004 for Aducanumab?",
        "output": "CT004 reports infusion reaction (Mild) and amyloid-related imaging abnormalities (Mild), both manageable with standard protocols.",
        "ground_truth": "Infusion reaction (Moderate), ARIA (Severe)",
        "is_correct": False,
        "deterministic_check": lambda o: "severe" in o.lower() or ("moderate" in o.lower() and "mild" not in o.lower()),
        "why_subtle": "Downgrading both AEs from Moderate/Severe to Mild is dangerous. ARIA in Aducanumab trials is a known Severe event.",
    },
]


def judge_output(prompt, output, ground_truth):
    """Ask the LLM judge if the output is correct."""
    judge_prompt = f"""You are evaluating an AI assistant's response for factual accuracy.

Question asked: {prompt}
AI's response: {output}
Known correct answer: {ground_truth}

Is the AI's response factually correct and consistent with the known answer?
Reply with exactly PASS or FAIL on the first line, then explain why."""

    t0 = time.time()
    resp = bedrock.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    ms = (time.time() - t0) * 1000
    text = resp.content[0].text.strip()
    verdict = "PASS" if text.upper().startswith("PASS") else "FAIL"
    return verdict, text, ms


def run_all():
    run_id = f"poc_domain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = []

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  PoC DOMAIN EVAL: LLM Judge vs Deterministic                     ║")
    print("║  Testing with real pharma/biomedical data from our knowledge graph║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    for tc in TEST_CASES:
        print(f"{SEP}")
        print(f"  TEST: {tc['id']} ({tc['category']})")
        print(f"  Q: {tc['prompt']}")
        print(f"  Output: {tc['output'][:100]}...")
        print(f"  Ground truth: {tc['ground_truth']}")
        print(f"  Actually correct: {tc['is_correct']}")

        # Deterministic check
        det_result = tc["deterministic_check"](tc["output"])
        det_correct = det_result == tc["is_correct"]

        # LLM judge
        judge_verdict, judge_explanation, judge_ms = judge_output(
            tc["prompt"], tc["output"], tc["ground_truth"]
        )
        judge_passed = judge_verdict == "PASS"
        judge_correct = judge_passed == tc["is_correct"]

        is_false_pass = not tc["is_correct"] and judge_passed
        is_false_fail = tc["is_correct"] and not judge_passed

        marker = ""
        if is_false_pass:
            marker = "  ✗✗✗ FALSE PASS — JUDGE APPROVED BAD OUTPUT ✗✗✗"
        elif is_false_fail:
            marker = "  ✗ FALSE FAIL — JUDGE REJECTED CORRECT OUTPUT"

        print(f"\n  DETERMINISTIC: {'PASS' if det_result else 'FAIL'} (correct={det_correct})")
        print(f"  LLM JUDGE:     {judge_verdict} (correct={judge_correct}) [{judge_ms:.0f}ms]")
        print(f"  Judge says:    {judge_explanation[:150]}...")
        if marker:
            print(marker)
        if not tc["is_correct"]:
            print(f"  Why subtle:    {tc['why_subtle']}")
        print()

        results.append({
            "test_id": tc["id"],
            "category": tc["category"],
            "prompt": tc["prompt"],
            "output": tc["output"],
            "ground_truth": tc["ground_truth"],
            "is_correct": tc["is_correct"],
            "judge_verdict": judge_verdict,
            "judge_correct": judge_correct,
            "judge_explanation": judge_explanation,
            "judge_ms": judge_ms,
            "det_result": det_result,
            "det_correct": det_correct,
            "is_false_pass": is_false_pass,
            "is_false_fail": is_false_fail,
            "why_subtle": tc.get("why_subtle", ""),
        })

    # ── Summary ──
    total = len(results)
    bad_outputs = [r for r in results if not r["is_correct"]]
    good_outputs = [r for r in results if r["is_correct"]]
    judge_false_passes = [r for r in results if r["is_false_pass"]]
    judge_false_fails = [r for r in results if r["is_false_fail"]]
    judge_correct_count = sum(1 for r in results if r["judge_correct"])
    det_correct_count = sum(1 for r in results if r["det_correct"])

    print(f"\n{SEP}")
    print("  RESULTS SUMMARY")
    print(SEP)
    print(f"  Total test cases:         {total}")
    print(f"  Bad outputs tested:       {len(bad_outputs)}")
    print(f"  Good outputs (controls):  {len(good_outputs)}")
    print(f"")
    print(f"  LLM JUDGE accuracy:       {judge_correct_count}/{total} ({judge_correct_count/total*100:.0f}%)")
    print(f"  DETERMINISTIC accuracy:   {det_correct_count}/{total} ({det_correct_count/total*100:.0f}%)")
    print(f"")
    print(f"  FALSE PASSES (judge approved bad output):")
    fp_rate = len(judge_false_passes) / len(bad_outputs) * 100 if bad_outputs else 0
    print(f"    {len(judge_false_passes)} of {len(bad_outputs)} bad outputs = {fp_rate:.0f}% false pass rate")
    for r in judge_false_passes:
        print(f"    ✗ {r['test_id']}: {r['category']} — {r['why_subtle'][:80]}")
    print(f"")
    print(f"  FALSE FAILS (judge rejected correct output):")
    for r in judge_false_fails:
        print(f"    ✗ {r['test_id']}: judge said FAIL on correct output")
    if not judge_false_fails:
        print(f"    (none)")

    # Category breakdown
    print(f"\n{SEP}")
    print("  FALSE PASS RATE BY CATEGORY")
    print(SEP)
    cats = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = {"total_bad": 0, "false_passes": 0}
        if not r["is_correct"]:
            cats[c]["total_bad"] += 1
            if r["is_false_pass"]:
                cats[c]["false_passes"] += 1
    for c, s in sorted(cats.items(), key=lambda x: -x[1]["false_passes"]):
        if s["total_bad"] > 0:
            rate = s["false_passes"] / s["total_bad"] * 100
            print(f"  {c:25s}  {s['false_passes']}/{s['total_bad']} = {rate:.0f}%")

    # ── Save to Neo4j ──
    print(f"\n{SEP}")
    print("  SYNCING TO NEO4J + QDRANT...")
    print(SEP)

    with neo4j_driver.session(database=NEO4J_DB) as session:
        # Create run node
        session.run("""
            MERGE (r:Eval_Run {run_id: $run_id})
            SET r.model = $model, r.timestamp = $ts,
                r.config = 'poc_domain', r.test_count = $cnt
        """, run_id=run_id, model=MODEL, ts=datetime.now().isoformat(), cnt=total)

        # Link to proof
        session.run("""
            MATCH (p:Eval_Proof)
            MATCH (r:Eval_Run {run_id: $run_id})
            MERGE (p)-[:EVIDENCED_BY]->(r)
        """, run_id=run_id)

        for r in results:
            # Test case
            session.run("""
                MERGE (t:Eval_TestCase {test_id: $tid})
                SET t.prompt = $prompt, t.output = $output,
                    t.ground_truth = $gt, t.is_correct = $correct,
                    t.category = $cat, t.domain = 'pharma'
            """, tid=r["test_id"], prompt=r["prompt"], output=r["output"],
                gt=r["ground_truth"], correct=r["is_correct"], cat=r["category"])

            # LLM Judge verdict
            session.run("""
                MATCH (t:Eval_TestCase {test_id: $tid})
                MERGE (v:Eval_Verdict {test_id: $tid, method: 'llm_judge', run_id: $run})
                SET v.result = $result, v.correct = $correct,
                    v.explanation = $expl, v.latency_ms = $ms
                MERGE (t)-[:RECEIVED_VERDICT]->(v)
                MERGE (r:Eval_Run {run_id: $run})
                MERGE (r)-[:CONTAINS_VERDICT]->(v)
            """, tid=r["test_id"], run=run_id, result=r["judge_verdict"],
                correct=r["judge_correct"], expl=r["judge_explanation"][:500], ms=r["judge_ms"])

            # Deterministic verdict
            session.run("""
                MATCH (t:Eval_TestCase {test_id: $tid})
                MERGE (v:Eval_Verdict {test_id: $tid, method: 'deterministic', run_id: $run})
                SET v.result = $result, v.correct = $correct, v.latency_ms = 0
                MERGE (t)-[:RECEIVED_VERDICT]->(v)
                MERGE (r:Eval_Run {run_id: $run})
                MERGE (r)-[:CONTAINS_VERDICT]->(v)
            """, tid=r["test_id"], run=run_id,
                result="PASS" if r["det_result"] else "FAIL", correct=r["det_correct"])

            # Category
            session.run("""
                MATCH (t:Eval_TestCase {test_id: $tid})
                MERGE (c:Eval_Category {name: $cat})
                MERGE (t)-[:BELONGS_TO]->(c)
            """, tid=r["test_id"], cat=r["category"])

            # Failure mode
            if r["is_false_pass"]:
                session.run("""
                    MATCH (v:Eval_Verdict {test_id: $tid, method: 'llm_judge', run_id: $run})
                    MERGE (f:Eval_FailureMode {name: 'FalsePass_Domain'})
                    MERGE (v)-[:EXHIBITS]->(f)
                """, tid=r["test_id"], run=run_id)

        # Update proof node
        session.run("""
            MATCH (p:Eval_Proof)
            SET p.domain_false_passes = $fp,
                p.domain_total_bad = $tb,
                p.domain_false_pass_rate_pct = $rate,
                p.domain_run_id = $run
        """, fp=len(judge_false_passes), tb=len(bad_outputs), rate=fp_rate, run=run_id)

    print("  Neo4j: synced")

    # ── Embed to Qdrant ──
    points = []
    for i, r in enumerate(results):
        vec = embed(f"{r['prompt']} {r['output']} {r['ground_truth']}")
        points.append(PointStruct(
            id=100 + i,
            vector=vec,
            payload={
                "test_id": r["test_id"],
                "category": r["category"],
                "is_correct": r["is_correct"],
                "judge_verdict": r["judge_verdict"],
                "judge_correct": r["judge_correct"],
                "det_result": "PASS" if r["det_result"] else "FAIL",
                "det_correct": r["det_correct"],
                "is_false_pass": r["is_false_pass"],
                "domain": "pharma",
                "run_id": run_id,
            },
        ))

    qdrant.upsert("eval_verdicts", points)
    print(f"  Qdrant: {len(points)} domain vectors upserted")

    # Final
    print(f"\n{SEP}")
    print(f"  PROOF: LLM Judge false pass rate on domain data = {fp_rate:.0f}%")
    print(f"  Deterministic checks: {det_correct_count}/{total} correct ({det_correct_count/total*100:.0f}%)")
    print(f"  These are PATIENT SAFETY errors the judge missed.")
    print(SEP)

    # Save results
    with open("eval_poc_results.json", "w") as f:
        json.dump({"run_id": run_id, "results": results, "summary": {
            "total": total, "judge_accuracy": judge_correct_count/total,
            "det_accuracy": det_correct_count/total,
            "false_pass_rate": fp_rate, "false_passes": len(judge_false_passes),
        }}, f, indent=2)
    print(f"  Results saved to eval_poc_results.json")


if __name__ == "__main__":
    run_all()
    neo4j_driver.close()
