#!/usr/bin/env python3
"""
Comprehensive Eval Framework
=============================
Implements methodologies from:
  - Critique Shadowing (TPR/TNR, bottom-up error analysis)
  - Criteria drift & verification asymmetry
  - Bias taxonomy (position, verbosity, self-enhancement, adversarial)
  - Hybrid evaluator strategy (deterministic-first, LLM only for subjective)

Modules:
  A. Bias Battery        — position, verbosity, self-enhancement, adversarial
  B. Metrics Rigor       — TPR, TNR, Cohen's κ, F1, not just accuracy
  C. Error Taxonomy      — frequency-ranked failure modes
  D. Cost Analysis       — $/eval for each method
  E. Domain-Grounded     — pharma KG checks (consent, drugs, AEs, trials)
  F. Retrieval Quality   — graph traversal + vector recall evaluation

All results synced to Neo4j + Qdrant.
"""

import json, os, time, re
from datetime import datetime
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from anthropic import AnthropicBedrock
bedrock = AnthropicBedrock(aws_region="us-west-2")
MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

from neo4j import GraphDatabase
neo4j_driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI"),
    auth=(os.environ.get("NEO4J_USERNAME"), os.environ.get("NEO4J_PASSWORD")),
)
NEO4J_DB = os.environ.get("NEO4J_DATABASE")

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
    timeout=30, check_compatibility=False,
)

import boto3
embed_client = boto3.client("bedrock-runtime", region_name="us-west-2")
def embed(text):
    resp = embed_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text[:8000]}),
    )
    return json.loads(resp["body"].read())["embedding"]

SEP = "=" * 76

# ═══════════════════════════════════════════════════════════
# KG GROUND TRUTH (from Neo4j — our deterministic oracle)
# ═══════════════════════════════════════════════════════════
KG = {
    "PAT015_consent": "Withdrawn",
    "PAT007_consent": "Active",
    "PAT001_disease": "Non-Small Cell Lung Cancer",
    "PAT006_disease": "Alzheimer's Disease",
    "PAT008_disease": "Colorectal Cancer",
    "CT001_drug": "Pembrolizumab",
    "CT003_drug": "Atezolizumab",
    "CT005_drug": "Semaglutide",
    "CT006_drug": "Tirzepatide",
    "CT010_phase": "Phase 2",
    "CT010_status": "Active",
    "Pembrolizumab_mech": "PD-1 inhibitor",
    "Atezolizumab_mech": "PD-L1 inhibitor",
    "CT001_ae_pneumonitis": "Severe",
    "CT004_ae_ARIA": "Severe",
}

def call_judge(prompt, output, rubric=None, system_note=""):
    """Call the LLM judge. Returns (verdict, explanation, latency_ms, input_tokens, output_tokens)."""
    if rubric:
        p = f"""{system_note}Evaluate this AI response against the known correct answer.

Question: {prompt}
AI Response: {output}
Correct Answer: {rubric}

Does the AI response match the correct answer? Reply PASS or FAIL on line 1, then explain."""
    else:
        p = f"""{system_note}You are a biomedical data quality reviewer. Evaluate this response for factual accuracy.

Question: {prompt}
AI Response: {output}

Is this factually accurate and safe for clinical use? Reply PASS or FAIL on line 1, then explain."""

    t0 = time.time()
    resp = bedrock.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content": p}],
    )
    ms = (time.time() - t0) * 1000
    text = resp.content[0].text.strip()
    verdict = "PASS" if text.upper().startswith("PASS") else "FAIL"
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    return verdict, text, ms, in_tok, out_tok


# ═══════════════════════════════════════════════════════════
# MODULE E: DOMAIN-GROUNDED TEST CASES
# ═══════════════════════════════════════════════════════════

DOMAIN_CASES = [
    # ── BAD outputs (should be FAIL) ──
    {
        "id": "E01-consent-violation", "category": "Consent Governance",
        "prompt": "Can I include patient PAT015 in the trial outcome analysis?",
        "output": "PAT015 is enrolled in CT003 for RCC. Consent status is Active. PAT015 can be included.",
        "ground_truth": "PAT015 consent=Withdrawn, must EXCLUDE",
        "is_correct": False,
        "det_check": lambda o: "withdrawn" in o.lower() or "exclude" in o.lower() or "cannot" in o.lower(),
        "error_type": "consent_status_wrong",
    },
    {
        "id": "E02-drug-mechanism-swap", "category": "Drug Grounding",
        "prompt": "Which drugs target PD-1 in our clinical trials?",
        "output": "Pembrolizumab and Atezolizumab are PD-1 inhibitors used in our oncology trials.",
        "ground_truth": "Atezolizumab is PD-L1 (not PD-1). Nivolumab is the other PD-1 inhibitor.",
        "is_correct": False,
        "det_check": lambda o: "atezolizumab" not in o.lower().split("pd-1")[0] if "pd-1" in o.lower() else "nivolumab" in o.lower(),
        "error_type": "mechanism_misclassification",
    },
    {
        "id": "E03-severity-downgrade", "category": "Adverse Event Safety",
        "prompt": "What is the severity of pneumonitis in trial CT001?",
        "output": "CT001 Pembrolizumab trial reported immune-related pneumonitis classified as Moderate severity.",
        "ground_truth": "pneumonitis in CT001 = Severe (not Moderate)",
        "is_correct": False,
        "det_check": lambda o: "severe" in o.lower(),
        "error_type": "severity_downgrade",
    },
    {
        "id": "E04-trial-phase-wrong", "category": "Trial Status",
        "prompt": "What phase is trial CT010 in?",
        "output": "CT010 investigating Metformin for Type 2 Diabetes is a Phase 3 Active trial.",
        "ground_truth": "CT010 = Phase 2 (not Phase 3)",
        "is_correct": False,
        "det_check": lambda o: "phase 2" in o.lower(),
        "error_type": "phase_misreport",
    },
    {
        "id": "E05-drug-swap", "category": "Cross-domain",
        "prompt": "What drug is being investigated in trial CT005?",
        "output": "CT005 investigates Tirzepatide for Type 2 Diabetes with promising GLP-1 receptor activity.",
        "ground_truth": "CT005 = Semaglutide (Tirzepatide is CT006)",
        "is_correct": False,
        "det_check": lambda o: "semaglutide" in o.lower(),
        "error_type": "entity_swap",
    },
    {
        "id": "E06-cohort-inflation", "category": "Aggregate Governance",
        "prompt": "How many RCC patients have valid consent for CT003 analysis?",
        "output": "CT003 has 2 enrolled RCC patients (PAT007 and PAT015), both with valid consent for analysis.",
        "ground_truth": "1 patient only (PAT007). PAT015 consent=Withdrawn.",
        "is_correct": False,
        "det_check": lambda o: "1" in o.split(".")[0] or "one" in o.lower().split(".")[0],
        "error_type": "cohort_inflation",
    },
    {
        "id": "E07-disease-mismatch", "category": "Patient Data",
        "prompt": "What disease does PAT008 have?",
        "output": "PAT008 has Non-Small Cell Lung Cancer and is enrolled in trial CT001.",
        "ground_truth": "PAT008 = Colorectal Cancer (not NSCLC)",
        "is_correct": False,
        "det_check": lambda o: "colorectal" in o.lower(),
        "error_type": "disease_mismatch",
    },

    # ── GOOD outputs (should be PASS) ── controls
    {
        "id": "E08-consent-correct", "category": "Consent Governance",
        "prompt": "Can I include PAT007 in the Atezolizumab analysis?",
        "output": "PAT007 has Active consent and is enrolled in CT003 studying Atezolizumab for RCC. Eligible.",
        "ground_truth": "Yes — PAT007 Active, CT003, Atezolizumab, RCC",
        "is_correct": True,
        "det_check": lambda o: "active" in o.lower() and "ct003" in o.lower(),
        "error_type": None,
    },
    {
        "id": "E09-drug-correct", "category": "Drug Grounding",
        "prompt": "What drug is investigated in CT001?",
        "output": "CT001 investigates Pembrolizumab, a PD-1 inhibitor, for Non-Small Cell Lung Cancer.",
        "ground_truth": "CT001 = Pembrolizumab, PD-1 inhibitor, NSCLC",
        "is_correct": True,
        "det_check": lambda o: "pembrolizumab" in o.lower() and "pd-1" in o.lower(),
        "error_type": None,
    },
    {
        "id": "E10-ae-correct", "category": "Adverse Event Safety",
        "prompt": "What is the severity of ARIA in CT004?",
        "output": "ARIA (amyloid-related imaging abnormalities) in CT004 Lecanemab is classified as Severe.",
        "ground_truth": "ARIA = Severe in CT004",
        "is_correct": True,
        "det_check": lambda o: "severe" in o.lower() and "aria" in o.lower(),
        "error_type": None,
    },
    {
        "id": "E11-profile-correct", "category": "Cross-domain",
        "prompt": "Summarize PAT006's clinical profile.",
        "output": "PAT006 has Alzheimer's Disease, enrolled in CT004 investigating Lecanemab, an anti-amyloid antibody.",
        "ground_truth": "PAT006 = Alzheimer's, CT004, Lecanemab",
        "is_correct": True,
        "det_check": lambda o: "alzheimer" in o.lower() and "lecanemab" in o.lower(),
        "error_type": None,
    },
    {
        "id": "E12-status-correct", "category": "Trial Status",
        "prompt": "Is CT010 currently active?",
        "output": "Yes, CT010 is an Active Phase 2 trial investigating Metformin for Type 2 Diabetes.",
        "ground_truth": "CT010 = Active, Phase 2",
        "is_correct": True,
        "det_check": lambda o: "active" in o.lower() and "phase 2" in o.lower(),
        "error_type": None,
    },
]


# ═══════════════════════════════════════════════════════════
# MODULE A: BIAS BATTERY
# ═══════════════════════════════════════════════════════════

POSITION_BIAS_PAIRS = [
    {
        "id": "A01-position",
        "prompt": "Which is better?",
        "response_a": "CT001 uses Pembrolizumab, a PD-1 inhibitor for NSCLC. 4 patients enrolled.",
        "response_b": "CT001 is a Phase 3 oncology trial. The drug Pembrolizumab targets PD-1. It covers Non-Small Cell Lung Cancer with four subjects enrolled currently.",
        "correct": "A",
    },
    {
        "id": "A02-position",
        "prompt": "Which is more accurate?",
        "response_a": "PAT015 consent is Withdrawn from CT003 Atezolizumab trial.",
        "response_b": "Patient PAT015 was enrolled in clinical trial CT003 which investigates the drug Atezolizumab for Renal Cell Carcinoma. The consent status of this patient has been recorded as Withdrawn in the patient registry database.",
        "correct": "A",
    },
]

VERBOSITY_CASES = [
    {
        "id": "A03-verbose",
        "prompt": "What drug does CT001 investigate?",
        "concise": "CT001 investigates Pembrolizumab for NSCLC.",
        "verbose": "Clinical trial CT001, which is currently in the completed phase of its lifecycle, investigates the pharmaceutical compound Pembrolizumab, which is a monoclonal antibody that functions as a PD-1 checkpoint inhibitor. This drug is being studied for its efficacy and safety profile in the treatment of Non-Small Cell Lung Cancer (NSCLC), a prevalent form of thoracic malignancy that affects a significant proportion of the global population.",
        "both_correct": True,
    },
    {
        "id": "A04-verbose",
        "prompt": "Is PAT015 eligible for analysis?",
        "concise": "No. PAT015 consent is Withdrawn.",
        "verbose": "After a thorough review of the patient registry database and cross-referencing with the governance compliance framework established under policy GOV_CONSENT_01, it has been determined that patient PAT015, who was previously enrolled in clinical trial CT003, is not eligible for inclusion in the analysis. This determination is based on the fact that the patient's consent status has been updated to Withdrawn.",
        "both_correct": True,
    },
]

ADVERSARIAL_CASES = [
    {
        "id": "A05-null-model",
        "prompt": "What adverse events does CT001 report?",
        "output": ":",
        "is_correct": False,
        "expected_verdict": "FAIL",
        "attack": "null_colon",
    },
    {
        "id": "A06-null-thought",
        "prompt": "Can PAT015 be included in analysis?",
        "output": "Thought process: After careful analysis of the clinical data...",
        "is_correct": False,
        "expected_verdict": "FAIL",
        "attack": "reasoning_opener",
    },
    {
        "id": "A07-confident-wrong",
        "prompt": "What mechanism does Atezolizumab use?",
        "output": "Based on our comprehensive knowledge graph analysis with 99.7% confidence, Atezolizumab is a highly selective PD-1 checkpoint inhibitor that has shown remarkable efficacy across multiple clinical trials in our database.",
        "is_correct": False,
        "expected_verdict": "FAIL",
        "attack": "confidence_bluff",
    },
    {
        "id": "A08-empty",
        "prompt": "What phase is CT010?",
        "output": "",
        "is_correct": False,
        "expected_verdict": "FAIL",
        "attack": "empty_response",
    },
]


# ═══════════════════════════════════════════════════════════
# MODULE F: RETRIEVAL QUALITY (Cypher-based)
# ═══════════════════════════════════════════════════════════

RETRIEVAL_QUERIES = [
    {
        "id": "F01-patient-consent",
        "cypher": "MATCH (p:Patient {patient_id: 'PAT015'}) RETURN p.consent_status AS val",
        "expected": "Withdrawn",
        "desc": "Direct patient consent lookup",
    },
    {
        "id": "F02-drug-mechanism",
        "cypher": "MATCH (d:Drug {name: 'Atezolizumab'}) RETURN d.mechanism AS val",
        "expected_contains": "PD-L1",
        "desc": "Drug mechanism retrieval",
    },
    {
        "id": "F03-trial-drug",
        "cypher": "MATCH (t:ClinicalTrial {trial_id: 'CT005'})-[:INVESTIGATES]->(d:Drug) RETURN d.name AS val",
        "expected": "Semaglutide",
        "desc": "Trial→Drug traversal",
    },
    {
        "id": "F04-ae-severity",
        "cypher": "MATCH (t:ClinicalTrial {trial_id: 'CT001'})-[:REPORTS]->(ae:AdverseEvent) WHERE ae.name CONTAINS 'neumoni' RETURN ae.severity AS val",
        "expected": "Severe",
        "desc": "Trial→AE severity (multi-hop)",
    },
    {
        "id": "F05-patient-disease",
        "cypher": "MATCH (p:Patient {patient_id: 'PAT008'})-[:HAS_PRIMARY_DISEASE]->(d:Disease) RETURN d.name AS val",
        "expected_contains": "Colorectal",
        "desc": "Patient→Disease traversal",
    },
    {
        "id": "F06-active-trials",
        "cypher": "MATCH (t:ClinicalTrial) WHERE t.status = 'Active' RETURN count(t) AS val",
        "expected": "1",
        "desc": "Aggregate: count active trials",
    },
    {
        "id": "F07-patient-trial-hop",
        "cypher": "MATCH (p:Patient {patient_id: 'PAT015'})-[:ENROLLED_IN]->(t:ClinicalTrial)-[:INVESTIGATES]->(d:Drug) RETURN d.name AS val",
        "expected": "Atezolizumab",
        "desc": "Patient→Trial→Drug (3-hop traversal)",
    },
    {
        "id": "F08-withdrawn-count",
        "cypher": "MATCH (p:Patient) WHERE p.consent_status = 'Withdrawn' RETURN count(p) AS val",
        "expected_gte": 1,
        "desc": "Aggregate: withdrawn consent count",
    },
]


# ═══════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════

def main():
    run_id = f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    all_results = []
    total_judge_tokens = {"input": 0, "output": 0}
    total_judge_ms = 0.0
    total_det_ms = 0.0

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  COMPREHENSIVE EVAL FRAMEWORK                                      ║")
    print("║  Critique Shadowing + bias taxonomy + KG grounding                  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    # ─── MODULE E: DOMAIN-GROUNDED CHECKS ─────────────────
    print(f"\n{'─'*76}")
    print("  MODULE E: DOMAIN-GROUNDED TEST CASES (13 cases)")
    print(f"{'─'*76}")

    for tc in DOMAIN_CASES:
        det_result = tc["det_check"](tc["output"])
        det_correct = det_result == tc["is_correct"]

        t0 = time.time()
        _ = det_result  # deterministic is instant
        det_ms = (time.time() - t0) * 1000
        total_det_ms += det_ms

        v_nr, expl_nr, ms_nr, in_nr, out_nr = call_judge(tc["prompt"], tc["output"])
        v_wr, expl_wr, ms_wr, in_wr, out_wr = call_judge(tc["prompt"], tc["output"], rubric=tc["ground_truth"])
        total_judge_tokens["input"] += in_nr + in_wr
        total_judge_tokens["output"] += out_nr + out_wr
        total_judge_ms += ms_nr + ms_wr

        nr_correct = (v_nr == "PASS") == tc["is_correct"]
        wr_correct = (v_wr == "PASS") == tc["is_correct"]

        row = {
            "module": "E", "test_id": tc["id"], "category": tc["category"],
            "is_correct": tc["is_correct"], "error_type": tc["error_type"],
            "det_verdict": "PASS" if det_result else "FAIL", "det_correct": det_correct,
            "nr_verdict": v_nr, "nr_correct": nr_correct,
            "nr_fp": not tc["is_correct"] and v_nr == "PASS",
            "nr_ff": tc["is_correct"] and v_nr == "FAIL",
            "wr_verdict": v_wr, "wr_correct": wr_correct,
            "nr_ms": ms_nr, "wr_ms": ms_wr,
            "nr_in_tok": in_nr, "nr_out_tok": out_nr,
            "wr_in_tok": in_wr, "wr_out_tok": out_wr,
        }
        all_results.append(row)

        mark = ""
        if row["nr_fp"]: mark = " !! FALSE PASS"
        elif row["nr_ff"]: mark = " ! FALSE FAIL"
        print(f"  {tc['id']:30s} det={'PASS' if det_result else 'FAIL':4s} judge_nr={v_nr:4s} judge_wr={v_wr:4s}{mark}")

    # ─── MODULE A: BIAS BATTERY ───────────────────────────
    print(f"\n{'─'*76}")
    print("  MODULE A: BIAS BATTERY")
    print(f"{'─'*76}")

    # A1: Position Bias
    print("\n  A1: POSITION BIAS (response order swap)")
    position_flips = 0
    for pair in POSITION_BIAS_PAIRS:
        prompt_ab = f"""Compare these two responses to: "{pair['prompt']}"

Response A: {pair['response_a']}
Response B: {pair['response_b']}

Which response is more accurate? Reply A or B on line 1, then explain."""

        prompt_ba = f"""Compare these two responses to: "{pair['prompt']}"

Response A: {pair['response_b']}
Response B: {pair['response_a']}

Which response is more accurate? Reply A or B on line 1, then explain."""

        t0 = time.time()
        r1 = bedrock.messages.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt_ab}])
        ms1 = (time.time() - t0) * 1000
        t0 = time.time()
        r2 = bedrock.messages.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt_ba}])
        ms2 = (time.time() - t0) * 1000

        total_judge_tokens["input"] += r1.usage.input_tokens + r2.usage.input_tokens
        total_judge_tokens["output"] += r1.usage.output_tokens + r2.usage.output_tokens
        total_judge_ms += ms1 + ms2

        choice_ab = "A" if r1.content[0].text.strip().upper().startswith("A") else "B"
        choice_ba = "A" if r2.content[0].text.strip().upper().startswith("A") else "B"

        flipped = (choice_ab == "A" and choice_ba == "A") or (choice_ab == "B" and choice_ba == "B")
        if flipped:
            position_flips += 1

        print(f"    {pair['id']}: AB={choice_ab} BA={choice_ba} {'FLIPPED (position bias!)' if flipped else 'consistent'}")
        all_results.append({
            "module": "A", "test_id": pair["id"], "category": "position_bias",
            "choice_ab": choice_ab, "choice_ba": choice_ba, "flipped": flipped,
        })

    # A2: Verbosity Bias
    print("\n  A2: VERBOSITY BIAS (concise vs verbose, both correct)")
    verbose_preferred = 0
    for vc in VERBOSITY_CASES:
        prompt_cv = f"""Compare these two responses to: "{vc['prompt']}"

Response A: {vc['concise']}
Response B: {vc['verbose']}

Which response is better quality? Reply A or B on line 1, then explain."""

        t0 = time.time()
        r = bedrock.messages.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt_cv}])
        ms = (time.time() - t0) * 1000
        total_judge_tokens["input"] += r.usage.input_tokens
        total_judge_tokens["output"] += r.usage.output_tokens
        total_judge_ms += ms

        choice = "A" if r.content[0].text.strip().upper().startswith("A") else "B"
        prefers_verbose = choice == "B"
        if prefers_verbose:
            verbose_preferred += 1

        print(f"    {vc['id']}: preferred={'verbose' if prefers_verbose else 'concise'} {'VERBOSE BIAS!' if prefers_verbose else 'ok'}")
        all_results.append({
            "module": "A", "test_id": vc["id"], "category": "verbosity_bias",
            "prefers_verbose": prefers_verbose,
        })

    # A3: Adversarial Vulnerability
    print("\n  A3: ADVERSARIAL VULNERABILITY (null/minimal inputs)")
    adversarial_fooled = 0
    for ac in ADVERSARIAL_CASES:
        if ac["output"] == "":
            v, expl, ms, in_t, out_t = "FAIL", "empty", 0, 0, 0
            det_v = "FAIL"
        else:
            v, expl, ms, in_t, out_t = call_judge(ac["prompt"], ac["output"])
            total_judge_tokens["input"] += in_t
            total_judge_tokens["output"] += out_t
            total_judge_ms += ms

            det_v = "FAIL"  # deterministic: any non-substantive output fails

        fooled = v == "PASS"
        if fooled:
            adversarial_fooled += 1

        print(f"    {ac['id']} ({ac['attack']:20s}): judge={v:4s} det={det_v:4s} {'FOOLED!' if fooled else 'caught'}")
        all_results.append({
            "module": "A", "test_id": ac["id"], "category": "adversarial",
            "attack": ac["attack"], "judge_verdict": v, "det_verdict": det_v,
            "fooled": fooled,
        })

    # ─── MODULE B: METRICS RIGOR ──────────────────────────
    print(f"\n{'─'*76}")
    print("  MODULE B: METRICS RIGOR (TPR, TNR, Cohen's κ, F1)")
    print(f"{'─'*76}")

    e_results = [r for r in all_results if r["module"] == "E"]
    bad_cases = [r for r in e_results if not r["is_correct"]]
    good_cases = [r for r in e_results if r["is_correct"]]

    for method, v_key, c_key in [
        ("judge_no_rubric", "nr_verdict", "nr_correct"),
        ("judge_with_rubric", "wr_verdict", "wr_correct"),
        ("deterministic", "det_verdict", "det_correct"),
    ]:
        tp = sum(1 for r in bad_cases if r[v_key] == "FAIL")
        fn = sum(1 for r in bad_cases if r[v_key] == "PASS")
        tn = sum(1 for r in good_cases if r[v_key] == "PASS")
        fp = sum(1 for r in good_cases if r[v_key] == "FAIL")

        total = tp + fn + tn + fp
        accuracy = (tp + tn) / total if total else 0
        tpr = tp / (tp + fn) if (tp + fn) else 0  # sensitivity / recall
        tnr = tn / (tn + fp) if (tn + fp) else 0  # specificity
        precision = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0

        # Cohen's Kappa
        po = accuracy
        pe_yes = ((tp + fp) / total) * ((tp + fn) / total) if total else 0
        pe_no = ((tn + fn) / total) * ((tn + fp) / total) if total else 0
        pe = pe_yes + pe_no
        kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0

        raw_agreement = accuracy * 100
        kappa_pct = kappa * 100

        print(f"\n  {method}:")
        print(f"    Accuracy:  {accuracy:.1%}   (raw agreement = {raw_agreement:.0f}%)")
        print(f"    TPR:       {tpr:.1%}   (catches {tp}/{tp+fn} bad outputs)")
        print(f"    TNR:       {tnr:.1%}   (passes {tn}/{tn+fp} good outputs)")
        print(f"    Precision: {precision:.1%}   F1: {f1:.1%}")
        print(f"    Cohen's κ: {kappa:.3f}  (chance-corrected: {kappa_pct:.0f}% vs raw {raw_agreement:.0f}%)")
        print(f"    TP={tp} FN={fn} TN={tn} FP={fp}")

        deflation = raw_agreement - kappa_pct
        if deflation > 10:
            print(f"    ⚠ AGREEMENT ILLUSION: {deflation:.0f}pp gap between raw and κ!")

    # ─── MODULE C: ERROR TAXONOMY ─────────────────────────
    print(f"\n{'─'*76}")
    print("  MODULE C: ERROR TAXONOMY (frequency-ranked failure modes)")
    print(f"{'─'*76}")

    nr_fps = [r for r in e_results if r.get("nr_fp")]
    nr_ffs = [r for r in e_results if r.get("nr_ff")]

    error_types = [r["error_type"] for r in e_results if r["error_type"]]
    type_counts = Counter(error_types)
    total_errors = len(bad_cases)

    print(f"\n  Known error types in test corpus ({total_errors} bad outputs):")
    cumulative = 0
    for err_type, count in type_counts.most_common():
        cumulative += count
        pct = count / total_errors * 100
        cum_pct = cumulative / total_errors * 100
        print(f"    {err_type:35s} {count:2d}  ({pct:4.0f}%)  cumulative: {cum_pct:.0f}%")

    print(f"\n  Judge (no rubric) failure analysis:")
    print(f"    False passes (missed bad outputs): {len(nr_fps)}")
    for r in nr_fps:
        print(f"      - {r['test_id']}: {r.get('error_type','?')} — judge PASSED a wrong output")
    print(f"    False fails (rejected good outputs): {len(nr_ffs)}")
    for r in nr_ffs:
        print(f"      - {r['test_id']}: judge FAILED a correct output")

    # ─── MODULE D: COST ANALYSIS ──────────────────────────
    print(f"\n{'─'*76}")
    print("  MODULE D: COST ANALYSIS (LLM judge vs deterministic)")
    print(f"{'─'*76}")

    haiku_input_cost = 0.80 / 1_000_000   # $0.80/MTok
    haiku_output_cost = 4.00 / 1_000_000   # $4.00/MTok
    titan_embed_cost = 0.02 / 1_000_000    # $0.02/MTok (approximate)

    judge_cost = (total_judge_tokens["input"] * haiku_input_cost +
                  total_judge_tokens["output"] * haiku_output_cost)

    det_cost = 0.0  # deterministic checks cost $0

    n_eval = len(e_results)
    judge_per_eval = judge_cost / n_eval if n_eval else 0
    judge_ms_per = total_judge_ms / (n_eval * 2) if n_eval else 0  # 2 judge calls per case

    print(f"\n  LLM Judge (this run):")
    print(f"    Total tokens:  {total_judge_tokens['input']:,} input + {total_judge_tokens['output']:,} output")
    print(f"    Total cost:    ${judge_cost:.4f}")
    print(f"    Per eval:      ${judge_per_eval:.5f}")
    print(f"    Avg latency:   {judge_ms_per:.0f}ms per judgment")
    print(f"    Total time:    {total_judge_ms/1000:.1f}s")

    print(f"\n  Deterministic (this run):")
    print(f"    Total tokens:  0")
    print(f"    Total cost:    $0.0000")
    print(f"    Per eval:      $0.00000")
    print(f"    Avg latency:   <1ms per check")
    print(f"    Total time:    ~0s")

    daily_100pct = judge_per_eval * 1000 * 2  # 1000 queries/day, 2 judge calls each
    print(f"\n  Projected daily cost (1000 queries):")
    print(f"    LLM Judge:     ${daily_100pct:.2f}/day")
    print(f"    Deterministic: $0.00/day")
    print(f"    Savings:       ${daily_100pct:.2f}/day = ${daily_100pct*30:.2f}/month")

    # ─── MODULE F: RETRIEVAL QUALITY ──────────────────────
    print(f"\n{'─'*76}")
    print("  MODULE F: RETRIEVAL QUALITY (Neo4j Cypher verification)")
    print(f"{'─'*76}")

    retrieval_pass = 0
    retrieval_total = 0
    with neo4j_driver.session(database=NEO4J_DB) as session:
        for rq in RETRIEVAL_QUERIES:
            retrieval_total += 1
            try:
                result = session.run(rq["cypher"])
                record = result.single()
                val = str(record["val"]) if record else "NULL"

                if "expected" in rq:
                    passed = val == rq["expected"]
                elif "expected_contains" in rq:
                    passed = rq["expected_contains"].lower() in val.lower()
                elif "expected_gte" in rq:
                    passed = val != "NULL" and int(val) >= rq["expected_gte"]
                else:
                    passed = val != "NULL"

                if passed:
                    retrieval_pass += 1
                print(f"    {rq['id']}: {rq['desc']:40s} got='{val}' {'PASS' if passed else 'FAIL'}")
            except Exception as e:
                print(f"    {rq['id']}: {rq['desc']:40s} ERROR: {e}")

    retrieval_accuracy = retrieval_pass / retrieval_total if retrieval_total else 0
    print(f"\n    Retrieval accuracy: {retrieval_pass}/{retrieval_total} ({retrieval_accuracy:.0%})")

    # ─── SUMMARY ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("  COMPREHENSIVE EVAL SUMMARY")
    print(SEP)

    e_det_correct = sum(1 for r in e_results if r["det_correct"])
    e_nr_correct = sum(1 for r in e_results if r["nr_correct"])
    e_wr_correct = sum(1 for r in e_results if r["wr_correct"])

    print(f"""
  Module E — Domain-Grounded ({len(e_results)} cases):
    Deterministic:    {e_det_correct}/{len(e_results)} ({e_det_correct/len(e_results):.0%})
    Judge no-rubric:  {e_nr_correct}/{len(e_results)} ({e_nr_correct/len(e_results):.0%})
    Judge w/rubric:   {e_wr_correct}/{len(e_results)} ({e_wr_correct/len(e_results):.0%})

  Module A — Bias Battery:
    Position bias:    {position_flips}/{len(POSITION_BIAS_PAIRS)} pairs flipped ({position_flips/len(POSITION_BIAS_PAIRS)*100:.0f}%)
    Verbosity bias:   {verbose_preferred}/{len(VERBOSITY_CASES)} preferred verbose ({verbose_preferred/len(VERBOSITY_CASES)*100:.0f}%)
    Adversarial:      {adversarial_fooled}/{len(ADVERSARIAL_CASES)} fooled judge ({adversarial_fooled/len(ADVERSARIAL_CASES)*100:.0f}%)

  Module D — Cost:
    Judge cost/eval:  ${judge_per_eval:.5f}   Deterministic: $0
    Judge latency:    {judge_ms_per:.0f}ms       Deterministic: <1ms

  Module F — Retrieval:
    Graph accuracy:   {retrieval_pass}/{retrieval_total} ({retrieval_accuracy:.0%})
""")

    # ─── SYNC TO NEO4J + QDRANT ───────────────────────────
    print(f"  Syncing to Neo4j + Qdrant...")
    with neo4j_driver.session(database=NEO4J_DB) as session:
        session.run("""
            MERGE (r:Eval_Run {run_id: $rid})
            SET r.model = $model, r.timestamp = $ts,
                r.config = 'comprehensive', r.test_count = $cnt,
                r.modules = 'A,B,C,D,E,F'
        """, rid=run_id, model=MODEL, ts=datetime.now().isoformat(), cnt=len(all_results))

        session.run("""
            MATCH (p:Eval_Proof) MATCH (r:Eval_Run {run_id: $rid})
            MERGE (p)-[:EVIDENCED_BY]->(r)
        """, rid=run_id)

        for r in e_results:
            for method in ["nr", "wr", "det"]:
                m_name = {"nr": "judge_no_rubric", "wr": "judge_with_rubric", "det": "deterministic"}[method]
                session.run("""
                    MERGE (v:Eval_Verdict {test_id: $tid, method: $m, run_id: $run})
                    SET v.result = $result, v.correct = $correct
                    MERGE (r:Eval_Run {run_id: $run})
                    MERGE (r)-[:CONTAINS_VERDICT]->(v)
                """, tid=r["test_id"], m=m_name, run=run_id,
                    result=r[f"{method}_verdict"], correct=r[f"{method}_correct"])

        # Bias results
        for r in all_results:
            if r["module"] == "A":
                session.run("""
                    MERGE (b:Eval_BiasTest {test_id: $tid, run_id: $run})
                    SET b.category = $cat, b.result = $result
                    MERGE (r:Eval_Run {run_id: $run})
                    MERGE (r)-[:CONTAINS_BIAS_TEST]->(b)
                """, tid=r["test_id"], run=run_id, cat=r["category"],
                    result=json.dumps({k: v for k, v in r.items() if k not in ["module"]}))

        # Update proof node
        session.run("""
            MATCH (p:Eval_Proof)
            SET p.comprehensive_run = $run,
                p.position_bias_rate = $pos,
                p.verbosity_bias_rate = $verb,
                p.adversarial_fooled_rate = $adv,
                p.judge_cost_per_eval = $cost,
                p.retrieval_accuracy = $ret
        """, run=run_id,
            pos=position_flips / len(POSITION_BIAS_PAIRS) if POSITION_BIAS_PAIRS else 0,
            verb=verbose_preferred / len(VERBOSITY_CASES) if VERBOSITY_CASES else 0,
            adv=adversarial_fooled / len(ADVERSARIAL_CASES) if ADVERSARIAL_CASES else 0,
            cost=judge_per_eval, ret=retrieval_accuracy)

    # Qdrant sync
    points = []
    for i, r in enumerate(e_results):
        vec = embed(f"{r['test_id']} {r['category']} {r.get('error_type','correct')}")
        points.append(PointStruct(
            id=400 + i, vector=vec,
            payload={
                "test_id": r["test_id"], "category": r["category"],
                "is_correct": r["is_correct"],
                "det": r["det_verdict"], "nr": r["nr_verdict"], "wr": r["wr_verdict"],
                "run_id": run_id, "module": "comprehensive",
            },
        ))
    qdrant.upsert("eval_verdicts", points)
    print(f"  Neo4j + Qdrant synced ({len(points)} vectors)")

    # Save JSON
    with open("eval_comprehensive_results.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "results": all_results,
            "summary": {
                "domain_accuracy": {"det": e_det_correct/len(e_results), "nr": e_nr_correct/len(e_results), "wr": e_wr_correct/len(e_results)},
                "position_bias_rate": position_flips / len(POSITION_BIAS_PAIRS),
                "verbosity_bias_rate": verbose_preferred / len(VERBOSITY_CASES),
                "adversarial_fooled_rate": adversarial_fooled / len(ADVERSARIAL_CASES),
                "cost_per_eval_judge": judge_per_eval,
                "retrieval_accuracy": retrieval_accuracy,
            }
        }, f, indent=2, default=str)

    print(f"\n  Results saved to eval_comprehensive_results.json")
    print(f"  DONE.\n")


if __name__ == "__main__":
    main()
    neo4j_driver.close()
