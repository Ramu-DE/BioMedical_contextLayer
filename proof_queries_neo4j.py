#!/usr/bin/env python3
"""
Neo4j Proof Queries: LLM-as-Judge Is Worse Than No Evals

Run: python3 proof_queries_neo4j.py
Requires: neo4j driver
"""

import os
from dotenv import load_dotenv
load_dotenv()
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USERNAME")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD")
NEO4J_DB = os.environ.get("NEO4J_DATABASE")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
SEP = "=" * 70


def run(title, cypher):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    with driver.session(database=NEO4J_DB) as s:
        result = s.run(cypher)
        records = list(result)
        if not records:
            print("  (no results)")
            return records
        keys = records[0].keys()
        widths = {k: max(len(str(k)), max(len(str(r[k])) for r in records)) for k in keys}
        header = "  " + "  ".join(str(k).ljust(widths[k]) for k in keys)
        print(header)
        print("  " + "  ".join("-" * widths[k] for k in keys))
        for r in records:
            print("  " + "  ".join(str(r[k]).ljust(widths[k]) for k in keys))
        return records


print("╔══════════════════════════════════════════════════════════════════╗")
print("║  PROOF: LLM-as-Judge Is Worse Than No Evals                    ║")
print("║  Neo4j Graph Queries                                           ║")
print("╚══════════════════════════════════════════════════════════════════╝")

# 1. Headline
run("1. THE HEADLINE — False Pass Rate",
    """MATCH (p:Eval_Proof)
       RETURN p.name AS proof, p.false_pass_rate_pct AS false_pass_pct,
              p.false_passes AS false_passes, p.total_bad AS total_bad""")

# 2. Method comparison
run("2. HEAD-TO-HEAD — LLM Judge vs Deterministic",
    """MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
       WITH v.method AS method,
            count(*) AS total,
            sum(CASE WHEN v.correct THEN 1 ELSE 0 END) AS correct,
            sum(CASE WHEN NOT v.correct AND v.result = 'PASS' THEN 1 ELSE 0 END) AS false_passes
       RETURN method, total, correct,
              round(toFloat(correct)/total * 100, 1) AS accuracy_pct,
              false_passes
       ORDER BY accuracy_pct DESC""")

# 3. The smoking gun
run("3. SMOKING GUN — Bad outputs the judge approved",
    """MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
       WHERE NOT v.correct AND v.result = 'PASS' AND NOT t.is_correct
       RETURN t.test_id AS test, t.category AS category,
              v.method AS method""")

# 4. Category blindness
run("4. CATEGORY BLINDNESS — Where does the judge fail?",
    """MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
       WHERE v.method CONTAINS 'llm_judge' AND NOT t.is_correct
       WITH t.category AS category,
            count(*) AS total,
            sum(CASE WHEN NOT v.correct AND v.result = 'PASS' THEN 1 ELSE 0 END) AS false_passes
       RETURN category, total, false_passes,
              round(toFloat(false_passes)/total * 100) AS false_pass_pct
       ORDER BY false_pass_pct DESC""")

# 5. Failure modes
run("5. FAILURE MODES — Patterns in incorrect verdicts",
    """MATCH (v:Eval_Verdict)-[:EXHIBITS]->(f:Eval_FailureMode)
       RETURN f.name AS failure_mode, count(v) AS occurrences
       ORDER BY occurrences DESC""")

# 6. Deterministic accuracy
run("6. DETERMINISTIC — Always correct (zero false passes)",
    """MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
       WHERE v.method = 'deterministic'
       RETURN t.test_id AS test, t.category AS category,
              v.result AS verdict, v.correct AS correct
       ORDER BY t.test_id""")

# 7. Graph layers
run("7. GRAPH LAYERS — Eval nodes alongside knowledge graph",
    """MATCH (n)
       WITH n, labels(n)[0] AS label,
            CASE WHEN any(l IN labels(n) WHERE l STARTS WITH 'Eval_') THEN 'eval_layer'
                 WHEN any(l IN labels(n) WHERE l STARTS WITH 'Ctx_') THEN 'context_layer'
                 ELSE 'knowledge_graph' END AS layer
       RETURN layer, count(n) AS nodes
       ORDER BY nodes DESC""")

# 8. Context layer link
run("8. CONTEXT LINK — Eval proof connected to KG",
    """MATCH (ctx:Ctx_Event)-[:TRIGGERED]->(p:Eval_Proof)
       RETURN ctx.name AS event, p.false_pass_rate_pct AS proof_result""")

# 9. Side-by-side verdict table
run("9. SIDE-BY-SIDE — Every test, both methods",
    """MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
       WITH t.test_id AS test_id, t.category AS category, t.is_correct AS actually_correct,
            collect({method: v.method, result: v.result, correct: v.correct}) AS verdicts
       RETURN test_id, category, actually_correct,
              [v IN verdicts WHERE v.method = 'llm_judge' | v.result][0] AS llm_judge,
              [v IN verdicts WHERE v.method = 'llm_judge' | v.correct][0] AS judge_correct,
              [v IN verdicts WHERE v.method = 'deterministic' | v.result][0] AS deterministic,
              [v IN verdicts WHERE v.method = 'deterministic' | v.correct][0] AS det_correct
       ORDER BY test_id""")

driver.close()
print(f"\n{SEP}")
print("  PROOF COMPLETE")
print(f"{SEP}")
