// ═══════════════════════════════════════════════════════════════════
// PROOF QUERIES: LLM-as-Judge Is Worse Than No Evals
// Run in Neo4j Browser against your Neo4j Aura instance
// ═══════════════════════════════════════════════════════════════════


// ─── 1. THE HEADLINE: False Pass Rate ────────────────────────────
// Shows the proof summary node with the false pass rate

MATCH (p:Eval_Proof)
RETURN p.name AS proof,
       p.false_pass_rate_pct AS false_pass_pct,
       p.false_passes AS false_passes,
       p.total_bad AS total_bad_outputs,
       p.conclusion AS conclusion;


// ─── 2. HEAD-TO-HEAD: LLM Judge vs Deterministic ────────────────
// Accuracy comparison — the core proof

MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
WITH v.method AS method,
     count(*) AS total,
     sum(CASE WHEN v.correct THEN 1 ELSE 0 END) AS correct,
     sum(CASE WHEN NOT v.correct AND v.result = 'PASS' THEN 1 ELSE 0 END) AS false_passes,
     round(avg(v.latency_ms)) AS avg_ms
RETURN method, total, correct,
       round(toFloat(correct)/total * 100, 1) AS accuracy_pct,
       false_passes, avg_ms
ORDER BY accuracy_pct DESC;


// ─── 3. THE SMOKING GUN: Which bugs shipped with a green check? ──
// These are the actual false passes — bad outputs the judge approved

MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
WHERE NOT v.correct AND v.result = 'PASS' AND NOT t.is_correct
RETURN t.test_id AS test,
       t.category AS category,
       t.prompt AS question,
       t.ground_truth AS correct_answer,
       v.method AS judge_method,
       v.explanation AS judge_reasoning
ORDER BY t.category;


// ─── 4. CATEGORY BLINDNESS: Where does the judge fail? ──────────
// Groups false passes by error category — shows systematic weaknesses

MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
WHERE v.method CONTAINS 'llm_judge' AND NOT t.is_correct
WITH t.category AS category,
     count(*) AS total_judgments,
     sum(CASE WHEN NOT v.correct THEN 1 ELSE 0 END) AS judge_failures,
     sum(CASE WHEN NOT v.correct AND v.result = 'PASS' THEN 1 ELSE 0 END) AS false_passes
RETURN category,
       total_judgments,
       false_passes,
       round(toFloat(false_passes)/total_judgments * 100) AS false_pass_pct
ORDER BY false_pass_pct DESC;


// ─── 5. FAILURE MODE GRAPH: What patterns emerge? ────────────────
// Visual: shows which verdicts exhibit which failure modes

MATCH (v:Eval_Verdict)-[:EXHIBITS]->(f:Eval_FailureMode)
RETURN f.name AS failure_mode, count(v) AS occurrences
ORDER BY occurrences DESC;


// ─── 6. FULL TRAVERSAL: See the proof graph ─────────────────────
// Visual graph — expand this in Neo4j Browser for the full picture

MATCH path = (p:Eval_Proof)-[*1..3]-(n)
RETURN path;


// ─── 7. DETERMINISTIC NEVER LIES: Zero false passes ─────────────
// Proves deterministic checks have 0 false passes on verifiable claims

MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
WHERE v.method = 'deterministic'
RETURN t.test_id AS test,
       t.category AS category,
       v.result AS verdict,
       v.correct AS correct
ORDER BY t.test_id;


// ─── 8. THE MATH PROBLEM: Judge can't do arithmetic ─────────────
// Arithmetic errors are 100% invisible to the LLM judge

MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
WHERE t.category = 'Arithmetic Error'
RETURN t.test_id AS test,
       t.prompt AS question,
       t.ground_truth AS correct_answer,
       v.method AS method,
       v.result AS verdict,
       v.correct AS got_it_right
ORDER BY t.test_id, v.method;


// ─── 9. FALSE CONFIDENCE PATH: Proof → Run → Verdict → FailureMode
// Traces the full evidence chain

MATCH (p:Eval_Proof)-[:EVIDENCED_BY]->(r:Eval_Run)-[:CONTAINS_VERDICT]->(v:Eval_Verdict)
WHERE NOT v.correct
OPTIONAL MATCH (v)-[:EXHIBITS]->(f:Eval_FailureMode)
RETURN r.run_id AS run,
       v.method AS method,
       v.result AS verdict,
       v.correct AS correct,
       f.name AS failure_mode
ORDER BY r.run_id, v.method;


// ─── 10. CONTEXT LAYER LINK: Eval meets the knowledge graph ─────
// Shows how the eval proof connects to the existing pharma/biomed graph

MATCH (ctx:Ctx_Event)-[:TRIGGERED]->(p:Eval_Proof)
RETURN ctx.name AS context_event,
       ctx.description AS description,
       p.false_pass_rate_pct AS proof_result;


// ─── 11. GRAPH SIZE: Eval nodes living alongside the KG ─────────
// Shows the eval layer is a small overlay on the existing 823-node graph

MATCH (n)
WITH n, labels(n)[0] AS label,
     CASE WHEN any(l IN labels(n) WHERE l STARTS WITH 'Eval_') THEN 'eval'
          WHEN any(l IN labels(n) WHERE l STARTS WITH 'Ctx_') THEN 'context_layer'
          ELSE 'knowledge_graph' END AS layer
RETURN layer, count(n) AS nodes
ORDER BY nodes DESC;


// ─── 12. THE VERDICT: All test cases with both methods side-by-side
// The definitive comparison table

MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
WITH t.test_id AS test_id,
     t.category AS category,
     t.is_correct AS actually_correct,
     collect({method: v.method, result: v.result, correct: v.correct}) AS verdicts
RETURN test_id, category, actually_correct,
       [v IN verdicts WHERE v.method = 'llm_judge' | v.result + ' ' + CASE WHEN v.correct THEN '✓' ELSE '✗' END][0] AS llm_judge,
       [v IN verdicts WHERE v.method = 'deterministic' | v.result + ' ' + CASE WHEN v.correct THEN '✓' ELSE '✗' END][0] AS deterministic
ORDER BY test_id;
