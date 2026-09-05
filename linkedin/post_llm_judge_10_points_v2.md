**LLM-as-Judge is Worse Than No Evals.**

I tested it on a live knowledge graph built from biomedical. Here are 10 things I learned:

1. An LLM judge without ground truth scored **64% accuracy** — it rejected every correct output because it couldn't verify facts against the domain.

2. The same judge with a "**domain expert**" prompt scored the **exact same 64%**. Prompt engineering doesn't fix a structural problem.

3. The only judge variant that scored 100% was the one **given the correct answer upfront**. But if you already have the answer, why do you need a judge?

4. This creates a **circular dependency** nobody talks about: LLM-as-Judge needs ground truth to work, but if you have ground truth, you don't need LLM-as-Judge.

5. Deterministic checks grounded in a knowledge graph scored **100% accuracy** — zero false passes, zero false fails, zero ambiguity.

6. The judge has no way to **verify** whether "Severe" or "Moderate" is correct — without ground truth it rejects both. In clinical governance, you need certainty, not guessing.

7. The judge can't distinguish similar-sounding entities — Semaglutide vs Tirzepatide, PD-1 vs PD-L1 — it has no ontological grounding. A knowledge graph resolves them by node identity, not string similarity.

8. In production you get two choices: **trust the judge** (block **100% of correct responses**) or **ignore the judge** (same as having no evals). Both options are worse than no evals.

9. What works instead? **Context you build** — ontology, rules, identifiers, and a knowledge graph. Not context you buy — another LLM API call.

10. The real eval isn't "does the AI sound right?" It's **"does the AI match what the graph knows?"** That question has a deterministic answer. Every time.

Your context is your edge.

**Stop evaluating AI with more AI. Start building context that knows the difference.**

---

### Validation Notes (data-backed corrections from v1)

| Point | v1 Claim | Actual Data | Correction |
|-------|----------|-------------|------------|
| 6 | "couldn't tell Severe from Moderate" | Judge FAILED both (correctly rejected bad output) | Reworded: judge can't *verify* which is correct without ground truth |
| 7 | "swapped entities and called it correct" | Judge FAILED entity swaps (0 false passes) | Reworded: judge can't *distinguish* them, not that it approved swaps |
| 8 | "block 36% of correct responses" | Judge blocked 5/5 = **100%** of correct outputs | Fixed: 100%, not 36% |

**Core failure mode:** The no-rubric judge is a **blind rejector** (100% false fail on controls), not a blind approver (0% false pass). It says FAIL to everything because it has no ground truth to verify against.

---

#NeurosymbolicAI #KnowledgeGraph #LLMEvals #GraphEngineering #AIGovernance #ContextEngineering #Ontology #DataQuality #AIReliability #GenerativeAI
