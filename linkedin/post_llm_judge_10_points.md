**LLM-as-Judge is Worse Than No Evals.**

I tested it on a live knowledge graph. Here are 10 things I learned:

1. An LLM judge without ground truth scored **64% accuracy** — it rejected every correct output because it couldn't verify facts against the domain.

2. The same judge with a "domain expert" prompt scored the **exact same 64%**. Prompt engineering doesn't fix a structural problem.

3. The only judge variant that scored 100% was the one **given the correct answer upfront**. But if you already have the answer, why do you need a judge?

4. This creates a **circular dependency** nobody talks about: LLM-as-Judge needs ground truth to work, but if you have ground truth, you don't need LLM-as-Judge.

5. Deterministic checks grounded in a knowledge graph scored **100% accuracy** — zero false passes, zero false fails, zero ambiguity.

6. The judge **couldn't tell** the difference between "Severe" and "Moderate" adverse events. In clinical governance, that changes the entire risk classification.

7. The judge **swapped similar-sounding entities** — two drugs in the same therapeutic class, two patients in the same trial — and called it correct. A knowledge graph never confuses them.

8. In production you get two choices: trust the judge (block 36% of correct responses) or ignore the judge (same as having no evals). **Both options are worse than no evals.**

9. What works instead? **Context you build** — ontology, rules, identifiers, and a knowledge graph. Not context you buy — another LLM API call.

10. The real eval isn't "does the AI sound right?" It's **"does the AI match what the graph knows?"** That question has a deterministic answer. Every time.

Your context is your edge.

Stop evaluating AI with more AI. Start building context that knows the difference.

---

#NeurosymbolicAI #KnowledgeGraph #LLMEvals #GraphEngineering #AIGovernance #ContextEngineering #Ontology #DataQuality #AIReliability #GenerativeAI

---

**Image:** linkedin_edge.jpg
