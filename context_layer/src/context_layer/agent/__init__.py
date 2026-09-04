"""Neurosymbolic Agent. Tasks 6.3-6.5.

Diagram: [AI-1] Neurosymbolic Reasoning, [AI-2] Agentic AI Applications,
         [AI-3] Decision Intelligence, [AI-4] AI Automation & Planning.

Contracts:
    tools: graph_lookup, vector_search, check_rules, prior_decisions
    agent.run(question) -> GovernedResponse

Ordering is enforced by the governance gate, not by the prompt: rules are
evaluated on the assembled package before any answer is released, and a block
verdict suppresses the answer regardless of what the model produced (R9.2).

Hosting matches module-5-langgraph-agent: POST /invoke and GET /health on
8080, MicroVM ready hook on 9000.
"""
