"""Runtime Context: the living model. Tasks 3.1-3.4.

Diagram: [RC-1] Events & Triggers, [RC-2] Decisions & Outcomes,
         [RC-3] Actions & Responses, [RC-4] Continuous Incorporation,
         [RC-5] Living Enterprise Model. Hub node: Runtime Context.

Replaces the in-memory audit_logger from the upstream MCP repo with a durable
append-only graph store.

Contracts:
    store.append_event(event) -> str
    store.append_decision(decision) -> str
    store.append_action(action) -> str
    store.append_outcome(outcome) -> str
    store.recent_decisions(embedding, k) -> list[Decision]   (R7.1)
    store.as_of(timestamp) -> list[Decision]                 (R7.3)

Append-only is structural: no update or delete verb exists (R6.4). Superseding
a record means writing a new version with valid_from, and setting valid_to on
the prior version (R7.2). All writes confined to NEO4J_CTX_DATABASE or the
CTX_LABEL_PREFIX namespace (R5.3).
"""
