"""Linked Context: the join between text and graph. Tasks 2.1-2.2.

Diagram: [EK-5] Linked Enterprise Context. Hub node: Linked Context.

This is the keystone component: verified absent from all surveyed repos, and
every distinctive capability downstream depends on it.

Contracts:
    term_index.build(bundle) -> int
        Embeds every ontology label and synonym into OPENSEARCH_TERM_INDEX,
        keyed by CURIE (R4.1).
    entity_linker.link(chunk) -> list[EntityMention]
        k-NN over the term index. Below ENTITY_LINK_THRESHOLD emits
        curie=None rather than a low-confidence binding (R4.4).
        Verifies the CURIE exists in Neo4j before linking (R4.5).
"""
