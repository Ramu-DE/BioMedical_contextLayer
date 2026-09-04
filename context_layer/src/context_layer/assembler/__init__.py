"""Context Assembly: what enters the model window. Tasks 5.1-5.2.

The binding component. Adapted from rag-ai-factory's Assembler pattern.

Contracts:
    retrievers.VectorRetriever.retrieve(question)  -> list[ContextElement]
    retrievers.GraphRetriever.retrieve(question)   -> list[ContextElement]
    retrievers.RuntimeRetriever.retrieve(question) -> list[ContextElement]
        Each degrades independently and reports its degradation (R8.5).
    package.assemble(question) -> ContextPackage
        Merges all three, dedups by CURIE, ranks, enforces
        CONTEXT_TOKEN_BUDGET, reports truncation, retains provenance
        per element (R8.2-8.4).
"""
