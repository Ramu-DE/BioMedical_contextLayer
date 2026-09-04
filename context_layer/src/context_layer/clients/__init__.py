"""Infrastructure adapters. Tasks 1.1.

Contracts:
    neo4j_client.query(cypher, params, database) -> list[dict]
        Parameterized only. Rejects f-string interpolation (R10.6).
    opensearch_client.knn_search(index, vector, k, filters) -> list[Hit]
    opensearch_client.hybrid_search(index, text, vector, k) -> list[Hit]
    opensearch_client.bulk_index(index, docs) -> int
    embeddings.embed(texts) -> list[list[float]]
        Batched, retried with exponential backoff.
    llm.chat(messages) -> str   # ChatBedrockConverse, matches module 5
"""
