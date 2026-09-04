"""Retrievers feeding the context assembler. Task 5.1, Requirement 8.1, 8.5.

Three independent sources. Each one reports its own failure as a *degradation*
rather than raising, so a thin context package is never mistaken for a
confident one. The assembler decides what to do about it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, Sequence

from context_layer.types import ContextElement, Provenance

if TYPE_CHECKING:
    from config import Config

GOV_COLLECTION = "ctx_gov_chunks"
BIOMED_COLLECTION = "ctx_biomed_chunks"


def _prov(source: str, ingested: str | None, confidence: float, record_id: str | None):
    when = datetime.now(timezone.utc)
    if ingested:
        try:
            when = datetime.fromisoformat(ingested.replace("Z", "+00:00"))
        except ValueError:
            pass
    return Provenance(
        source=source, ingested_at=when, confidence=confidence, record_id=record_id
    )


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, k: int) -> tuple[list[ContextElement], list[str]]:
        """Returns (elements, degradations). Never raises for expected failures."""
        ...


# ── Vector retrieval over both Qdrant collections ─────────────────────────────


class VectorRetriever:
    name = "vector"

    def __init__(self, cfg: Config, embedder=None, store=None,
                 collections: Sequence[str] = (BIOMED_COLLECTION, GOV_COLLECTION)) -> None:
        self.cfg = cfg
        self.collections = tuple(collections)
        self._embedder = embedder
        self._store = store

    def _lazy(self):
        if self._embedder is None:
            from context_layer.clients.embeddings import BedrockEmbeddings

            self._embedder = BedrockEmbeddings(self.cfg)
        if self._store is None:
            from context_layer.clients.vector_store import QdrantVectorStore

            self._store = QdrantVectorStore(self.cfg)
        return self._embedder, self._store

    def retrieve(self, question: str, k: int = 6):
        elements: list[ContextElement] = []
        degradations: list[str] = []
        try:
            embedder, store = self._lazy()
            vector = embedder.embed_one(question)
        except Exception as e:  # noqa: BLE001
            return [], [f"vector_retriever_unavailable: {type(e).__name__}"]

        for collection in self.collections:
            try:
                store.collection = collection
                for hit in store.search(vector, k=k):
                    if not hit.text:
                        continue
                    elements.append(
                        ContextElement(
                            element_id=f"{collection}:{hit.point_id}",
                            kind="chunk",
                            content=hit.text,
                            provenance=_prov(
                                hit.payload.get("source", collection),
                                hit.payload.get("ingested_at"),
                                float(hit.payload.get("confidence", 1.0)),
                                hit.payload.get("source_id"),
                            ),
                            score=hit.score,
                            curies=hit.curies,
                        )
                    )
            except Exception as e:  # noqa: BLE001
                degradations.append(
                    f"vector_collection_unavailable:{collection}: {type(e).__name__}"
                )
        return elements, degradations


# ── Graph retrieval: expand the CURIEs the chunks mentioned ────────────────────

_GRAPH_EXPAND = """
MATCH (e)-[:MAPS_TO]->(c:ExternalConcept)
WHERE c.curie IN $curies
WITH DISTINCT e LIMIT $limit
OPTIONAL MATCH (e)-[r]->(m)
WHERE type(r) IN $rels
WITH e, collect(DISTINCT type(r) + ' -> ' + coalesce(m.name, m.title, m.curie, ''))[0..8] AS edges
OPTIONAL MATCH (e)-[:MAPS_TO]->(cc:ExternalConcept)
RETURN labels(e)[0] AS label,
       coalesce(e.name, e.title, '?') AS name,
       properties(e) AS props,
       edges AS edges,
       collect(DISTINCT cc.curie) AS curies
"""

_GRAPH_KEYWORD = """
CALL () {
  MATCH (p:Policy) RETURN p AS n, 'Policy' AS label
  UNION
  MATCH (p:DataGovernancePolicy) RETURN p AS n, 'DataGovernancePolicy' AS label
}
WITH n, label
WHERE any(t IN $terms WHERE
      toLower(coalesce(n.name, n.title, '')) CONTAINS t
      OR toLower(coalesce(n.description, n.scope, '')) CONTAINS t)
RETURN label,
       coalesce(n.name, n.title, '?') AS name,
       properties(n) AS props,
       [] AS edges,
       [] AS curies
LIMIT $limit
"""

INTERESTING_RELS = [
    "TREATS", "TARGETS", "HAS_PHENOTYPE", "ASSOCIATED_WITH", "REPORTS",
    "INVESTIGATES", "STUDIES", "GOVERNS_ENTITY", "GOVERNED_BY",
    "HAS_PRIMARY_DISEASE", "PREDICTS_RESPONSE_TO", "MAPS_TO",
]

STOPWORDS = {
    "what", "which", "who", "how", "why", "when", "the", "a", "an", "is", "are",
    "for", "of", "in", "on", "to", "and", "or", "can", "we", "i", "do", "does",
    "with", "from", "about", "should", "would", "any", "there", "this", "that",
    "use", "used", "data", "please", "tell", "me",
}


class GraphRetriever:
    name = "graph"

    def __init__(self, cfg: Config, driver=None) -> None:
        self.cfg = cfg
        self._driver = driver
        self._owns = driver is None

    def _lazy(self):
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.cfg.neo4j_uri,
                auth=(self.cfg.neo4j_user, str(self.cfg.neo4j_password)),
            )
        return self._driver

    def close(self) -> None:
        if self._owns and self._driver is not None:
            self._driver.close()
            self._driver = None

    @staticmethod
    def _terms(question: str) -> list[str]:
        words = [
            w.strip(".,?!:;()'\"").lower()
            for w in question.split()
        ]
        return [w for w in words if len(w) > 3 and w not in STOPWORDS][:8]

    def _element(self, row, idx: int, via: str) -> ContextElement:
        props = dict(row["props"] or {})
        edges = [e for e in (row["edges"] or []) if e and not e.endswith("-> ")]
        detail = "; ".join(f"{k}={v}" for k, v in list(props.items())[:8] if v not in (None, ""))
        text = f"{row['label']} {row['name']}. {detail}"
        if edges:
            text += ". Links: " + "; ".join(edges)
        return ContextElement(
            element_id=f"graph:{row['label']}:{row['name']}:{idx}",
            kind="graph_node",
            content=text,
            provenance=_prov(
                f"neo4j:{row['label']}",
                props.get("ingested_at"),
                float(props.get("confidence", 1.0)),
                None,
            ),
            score=0.75 if via == "curie" else 0.55,
            curies=tuple(c for c in (row["curies"] or []) if c),
        )

    def retrieve(self, question: str, k: int = 8, curies: Sequence[str] = ()):
        elements: list[ContextElement] = []
        degradations: list[str] = []
        try:
            driver = self._lazy()
            with driver.session(database=self.cfg.neo4j_kg_database) as s:
                if curies:
                    rows = list(
                        s.run(
                            _GRAPH_EXPAND,
                            curies=list(curies),
                            limit=k,
                            rels=INTERESTING_RELS,
                        )
                    )
                    elements += [
                        self._element(r, i, "curie") for i, r in enumerate(rows)
                    ]
                terms = self._terms(question)
                if terms:
                    rows = list(s.run(_GRAPH_KEYWORD, terms=terms, limit=k))
                    elements += [
                        self._element(r, i, "keyword") for i, r in enumerate(rows)
                    ]
        except Exception as e:  # noqa: BLE001
            degradations.append(f"graph_retriever_unavailable: {type(e).__name__}: {e}")
        return elements, degradations


# ── Prior decisions, for continuous incorporation ─────────────────────────────


class RuntimeRetriever:
    """Prior decisions as context. Requirement 7.1."""

    name = "runtime"

    def __init__(self, cfg: Config, store=None) -> None:
        self.cfg = cfg
        self._store = store

    def retrieve(self, question: str, k: int | None = None):
        if self._store is None:
            return [], ["runtime_retriever_unavailable: no decision store configured"]
        k = k or self.cfg.prior_decision_k
        try:
            decisions = self._store.recent_decisions(question, k)
        except Exception as e:  # noqa: BLE001
            return [], [f"runtime_retriever_unavailable: {type(e).__name__}"]
        elements = [
            ContextElement(
                element_id=f"decision:{d.decision_id}",
                kind="prior_decision",
                content=(
                    f"Prior decision on '{d.question}': rules fired "
                    f"{list(d.rules_fired) or 'none'}, grounding "
                    f"{d.grounding_confidence:.2f}, at {d.decided_at.isoformat()}."
                ),
                provenance=_prov("runtime:Ctx_Decision", None, 1.0, d.decision_id),
                score=0.5,
                curies=tuple(d.grounded_in),
            )
            for d in decisions
        ]
        return elements, []
