"""Vector store abstraction with a Qdrant backend. Task 1.1.

Deliberately narrow: the context layer needs upsert, search, and count. Keeping
the protocol small is what makes the backend swappable (Requirement 8.1) —
Qdrant today, Neo4j-native or OpenSearch later, without touching the assembler.

Payload contract: every stored point carries ``text`` (the real source text, not
a stringified vector), ``source``, ``ingested_at``, and optional ``curies``.
Requirement 2.2 — no assertion enters the layer without provenance.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from config import Config

# Payload keys we require on every point.
TEXT_KEY = "text"
SOURCE_KEY = "source"
INGESTED_KEY = "ingested_at"
CURIES_KEY = "curies"


@dataclass(frozen=True)
class VectorHit:
    point_id: str
    score: float
    text: str
    payload: dict[str, Any]

    @property
    def curies(self) -> tuple[str, ...]:
        return tuple(self.payload.get(CURIES_KEY) or ())

    @property
    def source(self) -> str:
        return str(self.payload.get(SOURCE_KEY, ""))


@dataclass(frozen=True)
class VectorRecord:
    """One point to store. ``text`` must be the source text."""

    text: str
    vector: list[float]
    payload: dict[str, Any]
    point_id: str | None = None

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("VectorRecord.text must be non-empty source text")
        # Guard against the defect found in the pre-existing idp_chunks
        # collection, where chunk_text held a stringified embedding.
        head = self.text.lstrip()[:24]
        if head and all(c in "-0123456789.,eE \n" for c in head):
            raise ValueError(
                "VectorRecord.text looks like a stringified vector, not source "
                f"text: {head!r}"
            )
        if not self.payload.get(SOURCE_KEY):
            raise ValueError("VectorRecord.payload requires a 'source' for provenance")


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, dim: int) -> None: ...
    def upsert(self, records: Sequence[VectorRecord]) -> int: ...
    def search(
        self, vector: list[float], k: int = 5, must: dict[str, Any] | None = None
    ) -> list[VectorHit]: ...
    def count(self) -> int: ...


class QdrantVectorStore:
    """Qdrant Cloud backend."""

    def __init__(self, cfg: Config, client=None) -> None:
        self.cfg = cfg
        self.collection = cfg.qdrant_collection
        if client is not None:
            self._c = client
        else:
            from qdrant_client import QdrantClient

            self._c = QdrantClient(
                url=cfg.qdrant_url,
                api_key=str(cfg.qdrant_api_key),
                timeout=cfg.qdrant_timeout,
            )

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        distance = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclid": Distance.EUCLID,
        }[self.cfg.vector_similarity.lower()]

        existing = {c.name for c in self._c.get_collections().collections}
        if self.collection in existing:
            info = self._c.get_collection(self.collection)
            actual = info.config.params.vectors.size
            if actual != dim:
                raise ValueError(
                    f"collection {self.collection!r} has dim {actual}, expected {dim}"
                )
            return
        self._c.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=distance),
        )

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        from qdrant_client.models import PointStruct

        points = []
        for r in records:
            payload = dict(r.payload)
            payload[TEXT_KEY] = r.text
            points.append(
                PointStruct(
                    id=r.point_id or str(uuid.uuid4()),
                    vector=r.vector,
                    payload=payload,
                )
            )
        self._c.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search(
        self, vector: list[float], k: int = 5, must: dict[str, Any] | None = None
    ) -> list[VectorHit]:
        query_filter = None
        if must:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=val))
                    for key, val in must.items()
                ]
            )
        res = self._c.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [
            VectorHit(
                point_id=str(p.id),
                score=float(p.score),
                text=str((p.payload or {}).get(TEXT_KEY, "")),
                payload=dict(p.payload or {}),
            )
            for p in res
        ]

    def count(self) -> int:
        return int(self._c.get_collection(self.collection).points_count or 0)


def build_vector_store(cfg: Config) -> VectorStore:
    """Factory. Keeps backend choice in configuration, not in callers."""
    if cfg.vector_backend == "qdrant":
        return QdrantVectorStore(cfg)
    raise NotImplementedError(
        f"VECTOR_BACKEND={cfg.vector_backend!r} has no client yet; "
        "'qdrant' is implemented"
    )
