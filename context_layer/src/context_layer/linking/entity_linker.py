"""Entity linker: text mentions to graph CURIEs. Task 2.2, Requirement 4.2-4.5.

The keystone of Linked Enterprise Context. Without it, retrieval and reasoning
operate on different identifiers and the layer is two silos.

Approach: candidate surface forms are extracted from text, embedded, and matched
by k-NN against the vocabulary term index. Below ``ENTITY_LINK_THRESHOLD`` a
mention is emitted as *unlinked* rather than bound to a low-confidence CURIE
(Requirement 4.4) — inventing a link is worse than admitting ignorance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from context_layer.types import EntityMention, TextChunk

if TYPE_CHECKING:
    from config import Config

    from context_layer.linking.term_index import TermIndex

# Candidate spans: capitalised phrases, and parenthesised generic names as used
# in this corpus, e.g. "Entyvio (vedolizumab)".
_CANDIDATE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,3})\b"
    r"|\(([a-z][a-z0-9\- ]{3,})\)"
)

# Words that start sentences but are never concepts here.
_NOISE = frozenset(
    {
        "the", "this", "that", "these", "those", "policy", "drug", "disease",
        "gene", "protein", "patient", "trial", "adverse", "event", "biomarker",
        "research", "paper", "published", "approval", "status", "treats",
        "targets", "mechanism", "clinical", "based", "context", "answer",
        "note", "scope", "owner", "category", "phase", "sponsor", "enrollment",
        "severity", "frequency", "reported", "trials", "none", "recorded",
        "unknown", "unspecified", "medication", "indicated", "quality",
        "governance", "biomedical", "data", "table", "layer", "hub",
    }
)

MIN_SURFACE_LEN = 3


@dataclass(frozen=True)
class LinkStats:
    considered: int = 0
    linked: int = 0
    unlinked: int = 0
    rejected_missing_curie: int = 0

    @property
    def link_rate(self) -> float:
        return self.linked / self.considered if self.considered else 0.0


def candidate_spans(text: str) -> list[tuple[str, int, int]]:
    """Surface forms worth attempting to link, with character offsets."""
    out: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for m in _CANDIDATE_RE.finditer(text or ""):
        surface = (m.group(1) or m.group(2) or "").strip()
        if len(surface) < MIN_SURFACE_LEN:
            continue
        if surface.casefold() in _NOISE:
            continue
        # A multi-word phrase made entirely of noise words is noise.
        words = [w.casefold() for w in surface.split()]
        if words and all(w in _NOISE for w in words):
            continue
        key = surface.casefold()
        if key in seen:
            continue
        seen.add(key)
        start = m.start(1) if m.group(1) else m.start(2)
        out.append((surface, start, start + len(surface)))
    return out


class EntityLinker:
    """Resolves text spans to graph CURIEs, or declines to."""

    def __init__(
        self,
        cfg: Config,
        term_index: TermIndex | None = None,
        threshold: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.threshold = (
            threshold if threshold is not None else cfg.entity_link_threshold
        )
        if term_index is not None:
            self.index = term_index
        else:
            from context_layer.linking.term_index import TermIndex as _TI

            self.index = _TI(cfg)
        self.stats = LinkStats()

    def close(self) -> None:
        self.index.close()

    def __enter__(self) -> EntityLinker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── linking ───────────────────────────────────────────────────────────────

    def link_span(
        self, chunk_id: str, surface: str, start: int, end: int
    ) -> EntityMention:
        candidates = self.index.nearest(surface, k=3)
        if not candidates:
            return EntityMention(chunk_id, surface, None, 0.0, start, end)

        term, score = candidates[0]
        if score < self.threshold:
            return EntityMention(chunk_id, surface, None, score, start, end)

        # R4.5: the target must exist in the graph.
        if not self.index.curie_exists(term.curie):
            return EntityMention(chunk_id, surface, None, score, start, end)

        return EntityMention(chunk_id, surface, term.curie, score, start, end)

    def link(self, chunk: TextChunk) -> list[EntityMention]:
        """R4.2: zero or more mentions, each with offsets and confidence."""
        mentions: list[EntityMention] = []
        considered = linked = unlinked = rejected = 0
        for surface, start, end in candidate_spans(chunk.text):
            considered += 1
            mention = self.link_span(chunk.chunk_id, surface, start, end)
            if mention.is_linked:
                linked += 1
            else:
                unlinked += 1
                if mention.confidence >= self.threshold:
                    rejected += 1
            mentions.append(mention)
        self.stats = LinkStats(
            considered=self.stats.considered + considered,
            linked=self.stats.linked + linked,
            unlinked=self.stats.unlinked + unlinked,
            rejected_missing_curie=self.stats.rejected_missing_curie + rejected,
        )
        return mentions

    def resolve(self, surface: str) -> str | None:
        """Convenience: one surface form to a CURIE, or None."""
        return self.link_span("adhoc", surface, 0, len(surface)).curie

    def canonical(self, surface: str) -> str | None:
        """Resolve to the canonical CURIE of the concept's equivalence class."""
        curie = self.resolve(surface)
        if curie is None:
            return None
        return self.index.canonical_curie(curie)

    def same_concept(self, a: str, b: str) -> bool:
        """R4.3: do two surface forms denote the same concept?

        Compares canonical CURIEs, so a brand name and its generic unify even
        when they are indexed under different vocabularies.
        """
        ca, cb = self.canonical(a), self.canonical(b)
        return ca is not None and ca == cb

    # ── persistence ───────────────────────────────────────────────────────────

    def persist(self, mentions: Sequence[EntityMention]) -> int:
        """Write linked mentions as graph edges. R4.5 / [EK-5].

        Creates a lightweight Ctx_Chunk stub linked to the concept, so the
        chunk-to-concept association is a real relationship that graph queries
        can traverse, while the text itself stays in the vector store.
        """
        linked = [m for m in mentions if m.is_linked]
        if not linked:
            return 0
        prefix = self.cfg.ctx_label_prefix or "Ctx_"
        rows = [
            {
                "chunk_id": m.chunk_id,
                "curie": m.curie,
                "surface": m.surface_form,
                "confidence": m.confidence,
                "start": m.start,
                "end": m.end,
            }
            for m in linked
        ]
        self.index._run(
            f"""
            UNWIND $rows AS row
            MERGE (ch:`{prefix}Chunk` {{chunk_id: row.chunk_id}})
            WITH ch, row
            MATCH (c:ExternalConcept {{curie: row.curie}})
            MERGE (ch)-[m:MENTIONS]->(c)
            SET m.surface_form = row.surface,
                m.confidence = row.confidence,
                m.start = row.start,
                m.end = row.end
            """,
            rows=rows,
        )
        return len(rows)
