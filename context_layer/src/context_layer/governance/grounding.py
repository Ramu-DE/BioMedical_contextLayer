"""Grounding measurement. Task 6.1, Requirements 10.2, 10.4.

Grounding is scored from evidence, not from the model's own confidence. Three
signals combine: whether the answer cites retrievable elements, how much of the
answer's substance overlaps the supplied context, and whether retrieval
degraded while assembling the package.
"""

from __future__ import annotations

import re

from context_layer.types import Citation, ContextPackage

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:\-]+)\]")

# Degradations that materially undermine confidence, with their penalties.
DEGRADATION_PENALTY = {
    "no_context_retrieved": 1.0,
    "vector_retriever_unavailable": 0.5,
    "graph_retriever_unavailable": 0.25,
    "vector_collection_unavailable": 0.15,
    "runtime_retriever_unavailable": 0.0,  # expected until the store exists
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "and",
    "or", "but", "if", "then", "than", "that", "this", "these", "those", "of",
    "in", "on", "at", "to", "for", "with", "from", "by", "as", "it", "its",
    "which", "who", "whom", "what", "when", "where", "why", "how", "can",
    "could", "should", "would", "may", "might", "must", "not", "no", "there",
    "their", "they", "we", "our", "you", "your", "has", "have", "had", "do",
    "does", "did", "any", "all", "some", "more", "most", "other", "such",
}


def _content_words(text: str) -> set[str]:
    return {
        w
        for w in (t.strip(".,;:!?()[]'\"").casefold() for t in text.split())
        if len(w) > 3 and w not in STOPWORDS
    }


def extract_citations(answer: str, package: ContextPackage) -> tuple[Citation, ...]:
    """Pull [element_id] markers out of the answer and resolve them.

    Only markers matching a real element are accepted; a fabricated id yields no
    citation, which then depresses the grounding score.
    """
    by_id = {e.element_id: e for e in package.elements}
    citations: list[Citation] = []
    seen: set[str] = set()
    for raw in CITATION_RE.findall(answer or ""):
        element = by_id.get(raw)
        if element is None or raw in seen:
            continue
        seen.add(raw)
        chunk_id = element.element_id if element.kind == "chunk" else None
        curie = element.curies[0] if element.curies else None
        if chunk_id is None and curie is None:
            # A graph node with no CURIE still needs a resolvable target.
            chunk_id = element.element_id
        citations.append(
            Citation(
                chunk_id=chunk_id,
                curie=curie,
                source=element.provenance.source,
                quote=element.content[:180],
            )
        )
    return tuple(citations)


def fabricated_citations(answer: str, package: ContextPackage) -> tuple[str, ...]:
    """Citation markers that do not correspond to any supplied element."""
    valid = {e.element_id for e in package.elements}
    return tuple(
        sorted({raw for raw in CITATION_RE.findall(answer or "") if raw not in valid})
    )


def overlap_ratio(answer: str, package: ContextPackage) -> float:
    """Share of the answer's content words that appear in the context."""
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    context_words = _content_words(" ".join(e.content for e in package.elements))
    if not context_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)


def degradation_penalty(package: ContextPackage) -> float:
    penalty = 0.0
    for degradation in package.degradations:
        head = degradation.split(":")[0]
        penalty += DEGRADATION_PENALTY.get(head, 0.1)
    return min(penalty, 1.0)


def score(answer: str, package: ContextPackage) -> float:
    """Grounding confidence in [0, 1].

    Weighted: citation presence 0.45, content overlap 0.55, then the
    degradation penalty applied multiplicatively, then a hard penalty for
    fabricated citation markers.
    """
    if package.is_empty:
        return 0.0
    if not answer or not answer.strip():
        return 0.0

    citations = extract_citations(answer, package)
    cited_ratio = min(len(citations) / 2.0, 1.0)  # two good citations saturates
    overlap = overlap_ratio(answer, package)

    base = 0.45 * cited_ratio + 0.55 * overlap
    base *= 1.0 - degradation_penalty(package)

    fabricated = fabricated_citations(answer, package)
    if fabricated:
        base *= 0.4  # citing something that does not exist is a serious defect

    return round(max(0.0, min(1.0, base)), 3)


def missing_context_note(package: ContextPackage) -> str:
    """Explain what was absent, for a refusal message (Requirement 10.2)."""
    parts: list[str] = []
    if package.is_empty:
        parts.append("no context was retrieved for this question")
    else:
        kinds = {e.kind for e in package.elements}
        if "graph_node" not in kinds:
            parts.append("no graph evidence was found")
        if "chunk" not in kinds:
            parts.append("no document evidence was found")
    for degradation in package.degradations:
        if degradation.split(":")[0] in DEGRADATION_PENALTY and not degradation.startswith(
            "runtime_retriever"
        ):
            parts.append(degradation)
    if package.truncated:
        parts.append(f"{len(package.truncated)} element(s) dropped by token budget")
    return "; ".join(parts) or "insufficient supporting evidence"
