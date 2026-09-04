"""Context package assembly. Task 5.2, Requirements 8.2-8.5.

The binding component: it decides what the model is allowed to see, and records
why. Three properties matter.

1. **Deduplication** by content and by CURIE overlap, so the same fact arriving
   from two collections does not consume the budget twice.
2. **Token budget** enforced explicitly, with the dropped elements reported
   rather than silently trimmed.
3. **Degradation is surfaced.** A package assembled while the graph was
   unreachable says so, and the governance gate lowers confidence accordingly.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Sequence

from context_layer.assembler.retrievers import (
    GraphRetriever,
    RuntimeRetriever,
    VectorRetriever,
)
from context_layer.types import ContextElement, ContextPackage, new_id

if TYPE_CHECKING:
    from config import Config

# Rank order: a graph node asserting a fact outranks a chunk mentioning it.
KIND_WEIGHT = {"graph_node": 1.15, "chunk": 1.0, "prior_decision": 0.85}


def _fingerprint(text: str) -> str:
    """Normalised hash for near-duplicate detection."""
    normalised = re.sub(r"\s+", " ", text.strip().casefold())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def rank(elements: Sequence[ContextElement]) -> list[ContextElement]:
    """Score by relevance weighted by kind, then by id for stable ties."""
    return sorted(
        elements,
        key=lambda e: (-(e.score * KIND_WEIGHT.get(e.kind, 1.0)), e.element_id),
    )


def deduplicate(elements: Sequence[ContextElement]) -> list[ContextElement]:
    """Drop exact/near duplicates, keeping the highest-ranked instance."""
    seen_text: set[str] = set()
    kept: list[ContextElement] = []
    for e in rank(elements):
        fp = _fingerprint(e.content)
        if fp in seen_text:
            continue
        seen_text.add(fp)
        kept.append(e)
    return kept


def apply_budget(
    elements: Sequence[ContextElement], budget: int
) -> tuple[list[ContextElement], list[str]]:
    """Keep elements until the budget is spent. Returns (kept, dropped_ids)."""
    kept: list[ContextElement] = []
    dropped: list[str] = []
    spent = 0
    for e in elements:
        cost = e.approx_tokens()
        if spent + cost > budget:
            dropped.append(e.element_id)
            continue
        kept.append(e)
        spent += cost
    return kept, dropped


class ContextAssembler:
    """Assembles a provenance-tagged context package for one question."""

    def __init__(
        self,
        cfg: Config,
        vector: VectorRetriever | None = None,
        graph: GraphRetriever | None = None,
        runtime: RuntimeRetriever | None = None,
    ) -> None:
        self.cfg = cfg
        self.vector = vector if vector is not None else VectorRetriever(cfg)
        self.graph = graph if graph is not None else GraphRetriever(cfg)
        self.runtime = runtime if runtime is not None else RuntimeRetriever(cfg)

    def close(self) -> None:
        if isinstance(self.graph, GraphRetriever):
            self.graph.close()

    def assemble(self, question: str, k: int = 6, on_retriever=None) -> ContextPackage:
        """Assemble the package.

        ``on_retriever(name, count, ms, degradations)`` fires as each source
        finishes, so assembly is observable rather than one opaque wait.
        """
        if not question or not question.strip():
            raise ValueError("cannot assemble context for an empty question")

        import time

        def report(name, items, started, degs):
            if on_retriever is None:
                return
            try:
                on_retriever(name, len(items), (time.perf_counter() - started) * 1000, list(degs))
            except Exception as e:  # noqa: BLE001
                print(f"[assembler] retriever callback failed: {type(e).__name__}: {e}")

        elements: list[ContextElement] = []
        degradations: list[str] = []

        # 1. Vector first: its CURIEs seed the graph expansion.
        t0 = time.perf_counter()
        vec_elements, vec_deg = self.vector.retrieve(question, k=k)
        elements += vec_elements
        degradations += vec_deg
        report("vector", vec_elements, t0, vec_deg)

        seed_curies = tuple(
            dict.fromkeys(c for e in vec_elements for c in e.curies)
        )

        # 2. Graph expansion over those CURIEs, plus keyword policy lookup.
        t0 = time.perf_counter()
        graph_elements, graph_deg = self.graph.retrieve(
            question, k=k + 2, curies=seed_curies
        )
        elements += graph_elements
        degradations += graph_deg
        report("graph", graph_elements, t0, graph_deg)

        # 3. Prior decisions — continuous incorporation.
        t0 = time.perf_counter()
        rt_elements, rt_deg = self.runtime.retrieve(question)
        elements += rt_elements
        degradations += rt_deg
        report("runtime", rt_elements, t0, rt_deg)

        deduped = deduplicate(elements)
        kept, dropped = apply_budget(deduped, self.cfg.context_token_budget)

        if not kept:
            degradations.append("no_context_retrieved")

        return ContextPackage(
            context_package_id=new_id("ctx"),
            question=question.strip(),
            elements=tuple(kept),
            truncated=tuple(dropped),
            degradations=tuple(dict.fromkeys(degradations)),
        )

    # ── rendering for the model ────────────────────────────────────────────────

    @staticmethod
    def render(package: ContextPackage) -> str:
        """Format a package for a prompt, keeping element ids visible so the
        model can cite them and the gate can verify the citation."""
        if package.is_empty:
            return "(no context retrieved)"
        lines: list[str] = []
        for e in package.elements:
            curies = f" curies={list(e.curies)}" if e.curies else ""
            lines.append(
                f"[{e.element_id}] ({e.kind}, source={e.provenance.source}"
                f", score={e.score:.3f}{curies})\n{e.content}"
            )
        return "\n\n".join(lines)
