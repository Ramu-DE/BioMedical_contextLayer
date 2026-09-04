"""Context assembly. Requirements 8.1-8.5."""

from __future__ import annotations

import pytest

from context_layer.assembler.package import (
    ContextAssembler,
    apply_budget,
    deduplicate,
    rank,
)
from context_layer.types import ContextElement, Provenance, utcnow

PROV = Provenance(source="test", ingested_at=utcnow(), confidence=1.0)


def el(eid, content="some content here", kind="chunk", score=0.5, curies=()):
    return ContextElement(
        element_id=eid, kind=kind, content=content, provenance=PROV,
        score=score, curies=curies,
    )


class FakeRetriever:
    def __init__(self, name, elements=(), degradations=()):
        self.name = name
        self._e = list(elements)
        self._d = list(degradations)

    def retrieve(self, question, k=6, **kw):
        return list(self._e), list(self._d)


def assembler(cfg, vector=None, graph=None, runtime=None):
    return ContextAssembler(
        cfg,
        vector=vector or FakeRetriever("vector"),
        graph=graph or FakeRetriever("graph"),
        runtime=runtime or FakeRetriever("runtime"),
    )


# ── ranking and dedup, R8.2 ───────────────────────────────────────────────────


def test_rank_prefers_higher_score():
    out = rank([el("a", score=0.2), el("b", score=0.9)])
    assert [e.element_id for e in out] == ["b", "a"]


def test_rank_weights_graph_nodes_above_chunks_at_equal_score():
    out = rank([el("chunk", score=0.5), el("node", kind="graph_node", score=0.5)])
    assert out[0].element_id == "node"


def test_rank_is_stable_for_identical_scores():
    a = rank([el("z", score=0.5), el("a", score=0.5)])
    b = rank([el("a", score=0.5), el("z", score=0.5)])
    assert [e.element_id for e in a] == [e.element_id for e in b]


def test_deduplicate_drops_identical_content():
    out = deduplicate([el("a", "same text"), el("b", "same text")])
    assert len(out) == 1


def test_deduplicate_ignores_whitespace_and_case():
    out = deduplicate([el("a", "Same  Text"), el("b", "same text")])
    assert len(out) == 1


def test_deduplicate_keeps_highest_ranked_duplicate():
    out = deduplicate([el("low", "dup", score=0.1), el("high", "dup", score=0.9)])
    assert out[0].element_id == "high"


def test_deduplicate_keeps_distinct_content():
    assert len(deduplicate([el("a", "first"), el("b", "second")])) == 2


# ── token budget, R8.3 ────────────────────────────────────────────────────────


def test_budget_keeps_within_limit_and_reports_drops():
    elements = [el(f"e{i}", "x" * 400, score=1 - i / 10) for i in range(10)]
    kept, dropped = apply_budget(elements, budget=300)
    assert kept and dropped
    assert sum(e.approx_tokens() for e in kept) <= 300
    assert len(kept) + len(dropped) == 10


def test_budget_reports_dropped_element_ids():
    elements = [el("keep", "x" * 40), el("drop", "y" * 4000)]
    kept, dropped = apply_budget(elements, budget=20)
    assert [e.element_id for e in kept] == ["keep"]
    assert dropped == ["drop"]


def test_budget_of_zero_drops_everything():
    kept, dropped = apply_budget([el("a")], budget=0)
    assert not kept and dropped == ["a"]


# ── assembly, R8.4, R8.5 ──────────────────────────────────────────────────────


def test_package_carries_id_and_provenance(cfg):
    a = assembler(cfg, vector=FakeRetriever("vector", [el("v1", curies=("X:1",))]))
    pkg = a.assemble("a question")
    assert pkg.context_package_id.startswith("ctx_")
    assert pkg.elements[0].provenance.source == "test"
    assert pkg.curies == ("X:1",)


def test_degradations_are_collected_from_every_retriever(cfg):
    a = assembler(
        cfg,
        vector=FakeRetriever("vector", [el("v")], ["vector_partial"]),
        graph=FakeRetriever("graph", [], ["graph_retriever_unavailable: boom"]),
        runtime=FakeRetriever("runtime", [], ["runtime_retriever_unavailable: none"]),
    )
    pkg = a.assemble("q")
    assert "vector_partial" in pkg.degradations
    assert any(d.startswith("graph_retriever_unavailable") for d in pkg.degradations)


def test_empty_retrieval_is_flagged_not_silent(cfg):
    pkg = assembler(cfg).assemble("q")
    assert pkg.is_empty
    assert "no_context_retrieved" in pkg.degradations


def test_degradations_are_deduplicated(cfg):
    a = assembler(
        cfg,
        vector=FakeRetriever("vector", [el("v")], ["same"]),
        graph=FakeRetriever("graph", [], ["same"]),
    )
    assert list(a.assemble("q").degradations).count("same") == 1


def test_empty_question_is_rejected(cfg):
    with pytest.raises(ValueError, match="empty question"):
        assembler(cfg).assemble("   ")


def test_render_exposes_element_ids_for_citation(cfg):
    a = assembler(cfg, vector=FakeRetriever("vector", [el("cite-me", "content")]))
    rendered = ContextAssembler.render(a.assemble("q"))
    assert "[cite-me]" in rendered
    assert "content" in rendered


def test_render_handles_empty_package():
    from context_layer.types import ContextPackage

    text = ContextAssembler.render(ContextPackage(context_package_id="c", question="q"))
    assert "no context" in text
