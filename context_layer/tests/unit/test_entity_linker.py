"""Entity linking. Requirements 4.1-4.5."""

from __future__ import annotations

import pytest

from context_layer.linking.entity_linker import (
    EntityLinker,
    candidate_spans,
)
from context_layer.linking.term_index import Term
from context_layer.types import Provenance, TextChunk, utcnow


class FakeIndex:
    """Term index stand-in: deterministic scores, controllable existence."""

    def __init__(self, matches=None, existing=None, canonical=None):
        self.matches = matches or {}
        self.existing = existing
        self.canonical_map = canonical or {}
        self.closed = False

    def nearest(self, text, k=3):
        return self.matches.get(text.casefold(), [])

    def curie_exists(self, curie):
        return curie in self.existing if self.existing is not None else True

    def canonical_curie(self, curie):
        return self.canonical_map.get(curie, curie)

    def close(self):
        self.closed = True


def term(curie, surface, etype="Drug"):
    return Term(curie=curie, surface=surface, entity_type=etype, entity_id="X1")


def chunk(text):
    return TextChunk(
        chunk_id="c1", text=text,
        provenance=Provenance(source="t", ingested_at=utcnow()),
    )


# ── candidate extraction ──────────────────────────────────────────────────────


def test_extracts_capitalised_phrases():
    spans = [s for s, _, _ in candidate_spans("Pembrolizumab treats Melanoma.")]
    assert "Pembrolizumab" in spans
    assert "Melanoma" in spans


def test_extracts_parenthesised_generic_names():
    spans = [s for s, _, _ in candidate_spans("Entyvio (vedolizumab) is indicated.")]
    assert "Entyvio" in spans
    assert "vedolizumab" in spans


def test_filters_noise_words():
    spans = [s for s, _, _ in candidate_spans("The Policy governs Data.")]
    assert "Policy" not in spans
    assert "Data" not in spans


def test_offsets_locate_the_surface_form():
    text = "Melanoma responds to Pembrolizumab."
    for surface, start, end in candidate_spans(text):
        assert text[start:end] == surface


def test_deduplicates_repeated_surfaces():
    spans = [s for s, _, _ in candidate_spans("Melanoma and Melanoma again.")]
    assert spans.count("Melanoma") == 1


def test_ignores_very_short_tokens():
    assert not [s for s, _, _ in candidate_spans("A B C")]


# ── threshold behaviour, R4.4 ─────────────────────────────────────────────────


def test_links_above_threshold(cfg):
    idx = FakeIndex({"pembrolizumab": [(term("RXNORM:1", "Pembrolizumab"), 0.95)]})
    linker = EntityLinker(cfg, term_index=idx, threshold=0.72)
    m = linker.link_span("c1", "Pembrolizumab", 0, 13)
    assert m.is_linked and m.curie == "RXNORM:1"


def test_declines_below_threshold_rather_than_guessing(cfg):
    idx = FakeIndex({"pd-1": [(term("RXNORM:9", "Something"), 0.51)]})
    linker = EntityLinker(cfg, term_index=idx, threshold=0.72)
    m = linker.link_span("c1", "PD-1", 0, 4)
    assert not m.is_linked
    assert m.curie is None
    assert m.confidence == 0.51  # score retained for debugging


def test_exact_threshold_links(cfg):
    idx = FakeIndex({"x": [(term("C:1", "X"), 0.72)]})
    linker = EntityLinker(cfg, term_index=idx, threshold=0.72)
    assert linker.link_span("c1", "X", 0, 1).is_linked


def test_no_candidates_yields_unlinked(cfg):
    linker = EntityLinker(cfg, term_index=FakeIndex(), threshold=0.5)
    m = linker.link_span("c1", "Unknown", 0, 7)
    assert not m.is_linked and m.confidence == 0.0


# ── graph verification, R4.5 ──────────────────────────────────────────────────


def test_rejects_curie_absent_from_the_graph(cfg):
    """A high score must not create a link to a nonexistent concept."""
    idx = FakeIndex(
        {"ghost": [(term("RXNORM:ghost", "Ghost"), 0.99)]},
        existing={"RXNORM:real"},
    )
    linker = EntityLinker(cfg, term_index=idx, threshold=0.72)
    m = linker.link_span("c1", "Ghost", 0, 5)
    assert not m.is_linked, "linked to a CURIE that does not exist in the graph"


def test_accepts_curie_present_in_the_graph(cfg):
    idx = FakeIndex(
        {"real": [(term("RXNORM:real", "Real"), 0.99)]}, existing={"RXNORM:real"}
    )
    linker = EntityLinker(cfg, term_index=idx, threshold=0.72)
    assert linker.link_span("c1", "Real", 0, 4).is_linked


# ── synonym equivalence, R4.3 ─────────────────────────────────────────────────


def test_synonyms_unify_via_canonical_curie(cfg):
    """Brand and generic indexed under different vocabularies must unify."""
    idx = FakeIndex(
        matches={
            "entyvio": [(term("RXNORM:RXN_1745276", "Entyvio (vedolizumab)"), 0.86)],
            "vedolizumab": [(term("OMOP:40161532", "vedolizumab"), 1.0)],
        },
        canonical={
            "RXNORM:RXN_1745276": "OMOP:40161532",
            "OMOP:40161532": "OMOP:40161532",
        },
    )
    linker = EntityLinker(cfg, term_index=idx, threshold=0.72)
    assert linker.same_concept("Entyvio", "vedolizumab")


def test_distinct_concepts_do_not_unify(cfg):
    idx = FakeIndex(
        matches={
            "pembrolizumab": [(term("RXNORM:1657993", "Pembrolizumab"), 1.0)],
            "nivolumab": [(term("RXNORM:1597876", "Nivolumab"), 1.0)],
        }
    )
    linker = EntityLinker(cfg, term_index=idx, threshold=0.72)
    assert not linker.same_concept("Pembrolizumab", "Nivolumab")


def test_unresolvable_surfaces_are_not_the_same_concept(cfg):
    linker = EntityLinker(cfg, term_index=FakeIndex(), threshold=0.5)
    assert not linker.same_concept("Nonsense", "Gibberish")


# ── chunk-level linking and stats ─────────────────────────────────────────────


def test_link_reports_stats(cfg):
    idx = FakeIndex(
        {
            "pembrolizumab": [(term("RXNORM:1", "Pembrolizumab"), 0.99)],
            "melanoma": [(term("SNOMED:2", "Melanoma"), 0.98)],
            "quarterly revenue": [(term("X:9", "Other"), 0.3)],
        }
    )
    linker = EntityLinker(cfg, term_index=idx, threshold=0.72)
    mentions = linker.link(chunk("Pembrolizumab treats Melanoma. Quarterly Revenue rose."))
    assert linker.stats.considered >= 3
    assert linker.stats.linked == 2
    assert linker.stats.unlinked >= 1
    assert 0.0 < linker.stats.link_rate <= 1.0
    assert all(m.chunk_id == "c1" for m in mentions)


def test_stats_accumulate_across_chunks(cfg):
    idx = FakeIndex({"melanoma": [(term("SNOMED:2", "Melanoma"), 0.98)]})
    linker = EntityLinker(cfg, term_index=idx, threshold=0.72)
    linker.link(chunk("Melanoma."))
    first = linker.stats.linked
    linker.link(chunk("Melanoma."))
    assert linker.stats.linked == first + 1


def test_empty_text_yields_no_mentions(cfg):
    linker = EntityLinker(cfg, term_index=FakeIndex(), threshold=0.5)
    assert linker.link(chunk("")) == []


def test_persist_skips_when_nothing_linked(cfg):
    linker = EntityLinker(cfg, term_index=FakeIndex(), threshold=0.5)
    assert linker.persist(linker.link(chunk("nothing here"))) == 0


def test_close_propagates_to_the_index(cfg):
    idx = FakeIndex()
    with EntityLinker(cfg, term_index=idx):
        pass
    assert idx.closed
