"""Name-based entity resolution.

People ask about "HER2" and "Chronic Myeloid Leukemia", not "G007" and "DIS004".
Before this existed, only ID-shaped mentions entered scope, so natural phrasing
could never bind a rule or light the traversal graph — a question would answer
from text while the graph stayed dark.

The risk is the opposite failure: over-matching. "APP" is a gene and also a
common word; "status" appears in a biomarker name and in ordinary prose.
"""

from __future__ import annotations

import pytest

from context_layer.rules.graph_facts import GraphFactBuilder


class FakeSession:
    def __init__(self, rows):
        self.rows = rows

    def run(self, cypher, **params):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self, rows):
        self.rows = rows

    def session(self, database=None):
        return FakeSession(self.rows)

    def close(self):
        pass


def row(label, key, name):
    return {"l": label, "k": key, "nm": name}


# Mirrors the real graph, including the awkward cases.
INDEX = [
    row("Drug", "D001", "Pembrolizumab"),
    row("Drug", "D004", "Trastuzumab"),
    row("Disease", "DIS002", "Melanoma"),
    row("Disease", "DIS004", "Chronic Myeloid Leukemia"),
    row("Disease", "DIS007", "Obesity"),
    row("Gene", "G007", "HER2"),
    row("Gene", "G009", "APP"),
    row("Gene", "G001", "EGFR"),
    row("Biomarker", "B007", "HER2 Status"),
    row("Protein", "P009", "Tau"),
    row("Protein", "P001", "PD-1"),
    row("AdverseEvent", "AE001", "Immune-related pneumonitis"),
]


@pytest.fixture
def builder(cfg):
    return GraphFactBuilder(cfg, driver=FakeDriver(INDEX))


# ── the index ─────────────────────────────────────────────────────────────────


def test_index_is_sorted_longest_first(builder):
    lengths = [len(name) for _, _, name in builder.name_index()]
    assert lengths == sorted(lengths, reverse=True)


def test_index_is_cached(builder):
    assert builder.name_index() is builder.name_index()


def test_stoplist_words_are_excluded(builder):
    names = {n.casefold() for _, _, n in builder.name_index()}
    assert not names & GraphFactBuilder._NAME_STOPLIST


# ── positive matching ─────────────────────────────────────────────────────────


def test_matches_a_single_word_name(builder):
    assert builder.resolve_by_name("Tell me about Pembrolizumab") == {"Drug": ["D001"]}


def test_matches_a_multi_word_name(builder):
    got = builder.resolve_by_name("Which drug treats Chronic Myeloid Leukemia?")
    assert got == {"Disease": ["DIS004"]}


def test_matches_a_hyphenated_name(builder):
    assert builder.resolve_by_name("Which drugs target PD-1?") == {"Protein": ["P001"]}


def test_matching_is_case_insensitive_for_longer_names(builder):
    assert builder.resolve_by_name("about pembrolizumab") == {"Drug": ["D001"]}


def test_multiple_entities_in_one_question(builder):
    got = builder.resolve_by_name("Does Pembrolizumab treat Melanoma?")
    assert got["Drug"] == ["D001"]
    assert got["Disease"] == ["DIS002"]


def test_overlapping_names_of_different_types_both_match(builder):
    """'HER2 Status' is a Biomarker and 'HER2' a Gene; a question about one
    usually concerns the other."""
    got = builder.resolve_by_name("which gene is the biomarker for HER2 status?")
    assert got["Biomarker"] == ["B007"]
    assert got["Gene"] == ["G007"]


# ── false positives, the real risk ────────────────────────────────────────────


def test_short_names_require_exact_case(builder):
    """'APP' is a gene; 'app' is an ordinary word."""
    assert builder.resolve_by_name("Is the app working?") == {}
    assert builder.resolve_by_name("The APP gene") == {"Gene": ["G009"]}


def test_lowercase_tau_does_not_match_the_protein(builder):
    assert builder.resolve_by_name("I need the tau of this") == {}


def test_substring_of_a_longer_word_does_not_match(builder):
    """'Melanoma' must not fire inside 'Melanomas' or 'premelanoma'."""
    assert builder.resolve_by_name("premelanomatous lesions") == {}


def test_hyphenated_boundary_is_respected(builder):
    assert builder.resolve_by_name("anti-PD-1-directed therapy") == {}


def test_empty_and_irrelevant_text_match_nothing(builder):
    assert builder.resolve_by_name("") == {}
    assert builder.resolve_by_name("Nothing relevant here at all") == {}


def test_limit_caps_the_number_of_entities(builder):
    text = "Pembrolizumab Trastuzumab Melanoma Obesity HER2 EGFR"
    total = sum(len(v) for v in builder.resolve_by_name(text, limit=2).values())
    assert total <= 2


def test_unfetchable_labels_are_skipped(cfg):
    """A label the fact builder cannot query must never be returned."""
    from context_layer.rules.graph_facts import FACT_KEYS

    b = GraphFactBuilder(cfg, driver=FakeDriver([row("Nonsense", "X1", "Widgetol")]))
    assert b.resolve_by_name("Widgetol is great") == {}
    assert "Nonsense" not in FACT_KEYS
