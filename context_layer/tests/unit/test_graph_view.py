"""Traversal graph builder for the UI.

The panel exists to make "rules read the graph, not the prompt" visible, so the
colour semantics are load-bearing: a blocked entity must be unmistakable.
"""

from __future__ import annotations

from context_layer.rules.graph_facts import TRACKED_RELATIONS
from context_layer.rules.models import Fact, FactSet
from context_layer.types import FiredRule, RuleVerdict
from context_layer.ui import graph_view as gv


def facts(*items):
    return FactSet(facts=tuple(items))


def fact(eid, etype="Table", props=None, relations=None):
    return Fact(
        entity_id=eid, entity_type=etype,
        properties=props or {"name": eid}, relations=relations or {},
    )


def test_empty_option_has_a_message():
    opt = gv.empty_option("nothing yet")
    assert opt["title"]["text"] == "nothing yet"
    assert opt["series"] == []


def test_blocked_entities_are_red_and_largest():
    fs = facts(fact("hub.customer_master"), fact("hub.other"))
    verdict = RuleVerdict(
        fired=(FiredRule("g.dq", "block", "why", ("hub.customer_master",)),)
    )
    nodes = {n["id"]: n for n in gv.build_option(fs, verdict)["series"][0]["data"]}
    blocked, clean = nodes["hub.customer_master"], nodes["hub.other"]
    assert blocked["itemStyle"]["color"] == gv.RED
    assert blocked["category"] == "blocked by policy"
    assert blocked["symbolSize"] > clean["symbolSize"]


def test_warning_entities_are_amber():
    fs = facts(fact("ARD_X", "ARD"))
    verdict = RuleVerdict(fired=(FiredRule("g.dq", "warn", "why", ("ARD_X",)),))
    node = gv.build_option(fs, verdict)["series"][0]["data"][0]
    assert node["itemStyle"]["color"] == gv.AMBER


def test_in_scope_entities_are_violet():
    node = gv.build_option(facts(fact("T1")), RuleVerdict())["series"][0]["data"][0]
    assert node["itemStyle"]["color"] == gv.VIOLET


def test_edges_are_built_from_relations():
    fs = facts(fact("ARD_CM", "ARD", relations={"CERTIFIED_VIEW_OF": ("hub.cm",)}))
    s = gv.build_option(fs, RuleVerdict(), names={"hub.cm": ("Table", "customer_master")})["series"][0]
    assert len(s["links"]) == 1
    assert s["links"][0]["value"] == "CERTIFIED_VIEW_OF"
    assert any(n["name"] == "customer_master" for n in s["data"])


def test_edge_into_a_blocked_node_is_highlighted():
    fs = facts(fact("ARD_CM", "ARD", relations={"CERTIFIED_VIEW_OF": ("hub.cm",)}))
    verdict = RuleVerdict(fired=(FiredRule("g.dq", "block", "w", ("hub.cm",)),))
    link = gv.build_option(fs, verdict, names={"hub.cm": ("Table", "customer_master")})["series"][0]["links"][0]
    assert link["lineStyle"]["color"] == gv.RED
    # Blocked paths are drawn thicker than ordinary edges, not a fixed width.
    clean = gv.build_option(
        facts(fact("X", relations={"FEEDS_INTO": ("Y",)})), RuleVerdict()
    )["series"][0]["links"][0]
    assert link["lineStyle"]["width"] > clean["lineStyle"]["width"]
    assert link["lineStyle"]["opacity"] >= clean["lineStyle"]["opacity"]


def test_hidden_relations_are_not_drawn():
    fs = facts(fact("D1", "Drug", relations={"MAPS_TO": ("RXNORM:1",)}))
    assert gv.build_option(fs, RuleVerdict())["series"][0]["links"] == []


def test_self_referential_edges_are_dropped():
    fs = facts(fact("PAT015", "Patient", relations={"HAS_OUTCOME": ("PAT015",)}))
    assert gv.build_option(fs, RuleVerdict())["series"][0]["links"] == []


def test_unresolved_ids_fall_back_to_the_id():
    fs = facts(fact("A", relations={"FEEDS_INTO": ("mystery-id",)}))
    names = {n["name"] for n in gv.build_option(fs, RuleVerdict())["series"][0]["data"]}
    assert "mystery-id" in names


def test_node_cap_is_respected():
    fs = facts(*[fact(f"T{i}") for i in range(200)])
    assert len(gv.build_option(fs, RuleVerdict(), max_nodes=25)["series"][0]["data"]) == 25


def test_blocked_nodes_survive_the_cap():
    """Truncation must never hide the reason an answer was blocked."""
    fs = facts(*[fact(f"T{i}") for i in range(200)], fact("CULPRIT"))
    verdict = RuleVerdict(fired=(FiredRule("r", "block", "w", ("CULPRIT",)),))
    ids = {n["id"] for n in gv.build_option(fs, verdict, max_nodes=10)["series"][0]["data"]}
    assert "CULPRIT" in ids


def test_summarise_reports_entities_edges_and_triggers():
    fs = facts(fact("A", relations={"FEEDS_INTO": ("B", "C")}))
    verdict = RuleVerdict(fired=(FiredRule("r", "block", "w", ("A",)),))
    text = gv.summarise(fs, verdict)
    assert "1 entities" in text and "2 edges" in text and "1 triggered" in text


def test_every_relation_the_view_draws_is_actually_fetched():
    """A relation the fact builder never loads could never appear in the view."""
    assert gv.HIDDEN_RELATIONS <= set(TRACKED_RELATIONS)


def test_legend_covers_every_category_the_builder_emits():
    fs = facts(fact("blk"), fact("wrn"), fact("scope"),
               fact("hub.x", relations={"FEEDS_INTO": ("neigh",)}))
    verdict = RuleVerdict(fired=(
        FiredRule("a", "block", "w", ("blk",)),
        FiredRule("b", "warn", "w", ("wrn",)),
    ))
    opt = gv.build_option(fs, verdict)
    legend = set(opt["legend"][0]["data"])
    emitted = {n["category"] for n in opt["series"][0]["data"]}
    assert emitted <= legend, f"categories missing from legend: {emitted - legend}"


# ── legibility ────────────────────────────────────────────────────────────────
#
# Contrast is a correctness property here, not decoration. Two real regressions
# are guarded: an early dark palette used EDGE="#2a2a4a" (1.34:1) so the
# relationships were invisible, and the whole dark palette then failed again when
# the theme moved to white (amber measured 1.54:1). The reference background is
# read from the module so the test cannot drift from the theme.

CARD_BG = gv.CARD_BG


def _relative_luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(fg: str, bg: str = CARD_BG) -> float:
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def test_reference_background_is_the_theme_background():
    assert gv.CARD_BG == "#ffffff", "palette is verified against white"


def test_graphic_colours_meet_3to1():
    for name in ("RED", "AMBER", "VIOLET", "CYAN", "GREY", "EDGE"):
        colour = getattr(gv, name)
        assert contrast(colour) >= 3.0, f"{name}={colour} is {contrast(colour):.2f}:1"


def test_three_graph_colours_are_also_legible():
    from context_layer.ui import three_graph as tg

    for name in ("LEARNING", "DOING", "KNOWING_OK"):
        colour = getattr(tg, name)
        assert contrast(colour) >= 3.0, f"{name}={colour} is {contrast(colour):.2f}:1"


def test_label_halo_matches_the_background():
    """The outline behind label text must be the background colour, or the halo
    becomes a dark smear on a light theme."""
    assert gv.LABEL_HALO == gv.CARD_BG


def test_text_colours_meet_4point5to1():
    for name in ("EDGE_LABEL", "NODE_LABEL"):
        colour = getattr(gv, name)
        assert contrast(colour) >= 4.5, f"{name}={colour} is {contrast(colour):.2f}:1"


def test_edges_are_not_washed_out_by_opacity():
    """A legible colour at 0.45 opacity is an illegible line."""
    fs = facts(fact("A", relations={"FEEDS_INTO": ("B",)}))
    link = gv.build_option(fs, RuleVerdict())["series"][0]["links"][0]
    assert link["lineStyle"]["opacity"] >= 0.7
    assert link["lineStyle"]["width"] >= 1.5


def test_relationship_names_are_visible_without_hovering():
    fs = facts(fact("A", relations={"CERTIFIED_VIEW_OF": ("B",)}))
    link = gv.build_option(fs, RuleVerdict())["series"][0]["links"][0]
    assert link["label"]["show"] is True
    assert link["label"]["formatter"] == "CERTIFIED_VIEW_OF"


def test_every_node_is_labelled():
    fs = facts(fact("small"), fact("hub", relations={"FEEDS_INTO": ("x", "y", "z")}))
    for node in gv.build_option(fs, RuleVerdict())["series"][0]["data"]:
        assert node["label"]["show"] is True, f"{node['id']} unlabelled"


def test_blocked_nodes_get_extra_emphasis():
    fs = facts(fact("culprit"), fact("clean"))
    verdict = RuleVerdict(fired=(FiredRule("r", "block", "w", ("culprit",)),))
    nodes = {n["id"]: n for n in gv.build_option(fs, verdict)["series"][0]["data"]}
    assert nodes["culprit"]["itemStyle"]["borderWidth"] > nodes["clean"]["itemStyle"]["borderWidth"]
    assert nodes["culprit"]["itemStyle"]["shadowBlur"] > 0
    assert nodes["culprit"]["label"]["fontWeight"] == "bold"
