"""Three-graph view: Learning · Doing · Knowing.

Mirrors ontology/graph_engineering.png. The structural invariants matter because
the layout is fixed — a viewer learns the shape once, so zones must always be
present and always in the same place, even before a query runs.
"""

from __future__ import annotations

import math

import pytest

from context_layer.rules.models import Fact, FactSet
from context_layer.types import FiredRule, RuleVerdict
from context_layer.ui import three_graph as tg


def facts(*items):
    return FactSet(facts=tuple(items))


def fact(eid, etype, relations=None):
    return Fact(
        entity_id=eid, entity_type=etype,
        properties={"name": eid}, relations=relations or {},
    )


def nodes_of(option, prefix):
    return [n for n in option["series"][0]["data"] if n["id"].startswith(prefix)]


# ── structure is always present ───────────────────────────────────────────────


def test_all_three_zones_exist_even_when_empty():
    opt = tg.build_three_graph_option(None, None)
    assert len(nodes_of(opt, "__llm__")) == 1
    assert len(nodes_of(opt, "__doing__")) == len(tg.DOING_STAGES)
    assert len(nodes_of(opt, "__k__")) == len(tg.CORE_ENTITIES)


def test_layout_is_fixed_not_force():
    """Zones must not drift between runs."""
    assert tg.build_three_graph_option(None, None)["series"][0]["layout"] == "none"


def test_every_node_has_explicit_coordinates():
    for n in tg.build_three_graph_option(None, None)["series"][0]["data"]:
        assert "x" in n and "y" in n, n["id"]


def test_zones_are_concentric_and_ordered():
    opt = tg.build_three_graph_option(None, None)
    radius = lambda n: math.hypot(n["x"], n["y"])  # noqa: E731
    centre = radius(nodes_of(opt, "__llm__")[0])
    inner = [radius(n) for n in nodes_of(opt, "__doing__")]
    outer = [radius(n) for n in nodes_of(opt, "__k__")]
    assert centre == 0
    assert max(inner) < min(outer), "doing ring must sit inside knowing ring"


def test_all_six_core_entities_are_drawn():
    names = {n["id"].removeprefix("__k__") for n in
             nodes_of(tg.build_three_graph_option(None, None), "__k__")}
    assert names == set(tg.CORE_ENTITIES)


def test_all_nine_relationships_are_drawn():
    links = tg.build_three_graph_option(None, None)["series"][0]["links"]
    drawn = {l["value"] for l in links if l["value"]}
    assert drawn == {display for display, _, _, _ in tg.CORE_RELATIONS}
    assert len(tg.CORE_RELATIONS) == 9


def test_doing_ring_forms_a_closed_loop():
    links = tg.build_three_graph_option(None, None)["series"][0]["links"]
    loop = [l for l in links
            if l["source"].startswith("__doing__") and l["target"].startswith("__doing__")]
    assert len(loop) == len(tg.DOING_STAGES)  # closed cycle


def test_doing_stages_match_the_published_loop():
    assert [label for label, _ in tg.DOING_STAGES] == [
        "observe", "assess", "check", "decide", "act"
    ]


# ── populated from a query ────────────────────────────────────────────────────


def test_touched_entities_are_highlighted_untouched_are_dimmed():
    fs = facts(fact("D001", "Drug"))
    opt = tg.build_three_graph_option(fs, RuleVerdict())
    by_id = {n["id"]: n for n in nodes_of(opt, "__k__")}
    assert by_id["__k__Drug"]["itemStyle"]["color"] == tg.KNOWING_OK
    assert by_id["__k__Gene"]["itemStyle"]["color"] == tg.GREY


def test_instance_counts_appear_on_entity_labels():
    fs = facts(fact("D001", "Drug"), fact("D002", "Drug"))
    node = {n["id"]: n for n in nodes_of(
        tg.build_three_graph_option(fs, RuleVerdict()), "__k__")}["__k__Drug"]
    assert "(2)" in node["name"]


def test_traversed_relationships_get_counts_and_are_lit():
    fs = facts(fact("D001", "Drug", {"TREATS": ("DIS1", "DIS2")}))
    counts = tg.relation_counts_from(fs)
    assert counts == {"TREATS": 2}
    link = next(l for l in tg.build_three_graph_option(fs, RuleVerdict(), counts)
                ["series"][0]["links"] if l["value"] == "treats")
    assert link["label"]["show"] is True
    assert "×2" in link["label"]["formatter"]
    assert link["lineStyle"]["opacity"] > 0.5


def test_untraversed_relationships_stay_dim_and_unlabelled():
    link = next(l for l in tg.build_three_graph_option(None, None)["series"][0]["links"]
                if l["value"] == "treats")
    assert link["label"]["show"] is False
    assert link["lineStyle"]["opacity"] < 0.5


def test_blocked_entity_type_turns_red_with_emphasis():
    fs = facts(fact("PAT015", "Patient"))
    verdict = RuleVerdict(fired=(FiredRule("consent", "block", "w", ("PAT015",)),))
    node = {n["id"]: n for n in nodes_of(
        tg.build_three_graph_option(fs, verdict), "__k__")}["__k__Patient"]
    assert node["itemStyle"]["color"] == tg.RED
    assert node["itemStyle"]["shadowBlur"] > 0


def test_warning_entity_type_turns_amber():
    fs = facts(fact("ARD1", "Drug"))
    verdict = RuleVerdict(fired=(FiredRule("r", "warn", "w", ("ARD1",)),))
    node = {n["id"]: n for n in nodes_of(
        tg.build_three_graph_option(fs, verdict), "__k__")}["__k__Drug"]
    assert node["itemStyle"]["color"] == tg.AMBER


def test_check_stage_turns_red_when_blocked():
    opt = tg.build_three_graph_option(
        facts(fact("X", "Drug")), RuleVerdict(),
        stage_state={"symbolic": "blocked"},
    )
    check = next(n for n in nodes_of(opt, "__doing__") if n["name"] == "check")
    assert check["itemStyle"]["color"] == tg.RED


def test_completed_stages_are_lit_pending_are_grey():
    opt = tg.build_three_graph_option(
        None, None, stage_state={"assemble": "done"},
    )
    by_name = {n["name"]: n for n in nodes_of(opt, "__doing__")}
    assert by_name["observe"]["itemStyle"]["color"] == tg.DOING
    assert by_name["act"]["itemStyle"]["color"] == tg.GREY


def test_llm_node_grows_with_grounding_confidence():
    small = tg.build_three_graph_option(None, None, grounding=0.0)
    large = tg.build_three_graph_option(None, None, grounding=1.0)
    assert (nodes_of(large, "__llm__")[0]["symbolSize"]
            > nodes_of(small, "__llm__")[0]["symbolSize"])


def test_blocked_banner_only_shows_when_blocked():
    verdict = RuleVerdict(fired=(FiredRule("r", "block", "w", ("X",)),))
    on = tg.build_three_graph_option(facts(fact("X", "Drug")), verdict, blocked=True)
    off = tg.build_three_graph_option(None, None, blocked=False)
    assert any(g["style"]["text"] == "BLOCKED" for g in on["graphic"])
    assert not any(g["style"]["text"] == "BLOCKED" for g in off["graphic"])


def test_relation_counts_ignore_non_core_relations():
    fs = facts(fact("D1", "Drug", {"TREATS": ("X",), "MAPS_TO": ("RXNORM:1",)}))
    assert tg.relation_counts_from(fs) == {"TREATS": 1}


def test_legend_covers_every_emitted_category():
    fs = facts(fact("PAT015", "Patient"), fact("D1", "Drug"))
    verdict = RuleVerdict(fired=(FiredRule("c", "block", "w", ("PAT015",)),))
    opt = tg.build_three_graph_option(fs, verdict)
    legend = set(opt["legend"][0]["data"])
    emitted = {n["category"] for n in opt["series"][0]["data"]}
    assert emitted <= legend, f"missing from legend: {emitted - legend}"


def test_core_relations_are_all_fetched():
    """A relation absent from TRACKED_RELATIONS can never light up.

    Adding an edge to Neo4j is not enough — the fact builder must load it, or
    the view silently draws it dim forever.
    """
    from context_layer.rules.graph_facts import TRACKED_RELATIONS

    missing = sorted(
        rel for _, rel, _, _ in tg.CORE_RELATIONS if rel not in TRACKED_RELATIONS
    )
    assert not missing, f"drawn but never fetched: {missing}"


def test_core_entities_are_all_resolvable_by_the_fact_builder():
    from context_layer.rules.graph_facts import FACT_KEYS

    missing = sorted(e for e in tg.CORE_ENTITIES if e not in FACT_KEYS)
    assert not missing, f"entities the builder cannot fetch: {missing}"
