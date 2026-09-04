"""Three-graph view: Learning · Doing · Knowing.

Reproduces the structure of ontology/graph_engineering.png and populates it from
a real query, so the diagram stops being a picture and becomes an instrument.

    centre       Graph of Learning  -- the LLM, sized by grounding confidence
    inner ring   Graph of Doing     -- observe, assess, check, decide, act
    outer ring   Graph of Knowing   -- the six core entities and their nine
                                      typed relationships, with live counts

Layout is fixed (``layout: "none"`` with explicit coordinates) rather than
force-directed: the whole point is that the three concentric zones are always in
the same place, so the eye learns the shape once and then reads the data.

Spokes connect each Doing stage to the zone it actually touched, which is what
makes "which layer is the agent using right now" visible.
"""

from __future__ import annotations

import math
from typing import Any

from context_layer.ui.graph_view import (
    AMBER,
    CARD_BG,
    LABEL_HALO,
    CYAN,
    EDGE,
    EDGE_LABEL,
    GREY,
    NODE_LABEL,
    RED,
    VIOLET,
)

# Light theme; measured against white like the rest of the palette.
LEARNING = "#0369a1"    # 6.10:1  centre: the model
DOING = "#1d4ed8"       # 6.99:1  inner ring: the agent loop
KNOWING_OK = "#6d28d9"  # 7.10:1  outer ring: governed domain

# The six core entities, in the order the published diagram arranges them.
CORE_ENTITIES = ("Drug", "Disease", "ClinicalTrial", "Patient", "Gene", "Biomarker")
ENTITY_DISPLAY = {"ClinicalTrial": "Clinical\nTrial"}

# The nine typed relationships: (display name, neo4j type, source, target).
CORE_RELATIONS = (
    ("treats", "TREATS", "Drug", "Disease"),
    ("predicts_response_to", "PREDICTS_RESPONSE_TO", "Biomarker", "Drug"),
    ("associated_with", "ASSOCIATED_WITH", "Gene", "Disease"),
    ("participates_in", "ENROLLED_IN", "Patient", "ClinicalTrial"),
    ("investigates", "INVESTIGATES", "ClinicalTrial", "Drug"),
    ("studies", "STUDIES", "ClinicalTrial", "Disease"),
    ("is_biomarker_for", "IS_BIOMARKER_FOR", "Gene", "Biomarker"),
    ("indicates", "INDICATES", "Biomarker", "Disease"),
    ("evaluated_in", "EVALUATED_IN", "Drug", "ClinicalTrial"),
)

# The agent loop, in execution order, mapped to our pipeline stages.
DOING_STAGES = (
    ("observe", "assemble"),
    ("assess", "neural"),
    ("check", "symbolic"),
    ("decide", "gate"),
    ("act", "record"),
)

OUTER_R = 74.0
INNER_R = 30.0


def _ring(count: int, radius: float, rotate: float = -90.0) -> list[tuple[float, float]]:
    out = []
    for i in range(count):
        a = math.radians(rotate + (360.0 / count) * i)
        out.append((radius * math.cos(a), radius * math.sin(a)))
    return out


def build_three_graph_option(
    facts,
    verdict,
    relation_counts: dict[str, int] | None = None,
    stage_state: dict[str, str] | None = None,
    grounding: float = 0.0,
    blocked: bool = False,
) -> dict[str, Any]:
    """Assemble the three-ring option.

    ``relation_counts`` maps the neo4j relationship type to how many instances
    the query actually traversed. ``stage_state`` maps a pipeline stage name to
    "done" | "blocked" | "pending".
    """
    relation_counts = relation_counts or {}
    stage_state = stage_state or {}

    touched_types = {f.entity_type for f in facts.facts} if facts else set()
    blocking = (
        {t for f in verdict.fired if f.severity == "block" for t in f.triggered_by}
        if verdict
        else set()
    )
    warned = (
        {t for f in verdict.fired if f.severity == "warn" for t in f.triggered_by}
        if verdict
        else set()
    )
    # Per-entity-type counts, and whether any instance of the type tripped a rule.
    per_type: dict[str, int] = {}
    flagged: set[str] = set()
    warned_types: set[str] = set()
    if facts:
        for f in facts.facts:
            per_type[f.entity_type] = per_type.get(f.entity_type, 0) + 1
            if f.entity_id in blocking:
                flagged.add(f.entity_type)
            elif f.entity_id in warned:
                warned_types.add(f.entity_type)

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    # ── centre: Graph of Learning ────────────────────────────────────────────
    conf = max(0.0, min(1.0, grounding))
    nodes.append(
        {
            "id": "__llm__",
            "name": "LLM",
            "x": 0.0,
            "y": 0.0,
            "symbolSize": 30 + conf * 22,
            "category": "graph of learning",
            "itemStyle": {
                "color": LEARNING,
                "borderColor": NODE_LABEL,
                "borderWidth": 2,
                "shadowBlur": 26,
                "shadowColor": LEARNING,
            },
            "label": {
                "show": True,
                "color": "#ffffff",
                "fontSize": 10,
                "fontWeight": "bold",
                "position": "inside",
            },
            "tooltip": {
                "formatter": (
                    "<b>Graph of Learning</b><br/>parameterised model<br/>"
                    f"grounding confidence {grounding:.3f}"
                )
            },
        }
    )

    # ── inner ring: Graph of Doing ───────────────────────────────────────────
    doing_pos = _ring(len(DOING_STAGES), INNER_R)
    for (label, stage), (x, y) in zip(DOING_STAGES, doing_pos):
        state = stage_state.get(stage, "pending")
        colour = {"done": DOING, "blocked": RED, "pending": GREY}[state]
        nodes.append(
            {
                "id": f"__doing__{stage}",
                "name": label,
                "x": x,
                "y": y,
                "symbolSize": 24 if state != "pending" else 18,
                "category": "graph of doing",
                "itemStyle": {
                    "color": colour,
                    "borderColor": NODE_LABEL if state == "blocked" else colour,
                    "borderWidth": 2 if state == "blocked" else 1,
                },
                "label": {
                    "show": True,
                    "color": NODE_LABEL,
                    "fontSize": 10,
                    "fontWeight": "bold" if state == "blocked" else "normal",
                    "position": "inside" if state != "pending" else "bottom",
                    "textBorderColor": LABEL_HALO,
                    "textBorderWidth": 3,
                },
                "tooltip": {
                    "formatter": f"<b>{label}</b><br/>pipeline stage: {stage}<br/>{state}"
                },
            }
        )
    # the loop itself
    for i, (label, stage) in enumerate(DOING_STAGES):
        nxt = DOING_STAGES[(i + 1) % len(DOING_STAGES)]
        both_done = (
            stage_state.get(stage) in ("done", "blocked")
            and stage_state.get(nxt[1]) in ("done", "blocked")
        )
        links.append(
            {
                "source": f"__doing__{stage}",
                "target": f"__doing__{nxt[1]}",
                "value": "",
                "lineStyle": {
                    "color": DOING if both_done else EDGE,
                    "width": 2.4 if both_done else 1.2,
                    "opacity": 0.95 if both_done else 0.45,
                    "curveness": 0.28,
                },
                "label": {"show": False},
            }
        )
    # centre <-> assess: the model is consulted by the loop
    links.append(
        {
            "source": "__doing__neural",
            "target": "__llm__",
            "value": "",
            "lineStyle": {"color": LEARNING, "width": 2, "opacity": 0.8, "curveness": 0},
            "label": {"show": False},
        }
    )

    # ── outer ring: Graph of Knowing ─────────────────────────────────────────
    outer_pos = dict(zip(CORE_ENTITIES, _ring(len(CORE_ENTITIES), OUTER_R)))
    for entity in CORE_ENTITIES:
        x, y = outer_pos[entity]
        n = per_type.get(entity, 0)
        if entity in flagged:
            colour, cat = RED, "blocked by policy"
        elif entity in warned_types:
            colour, cat = AMBER, "policy warning"
        elif entity in touched_types:
            colour, cat = KNOWING_OK, "graph of knowing (in scope)"
        else:
            colour, cat = GREY, "graph of knowing (not touched)"
        nodes.append(
            {
                "id": f"__k__{entity}",
                "name": ENTITY_DISPLAY.get(entity, entity) + (f"\n({n})" if n else ""),
                "x": x,
                "y": y,
                "symbolSize": 34 + min(n, 12) * 1.6,
                "category": cat,
                "symbol": "roundRect",
                "itemStyle": {
                    "color": colour,
                    "borderColor": NODE_LABEL if entity in flagged else colour,
                    "borderWidth": 2 if entity in flagged else 1,
                    "shadowBlur": 14 if entity in flagged else 0,
                    "shadowColor": colour,
                },
                "label": {
                    "show": True,
                    "color": "#ffffff" if entity in touched_types or entity in flagged else NODE_LABEL,
                    "fontSize": 10,
                    "fontWeight": "bold",
                    "position": "inside",
                },
                "tooltip": {
                    "formatter": (
                        f"<b>{entity}</b><br/>Graph of Knowing<br/>"
                        f"{n} instance(s) in scope<br/>{cat}"
                    )
                },
            }
        )

    for display, rel_type, src, dst in CORE_RELATIONS:
        n = relation_counts.get(rel_type, 0)
        hot = src in flagged or dst in flagged
        live = n > 0
        links.append(
            {
                "source": f"__k__{src}",
                "target": f"__k__{dst}",
                "value": display,
                "label": {
                    "show": live,
                    "formatter": f"{display}" + (f" ×{n}" if n else ""),
                    "fontSize": 9,
                    "color": EDGE_LABEL,
                    "textBorderColor": LABEL_HALO,
                    "textBorderWidth": 3,
                },
                "lineStyle": {
                    "color": RED if hot else (KNOWING_OK if live else EDGE),
                    "width": 3 if hot else (2.2 if live else 1),
                    "opacity": 1.0 if hot else (0.9 if live else 0.35),
                    "curveness": 0.18,
                },
            }
        )

    # spokes: which zone each loop stage reached
    for stage, entity in (("assemble", "Disease"), ("symbolic", "Drug"), ("record", "Patient")):
        if stage_state.get(stage) in ("done", "blocked"):
            links.append(
                {
                    "source": f"__doing__{stage}",
                    "target": f"__k__{entity}",
                    "value": "",
                    "lineStyle": {
                        "color": RED if stage_state.get(stage) == "blocked" else DOING,
                        "width": 1.4,
                        "opacity": 0.5,
                        "type": "dashed",
                        "curveness": 0.1,
                    },
                    "label": {"show": False},
                }
            )

    categories = [
        {"name": "graph of learning", "itemStyle": {"color": LEARNING}},
        {"name": "graph of doing", "itemStyle": {"color": DOING}},
        {"name": "graph of knowing (in scope)", "itemStyle": {"color": KNOWING_OK}},
        {"name": "blocked by policy", "itemStyle": {"color": RED}},
        {"name": "policy warning", "itemStyle": {"color": AMBER}},
        {"name": "graph of knowing (not touched)", "itemStyle": {"color": GREY}},
    ]

    return {
        "backgroundColor": "transparent",
        "tooltip": {
            "trigger": "item",
            "backgroundColor": CARD_BG,
            "borderColor": EDGE,
            "textStyle": {"color": NODE_LABEL, "fontSize": 11},
        },
        "legend": [
            {
                "data": [c["name"] for c in categories],
                "textStyle": {"color": EDGE_LABEL, "fontSize": 9},
                "top": 0,
                "icon": "circle",
                "itemWidth": 8,
                "itemHeight": 8,
                "itemGap": 8,
            }
        ],
        "graphic": [
            {
                "type": "text",
                "left": "center",
                "bottom": 2,
                "style": {
                    "text": "Graph of learning   ·   Graph of doing   ·   Graph of knowing",
                    "fill": EDGE_LABEL,
                    "fontSize": 10,
                    "fontStyle": "italic",
                },
            },
            {
                "type": "text",
                "right": 6,
                "top": 26,
                "style": {
                    "text": "BLOCKED" if blocked else "",
                    "fill": RED,
                    "fontSize": 12,
                    "fontWeight": "bold",
                },
            },
        ],
        "series": [
            {
                "type": "graph",
                "layout": "none",  # fixed zones: the shape must stay learnable
                "roam": True,
                "draggable": False,
                "data": nodes,
                "links": links,
                "categories": categories,
                "emphasis": {
                    "focus": "adjacency",
                    "label": {"show": True},
                    "edgeLabel": {"show": True, "fontSize": 10},
                },
                "edgeSymbol": ["none", "arrow"],
                "edgeSymbolSize": 7,
                "top": 26,
                "bottom": 20,
            }
        ],
    }


def relation_counts_from(facts) -> dict[str, int]:
    """Count instances of each core relationship the query actually traversed."""
    wanted = {rel for _, rel, _, _ in CORE_RELATIONS}
    counts: dict[str, int] = {}
    for f in (facts.facts if facts else ()):
        for rel, targets in f.relations.items():
            if rel in wanted:
                counts[rel] = counts.get(rel, 0) + len(targets)
    return counts
