"""Graph traversal view. Turns a FactSet + RuleVerdict into an ECharts force graph.

The point is intuition: the rule engine reads the *graph*, not the retrieved
paragraph, and that claim is abstract until you can see which nodes it walked and
which of them tripped a policy.

Colour carries the meaning:

    red     triggered a BLOCKING rule      -- why the answer was suppressed
    amber   triggered a WARNING rule
    violet  entity pulled into scope       -- what the question touched
    cyan    external concept / CURIE       -- the vocabulary join
    grey    neighbour referenced but not itself in scope

Node size follows degree, so hubs are visually obvious.
"""

from __future__ import annotations

from typing import Any, Iterable

# Light theme. Every value is measured for contrast against a WHITE background;
# graphics need >= 3:1 and text >= 4.5:1.
#
# The previous dark palette fails completely on white — #ffc93c amber measures
# 1.54:1 and #e8eef7 label text 1.17:1 — so these are not tints of the old
# values but a recomputed set.
CARD_BG = "#ffffff"
RED = "#dc2626"        # 4.83:1  blocked by a policy
AMBER = "#b45309"      # 5.02:1  policy warning
VIOLET = "#6d28d9"     # 7.10:1  pulled into scope
CYAN = "#0e7490"       # 5.36:1  external concept / CURIE
GREY = "#64748b"       # 4.76:1  neighbour
EDGE = "#64748b"       # 4.76:1  relationship line
EDGE_LABEL = "#334155" # 10.35:1 relationship name
NODE_LABEL = "#0f172a" # 17.85:1 node name
LABEL_HALO = "#ffffff" # outline behind label text, now white not near-black

# Relations that clutter the picture without adding meaning.
HIDDEN_RELATIONS = {"MAPS_TO"}

# Self-referential relation targets to drop (e.g. Patient HAS_OUTCOME -> own id).
def _is_noise(source: str, target: str) -> bool:
    return source == target


def _category(
    entity_id: str,
    entity_type: str,
    blocking: set[str],
    warning: set[str],
    in_scope: set[str],
) -> tuple[str, str, int]:
    """Returns (colour, category label, base size)."""
    if entity_id in blocking:
        return RED, "blocked by policy", 46
    if entity_id in warning:
        return AMBER, "policy warning", 38
    if entity_type == "ExternalConcept" or ":" in entity_id and entity_id.isupper():
        return CYAN, "external concept", 26
    if entity_id in in_scope:
        return VIOLET, "in scope", 34
    return GREY, "neighbour", 20


def build_option(
    facts,
    verdict,
    names: dict[str, tuple[str, str]] | None = None,
    max_nodes: int = 60,
) -> dict[str, Any]:
    """ECharts option for a force-directed traversal graph.

    ``names`` maps id -> (label, display) for targets that are not themselves
    facts; without it the graph shows raw identifiers.
    """
    names = names or {}
    blocking = {
        t for f in verdict.fired if f.severity == "block" for t in f.triggered_by
    }
    warning = {
        t for f in verdict.fired if f.severity == "warn" for t in f.triggered_by
    }
    in_scope = {f.entity_id for f in facts.facts}

    # Collect edges first so degree can drive node size.
    edges: list[tuple[str, str, str]] = []
    for f in facts.facts:
        for rel, targets in f.relations.items():
            if rel in HIDDEN_RELATIONS:
                continue
            for t in targets:
                if _is_noise(f.entity_id, t):
                    continue
                edges.append((f.entity_id, t, rel))

    degree: dict[str, int] = {}
    for a, b, _ in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    # Node set: every fact, plus referenced targets, capped for legibility.
    wanted: list[tuple[str, str, str]] = []
    for f in facts.facts:
        wanted.append((f.entity_id, f.entity_type, f.label()))
    for _, t, _ in edges:
        if t in in_scope:
            continue
        label, display = names.get(t, ("", t))
        wanted.append((t, label or "Concept", display))

    seen: set[str] = set()
    ranked = sorted(
        wanted,
        key=lambda w: (
            0 if w[0] in blocking else 1 if w[0] in warning else
            2 if w[0] in in_scope else 3,
            -degree.get(w[0], 0),
        ),
    )
    nodes: list[dict[str, Any]] = []
    for entity_id, entity_type, display in ranked:
        if entity_id in seen or len(nodes) >= max_nodes:
            continue
        seen.add(entity_id)
        colour, category, size = _category(
            entity_id, entity_type, blocking, warning, in_scope
        )
        size += min(degree.get(entity_id, 0) * 2, 16)
        nodes.append(
            {
                "id": entity_id,
                "name": display[:34],
                "symbolSize": size,
                "category": category,
                "itemStyle": {
                    "color": colour,
                    "borderColor": NODE_LABEL if entity_id in blocking else colour,
                    "borderWidth": 2 if entity_id in blocking else 1,
                    "shadowBlur": 12 if entity_id in blocking else 0,
                    "shadowColor": colour,
                },
                # Show every label: an unlabelled node conveys nothing.
                "label": {
                    "show": True,
                    "color": NODE_LABEL,
                    "fontSize": 11 if size >= 34 else 9,
                    "fontWeight": "bold" if entity_id in blocking else "normal",
                    "position": "right",
                    "distance": 6,
                    "textBorderColor": LABEL_HALO,
                    "textBorderWidth": 3,
                },
                "tooltip": {
                    "formatter": (
                        f"<b>{display}</b><br/>{entity_type}<br/>"
                        f"<code>{entity_id}</code><br/>{category}"
                    )
                },
            }
        )

    links = [
        {
            "source": a,
            "target": b,
            "value": rel,
            "label": {
                "show": True,
                "formatter": rel,
                "fontSize": 9,
                "color": EDGE_LABEL,
                "textBorderColor": LABEL_HALO,
                "textBorderWidth": 3,
            },
            "lineStyle": {
                "color": RED if (a in blocking or b in blocking) else EDGE,
                "width": 3 if (a in blocking or b in blocking) else 1.6,
                "curveness": 0.12,
                # Fully opaque: the earlier 0.45 compounded an already
                # low-contrast colour into invisibility.
                "opacity": 1.0 if (a in blocking or b in blocking) else 0.75,
            },
        }
        for a, b, rel in edges
        if a in seen and b in seen
    ]

    categories = [
        {"name": "blocked by policy", "itemStyle": {"color": RED}},
        {"name": "policy warning", "itemStyle": {"color": AMBER}},
        {"name": "in scope", "itemStyle": {"color": VIOLET}},
        {"name": "external concept", "itemStyle": {"color": CYAN}},
        {"name": "neighbour", "itemStyle": {"color": GREY}},
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
                "textStyle": {"color": EDGE_LABEL, "fontSize": 10},
                "top": 0,
                "icon": "circle",
                "itemWidth": 8,
                "itemHeight": 8,
            }
        ],
        "series": [
            {
                "type": "graph",
                "layout": "force",
                "roam": True,
                "draggable": True,
                "data": nodes,
                "links": links,
                "categories": categories,
                "force": {
                    "repulsion": 320,
                    "edgeLength": [70, 160],
                    "gravity": 0.08,
                    "friction": 0.2,
                },
                "emphasis": {
                    "focus": "adjacency",
                    "label": {"show": True, "fontSize": 11},
                    "edgeLabel": {"show": True, "fontSize": 9},
                },
                "lineStyle": {"color": "source"},
                "edgeSymbol": ["none", "arrow"],
                "edgeSymbolSize": 7,
                "top": 22,
            }
        ],
    }


def empty_option(message: str = "run a question to see the traversal") -> dict[str, Any]:
    return {
        "backgroundColor": "transparent",
        "title": {
            "text": message,
            "left": "center",
            "top": "middle",
            "textStyle": {"color": GREY, "fontSize": 11, "fontWeight": "normal"},
        },
        "series": [],
    }


def summarise(facts, verdict) -> str:
    """One-line description of what was walked."""
    rels: dict[str, int] = {}
    for f in facts.facts:
        for rel, targets in f.relations.items():
            if rel in HIDDEN_RELATIONS:
                continue
            rels[rel] = rels.get(rel, 0) + len(targets)
    top = ", ".join(f"{r}×{n}" for r, n in sorted(rels.items(), key=lambda x: -x[1])[:4])
    triggered = len({t for f in verdict.fired for t in f.triggered_by})
    return (
        f"{len(facts)} entities · {sum(rels.values())} edges · "
        f"{triggered} triggered a policy" + (f" · {top}" if top else "")
    )
