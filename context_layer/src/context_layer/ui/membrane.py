"""Neurosymbolic Proof — animated architecture diagram.

Maps the conceptual membrane model to the concrete PoC: Neo4j graph, Qdrant
vectors, Bedrock LLM, deterministic checks, and the eval proof. Pulls live
stats and renders with CSS + SVG animations.
"""

from __future__ import annotations

import sys
from pathlib import Path

from nicegui import ui

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module

CYAN = "#0e7490"
VIOLET = "#7c3aed"
AMBER = "#b45309"
RED = "#dc2626"
GREEN = "#047857"
DIM = "#475569"
INK = "#0f172a"
BG = "#ffffff"
CARD = "#ffffff"
BORDER = "#e2e8f0"
EDGE = "#cbd5e1"

ANIMATIONS_CSS = """
<style>
@keyframes spin-slow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes spin-reverse { from { transform: rotate(360deg); } to { transform: rotate(0deg); } }
@keyframes pulse-glow { 0%,100% { opacity: 0.12; r: 200; } 50% { opacity: 0.25; r: 220; } }
@keyframes pulse-core { 0%,100% { r: 24; opacity: 1; } 50% { r: 28; opacity: 0.85; } }
@keyframes node-breathe { 0%,100% { opacity: 0.55; } 50% { opacity: 0.9; } }
@keyframes dash-flow { to { stroke-dashoffset: -20; } }
@keyframes dash-flow-reverse { to { stroke-dashoffset: 20; } }
@keyframes threat-pulse { 0%,100% { stroke-opacity: 0.4; stroke-width: 1.5; } 50% { stroke-opacity: 1; stroke-width: 2.5; } }
@keyframes fade-in-up { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes shield-glow { 0%,100% { stroke-opacity: 0.2; } 50% { stroke-opacity: 0.5; } }
@keyframes float-y { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }
@keyframes float-y-alt { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes scan-line { 0% { opacity: 0; } 10% { opacity: 0.15; } 90% { opacity: 0.15; } 100% { opacity: 0; } }

.anim-spin-slow   { animation: spin-slow 45s linear infinite; transform-origin: 500px 340px; }
.anim-spin-rev    { animation: spin-reverse 60s linear infinite; transform-origin: 500px 340px; }
.anim-pulse-glow  { animation: pulse-glow 4s ease-in-out infinite; }
.anim-pulse-core  { animation: pulse-core 3s ease-in-out infinite; }
.anim-breathe     { animation: node-breathe 3s ease-in-out infinite; }
.anim-breathe-d1  { animation: node-breathe 3s ease-in-out 0.4s infinite; }
.anim-breathe-d2  { animation: node-breathe 3s ease-in-out 0.8s infinite; }
.anim-breathe-d3  { animation: node-breathe 3s ease-in-out 1.2s infinite; }
.anim-breathe-d4  { animation: node-breathe 3s ease-in-out 1.6s infinite; }
.anim-dash        { animation: dash-flow 1.2s linear infinite; }
.anim-dash-slow   { animation: dash-flow 2.5s linear infinite; }
.anim-dash-rev    { animation: dash-flow-reverse 1.5s linear infinite; }
.anim-threat      { animation: threat-pulse 1.8s ease-in-out infinite; }
.anim-shield      { animation: shield-glow 3s ease-in-out infinite; }
.anim-float       { animation: float-y 4s ease-in-out infinite; }
.anim-float-alt   { animation: float-y-alt 5s ease-in-out 1s infinite; }

.card-animate { animation: fade-in-up 0.6s ease-out both; }
.card-d1 { animation-delay: 0.1s; }
.card-d2 { animation-delay: 0.25s; }
.card-d3 { animation-delay: 0.4s; }
.card-d4 { animation-delay: 0.55s; }
.card-d5 { animation-delay: 0.7s; }
.card-d6 { animation-delay: 0.85s; }

.stat-counter {
    font-variant-numeric: tabular-nums;
    transition: color 0.3s ease;
}
.stat-counter:hover { color: #7c3aed !important; }

.build-card:hover { border-color: #047857 !important; box-shadow: 0 4px 12px rgba(4,120,87,0.12) !important; transition: all 0.3s ease; }
.buy-card:hover { border-color: #b45309 !important; box-shadow: 0 4px 12px rgba(180,83,9,0.12) !important; transition: all 0.3s ease; }
.threat-card:hover { border-color: #dc2626 !important; box-shadow: 0 4px 12px rgba(220,38,38,0.12) !important; transition: all 0.3s ease; }
</style>
"""


class MembranePage:
    def __init__(self) -> None:
        self.cfg = config_module.load()
        self._neo4j_driver = None
        self._qdrant_client = None

    def _neo4j(self):
        if self._neo4j_driver is None:
            from neo4j import GraphDatabase
            self._neo4j_driver = GraphDatabase.driver(
                self.cfg.neo4j_uri,
                auth=(self.cfg.neo4j_user, str(self.cfg.neo4j_password)),
            )
        return self._neo4j_driver

    def _qdrant(self):
        if self._qdrant_client is None:
            from qdrant_client import QdrantClient
            self._qdrant_client = QdrantClient(
                url=self.cfg.qdrant_url,
                api_key=str(self.cfg.qdrant_api_key),
                timeout=self.cfg.qdrant_timeout,
                check_compatibility=False,
            )
        return self._qdrant_client

    def _fetch_live_stats(self) -> dict:
        stats = {}
        try:
            with self._neo4j().session(database=self.cfg.neo4j_kg_database) as s:
                total = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                eval_n = s.run("MATCH (n) WHERE any(l IN labels(n) WHERE l STARTS WITH 'Eval_') RETURN count(n) AS c").single()["c"]
                proof = s.run("MATCH (p:Eval_Proof) RETURN p.false_pass_rate_pct AS fp LIMIT 1").single()
                stats["neo4j_total"] = total
                stats["neo4j_eval"] = eval_n
                stats["neo4j_kg"] = total - eval_n
                stats["false_pass_rate"] = proof["fp"] if proof else 0
        except Exception as e:
            stats["neo4j_error"] = str(e)
        try:
            q = self._qdrant()
            colls = q.get_collections().collections
            total_pts = 0
            coll_details = []
            for c in colls:
                info = q.get_collection(c.name)
                pts = info.points_count or 0
                total_pts += pts
                coll_details.append({"name": c.name, "points": pts})
            stats["qdrant_collections"] = len(colls)
            stats["qdrant_points"] = total_pts
            stats["qdrant_details"] = coll_details
        except Exception as e:
            stats["qdrant_error"] = str(e)
        return stats

    def build(self) -> None:
        ui.colors(primary=VIOLET, secondary=CYAN, accent=AMBER)
        ui.query("body").style(f"background:{BG}")
        ui.html(ANIMATIONS_CSS)

        with ui.row().classes("w-full items-center justify-between px-4 pt-4 card-animate"):
            with ui.column().classes("gap-0"):
                ui.label("NEUROSYMBOLIC PROOF").style(
                    f"color:{VIOLET};font-size:26px;font-weight:800;letter-spacing:.06em"
                )
                ui.label(
                    "live architecture proof — Neo4j graph + Qdrant vectors + Bedrock LLM + deterministic checks"
                ).style(f"color:{DIM};font-size:12px")
            with ui.row().classes("gap-4 items-center"):
                ui.link("← Pipeline", "/").style(f"color:{CYAN};font-size:12px;font-weight:600")
                ui.link("Eval Proof →", "/eval").style(f"color:{RED};font-size:12px;font-weight:600")

        self.container = ui.column().classes("w-full px-4 gap-4")
        self.loading = ui.label("Loading live stats from Neo4j + Qdrant...").style(
            f"color:{DIM};font-size:14px;padding:16px"
        )
        ui.timer(0.5, self._load, once=True)

    async def _load(self) -> None:
        import asyncio
        try:
            stats = await asyncio.to_thread(self._fetch_live_stats)
        except Exception as e:
            self.loading.set_text(f"Error: {e}")
            return
        self.loading.delete()
        with self.container:
            self._render(stats)

    def _card(self, title: str, colour: str = EDGE, extra_class: str = ""):
        cls = f"w-full card-animate {extra_class}".strip()
        card = ui.card().classes(cls).style(
            f"background:{CARD};border:1px solid {BORDER};"
            f"border-top:3px solid {colour};box-shadow:0 1px 2px rgba(15,23,42,.06);"
            "transition: all 0.3s ease;"
        )
        with card:
            ui.label(title.upper()).style(
                f"color:{colour};font-size:11px;letter-spacing:.14em;font-weight:700"
            )
        return card

    def _render(self, stats: dict) -> None:
        neo_total = stats.get("neo4j_total", 0)
        neo_eval = stats.get("neo4j_eval", 0)
        neo_kg = stats.get("neo4j_kg", 0)
        fp_rate = stats.get("false_pass_rate", 0)
        q_colls = stats.get("qdrant_collections", 0)
        q_pts = stats.get("qdrant_points", 0)

        # ── Live Stats Banner ──
        with ui.row().classes("w-full no-wrap gap-4 card-animate card-d1"):
            for value, label, colour in [
                (str(neo_total), "Neo4j Nodes", VIOLET),
                (str(neo_eval), "Eval Nodes", RED),
                (f"{q_pts}", "Qdrant Vectors", CYAN),
                (f"{fp_rate}%", "False Pass Rate", RED),
                (f"{q_colls}", "Collections", CYAN),
                ("100%", "Deterministic", GREEN),
            ]:
                with ui.card().classes("flex-1").style(
                    f"background:{CARD};border:1px solid {BORDER};"
                    f"border-top:3px solid {colour};text-align:center;padding:8px 4px;"
                    "box-shadow:0 1px 2px rgba(15,23,42,.06);transition:all 0.3s ease"
                ):
                    ui.label(value).classes("stat-counter").style(
                        f"color:{colour};font-size:24px;font-weight:800;line-height:1"
                    )
                    ui.label(label).style(f"color:{DIM};font-size:9px;letter-spacing:.08em")

        # ── The Thesis ──
        with self._card("The Argument", VIOLET, "card-d2"):
            for line in [
                "Your context is not a feature you bolt on. It is your model of your own world.",
                "Every system that survives holds a boundary between itself and the world.",
                "You can buy the models. You can buy the agents. You have to build the context.",
                "A context you can buy is a context your competitors can buy too.",
            ]:
                ui.label(line).style(f"color:{INK};font-size:13px;line-height:1.7")

        # ── SVG Diagram ──
        with self._card("Architecture — Neurosymbolic Proof", VIOLET, "card-d3"):
            svg = self._build_svg(stats)
            ui.html(svg).classes("w-full")

        # ── Three Columns ──
        with ui.row().classes("w-full no-wrap gap-4"):
            with ui.column().classes("flex-1 gap-4"):
                with self._card("What You Build (The Boundary)", GREEN, "card-d4"):
                    card_el = ui.element("div").classes("build-card").style("padding:0")
                    with card_el:
                        for title, detail, why in [
                            ("Neo4j Knowledge Graph", f"{neo_kg} nodes — Drug, Patient, ClinicalTrial, Ctx_Event", "Your ontology, your identifiers, your relationships"),
                            ("Qdrant Semantic Layer", f"{q_colls} collections, {q_pts} vectors — Titan v2 1024-dim", "Your documents, your meaning, your semantic space"),
                            ("Governance Rules", "Deterministic checks, clinical advice guard", "Your policies, your thresholds, your compliance"),
                            ("Eval Ground Truth", f"{neo_eval} eval nodes — your test cases, your known answers", "Your calibration, your false-pass rate measurement"),
                        ]:
                            with ui.column().classes("gap-0").style("margin-bottom:8px"):
                                ui.label(title).style(f"color:{GREEN};font-size:12px;font-weight:700")
                                ui.label(detail).style(f"color:{DIM};font-size:10px;font-family:monospace")
                                ui.label(why).style(f"color:{DIM};font-size:10px;font-style:italic")

            with ui.column().classes("flex-1 gap-4"):
                with self._card("What You Buy (Commodity)", AMBER, "card-d5"):
                    card_el = ui.element("div").classes("buy-card").style("padding:0")
                    with card_el:
                        for title, detail, why in [
                            ("Bedrock Claude (LLM)", "claude-haiku-4-5 — neural pattern recognition", "Replaceable. Any model works. Not your differentiator."),
                            ("Titan Embed (Embeddings)", "amazon.titan-embed-text-v2:0 — 1024-dim", "Commodity vectorization. The vectors matter, not the model."),
                            ("LLM-as-Judge", f"75% accuracy, {fp_rate}% false pass rate", "Unreliable. Creates false confidence. Worse than nothing."),
                        ]:
                            with ui.column().classes("gap-0").style("margin-bottom:8px"):
                                ui.label(title).style(f"color:{AMBER};font-size:12px;font-weight:700")
                                ui.label(detail).style(f"color:{DIM};font-size:10px;font-family:monospace")
                                ui.label(why).style(f"color:{DIM};font-size:10px;font-style:italic")

            with ui.column().classes("flex-1 gap-4"):
                with self._card("What Breaches the Boundary", RED, "card-d6"):
                    card_el = ui.element("div").classes("threat-card").style("padding:0")
                    with card_el:
                        for title, detail, why in [
                            ("False Passes", f"{fp_rate}% of bad outputs approved by the judge", "Bugs ship because you trust the green check"),
                            ("False Fails", "correct-palindrome rejected by judge", "You waste time investigating correct outputs"),
                            ("Sycophancy", "Judge approves well-written wrong answers", "Eloquence beats accuracy in LLM evaluation"),
                            ("Outsourced Context", "If a vendor defines your boundary, you're a component", "You're inside someone else's system"),
                        ]:
                            with ui.column().classes("gap-0").style("margin-bottom:8px"):
                                ui.label(title).style(f"color:{RED};font-size:12px;font-weight:700")
                                ui.label(detail).style(f"color:{DIM};font-size:10px;font-family:monospace")
                                ui.label(why).style(f"color:{DIM};font-size:10px;font-style:italic")

        # ── The Line ──
        with self._card("The Line Every Organisation Must Draw", VIOLET, "card-d6"):
            with ui.row().classes("w-full no-wrap gap-8"):
                for icon, text, colour in [
                    ("✓", "Build the context — your ontology, your meaning, your rules", GREEN),
                    ("↔", "Buy the models — commodity LLMs, embeddings, infrastructure", AMBER),
                    ("✗", "Never outsource the boundary — it's what makes you a system", RED),
                ]:
                    with ui.row().classes("flex-1 items-start gap-2"):
                        ui.label(icon).style(f"color:{colour};font-size:20px;font-weight:800")
                        ui.label(text).style(f"color:{INK};font-size:12px;line-height:1.6")
            ui.separator().style("margin:8px 0")
            ui.label(
                "Openness is not the opposite of ownership. It is the only durable form of it. "
                "Open standards (openCypher, open-source Qdrant, standard embeddings) — "
                "no proprietary lock-in. The meaning is yours."
            ).style(f"color:{DIM};font-size:11px;line-height:1.6;font-style:italic")

    def _build_svg(self, stats: dict) -> str:
        neo_total = stats.get("neo4j_total", 0)
        neo_eval = stats.get("neo4j_eval", 0)
        neo_kg = stats.get("neo4j_kg", 0)
        fp_rate = stats.get("false_pass_rate", 0)
        q_colls = stats.get("qdrant_collections", 0)
        q_pts = stats.get("qdrant_points", 0)

        return f'''<svg viewBox="0 0 1000 720" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
  <defs>
    <radialGradient id="cg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="{VIOLET}" stop-opacity="0.15"/>
      <stop offset="100%" stop-color="{VIOLET}" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="cs" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#a78bfa"/>
      <stop offset="100%" stop-color="{VIOLET}"/>
    </radialGradient>
    <filter id="glow-f">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="glow-red">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feFlood flood-color="{RED}" flood-opacity="0.3"/>
      <feComposite in2="blur" operator="in"/>
      <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="glow-green">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feFlood flood-color="{GREEN}" flood-opacity="0.3"/>
      <feComposite in2="blur" operator="in"/>
      <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="ma" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <path d="M0,0 L7,2.5 L0,5" fill="{DIM}"/>
    </marker>
    <marker id="mr" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <path d="M0,0 L7,2.5 L0,5" fill="{RED}"/>
    </marker>
    <marker id="mg" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <path d="M0,0 L7,2.5 L0,5" fill="{GREEN}"/>
    </marker>
    <marker id="mc" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <path d="M0,0 L7,2.5 L0,5" fill="{CYAN}"/>
    </marker>
  </defs>

  <!-- Animated glow -->
  <circle cx="500" cy="340" r="200" fill="url(#cg)" class="anim-pulse-glow"/>

  <!-- Scanning line (radar effect) -->
  <line x1="500" y1="340" x2="500" y2="100" stroke="{VIOLET}" stroke-width="1" opacity="0.08" class="anim-spin-slow"/>

  <!-- Membrane rings (spinning slowly in opposite directions) -->
  <g class="anim-spin-slow">
    <circle cx="500" cy="340" r="210" fill="none" stroke="{VIOLET}" stroke-width="2" opacity="0.2" stroke-dasharray="12,8" class="anim-shield"/>
  </g>
  <g class="anim-spin-rev">
    <circle cx="500" cy="340" r="240" fill="none" stroke="{VIOLET}" stroke-width="2" opacity="0.2" stroke-dasharray="8,12" class="anim-shield"/>
  </g>

  <!-- Static reference rings -->
  <circle cx="500" cy="340" r="210" fill="none" stroke="{VIOLET}" stroke-width="0.5" opacity="0.1"/>
  <circle cx="500" cy="340" r="240" fill="none" stroke="{VIOLET}" stroke-width="0.5" opacity="0.1"/>

  <!-- Segment arcs -->
  <path d="M 500 100 A 240 240 0 0 1 730 185" fill="none" stroke="{VIOLET}" stroke-width="28" opacity="0.06"/>
  <path d="M 730 185 A 240 240 0 0 1 740 495" fill="none" stroke="{VIOLET}" stroke-width="28" opacity="0.06"/>
  <path d="M 740 495 A 240 240 0 0 1 500 580" fill="none" stroke="{VIOLET}" stroke-width="28" opacity="0.06"/>
  <path d="M 500 580 A 240 240 0 0 1 260 495" fill="none" stroke="{VIOLET}" stroke-width="28" opacity="0.06"/>
  <path d="M 260 495 A 240 240 0 0 1 270 185" fill="none" stroke="{VIOLET}" stroke-width="28" opacity="0.06"/>
  <path d="M 270 185 A 240 240 0 0 1 500 100" fill="none" stroke="{VIOLET}" stroke-width="28" opacity="0.06"/>

  <!-- Segment labels -->
  <text x="630" y="135" text-anchor="middle" font-size="12" font-weight="800" fill="{VIOLET}" letter-spacing="2" transform="rotate(20,630,135)">ONTOLOGY</text>
  <text x="640" y="152" text-anchor="middle" font-size="8" fill="{DIM}" transform="rotate(20,640,152)">Drug &#8594; Patient &#8594; Trial</text>

  <text x="760" y="330" text-anchor="start" font-size="12" font-weight="800" fill="{VIOLET}" letter-spacing="2" transform="rotate(70,760,330)">IDENTIFIERS</text>
  <text x="740" y="370" text-anchor="start" font-size="8" fill="{DIM}" transform="rotate(70,740,370)">Entity Link @ 0.72</text>

  <text x="660" y="555" text-anchor="middle" font-size="12" font-weight="800" fill="{VIOLET}" letter-spacing="2" transform="rotate(130,660,555)">RULES</text>
  <text x="630" y="568" text-anchor="middle" font-size="8" fill="{DIM}" transform="rotate(130,630,568)">Governance + Deterministic</text>

  <text x="370" y="575" text-anchor="middle" font-size="12" font-weight="800" fill="{VIOLET}" letter-spacing="2" transform="rotate(-135,370,575)">MEANING</text>
  <text x="385" y="558" text-anchor="middle" font-size="8" fill="{DIM}" transform="rotate(-135,385,558)">Qdrant Embeddings</text>

  <text x="255" y="400" text-anchor="end" font-size="12" font-weight="800" fill="{VIOLET}" letter-spacing="2" transform="rotate(-65,255,400)">PROCESSES</text>
  <text x="275" y="380" text-anchor="end" font-size="8" fill="{DIM}" transform="rotate(-65,275,380)">Eval Pipeline</text>

  <text x="355" y="145" text-anchor="middle" font-size="12" font-weight="800" fill="{VIOLET}" letter-spacing="2" transform="rotate(-22,355,145)">CONTEXT</text>
  <text x="365" y="162" text-anchor="middle" font-size="8" fill="{DIM}" transform="rotate(-22,365,162)">8000 token budget</text>

  <!-- Center: Neo4j core (pulsing) -->
  <circle cx="500" cy="340" r="24" fill="url(#cs)" filter="url(#glow-f)" class="anim-pulse-core"/>
  <text x="500" y="344" text-anchor="middle" font-size="8" font-weight="700" fill="white">NEO4J</text>

  <!-- Graph nodes (breathing with stagger) -->
  <circle cx="450" cy="280" r="9" fill="#a78bfa" class="anim-breathe"/><text x="450" y="268" text-anchor="middle" font-size="7" fill="{VIOLET}">Drug</text>
  <circle cx="555" cy="275" r="9" fill="#a78bfa" class="anim-breathe-d1"/><text x="555" y="263" text-anchor="middle" font-size="7" fill="{VIOLET}">Patient</text>
  <circle cx="575" cy="370" r="9" fill="#a78bfa" class="anim-breathe-d2"/><text x="592" y="367" text-anchor="start" font-size="7" fill="{VIOLET}">Trial</text>
  <circle cx="430" cy="385" r="9" fill="#a78bfa" class="anim-breathe-d3"/><text x="413" y="382" text-anchor="end" font-size="7" fill="{VIOLET}">Event</text>
  <circle cx="500" cy="410" r="9" fill="#a78bfa" class="anim-breathe-d4"/><text x="500" y="428" text-anchor="middle" font-size="7" fill="{VIOLET}">Rule</text>
  <circle cx="500" cy="295" r="7" fill="#fca5a5" class="anim-breathe-d2"/><text x="500" y="285" text-anchor="middle" font-size="7" fill="{RED}" font-weight="600">Eval</text>

  <!-- Extra small orbiting nodes -->
  <circle cx="470" cy="310" r="4" fill="#c4b5fd" class="anim-breathe-d1" opacity="0.4"/>
  <circle cx="530" cy="305" r="4" fill="#c4b5fd" class="anim-breathe-d3" opacity="0.4"/>
  <circle cx="540" cy="385" r="4" fill="#c4b5fd" class="anim-breathe" opacity="0.4"/>
  <circle cx="460" cy="360" r="4" fill="#c4b5fd" class="anim-breathe-d2" opacity="0.4"/>

  <!-- Edges -->
  <line x1="500" y1="340" x2="450" y2="280" stroke="#c4b5fd" stroke-width="1.2" opacity="0.4"/>
  <line x1="500" y1="340" x2="555" y2="275" stroke="#c4b5fd" stroke-width="1.2" opacity="0.4"/>
  <line x1="500" y1="340" x2="575" y2="370" stroke="#c4b5fd" stroke-width="1.2" opacity="0.4"/>
  <line x1="500" y1="340" x2="430" y2="385" stroke="#c4b5fd" stroke-width="1.2" opacity="0.4"/>
  <line x1="500" y1="340" x2="500" y2="410" stroke="#c4b5fd" stroke-width="1.2" opacity="0.4"/>
  <line x1="450" y1="280" x2="555" y2="275" stroke="#c4b5fd" stroke-width="1" opacity="0.3"/>
  <line x1="555" y1="275" x2="575" y2="370" stroke="#c4b5fd" stroke-width="1" opacity="0.3"/>
  <line x1="430" y1="385" x2="500" y2="410" stroke="#c4b5fd" stroke-width="1" opacity="0.3"/>
  <line x1="500" y1="340" x2="500" y2="295" stroke="#fca5a5" stroke-width="1.2" opacity="0.5"/>
  <text x="500" y="455" text-anchor="middle" font-size="9" fill="{DIM}" font-weight="600">{neo_total} nodes &#183; {neo_eval} eval &#183; {neo_kg} KG</text>

  <!-- ═══ EXTERNAL FORCES ═══ -->

  <!-- LLM Judge (top — animated threat arrow) -->
  <g class="anim-float">
    <rect x="430" y="18" width="140" height="40" rx="6" fill="#fef2f2" stroke="{RED}" stroke-width="1.5" filter="url(#glow-red)"/>
    <text x="500" y="36" text-anchor="middle" font-size="11" font-weight="800" fill="{RED}">LLM JUDGE</text>
    <text x="500" y="50" text-anchor="middle" font-size="8" fill="{RED}">Haiku &#183; {fp_rate}% blind</text>
  </g>
  <line x1="500" y1="58" x2="500" y2="130" stroke="{RED}" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#mr)" class="anim-dash anim-threat"/>
  <text x="522" y="95" font-size="8" fill="{RED}" font-weight="700">BREACHES {fp_rate}%</text>

  <!-- Bedrock (right — floating) -->
  <g class="anim-float-alt">
    <rect x="810" y="300" width="125" height="42" rx="6" fill="#fff7ed" stroke="{AMBER}" stroke-width="1.5"/>
    <text x="872" y="318" text-anchor="middle" font-size="10" font-weight="700" fill="{AMBER}">BEDROCK LLM</text>
    <text x="872" y="334" text-anchor="middle" font-size="7" fill="{AMBER}">Claude &#183; Titan Embed</text>
  </g>
  <line x1="810" y1="325" x2="720" y2="335" stroke="{AMBER}" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#ma)" class="anim-dash-slow"/>
  <text x="760" y="318" font-size="7" fill="{AMBER}">commodity</text>

  <!-- Qdrant (bottom — floating) -->
  <g class="anim-float">
    <rect x="415" y="630" width="170" height="42" rx="6" fill="#ecfeff" stroke="{CYAN}" stroke-width="1.5"/>
    <text x="500" y="648" text-anchor="middle" font-size="10" font-weight="700" fill="{CYAN}">QDRANT VECTORS</text>
    <text x="500" y="662" text-anchor="middle" font-size="7" fill="{CYAN}">{q_colls} collections &#183; {q_pts} pts &#183; 1024-dim</text>
  </g>
  <line x1="500" y1="630" x2="500" y2="565" stroke="{CYAN}" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#mc)" class="anim-dash-rev"/>
  <text x="522" y="600" font-size="7" fill="{CYAN}">semantic grounding</text>

  <!-- Deterministic (left — shield glow) -->
  <g class="anim-float-alt">
    <rect x="45" y="305" width="135" height="42" rx="6" fill="#f0fdf4" stroke="{GREEN}" stroke-width="2" filter="url(#glow-green)"/>
    <text x="112" y="324" text-anchor="middle" font-size="10" font-weight="700" fill="{GREEN}">DETERMINISTIC</text>
    <text x="112" y="338" text-anchor="middle" font-size="7" fill="{GREEN}">Regex &#183; Exact &#183; Code Exec</text>
  </g>
  <line x1="180" y1="325" x2="280" y2="335" stroke="{GREEN}" stroke-width="2" marker-end="url(#mg)"/>
  <text x="205" y="316" font-size="9" fill="{GREEN}" font-weight="800">100%</text>

  <!-- Users (top-right) -->
  <g class="anim-float-alt">
    <rect x="760" y="120" width="100" height="36" rx="6" fill="#f0fdf4" stroke="{GREEN}" stroke-width="1"/>
    <text x="810" y="138" text-anchor="middle" font-size="9" font-weight="700" fill="{GREEN}">USERS</text>
    <text x="810" y="150" text-anchor="middle" font-size="7" fill="{GREEN}">Queries</text>
  </g>
  <line x1="762" y1="148" x2="680" y2="220" stroke="{DIM}" stroke-width="1" stroke-dasharray="3,3" marker-end="url(#ma)" class="anim-dash-slow"/>

  <!-- Regulation (bottom-right) -->
  <g class="anim-float">
    <rect x="770" y="510" width="120" height="36" rx="6" fill="#f0fdf4" stroke="{GREEN}" stroke-width="1"/>
    <text x="830" y="528" text-anchor="middle" font-size="9" font-weight="700" fill="{GREEN}">REGULATION</text>
    <text x="830" y="540" text-anchor="middle" font-size="7" fill="{GREEN}">Clinical Guard</text>
  </g>
  <line x1="772" y1="522" x2="680" y2="475" stroke="{GREEN}" stroke-width="1" stroke-dasharray="3,3" marker-end="url(#mg)" class="anim-dash-slow"/>

  <!-- Eval Pipeline (top-left) -->
  <g class="anim-float">
    <rect x="95" y="120" width="125" height="36" rx="6" fill="#f8fafc" stroke="{DIM}" stroke-width="1"/>
    <text x="157" y="138" text-anchor="middle" font-size="9" font-weight="700" fill="{DIM}">EVAL PIPELINE</text>
    <text x="157" y="150" text-anchor="middle" font-size="7" fill="{DIM}">4-phase &#183; 12 tests</text>
  </g>
  <line x1="218" y1="150" x2="330" y2="230" stroke="{DIM}" stroke-width="1" stroke-dasharray="3,3" marker-end="url(#ma)" class="anim-dash-slow"/>

  <!-- False Passes (bottom-left) -->
  <g class="anim-float-alt">
    <rect x="65" y="520" width="125" height="36" rx="6" fill="#fef2f2" stroke="{RED}" stroke-width="1"/>
    <text x="127" y="538" text-anchor="middle" font-size="9" font-weight="700" fill="{RED}">FALSE PASSES</text>
    <text x="127" y="550" text-anchor="middle" font-size="7" fill="{RED}">Subtle errors slip</text>
  </g>
  <line x1="188" y1="530" x2="315" y2="470" stroke="{RED}" stroke-width="1" stroke-dasharray="5,3" marker-end="url(#mr)" class="anim-dash anim-threat"/>

  <!-- Legend -->
  <rect x="20" y="632" width="170" height="78" rx="5" fill="white" stroke="{BORDER}" stroke-width="1" opacity="0.9"/>
  <text x="30" y="648" font-size="8" font-weight="700" fill="{DIM}">LEGEND</text>
  <circle cx="32" cy="660" r="4" fill="{GREEN}"/><text x="42" y="663" font-size="7" fill="{DIM}">Yours — built, not bought</text>
  <circle cx="32" cy="675" r="4" fill="{AMBER}"/><text x="42" y="678" font-size="7" fill="{DIM}">Commodity — replaceable</text>
  <circle cx="32" cy="690" r="4" fill="{RED}"/><text x="42" y="693" font-size="7" fill="{DIM}">Threat — breaches boundary</text>
  <circle cx="32" cy="705" r="4" fill="{VIOLET}"/><text x="42" y="708" font-size="7" fill="{DIM}">The boundary — your context</text>

  <!-- Title bar -->
  <rect x="275" y="680" width="450" height="30" rx="5" fill="{VIOLET}" opacity="0.95"/>
  <text x="500" y="700" text-anchor="middle" font-size="14" font-weight="800" fill="white" letter-spacing="3">NEUROSYMBOLIC PROOF</text>
</svg>'''

    def close(self) -> None:
        if self._neo4j_driver:
            self._neo4j_driver.close()
        self._neo4j_driver = None
        self._qdrant_client = None
