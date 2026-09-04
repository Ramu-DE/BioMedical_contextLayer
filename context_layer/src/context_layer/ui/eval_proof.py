"""Eval Proof panel: LLM-as-Judge Is Worse Than No Evals.

Pulls live results from Neo4j (graph) and Qdrant (vectors) and renders
the proof visually — false pass rates, judge consistency, semantic failure
clusters, and the cost matrix.

Integrated into the Context Layer dashboard as a second page at /eval.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from nicegui import ui

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module

BG = "#ffffff"
CARD = "#ffffff"
BORDER = "#e2e8f0"
CYAN = "#0e7490"
VIOLET = "#6d28d9"
AMBER = "#b45309"
RED = "#dc2626"
GREEN = "#047857"
DIM = "#475569"
INK = "#0f172a"
EDGE = "#cbd5e1"


class EvalProofPage:
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

    def _card(self, title: str, colour: str = EDGE):
        card = ui.card().classes("w-full").style(
            f"background:{CARD};border:1px solid {BORDER};"
            f"border-top:3px solid {colour};box-shadow:0 1px 2px rgba(15,23,42,.06)"
        )
        with card:
            ui.label(title.upper()).style(
                f"color:{colour};font-size:11px;letter-spacing:.14em;font-weight:700"
            )
        return card

    def _fetch_neo4j_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        with self._neo4j().session(database=self.cfg.neo4j_kg_database) as s:
            # Proof summary
            r = s.run("MATCH (p:Eval_Proof) RETURN p LIMIT 1")
            rec = r.single()
            data["proof"] = dict(rec["p"]) if rec else {}

            # All verdicts with test case info
            rows = s.run("""
                MATCH (t:Eval_TestCase)-[:RECEIVED_VERDICT]->(v:Eval_Verdict)
                RETURN t.test_id AS test_id, t.category AS category,
                       t.prompt AS prompt, t.output AS output,
                       t.ground_truth AS ground_truth, t.is_correct AS is_correct,
                       v.method AS method, v.result AS result,
                       v.correct AS correct, v.explanation AS explanation,
                       v.latency_ms AS latency_ms
                ORDER BY t.test_id, v.method
            """).data()
            data["verdicts"] = rows

            # Failure modes
            rows = s.run("""
                MATCH (v:Eval_Verdict)-[:EXHIBITS]->(f:Eval_FailureMode)
                RETURN f.name AS mode, count(v) AS count
                ORDER BY count DESC
            """).data()
            data["failure_modes"] = rows

            # Categories
            rows = s.run("""
                MATCH (t:Eval_TestCase)-[:BELONGS_TO]->(c:Eval_Category)
                RETURN c.name AS category, count(t) AS count
                ORDER BY count DESC
            """).data()
            data["categories"] = rows

            # Eval runs
            rows = s.run("""
                MATCH (r:Eval_Run)
                RETURN r.run_id AS run_id, r.model AS model,
                       r.timestamp AS timestamp, r.config AS config
                ORDER BY r.timestamp DESC
            """).data()
            data["runs"] = rows

            # Graph stats
            eval_count = s.run("""
                MATCH (n) WHERE any(l IN labels(n) WHERE l STARTS WITH 'Eval_')
                RETURN count(n) AS cnt
            """).single()["cnt"]
            total_count = s.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            data["graph_stats"] = {
                "eval_nodes": eval_count,
                "total_nodes": total_count,
                "existing_nodes": total_count - eval_count,
            }

        return data

    def _fetch_qdrant_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        try:
            q = self._qdrant()
            info = q.get_collection("eval_verdicts")
            pts = info.points_count or 0
            data["collection"] = {
                "name": "eval_verdicts",
                "points": pts,
                "vectors_count": getattr(info, "vectors_count", pts),
            }
            points = q.scroll(
                "eval_verdicts", limit=20,
                with_payload=True, with_vectors=False,
            )[0]
            data["points"] = [
                {k: v for k, v in p.payload.items() if k != "explanation"}
                for p in points
            ]

            # All collections
            colls = q.get_collections().collections
            data["all_collections"] = [
                {"name": c.name} for c in colls
            ]
        except Exception as e:
            data["error"] = str(e)
        return data

    def build(self) -> None:
        ui.colors(primary=CYAN, secondary=VIOLET, accent=AMBER)
        ui.query("body").style(f"background:{BG}")

        # Header
        with ui.row().classes("w-full items-center justify-between px-4 pt-4"):
            with ui.column().classes("gap-0"):
                ui.label("LLM-as-JUDGE IS WORSE THAN NO EVALS").style(
                    f"color:{RED};font-size:22px;font-weight:800;letter-spacing:.04em"
                )
                ui.label(
                    "live proof from Neo4j graph + Qdrant vectors + Bedrock LLM judge"
                ).style(f"color:{DIM};font-size:12px")
            with ui.row().classes("gap-4 items-center"):
                ui.link("← Pipeline", "/").style(
                    f"color:{CYAN};font-size:12px;font-weight:600"
                )

        # Load data
        self.data_container = ui.column().classes("w-full px-4 gap-4")
        self.loading = ui.label("Loading from Neo4j + Qdrant...").style(
            f"color:{DIM};font-size:14px;padding:16px"
        )

        ui.timer(0.5, self._load_data, once=True)

    async def _load_data(self) -> None:
        try:
            neo4j_data = await asyncio.to_thread(self._fetch_neo4j_data)
            qdrant_data = await asyncio.to_thread(self._fetch_qdrant_data)
        except Exception as e:
            self.loading.set_text(f"Error: {e}")
            return

        self.loading.delete()
        with self.data_container:
            self._render_proof(neo4j_data, qdrant_data)

    def _render_proof(self, neo4j_data: dict, qdrant_data: dict) -> None:
        proof = neo4j_data.get("proof", {})
        verdicts = neo4j_data.get("verdicts", [])
        graph_stats = neo4j_data.get("graph_stats", {})

        # ── Thesis + Key Metric ──────────────────────────
        with ui.row().classes("w-full no-wrap gap-4"):
            with ui.column().classes("w-2/3 gap-4"):
                with self._card("The Thesis", RED):
                    ui.label(
                        proof.get("thesis", "A permissive LLM judge creates false confidence that masks real failures")
                    ).style(f"color:{INK};font-size:14px;font-weight:600")
                    ui.label(
                        "False confidence is worse than no confidence. "
                        "Without evals, you know you're unprotected — you review manually. "
                        "With a sycophantic judge that passes everything, you THINK you're covered — and ship bugs."
                    ).style(f"color:{DIM};font-size:12px;line-height:1.6")

            with ui.column().classes("w-1/3 gap-4"):
                fp_rate = proof.get("false_pass_rate_pct", 0)
                with self._card("False Pass Rate", RED):
                    ui.label(f"{fp_rate}%").style(
                        f"color:{RED};font-size:48px;font-weight:800;line-height:1"
                    )
                    ui.label("of known-bad outputs got a green check from the LLM judge").style(
                        f"color:{DIM};font-size:11px"
                    )
                    fp_count = proof.get("false_passes", 0)
                    total = proof.get("total_bad", 0)
                    ui.label(f"{fp_count} false passes out of {total} bad outputs").style(
                        f"color:{RED};font-size:10px;font-weight:600"
                    )

        # ── Cost Matrix ──────────────────────────────────
        with self._card("The Cost Matrix — Why This Matters", AMBER):
            with ui.row().classes("w-full no-wrap gap-8"):
                for title, desc, colour, icon in [
                    ("No Evals + Manual Review",
                     "You KNOW you're unprotected → you review manually → bugs found proportional to review quality",
                     GREEN, "✓"),
                    ("Deterministic Checks",
                     "Fast, free, reproducible. 0% false passes on verifiable claims. Human review on the rest.",
                     GREEN, "✓✓"),
                    (f"Naive LLM Judge ({fp_rate}% blind spots)",
                     f"Misses {fp_rate}% of subtle bugs AND removes motivation for manual review. Net: removes more protection than it adds.",
                     RED, "✗"),
                ]:
                    with ui.column().classes("flex-1 gap-1"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(icon).style(f"color:{colour};font-size:18px;font-weight:800")
                            ui.label(title).style(f"color:{colour};font-size:12px;font-weight:700")
                        ui.label(desc).style(f"color:{DIM};font-size:11px;line-height:1.5")

        # ── Verdicts Table ───────────────────────────────
        with self._card("Test Results — Every Verdict from Neo4j", VIOLET):
            # Build method comparison
            methods: dict[str, dict] = {}
            for v in verdicts:
                m = v["method"]
                if m not in methods:
                    methods[m] = {"total": 0, "correct": 0, "false_passes": 0, "total_latency": 0}
                methods[m]["total"] += 1
                if v["correct"]:
                    methods[m]["correct"] += 1
                elif v["result"] == "PASS" and not v.get("is_correct", True):
                    methods[m]["false_passes"] += 1
                methods[m]["total_latency"] += (v.get("latency_ms") or 0)

            with ui.row().classes("w-full gap-6"):
                for method, stats in methods.items():
                    acc = round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] else 0
                    avg_ms = round(stats["total_latency"] / stats["total"]) if stats["total"] else 0
                    is_det = "deterministic" in method
                    colour = GREEN if is_det else (AMBER if acc >= 80 else RED)
                    with ui.column().classes("gap-0"):
                        ui.label(method).style(f"color:{colour};font-size:11px;font-weight:700")
                        ui.label(f"{acc}% accuracy").style(f"color:{colour};font-size:20px;font-weight:800")
                        ui.label(f"{stats['false_passes']} false passes · {avg_ms}ms avg").style(
                            f"color:{DIM};font-size:10px"
                        )

            # Detailed table
            table_rows = []
            seen = set()
            for v in verdicts:
                key = f"{v['test_id']}:{v['method']}"
                if key in seen:
                    continue
                seen.add(key)
                table_rows.append({
                    "test_id": v["test_id"],
                    "category": v["category"] or "",
                    "method": v["method"],
                    "verdict": v["result"],
                    "correct": "✓" if v["correct"] else "✗",
                    "prompt": (v.get("prompt") or "")[:60],
                })

            ui.table(
                columns=[
                    {"name": "test_id", "label": "Test", "field": "test_id", "align": "left"},
                    {"name": "category", "label": "Category", "field": "category", "align": "left"},
                    {"name": "method", "label": "Method", "field": "method", "align": "left"},
                    {"name": "verdict", "label": "Verdict", "field": "verdict"},
                    {"name": "correct", "label": "Correct?", "field": "correct"},
                    {"name": "prompt", "label": "Prompt", "field": "prompt", "align": "left"},
                ],
                rows=table_rows,
                row_key="test_id",
            ).classes("w-full").props("flat dense")

        # ── Failure Patterns Chart ───────────────────────
        with ui.row().classes("w-full no-wrap gap-4"):
            with ui.column().classes("w-1/2 gap-4"):
                with self._card("Failure Modes (Neo4j)", RED):
                    modes = neo4j_data.get("failure_modes", [])
                    if modes:
                        chart_data = [{"value": m["count"], "name": m["mode"]} for m in modes]
                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "item"},
                            "series": [{
                                "type": "pie",
                                "radius": ["40%", "70%"],
                                "avoidLabelOverlap": True,
                                "itemStyle": {"borderRadius": 6, "borderColor": "#fff", "borderWidth": 2},
                                "label": {"show": True, "fontSize": 10, "color": INK},
                                "data": chart_data,
                                "color": [RED, AMBER, VIOLET, CYAN, DIM],
                            }],
                        }).style("height:240px")
                    else:
                        ui.label("No failure modes recorded").style(f"color:{DIM}")

            with ui.column().classes("w-1/2 gap-4"):
                with self._card("Categories by False Pass Rate", AMBER):
                    # Compute per-category false pass rates
                    cat_stats: dict[str, dict] = {}
                    for v in verdicts:
                        cat = v.get("category", "unknown")
                        if "llm_judge" not in (v.get("method") or ""):
                            continue
                        if cat not in cat_stats:
                            cat_stats[cat] = {"total": 0, "false_passes": 0}
                        if not v.get("is_correct", True):
                            cat_stats[cat]["total"] += 1
                            if not v["correct"]:
                                cat_stats[cat]["false_passes"] += 1

                    bar_cats = []
                    bar_vals = []
                    bar_colors = []
                    for cat, st in sorted(cat_stats.items(), key=lambda x: -x[1]["false_passes"]):
                        if st["total"] == 0:
                            continue
                        rate = round(st["false_passes"] / st["total"] * 100)
                        bar_cats.append(cat[:25])
                        bar_vals.append(rate)
                        bar_colors.append(RED if rate > 50 else AMBER if rate > 0 else GREEN)

                    if bar_cats:
                        ui.echart({
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "axis"},
                            "grid": {"left": "30%", "right": "10%", "top": 10, "bottom": 10},
                            "xAxis": {"type": "value", "max": 100,
                                      "axisLabel": {"formatter": "{value}%", "color": DIM, "fontSize": 10}},
                            "yAxis": {"type": "category", "data": bar_cats,
                                      "axisLabel": {"color": INK, "fontSize": 10}},
                            "series": [{
                                "type": "bar",
                                "data": [{"value": v, "itemStyle": {"color": c}}
                                         for v, c in zip(bar_vals, bar_colors)],
                                "barWidth": 16,
                            }],
                        }).style("height:240px")
                    else:
                        ui.label("No category data").style(f"color:{DIM}")

        # ── Qdrant + Neo4j Infrastructure ────────────────
        with ui.row().classes("w-full no-wrap gap-4"):
            with ui.column().classes("w-1/2 gap-4"):
                with self._card("Qdrant Vector Store", CYAN):
                    qd = qdrant_data
                    if "error" in qd:
                        ui.label(f"Error: {qd['error']}").style(f"color:{RED};font-size:11px")
                    else:
                        coll = qd.get("collection", {})
                        ui.label(f"Collection: {coll.get('name', '?')}").style(
                            f"color:{CYAN};font-size:13px;font-weight:600"
                        )
                        ui.label(f"{coll.get('points', 0)} eval points embedded (1024-dim, Titan v2)").style(
                            f"color:{DIM};font-size:11px"
                        )
                        all_colls = qd.get("all_collections", [])
                        ui.label(f"All collections: {', '.join(c['name'] for c in all_colls)}").style(
                            f"color:{DIM};font-size:10px"
                        )

                        # Show false-pass points
                        false_pass_points = [p for p in qd.get("points", [])
                                             if not p.get("judge_correct") and not p.get("is_correct")]
                        if false_pass_points:
                            ui.label("Embedded false passes:").style(
                                f"color:{RED};font-size:10px;font-weight:600;margin-top:6px"
                            )
                            for p in false_pass_points:
                                ui.label(
                                    f"· {p.get('test_id', '?')} ({p.get('category', '?')}) "
                                    f"— judge said {p.get('judge_verdict', '?')}"
                                ).style(f"color:{DIM};font-size:10px;font-family:monospace")

            with ui.column().classes("w-1/2 gap-4"):
                with self._card("Neo4j Knowledge Graph", VIOLET):
                    gs = graph_stats
                    ui.label(f"{gs.get('total_nodes', 0)} total nodes").style(
                        f"color:{VIOLET};font-size:20px;font-weight:800"
                    )
                    ui.label(
                        f"{gs.get('eval_nodes', 0)} eval nodes + "
                        f"{gs.get('existing_nodes', 0)} existing (pharma KG + context layer)"
                    ).style(f"color:{DIM};font-size:11px")

                    # Show eval node breakdown
                    runs = neo4j_data.get("runs", [])
                    cats = neo4j_data.get("categories", [])
                    ui.label(f"Eval runs: {len(runs)}").style(
                        f"color:{CYAN};font-size:11px;margin-top:6px"
                    )
                    for r in runs[:3]:
                        ui.label(f"· {r['run_id']} — {r['model']}").style(
                            f"color:{DIM};font-size:10px;font-family:monospace"
                        )
                    ui.label(f"Categories: {', '.join(c['category'] for c in cats)}").style(
                        f"color:{DIM};font-size:10px;margin-top:4px"
                    )

        # ── What To Do Instead ───────────────────────────
        with self._card("What To Do Instead", GREEN):
            with ui.row().classes("w-full no-wrap gap-6"):
                for num, title, desc in [
                    ("1", "Deterministic checks first",
                     "Exact match, regex, JSON parse, code execution, unit tests. Fast, free, reliable."),
                    ("2", "LLM judge as TRIAGE not GATE",
                     "Use it to flag items for human review, never to auto-approve."),
                    ("3", "Calibrate before trusting",
                     "Run the judge against known-bad outputs and measure its false-pass rate FIRST."),
                    ("4", "Multiple judges + disagreement",
                     "If two judges disagree, escalate to human — don't average the scores."),
                ]:
                    with ui.column().classes("flex-1 gap-1"):
                        ui.label(num).style(
                            f"color:{GREEN};font-size:24px;font-weight:800;line-height:1"
                        )
                        ui.label(title).style(f"color:{GREEN};font-size:12px;font-weight:700")
                        ui.label(desc).style(f"color:{DIM};font-size:11px;line-height:1.5")

        # ── Run Live Test Button ─────────────────────────
        with self._card("Run Live Eval", CYAN):
            ui.label(
                "Re-run the LLM judge against the test suite and update Neo4j + Qdrant in real time."
            ).style(f"color:{DIM};font-size:11px")
            self.run_status = ui.label("").style(f"color:{CYAN};font-size:11px;font-family:monospace")
            ui.button("RUN EVAL NOW", on_click=self._run_live_eval).props(
                "unelevated"
            ).style(f"background:{CYAN};color:{BG};font-weight:700")

    async def _run_live_eval(self) -> None:
        self.run_status.set_text("Running eval pipeline...")
        try:
            import subprocess
            result = await asyncio.to_thread(
                subprocess.run,
                ["/usr/bin/python3",
                 str(ROOT.parent / "eval_full_pipeline.py")],
                capture_output=True, text=True, timeout=300,
                cwd=str(ROOT.parent),
                env={**os.environ, "PYTHONPATH": str(ROOT.parent)},
            )
            if result.returncode == 0:
                self.run_status.set_text("Done! Refresh the page to see updated results.")
                ui.notify("Eval complete — refresh to see new data", type="positive")
            else:
                self.run_status.set_text(f"Error: {result.stderr[-200:]}")
                ui.notify("Eval failed", type="negative")
        except Exception as e:
            self.run_status.set_text(f"Error: {e}")

    def close(self) -> None:
        if self._neo4j_driver:
            self._neo4j_driver.close()
        self._neo4j_driver = None
        self._qdrant_client = None
