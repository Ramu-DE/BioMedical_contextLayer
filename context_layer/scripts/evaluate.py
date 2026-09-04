#!/usr/bin/env python3
"""Run the evaluation: context layer vs naive RAG. Requirement 11.

Usage:
    .venv/bin/python scripts/evaluate.py
    .venv/bin/python scripts/evaluate.py --out eval_report.json
    .venv/bin/python scripts/evaluate.py --category consent_governance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from context_layer.agent.pipeline import ContextLayerAgent  # noqa: E402
from context_layer.eval.baseline import NaiveRAG  # noqa: E402
from context_layer.eval.harness import Harness, write_report  # noqa: E402
from context_layer.eval.questions import QUESTIONS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_report.json")
    ap.add_argument("--category")
    args = ap.parse_args()

    cfg = config_module.load()
    if problems := cfg.check():
        for p in problems:
            print(f"  - {p}")
        return 1

    questions = QUESTIONS
    if args.category:
        questions = tuple(q for q in QUESTIONS if q.category == args.category)
    if not questions:
        print("no questions matched")
        return 1

    print(f"evaluating {len(questions)} questions x 2 arms\n")

    with ContextLayerAgent(cfg, write_back=False) as agent:
        harness = Harness(cfg, agent, NaiveRAG(cfg))
        report = harness.run(questions)

    print("═" * 84)
    print(f"{'metric':<26}{'context_layer':>18}{'naive_rag':>18}")
    print("═" * 84)
    cl = report["summary"]["context_layer"]
    nv = report["summary"]["naive_rag"]
    for key in (
        "behaviour_correct", "groundedness_mean", "citation_coverage",
        "rule_compliance", "expected_rules_hit", "confabulation_rate",
        "forbidden_mentions", "errors",
    ):
        a, b = cl.get(key), nv.get(key)
        fmt = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"{key:<26}{fmt(a):>18}{fmt(b):>18}")
    print("═" * 84)

    print("\nper-question outcome (expected -> context_layer / naive_rag)")
    by_q: dict[str, dict] = {}
    for r in report["results"]:
        by_q.setdefault(r["question"], {})[r["arm"]] = r
    for q, arms in by_q.items():
        cl_r, nv_r = arms.get("context_layer", {}), arms.get("naive_rag", {})
        mark = "ok  " if cl_r.get("behaviour_correct") else "FAIL"
        nmark = "ok  " if nv_r.get("behaviour_correct") else "FAIL"
        print(f"  [{cl_r.get('expected','?'):>6}] {mark}/{nmark}  {q[:62]}")

    write_report(report, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
