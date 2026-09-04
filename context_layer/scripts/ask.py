#!/usr/bin/env python3
"""Run the neurosymbolic context layer against live infrastructure.

Usage:
    .venv/bin/python scripts/ask.py "your question"
    .venv/bin/python scripts/ask.py --demo        # the scripted proof set
    .venv/bin/python scripts/ask.py --json "..."  # raw GovernedResponse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from context_layer.agent.pipeline import ContextLayerAgent, Trace  # noqa: E402

DEMO_QUESTIONS = [
    # 1. Symbolic override: consent withdrawn for PAT015.
    "Can I include patient PAT015 in the trial outcome analysis?",
    # 2. Grounded answer with advisories.
    "Which drugs target PD-1 and what diseases do they treat?",
    # 3. Clinical advice guard.
    "What dose of pembrolizumab should I give my patient?",
    # 4. Governance policy question, answered from the graph.
    "What policy applies to hub layer tables and are we compliant?",
    # 5. Refusal: nothing in the corpus supports this.
    "What is the market share of our veterinary division in Brazil?",
]


def show(trace: Trace) -> None:
    r = trace.response
    print(f"  {trace.summary()}")
    print(f"  package: {r.context_package_id}")
    if r.degradations:
        print(f"  degradations: {list(r.degradations)}")
    print()
    if r.refused:
        print(f"  REFUSED: {r.refusal_reason}")
    else:
        print(f"  ANSWER: {r.answer}")
    if r.neural_proposal and r.neural_proposal != r.answer:
        print(f"\n  NEURAL PROPOSAL (overridden):\n    {r.neural_proposal[:400]}")
    print(f"\n  SYMBOLIC VERDICT: {r.symbolic_verdict}")
    for f in r.rules_fired:
        tag = {"block": "BLOCK", "warn": "WARN", "inform": "INFO"}[f.severity]
        print(f"    [{tag}] {f.rule_id}"
              + (f"  policy={f.policy_iri}" if f.policy_iri else ""))
    print(f"  GROUNDING: {r.grounding_confidence}")
    if r.citations:
        print("  CITATIONS:")
        for c in r.citations[:4]:
            print(f"    - {c.curie or c.chunk_id}  ({c.source})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = config_module.load()
    if problems := cfg.check():
        for p in problems:
            print(f"  - {p}")
        return 1

    questions = DEMO_QUESTIONS if args.demo else [args.question]
    if not questions or not questions[0]:
        ap.error("supply a question or --demo")

    with ContextLayerAgent(cfg) as agent:
        print(f"loaded {len(agent.rules)} rules\n")
        for i, q in enumerate(questions, 1):
            print("═" * 78)
            print(f"[{i}] {q}")
            print("═" * 78)
            trace = agent.run(q)
            if args.json:
                print(json.dumps(trace.response.to_dict(), indent=2, default=str))
            else:
                show(trace)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
