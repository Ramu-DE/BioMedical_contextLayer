#!/usr/bin/env python3
"""Build the vocabulary term index and verify entity linking. Tasks 2.1-2.2.

Usage:
    .venv/bin/python scripts/build_term_index.py            # build
    .venv/bin/python scripts/build_term_index.py --verify   # test linking only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from context_layer.clients.embeddings import BedrockEmbeddings  # noqa: E402
from context_layer.linking.entity_linker import EntityLinker  # noqa: E402
from context_layer.linking.term_index import TermIndex  # noqa: E402
from context_layer.types import Provenance, TextChunk, utcnow  # noqa: E402

# Pairs that must resolve to the same CURIE (R4.3), and pairs that must not.
SYNONYM_PROBES = [
    ("Entyvio", "vedolizumab", True),
    ("Vyvanse", "lisdexamfetamine", True),
    ("Ninlaro", "ixazomib", True),
    ("Pembrolizumab", "Nivolumab", False),
    ("Crohn's disease", "Ulcerative colitis", False),
]

SAMPLE_TEXT = (
    "Pembrolizumab is a PD-1 inhibitor approved for Non-Small Cell Lung Cancer "
    "and Melanoma. Entyvio (vedolizumab) is indicated for Ulcerative Colitis. "
    "Immune-related pneumonitis was reported as a severe adverse event. "
    "The Quarterly Revenue Dashboard is unrelated to any clinical concept."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="skip build, test only")
    args = ap.parse_args()

    cfg = config_module.load()
    if problems := cfg.check():
        for p in problems:
            print(f"  - {p}")
        return 1

    embedder = BedrockEmbeddings(cfg)
    with TermIndex(cfg, embedder=embedder) as index:
        if not args.verify:
            terms = index.collect_terms()
            print(f"collected {len(terms)} distinct (curie, surface) terms")
            by_type: dict[str, int] = {}
            for t in terms:
                by_type[t.entity_type] = by_type.get(t.entity_type, 0) + 1
            for k, v in sorted(by_type.items()):
                print(f"  {v:>4}  {k}")
            print(f"\nembedding and indexing into {index.index_name!r}...")
            print(f"indexed {index.build(terms)} terms; index holds {index.count()}")
            print("\nbuilding concept crosswalk (SAME_AS)...")
            for k, v in index.build_crosswalk().items():
                print(f"  {k}: {v}")
        else:
            print(f"index {index.index_name!r} holds {index.count()} terms")

        linker = EntityLinker(cfg, term_index=index)

        print(f"\n═══ synonym resolution (threshold {linker.threshold}) ═══")
        failures = []
        for a, b, should_match in SYNONYM_PROBES:
            ca, cb = linker.canonical(a), linker.canonical(b)
            matched = ca is not None and ca == cb
            ok = matched == should_match
            print(f"  {'ok  ' if ok else 'FAIL'} {a!r} vs {b!r}: "
                  f"{ca} / {cb} -> {'same' if matched else 'different'}"
                  f" (expected {'same' if should_match else 'different'})")
            if not ok:
                failures.append((a, b))

        print("\n═══ linking a passage ═══")
        chunk = TextChunk(
            chunk_id="probe-1",
            text=SAMPLE_TEXT,
            provenance=Provenance(source="probe", ingested_at=utcnow()),
        )
        mentions = linker.link(chunk)
        for m in mentions:
            state = f"-> {m.curie}" if m.is_linked else "-> unlinked"
            print(f"  {m.confidence:.3f}  {m.surface_form!r:<38} {state}")
        s = linker.stats
        print(f"\n  considered={s.considered} linked={s.linked} "
              f"unlinked={s.unlinked} link_rate={s.link_rate:.0%}")

        written = linker.persist(mentions)
        print(f"  persisted {written} MENTIONS edges")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
