"""Vector store payload guards.

The pre-existing ``idp_chunks`` collection stored stringified embedding vectors
in its ``chunk_text`` payload across all 524 points, which silently destroys
retrieval quality. VectorRecord refuses to represent that state.
"""

from __future__ import annotations

import pytest

from context_layer.clients.vector_store import (
    SOURCE_KEY,
    VectorHit,
    VectorRecord,
    build_vector_store,
)

GOOD_PAYLOAD = {SOURCE_KEY: "neo4j:Policy", "ingested_at": "2026-08-27T00:00:00Z"}
VEC = [0.1] * 8


def test_accepts_real_source_text():
    r = VectorRecord(
        text="Policy Hub tables require DQ: all hub layer tables need a critical rule",
        vector=VEC,
        payload=GOOD_PAYLOAD,
    )
    assert r.text.startswith("Policy Hub")


def test_rejects_stringified_vector_as_text():
    """This is exactly the idp_chunks defect."""
    corrupted = "0.020671460777521133,\n-0.032740384340286255,\n-0.05982661247253418"
    with pytest.raises(ValueError, match="looks like a stringified vector"):
        VectorRecord(text=corrupted, vector=VEC, payload=GOOD_PAYLOAD)


def test_rejects_empty_text():
    with pytest.raises(ValueError, match="non-empty source text"):
        VectorRecord(text="   ", vector=VEC, payload=GOOD_PAYLOAD)


def test_requires_provenance_source():
    with pytest.raises(ValueError, match="requires a 'source'"):
        VectorRecord(text="real text here", vector=VEC, payload={"ingested_at": "x"})


def test_numeric_looking_but_legitimate_text_is_allowed():
    """A sentence starting with digits must not be mistaken for a vector."""
    r = VectorRecord(
        text="2026 revenue rose after the policy change was adopted.",
        vector=VEC,
        payload=GOOD_PAYLOAD,
    )
    assert r.text.startswith("2026 revenue")


def test_hit_exposes_curies_and_source():
    h = VectorHit(
        point_id="p1",
        score=0.9,
        text="Policy PII must be masked",
        payload={SOURCE_KEY: "neo4j:Policy", "curies": ["POL_PII_MASK"]},
    )
    assert h.curies == ("POL_PII_MASK",)
    assert h.source == "neo4j:Policy"


def test_hit_tolerates_absent_optional_payload():
    h = VectorHit(point_id="p1", score=0.1, text="t", payload={})
    assert h.curies == ()
    assert h.source == ""


def test_factory_rejects_unimplemented_backend(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "opensearch")
    import config as config_module

    with pytest.raises(NotImplementedError, match="has no client yet"):
        build_vector_store(config_module.Config())
