"""Shared test fixtures.

Unit and property tests must pass with no credentials configured. Integration
tests are skipped unless a usable configuration is present.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from context_layer.types import (  # noqa: E402
    ContextElement,
    ContextPackage,
    Provenance,
    utcnow,
)


@pytest.fixture(scope="session")
def live_cfg():
    """Configuration with .env applied — for integration tests."""
    return config_module.load()


@pytest.fixture
def cfg():
    return config_module.Config()


def pytest_collection_modifyitems(config, items):
    """Skip integration tests that cannot run in this environment.

    Loads .env first, otherwise every integration test skips even when
    credentials are present.
    """
    cfg = config_module.load()
    problems = cfg.check()

    skip_no_creds = pytest.mark.skip(
        reason=f"config not ready ({len(problems)} problems)"
    )
    skip_not_backend = pytest.mark.skip(
        reason=f"VECTOR_BACKEND={cfg.vector_backend} — OpenSearch not in use"
    )

    for item in items:
        if "integration" not in item.keywords:
            continue
        if problems:
            item.add_marker(skip_no_creds)
        elif "opensearch" in item.keywords and cfg.vector_backend != "opensearch":
            item.add_marker(skip_not_backend)


@pytest.fixture
def provenance():
    return Provenance(
        source="test://fixture",
        ingested_at=utcnow() - timedelta(days=1),
        confidence=0.95,
        record_id="rec-1",
    )


@pytest.fixture
def element(provenance):
    def _make(
        content: str = "Aspirin inhibits COX-1.",
        kind: str = "chunk",
        curies: tuple[str, ...] = ("CHEBI:15365",),
        score: float = 0.9,
        element_id: str = "el-1",
    ) -> ContextElement:
        return ContextElement(
            element_id=element_id,
            kind=kind,  # type: ignore[arg-type]
            content=content,
            provenance=provenance,
            score=score,
            curies=curies,
        )

    return _make


@pytest.fixture
def package(element):
    return ContextPackage(
        context_package_id="ctx_test",
        question="How does aspirin act on COX-1?",
        elements=(
            element(),
            element(
                content="COX-1 is a prostaglandin synthase.",
                kind="graph_node",
                curies=("HGNC:9604",),
                element_id="el-2",
            ),
        ),
    )

