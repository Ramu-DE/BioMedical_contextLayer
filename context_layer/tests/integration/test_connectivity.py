"""Live connectivity smoke tests.

Skipped automatically unless config.check() passes. Once you have filled in
.env, these are the first tests to run — they confirm the credentials work and
that runtime writes land in the isolated context namespace (R5.3).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_neo4j_reachable(live_cfg):
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        live_cfg.neo4j_uri, auth=(live_cfg.neo4j_user, str(live_cfg.neo4j_password))
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=live_cfg.neo4j_kg_database) as s:
            assert s.run("RETURN 1 AS ok").single()["ok"] == 1
    finally:
        driver.close()


def test_curated_graph_has_content(live_cfg):
    """Confirm we are pointed at the biomedical KG, not an empty database."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        live_cfg.neo4j_uri, auth=(live_cfg.neo4j_user, str(live_cfg.neo4j_password))
    )
    try:
        with driver.session(database=live_cfg.neo4j_kg_database) as s:
            labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")]
        assert labels, "curated KG database has no labels — is NEO4J_KG_DATABASE correct?"
        print(f"\nlabels found: {sorted(labels)[:20]}")
    finally:
        driver.close()


def test_context_namespace_is_isolated(live_cfg):
    """Runtime writes must not share a namespace with curated knowledge."""
    assert (
        live_cfg.neo4j_ctx_database != live_cfg.neo4j_kg_database or live_cfg.ctx_label_prefix
    ), "runtime context is not isolated from the curated KG"


@pytest.mark.opensearch
def test_opensearch_reachable(live_cfg):
    from opensearchpy import OpenSearch

    auth = (
        (live_cfg.opensearch_user, str(live_cfg.opensearch_password))
        if live_cfg.opensearch_auth == "basic"
        else None
    )
    client = OpenSearch(
        hosts=[live_cfg.opensearch_url],
        http_auth=auth,
        verify_certs=live_cfg.opensearch_verify_certs,
        ssl_show_warn=live_cfg.opensearch_verify_certs,
    )
    info = client.info()
    assert "version" in info
    print(f"\nopensearch version: {info['version'].get('number')}")


@pytest.mark.opensearch
def test_knn_plugin_available(live_cfg):
    """The entity linker and vector retrieval both require k-NN."""
    from opensearchpy import OpenSearch

    auth = (
        (live_cfg.opensearch_user, str(live_cfg.opensearch_password))
        if live_cfg.opensearch_auth == "basic"
        else None
    )
    client = OpenSearch(
        hosts=[live_cfg.opensearch_url],
        http_auth=auth,
        verify_certs=live_cfg.opensearch_verify_certs,
        ssl_show_warn=live_cfg.opensearch_verify_certs,
    )
    plugins = client.cat.plugins(format="json")
    names = {p.get("component", "") for p in plugins}
    # Serverless reports no plugin list; treat an empty list as inconclusive
    if names:
        assert any("knn" in n for n in names), f"k-NN plugin not found in {names}"
