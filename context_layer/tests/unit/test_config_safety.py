"""Configuration safety: secrets must never surface in logs or reports (R10.5)."""

from __future__ import annotations

import config as config_module

SECRET = "s3cret-value-never-logged"


def test_secret_repr_is_redacted():
    s = config_module.Secret(SECRET)
    assert SECRET not in repr(s)
    assert SECRET not in f"{s!r}"
    assert s.fingerprint() in repr(s)


def test_secret_still_usable_as_a_string():
    s = config_module.Secret(SECRET)
    assert str(s) == SECRET  # clients need the real value


def test_unset_secret_fingerprints_as_unset():
    assert config_module.Secret("").fingerprint() == "unset"


def test_fingerprint_is_stable_and_distinguishing():
    a, b = config_module.Secret("aaa"), config_module.Secret("bbb")
    assert a.fingerprint() == config_module.Secret("aaa").fingerprint()
    assert a.fingerprint() != b.fingerprint()


def test_report_never_contains_secret_values(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", SECRET)
    monkeypatch.setenv("OPENSEARCH_PASSWORD", SECRET)
    report = config_module.Config().report()
    assert SECRET not in report
    assert "set [" in report


def test_check_flags_missing_required_settings(monkeypatch):
    for var in ("NEO4J_URI", "NEO4J_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    problems = config_module.Config().check()
    assert any("NEO4J_URI" in p for p in problems)
    assert any("NEO4J_PASSWORD" in p for p in problems)


def test_neo4j_backend_does_not_require_opensearch(monkeypatch):
    """With the Neo4j-native vector backend, OpenSearch settings are irrelevant."""
    monkeypatch.setenv("VECTOR_BACKEND", "neo4j")
    monkeypatch.delenv("OPENSEARCH_URL", raising=False)
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://x")
    monkeypatch.setenv("NEO4J_PASSWORD", "p")
    assert config_module.Config().check() == []


def test_opensearch_backend_requires_its_settings(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "opensearch")
    monkeypatch.delenv("OPENSEARCH_URL", raising=False)
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://x")
    monkeypatch.setenv("NEO4J_PASSWORD", "p")
    problems = config_module.Config().check()
    assert any("requires OPENSEARCH_URL" in p for p in problems)


def test_check_rejects_unsupported_backend(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")
    assert any("is not supported" in p for p in config_module.Config().check())


def test_check_rejects_shared_chunk_and_term_index(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "opensearch")
    monkeypatch.setenv("OPENSEARCH_URL", "https://x")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "p")
    monkeypatch.setenv("OPENSEARCH_CHUNK_INDEX", "same")
    monkeypatch.setenv("OPENSEARCH_TERM_INDEX", "same")
    assert any("must differ" in p for p in config_module.Config().check())


def test_check_rejects_unisolated_runtime_context(monkeypatch):
    monkeypatch.setenv("NEO4J_KG_DATABASE", "shared")
    monkeypatch.setenv("NEO4J_CTX_DATABASE", "shared")
    monkeypatch.setenv("CTX_LABEL_PREFIX", "")
    problems = config_module.Config().check()
    assert any("share a database with the curated KG" in p for p in problems)


def test_check_warns_on_disabled_cert_verification(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "opensearch")
    monkeypatch.setenv("OPENSEARCH_URL", "https://x")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "p")
    monkeypatch.setenv("OPENSEARCH_VERIFY_CERTS", "false")
    assert any("VERIFY_CERTS" in p for p in config_module.Config().check())


def test_defaults_target_qdrant_and_bedrock(monkeypatch):
    """Qdrant for chunk vectors (94ms vs Neo4j 222ms from this host); Bedrock
    for embeddings and reasoning. Both verified live."""
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    cfg = config_module.Config()
    assert cfg.vector_backend == "qdrant"
    assert cfg.embedding_provider == "bedrock"
    assert cfg.llm_provider == "bedrock"
    assert cfg.clinical_advice_guard is True  # research-only by default


def test_qdrant_backend_requires_url_and_key(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    problems = config_module.Config().check()
    assert any("requires QDRANT_URL" in p for p in problems)
    assert any("requires QDRANT_API_KEY" in p for p in problems)


def test_check_rejects_the_corrupted_idp_collection(monkeypatch):
    """Guard against pointing at the collection whose payloads hold vectors."""
    monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "https://x")
    monkeypatch.setenv("QDRANT_API_KEY", "k")
    monkeypatch.setenv("QDRANT_COLLECTION", "idp_chunks")
    problems = config_module.Config().check()
    assert any("corrupted payloads" in p for p in problems)


# ── .env precedence: non-empty values win over the ambient environment ────────


def test_dotenv_overrides_ambient_environment(monkeypatch, tmp_path, capsys):
    """An ambient AWS_REGION from an unrelated account must not silently win."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    env = tmp_path / ".env"
    env.write_text("AWS_REGION=us-west-2\n")
    env.chmod(0o600)
    config_module._load_dotenv(env)
    assert config_module.Config().aws_region == "us-west-2"
    assert "overrode ambient AWS_REGION" in capsys.readouterr().out


def test_empty_dotenv_value_defers_to_environment(monkeypatch, tmp_path):
    """Blank entries must not clobber role- or container-injected values."""
    monkeypatch.setenv("OPENSEARCH_URL", "https://injected.example")
    env = tmp_path / ".env"
    env.write_text("OPENSEARCH_URL=\n")
    env.chmod(0o600)
    config_module._load_dotenv(env)
    assert config_module.Config().opensearch_url == "https://injected.example"


def test_dotenv_warns_on_loose_permissions(tmp_path, capsys):
    env = tmp_path / ".env"
    env.write_text("AWS_REGION=us-west-2\n")
    env.chmod(0o644)
    config_module._load_dotenv(env)
    assert "group/world accessible" in capsys.readouterr().out
