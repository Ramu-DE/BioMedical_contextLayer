"""Configuration for the context layer.

Loads settings from the environment (or a local .env) and validates that
required credentials are present. Secret values are never printed: __repr__
and the check() report show only whether a value is set, plus a short
fingerprint so you can confirm *which* credential is loaded without
revealing it.

Usage:
    python config.py            # prints a redacted readiness report
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ENV_PATH = Path(__file__).with_name(".env")


def _load_dotenv(path: Path = ENV_PATH) -> None:
    """Load .env.

    Precedence: a **non-empty** value in .env wins over the ambient
    environment, and any key it overrides is reported by name (never by value).
    This is deliberate — the ambient shell may carry unrelated values such as an
    AWS_REGION belonging to a different account, and silently inheriting those
    sends requests to the wrong place. Empty entries in .env defer to the
    environment so that container- or role-injected values still work.
    """
    if not path.is_file():
        return
    mode = path.stat().st_mode & 0o077
    if mode:
        print(f"warning: {path} is group/world accessible — run: chmod 600 {path}")

    overridden: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not value:
            continue  # defer to the environment
        if key in os.environ and os.environ[key] != value:
            overridden.append(key)
        os.environ[key] = value

    if overridden:
        print(f"note: .env overrode ambient {', '.join(sorted(overridden))}")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


class Secret(str):
    """A string that refuses to reveal itself in logs or tracebacks."""

    def fingerprint(self) -> str:
        if not self:
            return "unset"
        return hashlib.sha256(self.encode()).hexdigest()[:8]

    def __repr__(self) -> str:  # noqa: D105
        return f"<Secret {self.fingerprint()}>"

    def __str__(self) -> str:  # noqa: D105
        # str() is used by clients that need the real value; keep it working
        # but ensure f-string debugging (!r) and repr() stay redacted.
        return super().__str__()


@dataclass(frozen=True)
class Config:
    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: Secret = field(
        default_factory=lambda: Secret(_env("NEO4J_PASSWORD"))
    )
    neo4j_kg_database: str = field(
        default_factory=lambda: _env("NEO4J_KG_DATABASE", "neo4j")
    )
    neo4j_ctx_database: str = field(
        default_factory=lambda: _env("NEO4J_CTX_DATABASE", "context")
    )
    ctx_label_prefix: str = field(
        default_factory=lambda: _env("CTX_LABEL_PREFIX", "Ctx_")
    )

    # Vector backend: neo4j | qdrant | opensearch
    vector_backend: str = field(
        default_factory=lambda: _env("VECTOR_BACKEND", "qdrant")
    )
    # Qdrant (chunk vectors + text payload)
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL"))
    qdrant_api_key: Secret = field(
        default_factory=lambda: Secret(_env("QDRANT_API_KEY"))
    )
    qdrant_collection: str = field(
        default_factory=lambda: _env("QDRANT_COLLECTION", "ctx_gov_chunks")
    )
    qdrant_timeout: int = field(
        default_factory=lambda: int(_env("QDRANT_TIMEOUT", "30"))
    )
    # Neo4j-native vector indexes (term index always lives here)
    neo4j_chunk_index: str = field(
        default_factory=lambda: _env("NEO4J_CHUNK_INDEX", "ctx_chunk_embedding")
    )
    neo4j_term_index: str = field(
        default_factory=lambda: _env("NEO4J_TERM_INDEX", "ctx_term_embedding")
    )
    neo4j_fulltext_index: str = field(
        default_factory=lambda: _env("NEO4J_FULLTEXT_INDEX", "ctx_chunk_fulltext")
    )
    vector_similarity: str = field(
        default_factory=lambda: _env("VECTOR_SIMILARITY", "cosine")
    )
    opensearch_url: str = field(default_factory=lambda: _env("OPENSEARCH_URL"))
    opensearch_auth: str = field(
        default_factory=lambda: _env("OPENSEARCH_AUTH", "basic")
    )
    opensearch_user: str = field(default_factory=lambda: _env("OPENSEARCH_USER"))
    opensearch_password: Secret = field(
        default_factory=lambda: Secret(_env("OPENSEARCH_PASSWORD"))
    )
    opensearch_verify_certs: bool = field(
        default_factory=lambda: _env("OPENSEARCH_VERIFY_CERTS", "true").lower()
        == "true"
    )
    opensearch_chunk_index: str = field(
        default_factory=lambda: _env("OPENSEARCH_CHUNK_INDEX", "biomed_chunks")
    )
    opensearch_term_index: str = field(
        default_factory=lambda: _env("OPENSEARCH_TERM_INDEX", "biomed_ontology_terms")
    )
    opensearch_knn_m: int = field(
        default_factory=lambda: int(_env("OPENSEARCH_KNN_M", "16"))
    )
    opensearch_knn_ef_construction: int = field(
        default_factory=lambda: int(_env("OPENSEARCH_KNN_EF_CONSTRUCTION", "128"))
    )

    # Embeddings
    embedding_provider: str = field(
        default_factory=lambda: _env("EMBEDDING_PROVIDER", "bedrock")
    )
    embedding_model_id: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
        )
    )
    embedding_dim: int = field(
        default_factory=lambda: int(_env("EMBEDDING_DIM", "1024"))
    )
    embedding_api_key: Secret = field(
        default_factory=lambda: Secret(_env("EMBEDDING_API_KEY"))
    )

    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "bedrock"))
    bedrock_model_id: str = field(
        default_factory=lambda: _env("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    )
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-west-2"))
    aws_profile: str = field(default_factory=lambda: _env("AWS_PROFILE", ""))
    llm_api_key: Secret = field(default_factory=lambda: Secret(_env("LLM_API_KEY")))

    # Governance
    min_grounding_confidence: float = field(
        default_factory=lambda: float(_env("MIN_GROUNDING_CONFIDENCE", "0.6"))
    )
    clinical_advice_guard: bool = field(
        default_factory=lambda: _env("CLINICAL_ADVICE_GUARD", "true").lower() == "true"
    )

    # Context assembly
    entity_link_threshold: float = field(
        default_factory=lambda: float(_env("ENTITY_LINK_THRESHOLD", "0.72"))
    )
    context_token_budget: int = field(
        default_factory=lambda: int(_env("CONTEXT_TOKEN_BUDGET", "8000"))
    )
    prior_decision_k: int = field(
        default_factory=lambda: int(_env("PRIOR_DECISION_K", "5"))
    )

    def check(self) -> list[str]:
        """Return a list of human-readable problems; empty means ready."""
        problems: list[str] = []
        if not self.neo4j_uri:
            problems.append("NEO4J_URI is not set")
        if not self.neo4j_password:
            problems.append("NEO4J_PASSWORD is not set")
        if self.vector_backend not in {"neo4j", "qdrant", "opensearch"}:
            problems.append(
                f"VECTOR_BACKEND '{self.vector_backend}' is not supported "
                "(use 'neo4j', 'qdrant', or 'opensearch')"
            )
        if self.vector_backend == "qdrant":
            if not self.qdrant_url:
                problems.append("VECTOR_BACKEND=qdrant requires QDRANT_URL")
            if not self.qdrant_api_key:
                problems.append("VECTOR_BACKEND=qdrant requires QDRANT_API_KEY")
            if self.qdrant_collection == "idp_chunks":
                problems.append(
                    "QDRANT_COLLECTION=idp_chunks holds corrupted payloads "
                    "(chunk_text contains stringified vectors) — use a new collection"
                )
        # OpenSearch settings are only required when it is the active backend.
        if self.vector_backend == "opensearch":
            if not self.opensearch_url:
                problems.append("VECTOR_BACKEND=opensearch requires OPENSEARCH_URL")
            if self.opensearch_auth not in {"basic", "awsv4"}:
                problems.append("OPENSEARCH_AUTH must be 'basic' or 'awsv4'")
            if self.opensearch_auth == "basic" and not self.opensearch_password:
                problems.append("OPENSEARCH_AUTH=basic requires OPENSEARCH_PASSWORD")
            if not self.opensearch_verify_certs:
                problems.append(
                    "OPENSEARCH_VERIFY_CERTS=false — local dev only, never against "
                    "a managed domain"
                )
            if self.opensearch_chunk_index == self.opensearch_term_index:
                problems.append(
                    "OPENSEARCH_CHUNK_INDEX and OPENSEARCH_TERM_INDEX must differ"
                )
        if self.embedding_provider == "openai" and not self.embedding_api_key:
            problems.append("EMBEDDING_PROVIDER=openai requires EMBEDDING_API_KEY")
        if self.llm_provider != "bedrock" and not self.llm_api_key:
            problems.append(f"LLM_PROVIDER={self.llm_provider} requires LLM_API_KEY")
        if self.embedding_dim <= 0:
            problems.append("EMBEDDING_DIM must be positive")
        if not 0.0 <= self.min_grounding_confidence <= 1.0:
            problems.append("MIN_GROUNDING_CONFIDENCE must be between 0 and 1")
        if not 0.0 <= self.entity_link_threshold <= 1.0:
            problems.append("ENTITY_LINK_THRESHOLD must be between 0 and 1")
        if self.neo4j_ctx_database == self.neo4j_kg_database and not self.ctx_label_prefix:
            problems.append(
                "runtime context would share a database with the curated KG and has "
                "no CTX_LABEL_PREFIX to isolate it"
            )
        return problems

    def report(self) -> str:
        """Redacted readiness report — safe to paste into a chat or a log."""
        lines = ["context layer configuration (secrets redacted)"]
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Secret):
                shown = f"set [{value.fingerprint()}]" if value else "MISSING"
            else:
                shown = str(value) if str(value) else "MISSING"
            lines.append(f"  {f.name:<28} {shown}")
        problems = self.check()
        lines.append("")
        if problems:
            lines.append(f"not ready — {len(problems)} problem(s):")
            lines += [f"  - {p}" for p in problems]
        else:
            lines.append("ready: all required settings present")
        return "\n".join(lines)


def load() -> Config:
    _load_dotenv()
    return Config()


if __name__ == "__main__":
    print(load().report())
