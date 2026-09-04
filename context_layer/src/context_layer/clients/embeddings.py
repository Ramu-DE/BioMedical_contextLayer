"""Bedrock embedding client. Task 1.1, Requirement 4.1.

Titan Text v2 embeds one input per call, so batching here means sequential calls
with bounded retry rather than a true batch API. Verified live in us-west-2:
amazon.titan-embed-text-v2:0 returns 1024 dimensions.
"""

from __future__ import annotations

import json
import random
import time
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from config import Config

MAX_ATTEMPTS = 5
BASE_DELAY = 0.5
RETRYABLE = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
}


class EmbeddingError(RuntimeError):
    """Embedding failed after exhausting retries."""


class BedrockEmbeddings:
    """Text -> vector, with exponential backoff and jitter."""

    def __init__(self, cfg: Config, client=None) -> None:
        self.cfg = cfg
        self.model_id = cfg.embedding_model_id
        self.dim = cfg.embedding_dim
        if client is not None:
            self._client = client
        else:
            import boto3

            session = boto3.Session(
                profile_name=cfg.aws_profile or None, region_name=cfg.aws_region
            )
            self._client = session.client("bedrock-runtime")

    # ── internals ─────────────────────────────────────────────────────────────

    def _body(self, text: str) -> str:
        if "cohere" in self.model_id:
            return json.dumps({"texts": [text], "input_type": "search_document"})
        return json.dumps({"inputText": text})

    @staticmethod
    def _extract(payload: dict) -> list[float]:
        if "embedding" in payload:
            return payload["embedding"]
        if "embeddings" in payload:
            return payload["embeddings"][0]
        raise EmbeddingError(f"no embedding in response keys={list(payload)}")

    def _invoke_once(self, text: str) -> list[float]:
        resp = self._client.invoke_model(
            modelId=self.model_id, body=self._body(text)
        )
        return self._extract(json.loads(resp["body"].read()))

    # ── public API ────────────────────────────────────────────────────────────

    def embed_one(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("cannot embed empty text")
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                vec = self._invoke_once(text)
            except Exception as e:  # noqa: BLE001
                code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
                if code not in RETRYABLE and type(e).__name__ not in RETRYABLE:
                    raise EmbeddingError(f"{type(e).__name__}: {e}") from e
                last = e
                time.sleep(BASE_DELAY * (2**attempt) + random.uniform(0, 0.3))  # noqa: S311
                continue
            if len(vec) != self.dim:
                raise EmbeddingError(
                    f"expected dim {self.dim}, got {len(vec)} from {self.model_id} — "
                    "EMBEDDING_DIM and the model disagree"
                )
            return vec
        raise EmbeddingError(f"exhausted {MAX_ATTEMPTS} attempts: {last}")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence, preserving order."""
        return [self.embed_one(t) for t in texts]
