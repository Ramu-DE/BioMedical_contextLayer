"""Naive RAG baseline. Task 7.1, Requirement 11.1.

The control arm: vector retrieval straight into the model, with no graph
expansion, no rule evaluation, no grounding gate and no refusal path. This is
what most "RAG on a knowledge base" implementations actually do.

It shares the same embedder, the same Qdrant collections and the same LLM as the
context layer, so any measured difference comes from the layer itself rather
than from a different model or corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config

BASELINE_PROMPT = """\
Answer the question using the context below.

{context}

Question: {question}
Answer:"""

COLLECTIONS = ("ctx_biomed_chunks", "ctx_gov_chunks")


@dataclass(frozen=True)
class BaselineAnswer:
    question: str
    answer: str | None
    chunks_used: int
    retrieved_texts: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    curies: tuple[str, ...] = ()
    error: str | None = None

    @property
    def refused(self) -> bool:
        """Naive RAG has no refusal path; it answers whatever it retrieves."""
        return False


class NaiveRAG:
    """Vector search, then generate. No governance of any kind."""

    def __init__(self, cfg: Config, embedder=None, store=None, llm=None, k: int = 6):
        self.cfg = cfg
        self.k = k
        self._embedder = embedder
        self._store = store
        self._llm = llm

    def _lazy(self):
        if self._embedder is None:
            from context_layer.clients.embeddings import BedrockEmbeddings

            self._embedder = BedrockEmbeddings(self.cfg)
        if self._store is None:
            from context_layer.clients.vector_store import QdrantVectorStore

            self._store = QdrantVectorStore(self.cfg)
        if self._llm is None:
            from langchain_aws import ChatBedrockConverse

            kwargs = {
                "model": self.cfg.bedrock_model_id,
                "region_name": self.cfg.aws_region,
                "temperature": 0,
                "max_tokens": 600,
            }
            if self.cfg.aws_profile:
                import boto3

                kwargs["client"] = boto3.Session(
                    profile_name=self.cfg.aws_profile, region_name=self.cfg.aws_region
                ).client("bedrock-runtime")
            self._llm = ChatBedrockConverse(**kwargs)
        return self._embedder, self._store, self._llm

    @staticmethod
    def _text(content) -> str:
        if isinstance(content, list):
            return "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        return str(content)

    def ask(self, question: str) -> BaselineAnswer:
        try:
            embedder, store, llm = self._lazy()
            vector = embedder.embed_one(question)
        except Exception as e:  # noqa: BLE001
            return BaselineAnswer(question, None, 0, error=f"{type(e).__name__}: {e}")

        hits = []
        for collection in COLLECTIONS:
            try:
                store.collection = collection
                hits += store.search(vector, k=self.k)
            except Exception:  # noqa: BLE001
                continue
        hits.sort(key=lambda h: -h.score)
        hits = hits[: self.k]

        context = "\n\n".join(h.text for h in hits if h.text) or "(no context)"
        try:
            result = llm.invoke(
                BASELINE_PROMPT.format(context=context, question=question)
            )
            answer = self._text(result.content).strip()
        except Exception as e:  # noqa: BLE001
            return BaselineAnswer(
                question, None, len(hits), error=f"{type(e).__name__}: {e}"
            )

        return BaselineAnswer(
            question=question,
            answer=answer or None,
            chunks_used=len(hits),
            retrieved_texts=tuple(h.text for h in hits if h.text),
            sources=tuple(dict.fromkeys(h.source for h in hits if h.source)),
            curies=tuple(dict.fromkeys(c for h in hits for c in h.curies)),
        )
