"""Shared contracts for the context layer.

Every component exchanges these types. They are frozen dataclasses so that a
context package cannot be mutated after assembly — an assembled package is
evidence, and evidence should not be editable.

Design reference: .kiro/specs/context-layer/design.md
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Severity = Literal["block", "warn", "inform"]
ElementKind = Literal["chunk", "graph_node", "prior_decision"]


def utcnow() -> datetime:
    """Timezone-aware UTC now. Never use naive datetimes in the temporal model."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Sortable-ish unique identifier, readable in logs."""
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


# ── Provenance (R2) ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Provenance:
    """Where an assertion came from. Required for grounded answers (R2.3)."""

    source: str
    ingested_at: datetime
    confidence: float = 1.0
    record_id: str | None = None

    @property
    def is_complete(self) -> bool:
        return bool(self.source) and self.confidence > 0.0


@dataclass(frozen=True)
class Citation:
    """A resolvable pointer backing a factual claim (R10.4)."""

    chunk_id: str | None = None
    curie: str | None = None
    source: str = ""
    quote: str | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id and not self.curie:
            raise ValueError("a citation must resolve to a chunk_id or a curie")


# ── Linked context (R4) ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class EntityMention:
    """A text span linked to an ontology concept.

    ``curie is None`` means deliberately unlinked: the best candidate scored
    below the threshold, so no low-confidence binding was invented (R4.4).
    """

    chunk_id: str
    surface_form: str
    curie: str | None
    confidence: float
    start: int
    end: int

    @property
    def is_linked(self) -> bool:
        return self.curie is not None


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    provenance: Provenance
    embedding: list[float] | None = None


# ── Rules (R3) ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FiredRule:
    rule_id: str
    severity: Severity
    rationale: str
    triggered_by: tuple[str, ...] = ()
    policy_iri: str | None = None


@dataclass(frozen=True)
class RuleVerdict:
    """Result of deterministic rule evaluation. No LLM involved (R3.5)."""

    fired: tuple[FiredRule, ...] = ()

    @property
    def blocked(self) -> bool:
        return any(r.severity == "block" for r in self.fired)

    @property
    def block_rationale(self) -> str | None:
        for r in self.fired:
            if r.severity == "block":
                return r.rationale
        return None

    def summary(self) -> str:
        if not self.fired:
            return "no rules fired"
        return ", ".join(f"{r.rule_id}({r.severity})" for r in self.fired)


# ── Context assembly (R8) ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContextElement:
    element_id: str
    kind: ElementKind
    content: str
    provenance: Provenance
    score: float = 0.0
    curies: tuple[str, ...] = ()

    def approx_tokens(self) -> int:
        """Cheap token estimate; ~4 chars per token. Avoids a tokenizer dep."""
        return max(1, len(self.content) // 4)


@dataclass(frozen=True)
class ContextPackage:
    """What the model is allowed to see, and why.

    ``degradations`` records any retrieval source that failed, so a thin package
    is never mistaken for a confident one (R8.5).
    """

    context_package_id: str
    question: str
    elements: tuple[ContextElement, ...] = ()
    truncated: tuple[str, ...] = ()
    degradations: tuple[str, ...] = ()
    assembled_at: datetime = field(default_factory=utcnow)

    @property
    def token_count(self) -> int:
        return sum(e.approx_tokens() for e in self.elements)

    @property
    def curies(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for e in self.elements:
            for c in e.curies:
                seen.setdefault(c, None)
        return tuple(seen)

    def by_kind(self, kind: ElementKind) -> tuple[ContextElement, ...]:
        return tuple(e for e in self.elements if e.kind == kind)

    @property
    def is_empty(self) -> bool:
        return not self.elements


# ── Runtime context (R5, R6) ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Event:
    event_id: str
    event_type: str
    occurred_at: datetime
    ingested_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    about_curies: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    """An append-only record of one AI decision (R6.1, R6.4)."""

    decision_id: str
    question: str
    context_package_id: str
    rules_fired: tuple[str, ...]
    answer_hash: str
    grounding_confidence: float
    decided_at: datetime
    grounded_in: tuple[str, ...] = ()
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    @staticmethod
    def hash_answer(answer: str | None) -> str:
        return hashlib.sha256((answer or "").encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Action:
    action_id: str
    decision_id: str
    tool_name: str
    arguments: dict[str, Any]
    duration_ms: float
    status: Literal["success", "failure", "partial"]
    at: datetime


@dataclass(frozen=True)
class Outcome:
    """A decision-quality outcome, not a clinical outcome. See spec scope."""

    outcome_id: str
    decision_id: str
    outcome_type: str
    observed_at: datetime
    detail: str = ""


# ── Governed response (R10) ───────────────────────────────────────────────────


@dataclass(frozen=True)
class GovernedResponse:
    """The single published output of the context layer.

    Exposes ``neural_proposal`` alongside ``symbolic_verdict`` so the
    interaction between the two halves of the neurosymbolic system is auditable
    rather than implied (R9.3).
    """

    context_package_id: str
    symbolic_verdict: str
    grounding_confidence: float
    answer: str | None = None
    refused: bool = False
    refusal_reason: str | None = None
    citations: tuple[Citation, ...] = ()
    rules_fired: tuple[FiredRule, ...] = ()
    neural_proposal: str | None = None
    degradations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.refused and self.answer is not None:
            raise ValueError("a refused response must not carry an answer")
        if self.refused and not self.refusal_reason:
            raise ValueError("a refused response must state a reason")
        if not self.refused and self.answer is None:
            raise ValueError("a non-refused response must carry an answer")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready. This payload is the proof artifact."""
        out = asdict(self)
        for key in ("citations", "rules_fired", "degradations"):
            out[key] = list(out[key])
        return out

    @classmethod
    def refusal(
        cls,
        *,
        context_package_id: str,
        reason: str,
        symbolic_verdict: str = "not evaluated",
        grounding_confidence: float = 0.0,
        rules_fired: tuple[FiredRule, ...] = (),
        degradations: tuple[str, ...] = (),
    ) -> GovernedResponse:
        return cls(
            context_package_id=context_package_id,
            symbolic_verdict=symbolic_verdict,
            grounding_confidence=grounding_confidence,
            answer=None,
            refused=True,
            refusal_reason=reason,
            rules_fired=rules_fired,
            degradations=degradations,
        )
