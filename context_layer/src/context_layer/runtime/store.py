"""Append-only runtime context store. Tasks 3.2-3.4, Requirements 5.x, 6.x, 7.x.

This is the Runtime Context column of the reference architecture: events,
decisions, actions and outcomes written back into the graph so the model of the
enterprise stays current.

Three properties are structural rather than conventional:

1. **Append-only.** No update or delete verb exists on this class. Superseding a
   record means writing a new version and closing the previous one's validity
   window, never mutating in place (Requirement 6.4).
2. **Bi-temporal.** ``occurred_at``/``ingested_at`` for events and
   ``valid_from``/``valid_to`` for decisions, so "what did we believe on date X"
   is answerable and out-of-order arrivals stay correct (Requirements 5.4, 7.2).
3. **Namespace isolation.** Every label is prefixed (default ``Ctx_``) so runtime
   writes never touch curated knowledge (Requirement 5.3).

Replaces the in-memory ``audit_logger`` from the upstream MCP platform, whose
records were lost on restart.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from context_layer.types import (
    Action,
    Decision,
    Event,
    Outcome,
    new_id,
    utcnow,
)

if TYPE_CHECKING:
    from config import Config

_ISO = "%Y-%m-%dT%H:%M:%S.%f%z"


def _to_dt(value) -> datetime:
    """Neo4j returns its own temporal types; normalise to aware datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if hasattr(value, "to_native"):
        native = value.to_native()
        return native if native.tzinfo else native.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return utcnow()


class RuntimeStore:
    """Durable append-only runtime context in Neo4j."""

    def __init__(self, cfg: Config, driver=None, embedder=None) -> None:
        self.cfg = cfg
        self.prefix = cfg.ctx_label_prefix or "Ctx_"
        self.database = cfg.neo4j_ctx_database or cfg.neo4j_kg_database
        self._embedder = embedder
        if driver is not None:
            self._driver = driver
            self._owns = False
        else:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                cfg.neo4j_uri, auth=(cfg.neo4j_user, str(cfg.neo4j_password))
            )
            self._owns = True

    # ── labels ────────────────────────────────────────────────────────────────

    def label(self, kind: str) -> str:
        return f"{self.prefix}{kind}"

    @property
    def decision_label(self) -> str:
        return self.label("Decision")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._owns:
            self._driver.close()

    def __enter__(self) -> RuntimeStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _run(self, cypher: str, **params):
        with self._driver.session(database=self.database) as s:
            return list(s.run(cypher, **params))

    # ── schema ────────────────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        """Constraints and the vector index for prior-decision retrieval."""
        for kind, key in (
            ("Event", "event_id"),
            ("Decision", "decision_id"),
            ("Action", "action_id"),
            ("Outcome", "outcome_id"),
        ):
            self._run(
                f"CREATE CONSTRAINT ctx_{kind.lower()}_key IF NOT EXISTS "
                f"FOR (n:`{self.label(kind)}`) REQUIRE n.{key} IS UNIQUE"
            )
        self._run(
            f"CREATE INDEX ctx_decision_time IF NOT EXISTS "
            f"FOR (n:`{self.decision_label}`) ON (n.decided_at)"
        )
        # Neo4j 5.27 native vector index for semantic prior-decision lookup.
        self._run(
            f"""
            CREATE VECTOR INDEX {self.cfg.neo4j_chunk_index} IF NOT EXISTS
            FOR (n:`{self.decision_label}`) ON (n.question_embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {int(self.cfg.embedding_dim)},
                `vector.similarity_function`: '{self.cfg.vector_similarity}'
            }}}}
            """
        )

    # ── append: events, R5.1-5.4 ──────────────────────────────────────────────

    def append_event(self, event: Event) -> str:
        self._run(
            f"""
            CREATE (e:`{self.label('Event')}` {{
                event_id: $event_id, event_type: $event_type,
                occurred_at: datetime($occurred_at),
                ingested_at: datetime($ingested_at),
                payload: $payload
            }})
            WITH e
            UNWIND CASE WHEN $curies = [] THEN [null] ELSE $curies END AS curie
            OPTIONAL MATCH (target)-[:MAPS_TO]->(c:ExternalConcept {{curie: curie}})
            FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END |
                MERGE (e)-[:ABOUT]->(target))
            """,
            event_id=event.event_id,
            event_type=event.event_type,
            occurred_at=event.occurred_at.isoformat(),
            ingested_at=event.ingested_at.isoformat(),
            # Neo4j properties must be primitives or arrays, never maps.
            payload=json.dumps(event.payload, default=str, sort_keys=True),
            curies=list(event.about_curies),
        )
        return event.event_id

    # ── append: decisions, R6.1, 6.4, 6.5 ─────────────────────────────────────

    def append_decision(
        self, decision: Decision, question_embedding: Sequence[float] | None = None
    ) -> str:
        embedding = list(question_embedding) if question_embedding else None
        if embedding is None and self._embedder is not None:
            try:
                embedding = self._embedder.embed_one(decision.question)
            except Exception:  # noqa: BLE001
                embedding = None
        self._run(
            f"""
            CREATE (d:`{self.decision_label}` {{
                decision_id: $decision_id,
                question: $question,
                context_package_id: $context_package_id,
                rules_fired: $rules_fired,
                answer_hash: $answer_hash,
                grounding_confidence: $grounding_confidence,
                decided_at: datetime($decided_at),
                valid_from: datetime($valid_from),
                grounded_in: $grounded_in
            }})
            WITH d
            FOREACH (_ IN CASE WHEN $embedding IS NULL THEN [] ELSE [1] END |
                SET d.question_embedding = $embedding)
            WITH d
            UNWIND CASE WHEN $grounded_in = [] THEN [null] ELSE $grounded_in END AS curie
            OPTIONAL MATCH (e)-[:MAPS_TO]->(c:ExternalConcept {{curie: curie}})
            FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
                MERGE (d)-[:GROUNDED_IN]->(e))
            """,
            decision_id=decision.decision_id,
            question=decision.question,
            context_package_id=decision.context_package_id,
            rules_fired=list(decision.rules_fired),
            answer_hash=decision.answer_hash,
            grounding_confidence=float(decision.grounding_confidence),
            decided_at=decision.decided_at.isoformat(),
            valid_from=(decision.valid_from or decision.decided_at).isoformat(),
            grounded_in=list(decision.grounded_in),
            embedding=embedding,
        )
        return decision.decision_id

    def supersede_decision(self, decision_id: str, at: datetime | None = None) -> None:
        """Close a decision's validity window. R7.2 — never an in-place edit.

        This sets valid_to only; the record's content is immutable. Superseding
        is how the living model advances without losing history.
        """
        self._run(
            f"""
            MATCH (d:`{self.decision_label}` {{decision_id: $decision_id}})
            WHERE d.valid_to IS NULL
            SET d.valid_to = datetime($at)
            """,
            decision_id=decision_id,
            at=(at or utcnow()).isoformat(),
        )

    # ── append: actions and outcomes, R6.2, 6.3 ───────────────────────────────

    def append_action(self, action: Action) -> str:
        self._run(
            f"""
            MATCH (d:`{self.decision_label}` {{decision_id: $decision_id}})
            CREATE (a:`{self.label('Action')}` {{
                action_id: $action_id, tool_name: $tool_name,
                arguments: $arguments, duration_ms: $duration_ms,
                status: $status, at: datetime($at)
            }})
            MERGE (d)-[:PRODUCED]->(a)
            """,
            decision_id=action.decision_id,
            action_id=action.action_id,
            tool_name=action.tool_name,
            arguments=json.dumps(action.arguments, default=str, sort_keys=True),
            duration_ms=float(action.duration_ms),
            status=action.status,
            at=action.at.isoformat(),
        )
        return action.action_id

    def append_outcome(self, outcome: Outcome) -> str:
        self._run(
            f"""
            MATCH (d:`{self.decision_label}` {{decision_id: $decision_id}})
            CREATE (o:`{self.label('Outcome')}` {{
                outcome_id: $outcome_id, outcome_type: $outcome_type,
                observed_at: datetime($observed_at), detail: $detail
            }})
            MERGE (d)-[:RESULTED_IN]->(o)
            """,
            decision_id=outcome.decision_id,
            outcome_id=outcome.outcome_id,
            outcome_type=outcome.outcome_type,
            observed_at=outcome.observed_at.isoformat(),
            detail=outcome.detail,
        )
        return outcome.outcome_id

    # ── read: continuous incorporation, R7.1 ──────────────────────────────────

    def _row_to_decision(self, row) -> Decision:
        return Decision(
            decision_id=row["decision_id"],
            question=row["question"],
            context_package_id=row["context_package_id"],
            rules_fired=tuple(row["rules_fired"] or ()),
            answer_hash=row["answer_hash"] or "",
            grounding_confidence=float(row["grounding_confidence"] or 0.0),
            decided_at=_to_dt(row["decided_at"]),
            grounded_in=tuple(row["grounded_in"] or ()),
            valid_from=_to_dt(row["valid_from"]) if row.get("valid_from") else None,
            valid_to=_to_dt(row["valid_to"]) if row.get("valid_to") else None,
        )

    _RETURN = """
        RETURN d.decision_id AS decision_id, d.question AS question,
               d.context_package_id AS context_package_id,
               d.rules_fired AS rules_fired, d.answer_hash AS answer_hash,
               d.grounding_confidence AS grounding_confidence,
               d.decided_at AS decided_at, d.grounded_in AS grounded_in,
               d.valid_from AS valid_from, d.valid_to AS valid_to
    """

    def recent_decisions(self, question: str, k: int = 5) -> list[Decision]:
        """Prior decisions relevant to this question.

        Semantic search via the native vector index when an embedder is
        available, falling back to recency so the loop still closes without
        Bedrock.
        """
        embedding = None
        if self._embedder is not None and question:
            try:
                embedding = self._embedder.embed_one(question)
            except Exception:  # noqa: BLE001
                embedding = None

        if embedding is not None:
            try:
                rows = self._run(
                    f"""
                    CALL db.index.vector.queryNodes($index, $k, $embedding)
                    YIELD node AS d, score
                    WHERE d.valid_to IS NULL
                    {self._RETURN}, score
                    """,
                    index=self.cfg.neo4j_chunk_index,
                    k=int(k),
                    embedding=embedding,
                )
                return [self._row_to_decision(r) for r in rows]
            except Exception:  # noqa: BLE001
                pass  # index may not exist yet; fall through to recency

        rows = self._run(
            f"""
            MATCH (d:`{self.decision_label}`)
            WHERE d.valid_to IS NULL
            WITH d ORDER BY d.decided_at DESC LIMIT $k
            {self._RETURN}
            """,
            k=int(k),
        )
        return [self._row_to_decision(r) for r in rows]

    def as_of(self, timestamp: datetime, limit: int = 50) -> list[Decision]:
        """Decisions considered valid at a point in time. R7.3."""
        rows = self._run(
            f"""
            MATCH (d:`{self.decision_label}`)
            WHERE d.valid_from <= datetime($ts)
              AND (d.valid_to IS NULL OR d.valid_to > datetime($ts))
            WITH d ORDER BY d.decided_at DESC LIMIT $limit
            {self._RETURN}
            """,
            ts=timestamp.isoformat(),
            limit=int(limit),
        )
        return [self._row_to_decision(r) for r in rows]

    def decision_by_id(self, decision_id: str) -> Decision | None:
        rows = self._run(
            f"""
            MATCH (d:`{self.decision_label}` {{decision_id: $decision_id}})
            {self._RETURN}
            """,
            decision_id=decision_id,
        )
        return self._row_to_decision(rows[0]) if rows else None

    def actions_for(self, decision_id: str) -> list[Action]:
        rows = self._run(
            f"""
            MATCH (d:`{self.decision_label}` {{decision_id: $decision_id}})
                  -[:PRODUCED]->(a:`{self.label('Action')}`)
            RETURN a.action_id AS action_id, a.tool_name AS tool_name,
                   a.arguments AS arguments, a.duration_ms AS duration_ms,
                   a.status AS status, a.at AS at
            ORDER BY a.at
            """,
            decision_id=decision_id,
        )
        return [
            Action(
                action_id=r["action_id"],
                decision_id=decision_id,
                tool_name=r["tool_name"],
                arguments=json.loads(r["arguments"]) if r["arguments"] else {},
                duration_ms=float(r["duration_ms"] or 0.0),
                status=r["status"],
                at=_to_dt(r["at"]),
            )
            for r in rows
        ]

    def outcomes_for(self, decision_id: str) -> list[Outcome]:
        rows = self._run(
            f"""
            MATCH (d:`{self.decision_label}` {{decision_id: $decision_id}})
                  -[:RESULTED_IN]->(o:`{self.label('Outcome')}`)
            RETURN o.outcome_id AS outcome_id, o.outcome_type AS outcome_type,
                   o.observed_at AS observed_at, o.detail AS detail
            ORDER BY o.observed_at
            """,
            decision_id=decision_id,
        )
        return [
            Outcome(
                outcome_id=r["outcome_id"],
                decision_id=decision_id,
                outcome_type=r["outcome_type"],
                observed_at=_to_dt(r["observed_at"]),
                detail=r["detail"] or "",
            )
            for r in rows
        ]

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kind in ("Event", "Decision", "Action", "Outcome"):
            rows = self._run(f"MATCH (n:`{self.label(kind)}`) RETURN count(n) AS c")
            out[kind] = rows[0]["c"] if rows else 0
        return out


# ── convenience constructors ──────────────────────────────────────────────────


def decision_from_response(question: str, response, grounded_in=()) -> Decision:
    """Build an append-ready Decision from a GovernedResponse. R6.1."""
    return Decision(
        decision_id=new_id("dec"),
        question=question,
        context_package_id=response.context_package_id,
        rules_fired=tuple(f.rule_id for f in response.rules_fired),
        answer_hash=Decision.hash_answer(response.answer),
        grounding_confidence=response.grounding_confidence,
        decided_at=utcnow(),
        grounded_in=tuple(grounded_in),
    )


def action(decision_id: str, tool_name: str, arguments: dict,
           duration_ms: float, status: str = "success") -> Action:
    return Action(
        action_id=new_id("act"),
        decision_id=decision_id,
        tool_name=tool_name,
        arguments=arguments,
        duration_ms=duration_ms,
        status=status,  # type: ignore[arg-type]
        at=utcnow(),
    )


def outcome(decision_id: str, outcome_type: str, detail: str = "") -> Outcome:
    return Outcome(
        outcome_id=new_id("out"),
        decision_id=decision_id,
        outcome_type=outcome_type,
        observed_at=utcnow(),
        detail=detail,
    )
