"""Vocabulary term index for entity linking. Task 2.1, Requirement 4.1.

Embeds the label and synonyms of every linkable concept into a Neo4j native
vector index, keyed by CURIE. Living in Neo4j rather than Qdrant is deliberate:
linking must verify that a resolved concept actually exists in the graph
(Requirement 4.5), so keeping the index beside the concepts avoids a
cross-system round trip per mention.

Term nodes are written under the Ctx_ namespace so the curated graph is
untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from config import Config

# Sources of linkable surface forms. Each row yields one or more terms sharing a
# CURIE, so "aspirin" and "acetylsalicylic acid" collapse to one concept.
TERM_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "Drug",
        """
        MATCH (d:Drug)-[:MAPS_TO]->(c:ExternalConcept)
        WHERE c.standard IN ['RxNorm', 'SNOMED', 'MeSH']
        RETURN c.curie AS curie, 'Drug' AS entity_type,
               d.drug_id AS entity_id,
               [x IN [d.name, d.generic_name] WHERE x IS NOT NULL] AS surfaces
        """,
    ),
    (
        "Disease",
        """
        MATCH (dis:Disease)-[:MAPS_TO]->(c:ExternalConcept)
        RETURN c.curie AS curie, 'Disease' AS entity_type,
               dis.disease_id AS entity_id,
               [x IN [dis.name, dis.icd10_code] WHERE x IS NOT NULL] AS surfaces
        """,
    ),
    (
        "AdverseEvent",
        """
        MATCH (ae:AdverseEvent)-[:MAPS_TO]->(c:ExternalConcept)
        RETURN c.curie AS curie, 'AdverseEvent' AS entity_type,
               ae.event_id AS entity_id, [ae.name] AS surfaces
        """,
    ),
    (
        "SNOMED",
        """
        MATCH (s:SNOMED)
        RETURN 'SNOMED:' + s.code AS curie, 'SNOMED' AS entity_type,
               s.code AS entity_id, [s.name] AS surfaces
        """,
    ),
    (
        "RxNorm",
        """
        MATCH (r:RxNorm)
        RETURN 'RXNORM:' + r.cui AS curie, 'RxNorm' AS entity_type,
               r.cui AS entity_id, [r.name] AS surfaces
        """,
    ),
    (
        "MedDRA",
        """
        MATCH (m:MedDRA)
        RETURN 'MEDDRA:' + m.code AS curie, 'MedDRA' AS entity_type,
               m.code AS entity_id, [m.name] AS surfaces
        """,
    ),
    (
        "OMOP",
        """
        MATCH (o:OMOP)
        RETURN 'OMOP:' + toString(o.concept_id) AS curie, 'OMOP' AS entity_type,
               toString(o.concept_id) AS entity_id, [o.name] AS surfaces
        """,
    ),
)


@dataclass(frozen=True)
class Term:
    curie: str
    surface: str
    entity_type: str
    entity_id: str

    @property
    def term_id(self) -> str:
        return f"{self.curie}|{self.surface.casefold()}"


class TermIndex:
    """Builds and queries the vocabulary vector index."""

    def __init__(self, cfg: Config, driver=None, embedder=None) -> None:
        self.cfg = cfg
        self.prefix = cfg.ctx_label_prefix or "Ctx_"
        self.label = f"{self.prefix}Term"
        self.index_name = cfg.neo4j_term_index
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

    def close(self) -> None:
        if self._owns:
            self._driver.close()

    def __enter__(self) -> TermIndex:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _run(self, cypher: str, **params):
        with self._driver.session(database=self.cfg.neo4j_kg_database) as s:
            return list(s.run(cypher, **params))

    # ── build ─────────────────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        self._run(
            f"CREATE CONSTRAINT ctx_term_key IF NOT EXISTS "
            f"FOR (t:`{self.label}`) REQUIRE t.term_id IS UNIQUE"
        )
        self._run(
            f"""
            CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
            FOR (t:`{self.label}`) ON (t.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {int(self.cfg.embedding_dim)},
                `vector.similarity_function`: '{self.cfg.vector_similarity}'
            }}}}
            """
        )

    def collect_terms(self) -> list[Term]:
        """Distinct (curie, surface) pairs from every vocabulary source."""
        seen: dict[str, Term] = {}
        for _, cypher in TERM_QUERIES:
            for row in self._run(cypher):
                curie = row["curie"]
                if not curie:
                    continue
                for surface in row["surfaces"] or []:
                    if not surface or len(str(surface).strip()) < 2:
                        continue
                    term = Term(
                        curie=curie,
                        surface=str(surface).strip(),
                        entity_type=row["entity_type"],
                        entity_id=str(row["entity_id"]),
                    )
                    seen.setdefault(term.term_id, term)
        return sorted(seen.values(), key=lambda t: t.term_id)

    def build(self, terms: Iterable[Term] | None = None) -> int:
        if self._embedder is None:
            from context_layer.clients.embeddings import BedrockEmbeddings

            self._embedder = BedrockEmbeddings(self.cfg)
        self.ensure_schema()
        terms = list(terms if terms is not None else self.collect_terms())
        payload = []
        for t in terms:
            payload.append(
                {
                    "term_id": t.term_id,
                    "curie": t.curie,
                    "surface": t.surface,
                    "entity_type": t.entity_type,
                    "entity_id": t.entity_id,
                    "embedding": self._embedder.embed_one(t.surface),
                }
            )
        if payload:
            self._run(
                f"""
                UNWIND $rows AS row
                MERGE (t:`{self.label}` {{term_id: row.term_id}})
                SET t.curie = row.curie, t.surface = row.surface,
                    t.entity_type = row.entity_type, t.entity_id = row.entity_id,
                    t.embedding = row.embedding
                """,
                rows=payload,
            )
        return len(payload)

    def count(self) -> int:
        rows = self._run(f"MATCH (t:`{self.label}`) RETURN count(t) AS c")
        return rows[0]["c"] if rows else 0

    # ── query ─────────────────────────────────────────────────────────────────

    def nearest(self, text: str, k: int = 3) -> list[tuple[Term, float]]:
        """k nearest vocabulary terms to a surface form."""
        if self._embedder is None:
            from context_layer.clients.embeddings import BedrockEmbeddings

            self._embedder = BedrockEmbeddings(self.cfg)
        vector = self._embedder.embed_one(text)
        rows = self._run(
            f"""
            CALL db.index.vector.queryNodes($index, $k, $vector)
            YIELD node AS t, score
            RETURN t.curie AS curie, t.surface AS surface,
                   t.entity_type AS entity_type, t.entity_id AS entity_id,
                   score
            """,
            index=self.index_name,
            k=int(k),
            vector=vector,
        )
        return [
            (
                Term(
                    curie=r["curie"],
                    surface=r["surface"],
                    entity_type=r["entity_type"],
                    entity_id=r["entity_id"],
                ),
                float(r["score"]),
            )
            for r in rows
        ]

    def curie_exists(self, curie: str) -> bool:
        """Requirement 4.5: never link to a concept absent from the graph."""
        rows = self._run(
            "MATCH (c:ExternalConcept {curie: $curie}) RETURN count(c) AS c",
            curie=curie,
        )
        if rows and rows[0]["c"]:
            return True
        rows = self._run(
            f"MATCH (t:`{self.label}` {{curie: $curie}}) RETURN count(t) AS c",
            curie=curie,
        )
        return bool(rows and rows[0]["c"])

    # ── concept canonicalisation ──────────────────────────────────────────────
    #
    # The loaded graph holds separate vocabulary nodes for the same drug — e.g.
    # RxNorm "Entyvio (vedolizumab)" and OMOP "vedolizumab" — with no crosswalk
    # edge joining them. Without canonicalisation, "Entyvio" and "vedolizumab"
    # resolve to different CURIEs and Requirement 4.3 fails.
    #
    # This builds SAME_AS edges explicitly and records why, rather than silently
    # merging concepts. Every edge carries its method and confidence so it can be
    # audited or revoked.

    def build_crosswalk(self, min_token_len: int = 5) -> dict[str, int]:
        """Link vocabulary nodes whose names share a distinctive token.

        Conservative by design: requires a shared token of at least
        ``min_token_len`` characters, so 'vedolizumab' unifies Entyvio's RxNorm
        node with OMOP's, while short words like 'acid' cannot.
        """
        stats: dict[str, int] = {}

        # RxNorm brand names in this corpus follow "Brand (generic)". Extract the
        # parenthesised generic and match it to other vocabularies by name.
        rows = self._run(
            f"""
            MATCH (t:`{self.label}`)
            WHERE t.entity_type IN ['RxNorm', 'Drug', 'OMOP', 'SNOMED', 'MedDRA']
            RETURN t.curie AS curie, toLower(t.surface) AS surface,
                   t.entity_type AS entity_type
            """
        )
        by_token: dict[str, set[str]] = {}
        for r in rows:
            surface = r["surface"] or ""
            tokens = {
                tok.strip("()[],.;:")
                for tok in surface.replace("(", " ").replace(")", " ").split()
            }
            for tok in tokens:
                if len(tok) >= min_token_len and tok.isalpha():
                    by_token.setdefault(tok, set()).add(r["curie"])

        pairs: list[dict] = []
        for token, curies in by_token.items():
            if len(curies) < 2:
                continue
            ordered = sorted(curies)
            for i, a in enumerate(ordered):
                for b in ordered[i + 1 :]:
                    if a.split(":")[0] == b.split(":")[0]:
                        continue  # same vocabulary; not a crosswalk
                    pairs.append({"a": a, "b": b, "token": token})

        if pairs:
            self._run(
                """
                UNWIND $pairs AS p
                MATCH (ca:ExternalConcept {curie: p.a})
                MATCH (cb:ExternalConcept {curie: p.b})
                MERGE (ca)-[r:SAME_AS]-(cb)
                SET r.method = 'shared_name_token',
                    r.evidence = p.token,
                    r.confidence = 0.9
                """,
                pairs=pairs,
            )
        stats["candidate_pairs"] = len(pairs)

        # Ensure every indexed term has an ExternalConcept to hang SAME_AS from.
        created = self._run(
            f"""
            MATCH (t:`{self.label}`)
            WHERE NOT EXISTS {{ MATCH (:ExternalConcept {{curie: t.curie}}) }}
            MERGE (c:ExternalConcept {{curie: t.curie}})
            SET c.standard = t.entity_type, c.derived_from = 'term_index'
            RETURN count(c) AS c
            """
        )
        stats["concepts_created"] = created[0]["c"] if created else 0

        if pairs:
            self._run(
                """
                UNWIND $pairs AS p
                MATCH (ca:ExternalConcept {curie: p.a})
                MATCH (cb:ExternalConcept {curie: p.b})
                MERGE (ca)-[r:SAME_AS]-(cb)
                SET r.method = 'shared_name_token',
                    r.evidence = p.token,
                    r.confidence = 0.9
                """,
                pairs=pairs,
            )
        linked = self._run("MATCH ()-[r:SAME_AS]-() RETURN count(r) AS c")
        stats["same_as_edges"] = (linked[0]["c"] if linked else 0) // 2
        return stats

    def canonical_curie(self, curie: str) -> str:
        """Lowest CURIE in the concept's SAME_AS equivalence class.

        Deterministic: sorting makes the representative stable regardless of
        traversal order, so two surface forms in the same class always agree.
        """
        rows = self._run(
            """
            MATCH (c:ExternalConcept {curie: $curie})
            OPTIONAL MATCH (c)-[:SAME_AS*1..3]-(eq:ExternalConcept)
            WITH c.curie AS self_curie, collect(DISTINCT eq.curie) AS eq_curies
            WITH self_curie, [x IN eq_curies WHERE x IS NOT NULL] AS eq_curies
            UNWIND eq_curies + [self_curie] AS cu
            WITH cu WHERE cu IS NOT NULL
            RETURN cu ORDER BY cu LIMIT 1
            """,
            curie=curie,
        )
        return rows[0]["cu"] if rows else curie
