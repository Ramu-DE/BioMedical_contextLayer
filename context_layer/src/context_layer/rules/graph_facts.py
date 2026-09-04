"""Build a FactSet from Neo4j. Task 4.2 support.

All graph I/O for rule evaluation lives here. The engine consumes the FactSet
and performs no I/O of its own, which is what keeps verdicts reproducible
(Requirement 3.6).

Facts are fetched by entity id or by CURIE, so the caller can scope evaluation
to exactly what a question touched rather than the whole graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

from context_layer.rules.models import Fact, FactSet

if TYPE_CHECKING:
    from config import Config

# Relationship types the rules ask about. Fetching a bounded set keeps the
# query cheap and makes the engine's inputs explicit.
TRACKED_RELATIONS: tuple[str, ...] = (
    "HAS_PRIMARY_DISEASE",
    "HAS_DQ_RULE",
    "HAS_PRIMARY_KEY",
    "MAPS_TO",
    "TREATS",
    "TARGETS",
    "ASSOCIATED_WITH",
    "REPORTS",
    "INVESTIGATES",
    "STUDIES",
    "GOVERNS_ENTITY",
    "GOVERNED_BY",
    "PREDICTS_RESPONSE_TO",
    # Derived by scripts/derive_relationships.py. Adding an edge to the graph is
    # not enough: unless it is fetched here it can never reach a rule or the
    # traversal view. test_core_relations_are_all_fetched guards this.
    "IS_BIOMARKER_FOR",
    "INDICATES",
    "EVALUATED_IN",
    "ENROLLED_IN",
    "HAS_OUTCOME",
    # Governance and data-lineage edges. Rules only assert on HAS_DQ_RULE and
    # HAS_PRIMARY_KEY, but fetching the surrounding structure is what makes the
    # traversal view legible: without these, Table/ARD/Entity facts arrive with
    # no relations at all and the graph renders as disconnected dots.
    "CERTIFIED_VIEW_OF",
    "FEEDS_INTO",
    "GOVERNS",
    "OWNS_ARD",
    "HAS_ARD",
    "WRITES_TO",
    "CONFIGURES",
    "CONTAINS_JOB",
    "COVERS_TA",
    "OWNS_DOMAIN",
    "HAS_FUNCTION",
    "OMOP_MAPS_TO",
)

# Entity types the rule packs reason over, with their natural key property.
FACT_KEYS: dict[str, str] = {
    "Patient": "patient_id",
    "Drug": "drug_id",
    "Disease": "disease_id",
    "AdverseEvent": "event_id",
    "ClinicalTrial": "trial_id",
    "Biomarker": "biomarker_id",
    # Gene completes the six core entities of the published model;
    # without it, Gene-[:IS_BIOMARKER_FOR]->Biomarker can never be
    # fetched from the Gene side.
    "Gene": "gene_id",
    "Protein": "protein_id",
    "Table": "fqn",
    "ARD": "ard_id",
    "Entity": "entity_id",
    "DataGovernancePolicy": "policy_id",
    "Policy": "policy_id",
}

_FETCH = """
UNWIND $specs AS spec
MATCH (n)
WHERE spec.label IN labels(n) AND n[spec.key] IN spec.ids
WITH n, spec
OPTIONAL MATCH (n)-[r]->(m)
WHERE type(r) IN $tracked
WITH n, spec, type(r) AS rel_type, collect(DISTINCT coalesce(
    m.patient_id, m.drug_id, m.disease_id, m.event_id, m.trial_id,
    m.biomarker_id, m.fqn, m.ard_id, m.entity_id, m.policy_id, m.curie,
    elementId(m)
)) AS targets
WITH n, spec, collect({rel: rel_type, targets: targets}) AS rels
OPTIONAL MATCH (n)-[:MAPS_TO]->(c:ExternalConcept)
RETURN spec.label AS label,
       n[spec.key] AS entity_id,
       properties(n) AS props,
       rels AS relations,
       collect(DISTINCT c.curie) AS curies
"""

_FETCH_ALL_OF_TYPE = """
MATCH (n)
WHERE $label IN labels(n)
WITH n LIMIT $limit
OPTIONAL MATCH (n)-[r]->(m)
WHERE type(r) IN $tracked
WITH n, type(r) AS rel_type, collect(DISTINCT coalesce(
    m.patient_id, m.drug_id, m.disease_id, m.event_id, m.trial_id,
    m.biomarker_id, m.fqn, m.ard_id, m.entity_id, m.policy_id, m.curie,
    elementId(m)
)) AS targets
WITH n, collect({rel: rel_type, targets: targets}) AS rels
OPTIONAL MATCH (n)-[:MAPS_TO]->(c:ExternalConcept)
RETURN n[$key] AS entity_id, properties(n) AS props,
       rels AS relations, collect(DISTINCT c.curie) AS curies
"""


def _to_fact(label: str, entity_id, props, relations, curies) -> Fact | None:
    if entity_id is None:
        return None
    rel_map: dict[str, tuple[str, ...]] = {}
    for entry in relations or []:
        rel = entry.get("rel")
        if not rel:
            continue
        targets = tuple(t for t in (entry.get("targets") or []) if t is not None)
        if targets:
            rel_map[rel] = tuple(sorted(set(rel_map.get(rel, ()) + targets)))
    return Fact(
        entity_id=str(entity_id),
        entity_type=label,
        properties=dict(props or {}),
        curies=tuple(sorted(c for c in (curies or []) if c)),
        relations=rel_map,
    )


class GraphFactBuilder:
    """Fetches facts from Neo4j. The only I/O in the symbolic path."""

    def __init__(self, cfg: Config, driver=None) -> None:
        self.cfg = cfg
        self._name_cache: list[tuple[str, str, str]] | None = None
        if driver is not None:
            self._driver = driver
            self._owns_driver = False
        else:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                cfg.neo4j_uri, auth=(cfg.neo4j_user, str(cfg.neo4j_password))
            )
            self._owns_driver = True

    def close(self) -> None:
        if self._owns_driver:
            self._driver.close()

    def __enter__(self) -> GraphFactBuilder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── fetch by explicit ids ─────────────────────────────────────────────────

    def for_entities(self, wanted: dict[str, Sequence[str]]) -> FactSet:
        """wanted maps entity label -> ids. Only tracked relations are loaded."""
        specs = [
            {"label": label, "key": FACT_KEYS[label], "ids": list(ids)}
            for label, ids in wanted.items()
            if label in FACT_KEYS and ids
        ]
        if not specs:
            return FactSet(source_note="no resolvable entity types requested")
        with self._driver.session(database=self.cfg.neo4j_kg_database) as s:
            rows = list(s.run(_FETCH, specs=specs, tracked=list(TRACKED_RELATIONS)))
        facts = [
            f
            for f in (
                _to_fact(r["label"], r["entity_id"], r["props"], r["relations"], r["curies"])
                for r in rows
            )
            if f is not None
        ]
        return FactSet(
            facts=tuple(sorted(facts, key=lambda f: (f.entity_type, f.entity_id))),
            source_note=f"neo4j:{self.cfg.neo4j_kg_database} by id",
        )

    # ── fetch a whole type, for scans and demos ───────────────────────────────

    def for_type(self, label: str, limit: int = 500) -> FactSet:
        key = FACT_KEYS.get(label)
        if key is None:
            return FactSet(source_note=f"unknown entity type {label!r}")
        with self._driver.session(database=self.cfg.neo4j_kg_database) as s:
            rows = list(
                s.run(
                    _FETCH_ALL_OF_TYPE,
                    label=label,
                    key=key,
                    limit=limit,
                    tracked=list(TRACKED_RELATIONS),
                )
            )
        facts = [
            f
            for f in (
                _to_fact(label, r["entity_id"], r["props"], r["relations"], r["curies"])
                for r in rows
            )
            if f is not None
        ]
        return FactSet(
            facts=tuple(sorted(facts, key=lambda f: f.entity_id)),
            source_note=f"neo4j:{self.cfg.neo4j_kg_database} all :{label}",
        )

    def for_types(self, labels: Iterable[str], limit: int = 500) -> FactSet:
        facts: list[Fact] = []
        for label in labels:
            facts.extend(self.for_type(label, limit).facts)
        return FactSet(
            facts=tuple(sorted(facts, key=lambda f: (f.entity_type, f.entity_id))),
            source_note=f"neo4j:{self.cfg.neo4j_kg_database} multi-type",
        )

    def for_curies(self, curies: Sequence[str], limit: int = 200) -> FactSet:
        """Resolve CURIEs (as retrieved from the vector store) back to facts."""
        if not curies:
            return FactSet(source_note="no curies supplied")
        with self._driver.session(database=self.cfg.neo4j_kg_database) as s:
            rows = list(
                s.run(
                    """
                    MATCH (e)-[:MAPS_TO]->(c:ExternalConcept)
                    WHERE c.curie IN $curies
                    WITH DISTINCT e, labels(e)[0] AS label LIMIT $limit
                    RETURN label, e AS node
                    """,
                    curies=list(curies),
                    limit=limit,
                )
            )
        wanted: dict[str, list[str]] = {}
        for r in rows:
            label = r["label"]
            key = FACT_KEYS.get(label)
            if key is None:
                continue
            value = r["node"].get(key)
            if value:
                wanted.setdefault(label, []).append(str(value))
        return self.for_entities(wanted)

    # ── name-based resolution ─────────────────────────────────────────────────
    #
    # ID patterns alone are not enough: people ask about "HER2" and "Chronic
    # Myeloid Leukemia", not "G007" and "DIS003". Without this, natural phrasing
    # never brings an entity into scope, so no rule can bind to it and the
    # traversal view stays dark.

    _NAME_QUERY = """
    CALL () {
      MATCH (n:Drug)          RETURN 'Drug' AS l, n.drug_id AS k, n.name AS nm
      UNION MATCH (n:Drug)    RETURN 'Drug' AS l, n.drug_id AS k, n.generic_name AS nm
      UNION MATCH (n:Disease) RETURN 'Disease' AS l, n.disease_id AS k, n.name AS nm
      UNION MATCH (n:Gene)    RETURN 'Gene' AS l, n.gene_id AS k, n.symbol AS nm
      UNION MATCH (n:Gene)    RETURN 'Gene' AS l, n.gene_id AS k, n.name AS nm
      UNION MATCH (n:Protein) RETURN 'Protein' AS l, n.protein_id AS k, n.name AS nm
      UNION MATCH (n:Biomarker) RETURN 'Biomarker' AS l, n.biomarker_id AS k, n.name AS nm
      UNION MATCH (n:AdverseEvent) RETURN 'AdverseEvent' AS l, n.event_id AS k, n.name AS nm
      UNION MATCH (n:ClinicalTrial) RETURN 'ClinicalTrial' AS l, n.trial_id AS k, n.nct_id AS nm
      UNION MATCH (n:ClinicalTrial) RETURN 'ClinicalTrial' AS l, n.trial_id AS k, n.title AS nm
      UNION MATCH (n:Patient) RETURN 'Patient' AS l, n.patient_id AS k, n.patient_id AS nm
      UNION MATCH (n:ARD)     RETURN 'ARD' AS l, n.ard_id AS k, n.name AS nm
      UNION MATCH (n:Table)   RETURN 'Table' AS l, n.fqn AS k, n.table_name AS nm
    }
    WITH l, k, nm WHERE nm IS NOT NULL AND size(toString(nm)) >= 3
    RETURN l, k, toString(nm) AS nm
    """

    # Names short enough to collide with ordinary prose. Matched case-sensitively
    # so "APP" hits and "the app" does not.
    _SHORT_NAME_LEN = 4

    # Names that are common words even at length >= 4 and would over-match.
    _NAME_STOPLIST = frozenset({"type", "status", "protein", "mutation", "expression"})

    def name_index(self) -> list[tuple[str, str, str]]:
        """Cached (label, key, name) triples, longest name first.

        Longest-first ordering matters: "Chronic Myeloid Leukemia" must win over
        any shorter substring that also appears in it.
        """
        if getattr(self, "_name_cache", None) is None:
            with self._driver.session(database=self.cfg.neo4j_kg_database) as s:
                rows = [
                    (r["l"], str(r["k"]), r["nm"].strip())
                    for r in s.run(self._NAME_QUERY)
                    if r["nm"] and r["k"]
                ]
            seen: set[tuple[str, str, str]] = set()
            uniq = []
            for row in rows:
                if row[2].casefold() in self._NAME_STOPLIST:
                    continue
                if row not in seen:
                    seen.add(row)
                    uniq.append(row)
            self._name_cache = sorted(uniq, key=lambda r: -len(r[2]))
        return self._name_cache

    def resolve_by_name(self, text: str, limit: int = 12) -> dict[str, list[str]]:
        """Entity ids whose name appears in the text, as {label: [ids]}."""
        if not text:
            return {}
        import re as _re

        found: dict[str, list[str]] = {}
        # Spans are tracked per label. "HER2 Status" (Biomarker) and "HER2"
        # (Gene) overlap textually but are different entities, and a question
        # about one usually concerns the other too. Suppressing the inner match
        # only within the same label keeps genuine co-mentions while still
        # preventing one long name from matching itself twice.
        matched_spans: dict[str, list[tuple[int, int]]] = {}
        total = 0
        for label, key, name in self.name_index():
            if total >= limit:
                break
            if label not in FACT_KEYS:
                continue
            flags = 0 if len(name) < self._SHORT_NAME_LEN else _re.IGNORECASE
            try:
                pattern = _re.compile(rf"(?<![\w-]){_re.escape(name)}(?![\w-])", flags)
            except _re.error:
                continue
            m = pattern.search(text)
            if not m:
                continue
            # Skip a match wholly inside a longer name of the SAME label.
            spans = matched_spans.setdefault(label, [])
            if any(s <= m.start() and m.end() <= e for s, e in spans):
                continue
            spans.append((m.start(), m.end()))
            ids = found.setdefault(label, [])
            if key not in ids:
                ids.append(key)
                total += 1
        return found

    # ── display-name resolution, for the traversal view ───────────────────────

    def resolve_names(self, ids: Sequence[str]) -> dict[str, tuple[str, str]]:
        """Map entity ids (or CURIEs, or elementIds) to (label, display_name).

        The relation targets returned by the fact queries are bare identifiers.
        Rendering those raw gives a graph of opaque codes, so resolve them to
        human-readable names. Any id that cannot be resolved is simply absent
        from the result and the caller falls back to the id.
        """
        ids = [i for i in dict.fromkeys(ids) if i]
        if not ids:
            return {}
        out: dict[str, tuple[str, str]] = {}
        with self._driver.session(database=self.cfg.neo4j_kg_database) as s:
            rows = list(
                s.run(
                    """
                    UNWIND $ids AS wanted
                    OPTIONAL MATCH (n)
                    WHERE wanted IN [
                        n.patient_id, n.drug_id, n.disease_id, n.event_id,
                        n.trial_id, n.biomarker_id, n.gene_id, n.protein_id,
                        n.fqn, n.ard_id, n.entity_id, n.policy_id, n.curie,
                        n.phenotype_id, n.pathway_id, n.anatomy_id,
                        n.outcome_id, n.site_id, n.batch_id,
                        n.job_id, n.domain_id, n.bu_id, n.bf_id, n.app_id,
                        n.code, n.cui, n.concept_id
                    ] OR elementId(n) = wanted
                    WITH wanted, n LIMIT 400
                    RETURN wanted,
                           labels(n)[0] AS label,
                           coalesce(n.name, n.title, n.symbol, n.table_name,
                                    n.fqn, n.curie, wanted) AS display
                    """,
                    ids=ids,
                )
            )
        for r in rows:
            if r["label"]:
                out[r["wanted"]] = (r["label"], str(r["display"]))
        return out
