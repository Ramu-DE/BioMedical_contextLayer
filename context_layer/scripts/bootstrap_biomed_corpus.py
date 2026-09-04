#!/usr/bin/env python3
"""Index the biomedical corpus into its own Qdrant collection. Task 2.3.

Two things make this a context layer rather than plain RAG:

1. **Graph-enriched text.** Each chunk is composed by traversing the graph, so a
   drug chunk states the diseases it treats and the proteins it targets. The
   relationship is inside the retrievable text, not just in the database.
2. **CURIE-linked payloads.** Every chunk carries the CURIEs of the entities it
   describes, pulled from the MAPS_TO crosswalk. That is the join key the rule
   engine and the graph retriever use (Requirements 4.3, 8.2).

Usage:
    .venv/bin/python scripts/bootstrap_biomed_corpus.py [--dry-run] [--recreate]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from context_layer.clients.embeddings import BedrockEmbeddings  # noqa: E402
from context_layer.clients.vector_store import (  # noqa: E402
    QdrantVectorStore,
    VectorRecord,
)
from neo4j import GraphDatabase  # noqa: E402

COLLECTION = "ctx_biomed_chunks"

# Each query returns: id, text, curies. Text is composed in Cypher so the
# traversal that enriches it is explicit and reviewable.
CORPUS_QUERIES: list[tuple[str, str]] = [
    (
        "Drug",
        """
        MATCH (d:Drug {dataset:'biomed'})
        OPTIONAL MATCH (d)-[t:TREATS]->(dis:Disease)
        OPTIONAL MATCH (d)-[:TARGETS]->(p:Protein)
        OPTIONAL MATCH (d)-[:MAPS_TO]->(c:ExternalConcept)
        WITH d,
             collect(DISTINCT dis.name) AS diseases,
             collect(DISTINCT p.name) AS proteins,
             collect(DISTINCT c.curie) AS curies
        RETURN d.drug_id AS id,
          'Drug ' + d.name + ' (' + coalesce(d.generic_name,'') + '), a ' +
          coalesce(d.drug_type,'therapeutic') + ' with mechanism: ' +
          coalesce(d.mechanism,'unspecified') + '. Approval status: ' +
          coalesce(d.approval_status,'unknown') +
          CASE WHEN d.approval_year IS NULL THEN '' ELSE ' since ' + toString(d.approval_year) END +
          '. Treats: ' + CASE WHEN size(diseases)=0 THEN 'no recorded indication'
                              ELSE reduce(s='', x IN diseases | s + x + '; ') END +
          'Targets protein: ' + CASE WHEN size(proteins)=0 THEN 'none recorded'
                                     ELSE reduce(s='', x IN proteins | s + x + '; ') END
          AS text,
          curies AS curies
        """,
    ),
    (
        "Disease",
        """
        MATCH (dis:Disease {dataset:'biomed'})
        OPTIONAL MATCH (dis)-[:HAS_PHENOTYPE]->(ph:Phenotype)
        OPTIONAL MATCH (g:Gene)-[:ASSOCIATED_WITH]->(dis)
        OPTIONAL MATCH (dis)-[:MAPS_TO]->(c:ExternalConcept)
        WITH dis, collect(DISTINCT ph.name) AS phenos,
             collect(DISTINCT g.symbol) AS genes,
             collect(DISTINCT c.curie) AS curies
        RETURN dis.disease_id AS id,
          'Disease ' + dis.name + ' in category ' + coalesce(dis.category,'unclassified') +
          ', ICD-10 ' + coalesce(dis.icd10_code,'n/a') + ', prevalence ' +
          coalesce(dis.prevalence,'unknown') + '. Phenotypes: ' +
          CASE WHEN size(phenos)=0 THEN 'none recorded'
               ELSE reduce(s='', x IN phenos | s + x + '; ') END +
          'Associated genes: ' + CASE WHEN size(genes)=0 THEN 'none recorded'
               ELSE reduce(s='', x IN genes | s + x + '; ') END
          AS text, curies AS curies
        """,
    ),
    (
        "Gene",
        """
        MATCH (g:Gene {dataset:'biomed'})
        OPTIONAL MATCH (g)-[:ASSOCIATED_WITH]->(dis:Disease)
        OPTIONAL MATCH (g)-[:PARTICIPATES_IN]->(pw:Pathway)
        WITH g, collect(DISTINCT dis.name) AS diseases, collect(DISTINCT pw.name) AS pathways
        RETURN g.gene_id AS id,
          'Gene ' + g.symbol + ' (' + coalesce(g.name,'') + ') on chromosome ' +
          coalesce(toString(g.chromosome),'?') + '. Function: ' + coalesce(g.function,'unknown') +
          '. Associated diseases: ' + CASE WHEN size(diseases)=0 THEN 'none recorded'
               ELSE reduce(s='', x IN diseases | s + x + '; ') END +
          'Pathways: ' + CASE WHEN size(pathways)=0 THEN 'none recorded'
               ELSE reduce(s='', x IN pathways | s + x + '; ') END
          AS text, [] AS curies
        """,
    ),
    (
        "Protein",
        """
        MATCH (p:Protein {dataset:'biomed'})
        OPTIONAL MATCH (d:Drug)-[:TARGETS]->(p)
        WITH p, collect(DISTINCT d.name) AS drugs
        RETURN p.protein_id AS id,
          'Protein ' + p.name + ' (UniProt ' + coalesce(p.uniprot_id,'n/a') + '), class ' +
          coalesce(p.protein_class,'unclassified') + ', located in ' +
          coalesce(p.cellular_location,'unknown') + '. Targeted by: ' +
          CASE WHEN size(drugs)=0 THEN 'no recorded drug'
               ELSE reduce(s='', x IN drugs | s + x + '; ') END
          AS text, [] AS curies
        """,
    ),
    (
        "ClinicalTrial",
        """
        MATCH (t:ClinicalTrial {dataset:'biomed'})
        OPTIONAL MATCH (t)-[:INVESTIGATES]->(d:Drug)
        OPTIONAL MATCH (t)-[:STUDIES]->(dis:Disease)
        OPTIONAL MATCH (t)-[:REPORTS]->(ae:AdverseEvent)
        WITH t, collect(DISTINCT d.name) AS drugs, collect(DISTINCT dis.name) AS diseases,
             collect(DISTINCT ae.name) AS events
        RETURN t.trial_id AS id,
          'Clinical trial ' + coalesce(t.nct_id,t.trial_id) + ': ' + t.title + '. ' +
          coalesce(t.phase,'phase unknown') + ', status ' + coalesce(t.status,'unknown') +
          ', sponsor ' + coalesce(t.sponsor,'unknown') + ', enrollment ' +
          coalesce(toString(t.enrollment),'unreported') + '. Investigates: ' +
          CASE WHEN size(drugs)=0 THEN 'none' ELSE reduce(s='', x IN drugs | s + x + '; ') END +
          'Studies: ' + CASE WHEN size(diseases)=0 THEN 'none' ELSE reduce(s='', x IN diseases | s + x + '; ') END +
          'Reported adverse events: ' + CASE WHEN size(events)=0 THEN 'none recorded'
               ELSE reduce(s='', x IN events | s + x + '; ') END
          AS text, [] AS curies
        """,
    ),
    (
        "AdverseEvent",
        """
        MATCH (ae:AdverseEvent {dataset:'biomed'})
        OPTIONAL MATCH (ae)-[:MAPS_TO]->(c:ExternalConcept)
        OPTIONAL MATCH (t:ClinicalTrial)-[r:REPORTS]->(ae)
        WITH ae, collect(DISTINCT c.curie) AS curies, collect(DISTINCT t.nct_id) AS trials
        RETURN ae.event_id AS id,
          'Adverse event ' + ae.name + ', severity ' + coalesce(ae.severity,'unknown') +
          ', category ' + coalesce(ae.category,'unclassified') + ', frequency ' +
          coalesce(ae.frequency,'unknown') + '. Reported in trials: ' +
          CASE WHEN size(trials)=0 THEN 'none' ELSE reduce(s='', x IN trials | s + x + '; ') END
          AS text, curies AS curies
        """,
    ),
    (
        "DataGovernancePolicy",
        """
        MATCH (p:DataGovernancePolicy {dataset:'biomed'})
        OPTIONAL MATCH (p)-[g:GOVERNS_ENTITY]->(e)
        WITH p, collect(DISTINCT labels(e)[0]) AS kinds, collect(DISTINCT g.enforcement_level) AS levels
        RETURN p.policy_id AS id,
          'Biomedical governance policy ' + p.title + ' in category ' +
          coalesce(p.category,'general') + '. Scope: ' + coalesce(p.scope,'unspecified') +
          '. Owner: ' + coalesce(p.owner,'unassigned') + '. Status: ' +
          coalesce(p.status,'unknown') + '. Governs entity types: ' +
          CASE WHEN size(kinds)=0 THEN 'none' ELSE reduce(s='', x IN kinds | s + x + '; ') END +
          'Enforcement: ' + CASE WHEN size(levels)=0 THEN 'unspecified'
               ELSE reduce(s='', x IN levels | s + x + '; ') END
          AS text, [] AS curies
        """,
    ),
    (
        "ClusterSummary",
        """
        MATCH (cs:ClusterSummary {dataset:'biomed'})
        RETURN cs.summary_id AS id,
          'Knowledge cluster summary: ' + coalesce(cs.summary_text,'') +
          ' Key entities: ' + coalesce(cs.key_entities,'none') +
          '. Therapeutic relevance: ' + coalesce(cs.therapeutic_relevance,'unstated')
          AS text, [] AS curies
        """,
    ),
    (
        "ResearchPaper",
        """
        MATCH (rp:ResearchPaper {dataset:'biomed'})
        OPTIONAL MATCH (rp)-[:MENTIONS_DRUG]->(d:Drug)
        OPTIONAL MATCH (rp)-[:MENTIONS_DISEASE]->(dis:Disease)
        WITH rp, collect(DISTINCT d.name) AS drugs, collect(DISTINCT dis.name) AS diseases
        RETURN rp.paper_id AS id,
          'Research paper: ' + rp.title + '. Published in ' + coalesce(rp.journal,'unknown') +
          CASE WHEN rp.publication_date IS NULL THEN '' ELSE ' on ' + toString(rp.publication_date) END +
          ', DOI ' + coalesce(rp.doi,'n/a') + ', cited ' + coalesce(toString(rp.citations),'0') +
          ' times. Mentions drugs: ' + CASE WHEN size(drugs)=0 THEN 'none'
               ELSE reduce(s='', x IN drugs | s + x + '; ') END +
          'Mentions diseases: ' + CASE WHEN size(diseases)=0 THEN 'none'
               ELSE reduce(s='', x IN diseases | s + x + '; ') END
          AS text, [] AS curies
        """,
    ),
    (
        "Biomarker",
        """
        MATCH (b:Biomarker {dataset:'biomed'})
        OPTIONAL MATCH (b)-[r:PREDICTS_RESPONSE_TO]->(d:Drug)
        WITH b, collect(DISTINCT d.name + ' (predictive value ' +
                        coalesce(toString(r.predictive_value),'?') + ')') AS preds
        RETURN b.biomarker_id AS id,
          'Biomarker ' + b.name + ' of type ' + coalesce(b.type,'unspecified') +
          ', measured in ' + coalesce(b.measurement_unit,'n/a') +
          '. Clinical significance: ' + coalesce(b.clinical_significance,'unstated') +
          '. Predicts response to: ' + CASE WHEN size(preds)=0 THEN 'none recorded'
               ELSE reduce(s='', x IN preds | s + x + '; ') END
          AS text, [] AS curies
        """,
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--recreate", action="store_true", help="drop the collection first")
    args = ap.parse_args()

    cfg = config_module.load()
    if problems := cfg.check():
        for p in problems:
            print(f"  - {p}")
        return 1

    driver = GraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_user, str(cfg.neo4j_password))
    )
    items: list[dict] = []
    try:
        with driver.session(database=cfg.neo4j_kg_database) as s:
            for label, cypher in CORPUS_QUERIES:
                rows = list(s.run(cypher))
                for r in rows:
                    text = (r["text"] or "").strip()
                    if not text:
                        continue
                    items.append(
                        {
                            "source_id": r["id"],
                            "source": f"neo4j:{label}",
                            "text": text,
                            "curies": list(r["curies"] or []),
                        }
                    )
                print(f"  {len(rows):>4}  {label}")
    finally:
        driver.close()

    linked = sum(1 for i in items if i["curies"])
    print(f"\n{len(items)} chunks composed; {linked} carry CURIEs")
    if items:
        print(f"\nsample:\n  {items[0]['text'][:240]}...")
        print(f"  curies: {items[0]['curies']}")
    if args.dry_run:
        print("\n--dry-run: stopping before embedding")
        return 0

    # Dedicated collection, separate from the governance corpus.
    store = QdrantVectorStore(cfg, client=None)
    store.collection = COLLECTION
    if args.recreate:
        try:
            store._c.delete_collection(COLLECTION)
            print(f"\ndropped {COLLECTION}")
        except Exception as e:  # noqa: BLE001
            print(f"\n(no existing collection to drop: {type(e).__name__})")
    store.ensure_collection(cfg.embedding_dim)
    print(f"collection {COLLECTION!r} ready (dim={cfg.embedding_dim})")

    embedder = BedrockEmbeddings(cfg)
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for it in items:
        records.append(
            VectorRecord(
                text=it["text"],
                vector=embedder.embed_one(it["text"]),
                payload={
                    "source": it["source"],
                    "source_id": it["source_id"],
                    "curies": it["curies"],
                    "dataset": "biomed",
                    "ingested_at": now,
                    "embedding_model": cfg.embedding_model_id,
                    "confidence": 1.0,
                },
            )
        )
    print(f"embedded {len(records)} chunks")
    print(f"upserted {store.upsert(records)}; collection holds {store.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
