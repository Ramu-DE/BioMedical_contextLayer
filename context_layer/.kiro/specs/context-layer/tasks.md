# Implementation Plan: Biomedical Context Layer

Ordered by dependency. Each task names the requirements it satisfies. Tasks marked
**[gate]** unblock multiple downstream tasks and should not be deferred.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## Phase 0 — Foundations

- [x] 0.1 Create module structure, `.config.kiro`, spec documents
  - `.kiro/specs/context-layer/{requirements,design,tasks}.md`

- [x] 0.2 Secret-safe configuration
  - `config.py` with `Secret` type, `check()`, redacted `report()`
  - `.env.example`; root `.gitignore` protecting `.env`
  - _Requirements: 10.5_

- [x] 0.3 Packaging and test scaffolding
  - `pyproject.toml` with deps: `neo4j`, `opensearch-py`, `boto3`, `rdflib`,
    `pyshacl`, `pydantic`, `pyyaml`, `langgraph`, `langchain-aws`
  - `pytest` config; `tests/conftest.py` with a `skip_without_credentials` marker
  - Verify: `pytest` collects and passes with no `.env` present

- [x] 0.4 Vendor the verified biomedical assets
  - Copy 8 TTL modules into `ontology/`, 10 SHACL shapes into `src/context_layer/knowledge/shapes/`
  - Copy the 69 node/relationship CSVs into `data/`
  - Record upstream commit SHA in `ontology/PROVENANCE.md`
  - _Requirements: 1.1, 2.1_

---

## Phase 1 — Enterprise Knowledge

- [x] 1.1 **[gate]** Client adapters
  - `clients/neo4j_client.py` — parameterized `query()`, database selection, retry
  - `clients/opensearch_client.py` — `knn_search`, `hybrid_search`, `bulk_index`,
    index creation with HNSW mapping at `EMBEDDING_DIM`
  - `clients/embeddings.py` — batched Bedrock Titan embedding with backoff
  - Unit tests with mocked transports; injection strings must not reach a query
  - _Requirements: 10.6_

- [x] 1.2 Ontology loader
  - `knowledge/ontology_loader.py` — load 8 modules, per-module triple counts,
    abort with module and line on parse failure
  - _Requirements: 1.1, 1.2_

- [x] 1.3 SHACL write gate
  - `knowledge/shacl_gate.py` wrapping `pyshacl`; reject non-conforming nodes with
    the violated constraint named
  - _Requirements: 1.3_

- [x] 1.4 IRI minting and policy instances
  - Deterministic IRI minter; governance policy instances requiring `policy_id`,
    `enforcement_level`, `jurisdiction`
  - _Requirements: 1.4, 1.5_

- [x] 1.5 Provenance and DCAT catalog
  - `knowledge/provenance.py` — attach `source`/`ingested_at`/`confidence`;
    flag `provenance_missing`; `chain_for(answer_id)`
  - `knowledge/catalog.py` — DCAT record on ingest
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 1.6 Load the knowledge graph
  - `scripts/load_kg.py` — CSVs into Neo4j through the SHACL gate, with provenance
  - Verify: node and relationship counts match source CSVs

---

## Phase 2 — Linked Context (the keystone)

- [x] 2.1 **[gate]** Ontology term index
  - `linking/term_index.py` — embed every label and synonym into
    `biomed_ontology_terms`, keyed by CURIE
  - `scripts/build_term_index.py`
  - _Requirements: 4.1_

- [x] 2.2 **[gate]** Entity linker
  - `linking/entity_linker.py` — k-NN over the term index, threshold, `unlinked`
    fallback, Neo4j existence verification
  - Acceptance: "aspirin" and "acetylsalicylic acid" resolve to one CURIE
  - _Requirements: 4.2, 4.3, 4.4, 4.5_

- [x] 2.3 Chunk ingestion with links
  - `scripts/ingest_chunks.py` — chunk, embed, index into `biomed_chunks`, persist
    `EntityMention` edges into Neo4j
  - Property test: no emitted CURIE is absent from the graph
  - _Requirements: 4.5_

---

## Phase 3 — Runtime Context

- [x] 3.1 Runtime models
  - `runtime/models.py` — frozen `Event`, `Decision`, `Action`, `Outcome`
  - _Requirements: 5.1, 6.1, 6.2, 6.3_

- [x] 3.2 **[gate]** Append-only store
  - `runtime/store.py` — append verbs only; no update or delete methods
  - Bi-temporal `valid_from`/`valid_to`, `occurred_at`/`ingested_at`
  - Writes confined to the context database or `Ctx_` prefix
  - Property test: durability across a reconnect; no decision is lost
  - _Requirements: 5.1–5.4, 6.4, 6.5, 7.2_

- [x] 3.3 Temporal queries
  - `as_of(timestamp)`; `recent_decisions(embedding, k)` for write-back retrieval
  - _Requirements: 7.1, 7.3_

- [x] 3.4 Event ingestion
  - `scripts/emit_event.py`; link events to entities via `ABOUT`
  - _Requirements: 5.2_

---

## Phase 4 — Symbolic Reasoning

- [x] 4.1 Rule pack schema and loader
  - `rules/loader.py` — validate `id`, `description`, `severity`, `when`, `rationale`
  - Fail at startup on a malformed pack, never at request time
  - _Requirements: 3.1_

- [x] 4.2 **[gate]** Rule engine
  - `rules/engine.py` — pure `evaluate(package) -> RuleVerdict`
  - No network, no LLM; identical input yields identical output
  - _Requirements: 3.2, 3.3, 3.5, 3.6_

- [x] 4.3 Author the rule packs
  - `rules/packs/contraindications.yaml`, `interactions.yaml`, `eligibility.yaml`
  - 15–20 rules grounded in the existing `drug_treats_disease`,
    `drug_targets_protein`, `exposure_affects_gene` relationships
  - Each rule cites a governance policy IRI
  - _Requirements: 3.1, 3.4_

---

## Phase 5 — Context Assembly

- [x] 5.1 Retrievers
  - `assembler/retrievers.py` — `VectorRetriever` (OpenSearch hybrid),
    `GraphRetriever` (CURIE traversal), `RuntimeRetriever` (prior decisions)
  - Each degrades independently and reports its degradation
  - _Requirements: 8.1, 8.5_

- [x] 5.2 **[gate]** Context package assembly
  - `assembler/package.py` — merge, dedup by CURIE, rank, enforce token budget,
    report truncation, mint `context_package_id`, retain per-element provenance
  - Adapt the `Assembler`/`AssembledPipeline` pattern from `rag-ai-factory`
  - _Requirements: 8.2, 8.3, 8.4_

---

## Phase 6 — Governance and Agent

- [x] 6.1 Grounding and guards
  - `governance/grounding.py` — confidence score, per-claim citation coverage
  - `governance/guards.py` — clinical advice guard, refusal messages
  - Refuse below `MIN_GROUNDING_CONFIDENCE` stating what context was missing
  - _Requirements: 10.2, 10.3, 10.4_

- [x] 6.2 Response contract
  - `governance/contract.py` — `GovernedResponse` with `neural_proposal` and
    `symbolic_verdict` both exposed
  - _Requirements: 10.1, 9.3_

- [x] 6.3 Agent tools
  - `agent/tools.py` — `graph_lookup`, `vector_search`, `check_rules`,
    `prior_decisions`
  - _Requirements: 9.4_

- [x] 6.4 **[gate]** Neurosymbolic agent
  - `agent/agent.py` — LangGraph ReAct on Bedrock; rules evaluated on the package
    before release; `block` suppresses the answer and returns the rationale
  - HTTP `POST /invoke`, `GET /health`, MicroVM ready hook on 9000, matching module 5
  - _Requirements: 9.1, 9.2, 3.4_

- [x] 6.5 Write-back loop
  - On every response, append `Ctx_Decision` and `Ctx_Action`; make them retrievable
    as context for the next question
  - Acceptance: decision N+1 retrieves decision N
  - _Requirements: 7.1, 7.4, 6.1, 6.2_

---

## Phase 7 — Proof

- [x] 7.1 Naive RAG baseline
  - Vector-only, no graph, no rules, no governance — the control arm
  - _Requirements: 11.1_

- [x] 7.2 Evaluation harness
  - Identical question set through both pipelines
  - Report groundedness, citation coverage, rule-compliance rate, refusal correctness
  - Machine-readable results file
  - _Requirements: 11.1, 11.2, 11.4_

- [x] 7.3 Adversarial scenarios
  - Port `conflicting_info`, `pattern_pollution`, `signal_drowning` generators from
    `AI_Agent_Context_Trap`
  - _Requirements: 11.3_

- [x] 7.4 Deployment
  - Dockerfile and `build-image.sh` following module 5
  - `README.md` with setup, the coverage matrix, and sample `GovernedResponse`

---

## Critical path

```
0.3 → 1.1 → 2.1 → 2.2 → 5.2 → 6.4 → 6.5 → 7.2
             ↑      ↑      ↑      ↑
           1.2    1.6    3.2    4.2
```

`2.2` (entity linker) and `4.2` (rule engine) are the two components that do not
exist in any surveyed repository. Everything distinctive about this build depends
on them.

## Definition of done

The claim is proven when a single `POST /invoke` returns a `GovernedResponse` that
traces question → context package → graph CURIE → fired rule → answer, **and** the
same question through the naive baseline demonstrably lacks that trace.
