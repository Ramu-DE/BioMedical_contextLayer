# Requirements: Biomedical Context Layer

## Introduction

This feature implements a **context layer** over an existing biomedical knowledge
graph, following the definition proposed by Forrester (Evelson & Bandyopadhyay,
Aug 2026):

> "A context layer is the next evolution of semantic layers and knowledge graphs,
> providing the foundation for neurosymbolic AI context engineering and agentic AI
> applications. It combines business semantics and governance of semantic layers
> with the ontological modeling of knowledge graphs. The context layer represents
> all enterprise knowledge across data, metadata, business concepts, policies, and
> processes through graph-based ontologies and linked context. In addition, it
> continuously incorporates runtime context such as events, decisions, actions, and
> outcomes, creating a living model of the enterprise that enables AI reasoning,
> automation, and decision intelligence."

The reference diagram (`../../../context_Layer/context_layer_linkedin.png`) defines
15 capabilities across three columns. Every requirement below carries a **trace tag**
`[EK-n]`, `[RC-n]`, or `[AI-n]` mapping it to a specific diagram bullet, so coverage
is auditable.

### Reuse baseline

Verified already present in `ramu-de/BioMedical_KnowledgeGraph_ontology_MCP`
(vendored, not rewritten): 8 OWL/TTL ontology modules, 10 SHACL shapes with a
working `pyshacl` validator, `vocab_aligner`, `iri_minter`, `dcat_catalog`,
`rdf_lpg_converter`, Neo4j client, and 69 node/relationship CSVs.

Verified **absent** across all surveyed repos, and therefore in scope here:
entity linking, a decision-rule engine, and a durable runtime context store.

### Scope boundaries

| Boundary | Decision |
|---|---|
| Domain | Biomedical research only. Reference implementation, not enterprise-wide |
| Clinical use | Research questions only. Treatment recommendation is out of scope and actively blocked |
| Patient data | Synthetic or public datasets only. No PHI |
| "Outcomes" | Decision-quality outcomes (answer accepted, rule held, retrieval sufficient). **Not** clinical outcomes |
| "Continuous incorporation" | Write-back loop that keeps the graph current. **Not** model self-training |

---

## Requirement 1: Ontology-Grounded Concepts and Policies

**User story:** As a knowledge engineer, I want business concepts and governance
policies expressed as a formal ontology, so that every downstream answer resolves
to a governed definition rather than a string.

**Trace:** `[EK-1] Business Concepts & Policies`, `[EK-4] Graph-Based Ontologies`,
hub node `Ontologies`

### Acceptance Criteria

1. WHEN the ontology loader runs THEN the system SHALL load all 8 TTL modules
   (`foundation`, `clinical`, `patient`, `governance`, `commercial`,
   `medical_affairs`, `supply_quality`, `biomedkg-data`) into an in-memory RDF graph
   and report a per-module triple count.
2. IF a TTL module fails to parse THEN the system SHALL abort loading and report the
   offending module and line number, rather than continuing with partial knowledge.
3. WHEN a node is written to the knowledge graph THEN the system SHALL validate it
   against its SHACL shape and SHALL reject non-conforming nodes with a report
   naming the violated constraint.
4. WHEN a governance policy instance is created THEN the system SHALL require
   `policy_id`, `enforcement_level` ∈ {Mandatory, Advisory, Informational}, and
   `jurisdiction`, per `governance.ttl`.
5. WHEN a concept is referenced by any component THEN the system SHALL resolve it to
   a stable IRI minted by the deterministic IRI minter.

---

## Requirement 2: Data and Metadata with Provenance

**User story:** As a compliance reviewer, I want every assertion to carry its origin,
so that I can audit where an answer came from.

**Trace:** `[EK-2] Data & Metadata`, pillar `Trusted / Governed Data`

### Acceptance Criteria

1. WHEN a dataset is ingested THEN the system SHALL emit a DCAT catalog record
   capturing source, ingestion timestamp, and record count.
2. WHEN a graph assertion is created THEN the system SHALL attach `source`,
   `ingested_at`, and `confidence` properties.
3. IF an assertion lacks provenance THEN the system SHALL flag it as
   `provenance_missing` and SHALL exclude it from grounded answers.
4. WHEN provenance is requested for any answer THEN the system SHALL return the
   full chain from answer to graph node to source record.

---

## Requirement 3: Executable Business Rules

**User story:** As a domain expert, I want clinical and research rules expressed as
declarative artifacts evaluated deterministically, so that the AI cannot talk its
way around a constraint.

**Trace:** `[EK-3] Processes & Rules`, `[AI-1] Neurosymbolic Reasoning`

### Acceptance Criteria

1. WHEN a rule pack is loaded THEN the system SHALL validate each rule has `id`,
   `description`, `severity` ∈ {block, warn, inform}, `when` predicate, and
   `rationale`.
2. WHEN the rule engine evaluates a context package THEN the system SHALL return an
   ordered list of fired rules, each with `rule_id`, `severity`, `rationale`, and the
   graph entities that triggered it.
3. WHEN no rule fires THEN the system SHALL return an empty list, and SHALL NOT
   infer or invent a rule.
4. IF a rule of severity `block` fires THEN the system SHALL suppress the generated
   answer and SHALL return the rule's rationale instead.
5. WHEN the same context package is evaluated twice THEN the system SHALL return
   identical results, with no LLM involvement in rule evaluation.
6. WHEN the rule engine runs THEN evaluation SHALL complete without any network call.

---

## Requirement 4: Linked Context Between Text and Graph

**User story:** As a researcher, I want passages of literature linked to the exact
graph entities they discuss, so that retrieval and reasoning operate on the same
identifiers.

**Trace:** `[EK-5] Linked Enterprise Context`, hub node `Linked Context`

### Acceptance Criteria

1. WHEN the term index is built THEN the system SHALL embed every ontology term
   label and synonym into an OpenSearch k-NN index keyed by CURIE.
2. WHEN a text chunk is linked THEN the system SHALL emit zero or more
   `EntityMention` records, each with `chunk_id`, `surface_form`, `curie`,
   `confidence`, and character offsets.
3. WHEN two different surface forms denote the same concept (for example "aspirin"
   and "acetylsalicylic acid") THEN the system SHALL resolve both to the same CURIE.
4. IF a mention's best match scores below the linking threshold THEN the system SHALL
   emit it as `unlinked` rather than binding it to a low-confidence CURIE.
5. WHEN a mention is linked THEN the system SHALL verify the target CURIE exists in
   the knowledge graph, and SHALL reject the link if it does not.

---

## Requirement 5: Runtime Events and Triggers

**User story:** As a platform operator, I want the system to record what is happening
now, so that context reflects the present rather than only the curated past.

**Trace:** `[RC-1] Events & Triggers`, hub node `Runtime Context`

### Acceptance Criteria

1. WHEN an event is ingested THEN the system SHALL persist it as a
   `Ctx_Event` node with `event_id`, `event_type`, `occurred_at`, `ingested_at`, and
   `payload`.
2. WHEN an event references a known entity THEN the system SHALL create an edge to
   that entity's graph node.
3. WHEN events are written THEN the system SHALL use the isolated context database
   or label prefix, and SHALL NOT modify curated knowledge graph nodes.
4. IF an event arrives out of order THEN the system SHALL preserve both `occurred_at`
   and `ingested_at` so that bi-temporal queries remain correct.

---

## Requirement 6: Decisions, Actions, and Outcomes

**User story:** As an auditor, I want every AI decision recorded with its inputs,
rules, and result, so that I can reconstruct why the system answered as it did.

**Trace:** `[RC-2] Decisions & Outcomes`, `[RC-3] Actions & Responses`,
`[AI-3] Decision Intelligence`

### Acceptance Criteria

1. WHEN the agent answers a question THEN the system SHALL persist a
   `Ctx_Decision` node containing `decision_id`, `question`, `context_package_id`,
   `rules_fired`, `answer_hash`, `grounding_confidence`, and `decided_at`.
2. WHEN the agent invokes a tool THEN the system SHALL persist a `Ctx_Action` node
   linked to the owning decision, with tool name, arguments, duration, and status.
3. WHEN an outcome is recorded THEN the system SHALL link it to its decision with
   `outcome_type`, `observed_at`, and `detail`.
4. WHEN a decision is persisted THEN the write SHALL be append-only; the system SHALL
   NOT support updating or deleting a decision record.
5. WHEN the process restarts THEN previously written decisions SHALL still be
   retrievable, proving durability rather than in-memory logging.

---

## Requirement 7: Continuous Incorporation and Living Model

**User story:** As a researcher, I want the layer to carry forward what it has already
learned about my questions, so that context accumulates instead of resetting.

**Trace:** `[RC-4] Continuous Incorporation`, `[RC-5] Living Enterprise Model`,
pillar `Runtime / Continuous`

### Acceptance Criteria

1. WHEN a new question is asked THEN the context assembler SHALL retrieve relevant
   prior decisions and SHALL include them in the context package.
2. WHEN an assertion's validity changes THEN the system SHALL set `valid_to` on the
   prior version and create a new version with `valid_from`, rather than overwriting.
3. WHEN a caller queries state "as of" a timestamp THEN the system SHALL return the
   assertions valid at that timestamp.
4. WHEN prior context is incorporated THEN the system SHALL NOT alter any model
   weights; incorporation SHALL be limited to graph write-back.

---

## Requirement 8: Context Assembly

**User story:** As an AI engineer, I want a single component that decides what enters
the model's window, so that retrieval is deliberate, bounded, and inspectable.

**Trace:** hub `Context Layer / Living Enterprise Model`, `[AI-4] AI Automation & Planning`

### Acceptance Criteria

1. WHEN a question is received THEN the assembler SHALL retrieve from all three
   sources: OpenSearch vector search, Neo4j graph traversal, and the runtime store.
2. WHEN sources are merged THEN the assembler SHALL deduplicate by CURIE and rank by
   combined relevance.
3. WHEN the package is built THEN the assembler SHALL enforce a configured token
   budget and SHALL report what was truncated.
4. WHEN the package is emitted THEN it SHALL carry a `context_package_id` and every
   element SHALL retain its provenance.
5. IF a retrieval source is unavailable THEN the assembler SHALL degrade gracefully,
   record the degradation in the package, and SHALL NOT fail silently.

---

## Requirement 9: Neurosymbolic Reasoning

**User story:** As a researcher, I want pattern recognition and rule following in one
answer, so that the system is both fluent and correct.

**Trace:** `[AI-1] Neurosymbolic Reasoning`, `[AI-2] Agentic AI Applications`,
pillars `Neurosymbolic`, `Agentic`

### Acceptance Criteria

1. WHEN a question is answered THEN the neural component SHALL propose an answer from
   the context package and the symbolic component SHALL evaluate rules independently.
2. WHEN a `block` rule contradicts the proposed answer THEN the symbolic result SHALL
   take precedence.
3. WHEN the response is returned THEN it SHALL expose both the neural output and the
   symbolic verdict, so the interaction between them is visible.
4. WHEN the agent runs THEN it SHALL select its own tools across graph, vector, and
   rule interfaces without a hardcoded sequence.

---

## Requirement 10: Governed AI Responses

**User story:** As a risk owner, I want the system to refuse rather than guess, so
that ungrounded output never reaches a user.

**Trace:** `[AI-5] Governed AI Responses`, hub node `Governance`

### Acceptance Criteria

1. WHEN a response is produced THEN the system SHALL return the contract fields
   `answer`, `citations`, `rules_fired`, `context_package_id`,
   `grounding_confidence`, and `degradations`.
2. IF `grounding_confidence` is below `MIN_GROUNDING_CONFIDENCE` THEN the system
   SHALL refuse to answer and SHALL state what context was missing.
3. IF `CLINICAL_ADVICE_GUARD` is enabled AND the question requests individual
   treatment guidance THEN the system SHALL decline and redirect to research framing.
4. WHEN an answer contains a factual claim THEN each claim SHALL carry at least one
   citation resolving to a chunk ID or a graph CURIE.
5. WHEN credentials are loaded THEN the system SHALL NOT write secret values to logs
   or to the response.
6. WHEN a query is issued to Neo4j or OpenSearch THEN it SHALL be parameterized, and
   the system SHALL reject attempts at Cypher or query-DSL injection.

---

## Requirement 11: Verifiable Proof of Value

**User story:** As a stakeholder, I want evidence that the context layer beats plain
RAG, so that the claim rests on measurement rather than assertion.

**Trace:** whole-diagram acceptance; supports the `EVOLUTION` strip

### Acceptance Criteria

1. WHEN the evaluation harness runs THEN the system SHALL execute an identical
   question set through a naive RAG baseline and through the context layer.
2. WHEN results are compared THEN the harness SHALL report groundedness, citation
   coverage, rule-compliance rate, and refusal correctness for both.
3. WHEN an adversarial scenario is injected (conflicting information, pattern
   pollution, signal drowning) THEN the harness SHALL report how each pipeline
   responded.
4. WHEN evaluation completes THEN the harness SHALL emit a machine-readable results
   file suitable for publication.

---

## Coverage Matrix

| # | Diagram element | Requirement |
|---|---|---|
| EK-1 | Business Concepts & Policies | R1 |
| EK-2 | Data & Metadata | R2 |
| EK-3 | Processes & Rules | R3 |
| EK-4 | Graph-Based Ontologies | R1 |
| EK-5 | Linked Enterprise Context | R4 |
| RC-1 | Events & Triggers | R5 |
| RC-2 | Decisions & Outcomes | R6 |
| RC-3 | Actions & Responses | R6 |
| RC-4 | Continuous Incorporation | R7 |
| RC-5 | Living Enterprise Model | R7 |
| AI-1 | Neurosymbolic Reasoning | R3, R9 |
| AI-2 | Agentic AI Applications | R9 |
| AI-3 | Decision Intelligence | R6 |
| AI-4 | AI Automation & Planning | R8, R9 |
| AI-5 | Governed AI Responses | R10 |
| Hub | Ontologies | R1 |
| Hub | Governance | R10 |
| Hub | Runtime Context | R5, R6 |
| Hub | Linked Context | R4 |
| — | Proof of value | R11 |

All 15 diagram bullets and all 4 hub nodes are covered.
