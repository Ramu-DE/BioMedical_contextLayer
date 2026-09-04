# Data provenance — who created what

Written so that any claim about this system can be traced to its origin. The
distinction matters: *"we found a real governance contradiction"* and *"we planted
one"* look identical in a screenshot.

Every node and edge falls into exactly one of four categories, and each is
distinguishable by its properties.

---

## Summary

| Origin | Nodes | Relationships | How to identify |
|---|---:|---:|---|
| **Pre-existing** — already in the graph | 124 | 493 | no `dataset`, no `source`, no `derivation` |
| **Loaded** — from biomedical CSVs | 387 | 683 | `dataset='biomed'` + `source=github:...` + `ingested_at` |
| **Derived** — inferred by this project | 34 | 21 | `derivation` + `evidence` + `confidence` |
| **Runtime** — written while answering | 278 | — | label prefixed `Ctx_` |

Nothing is ambiguous. If a node has no provenance properties, this project did
not create it.

---

## 1. Pre-existing — 124 nodes

An enterprise pharma **data governance graph** that was already in the Neo4j
instance before this project wrote anything. Surveyed 2026-08-27 ~16:16 UTC.

```
Policy(6)   RuleType(10)  Table(7)     ARD(7)      DataDomain(8)
DataLayer(4) Entity(6)    Job(7)       Batch(6)    Application(3)
BusinessUnit(3) BusinessFunction(6)    LoadType(4) ProcessingType(6)
TherapeuticArea(6)  SNOMED(8)  RxNorm(8)  MedDRA(10)  OMOP(9)
```

### The six policies were not written for this demo

This is the single most important provenance fact.

| policy_id | name | scope |
|---|---|---|
| `POL_HUB_DQ` | Hub tables require DQ | hub |
| `POL_PII_MASK` | PII must be masked | all |
| `POL_CROSS_BU` | Cross-BU write approval | all |
| `POL_INCREMENTAL_KEY` | Incremental needs keys | all |
| `POL_ARCHIVE_RAW` | Archive raw files | raw |
| `POL_SCD_AUDIT` | SCD requires audit columns | hub |

`POL_HUB_DQ` in full, exactly as found:

```
labels:      ['Policy']
policy_id:   'POL_HUB_DQ'
name:        'Hub tables require DQ'
description: 'All hub layer tables must have at least one critical DQ rule'
scope:       'hub'
```

Four properties. **No** `dataset`, `source`, `ingested_at`, `confidence` or
`derivation` — the markers every node this project wrote carries. Compare a node
we did load:

```
policy_id:   'POL001'
title:       'Patient Data Privacy Policy'
dataset:     'biomed'
source:      'github:ramu-de/BioMedical_KnowledgeGraph_ontology_MCP/data_governance_policies.csv'
ingested_at: '2026-08-27T17:30:49'
```

It is also wired into pre-existing structure: `DataLayer -[GOVERNED_BY]-> Policy`,
15 such edges.

**No authoring timestamp exists.** The node has no `created_at`, so the true
creation date lives in whatever pipeline seeded the instance, not in the graph.
What can be established: present before this project's first write, carrying none
of this project's provenance, already connected to `GOVERNED_BY` structure.

### Zero DQRule nodes

The graph contains **no `DQRule` instances at all**. That is why every hub table
violates `POL_HUB_DQ` — a genuine pre-existing condition, not a contrived one.

### What the project contributed

The policy existed as a *description*. Nothing executed it. This project added
the rule that enforces it:

```yaml
id: governance.hub_table_without_dq_rule
severity: block
policy_iri: "graph:Policy/POL_HUB_DQ"
when:
  entity: Table
  where: {layer: {equals: hub}}
  without_relation: HAS_DQ_RULE
```

**Honest framing:** your organisation wrote the policy. It sat in the graph as
prose nobody enforced. We made it executable — and the first thing it did was
block a certification sign-off.

---

## 2. Loaded — 387 nodes, 683 relationships

Synthetic biomedical data from
[`BioMedical_KnowledgeGraph_ontology_MCP`](https://github.com/ramu-de/BioMedical_KnowledgeGraph_ontology_MCP),
31 node CSVs and 37 relationship CSVs. Upstream commit pinned in
[`../ontology/PROVENANCE.md`](../ontology/PROVENANCE.md).

Every node carries `dataset='biomed'`, `source`, `ingested_at`, `confidence`.
Loading is idempotent — MERGE on natural keys, so re-running converges.

Two labels were renamed to avoid colliding with the pre-existing graph:

| CSV | Loaded as | Why |
|---|---|---|
| `data_governance_policies` | `:DataGovernancePolicy` | `:Policy` already held 6 enterprise nodes |
| `entities` | `:BioEntity` | `:Entity` is used by `Job -[:CONFIGURES]-> Entity` |

**All data is synthetic.** Patients are `PAT001`…`PAT015`. No PHI.

---

## 3. Derived — 34 nodes, 21 relationships

Inferred by this project, and **labelled as such** so an inference can never pass
as a curated fact. Reversible: `scripts/derive_relationships.py --purge`.

| Relationship | Method | Edges | Confidence |
|---|---|---:|---:|
| `evaluated_in` | `inverse_of:INVESTIGATES` | 10 | 1.0 |
| `is_biomarker_for` | `gene_symbol_in_biomarker_name` | 5 | 0.9 |
| `indicates` | `disease_name_in_clinical_significance` | 2 | 0.9 |
| `indicates` | `curated_abbreviation_map` | 4 | 0.8 |

The last one is a hand-written map, deliberately not fuzzy-matched:

```
"CML monitoring"                → Chronic Myeloid Leukemia
"Alzheimer's diagnostic marker" → Alzheimer's Disease
"Diabetes control marker"       → Type 2 Diabetes
"Diabetes screening"            → Type 2 Diabetes
```

No substring match would find these. Guessing them with fuzzy matching is exactly
the unaudited inference this layer exists to prevent, so each is reviewable.

Also derived: 34 `ExternalConcept` nodes created by the term index for CURIEs that
had no concept node, marked `derived_from='term_index'`, plus `SAME_AS` edges
joining vocabulary nodes that share a distinctive name token (method
`shared_name_token`, evidence recorded, confidence 0.9).

---

## 4. Runtime — 278 nodes

Written while answering questions. All prefixed `Ctx_` so they never mix with
curated knowledge.

| Label | Count | What it is |
|---|---:|---|
| `Ctx_Action` | 172 | Per-stage timings for each decision |
| `Ctx_Term` | 60 | Embedded vocabulary labels for entity linking |
| `Ctx_Decision` | 43 | One per answer: question, package id, rules fired, grounding |
| `Ctx_Outcome` | 2 | Decision-quality outcomes |
| `Ctx_Chunk` | 1 | Chunk stubs linking text to concepts |

Append-only: `RuntimeStore` exposes no update or delete verb. Superseding writes a
new version and closes the prior one's `valid_to`.

---

## Which rules cite a real policy node

Of 13 rules, 6 cite a policy that exists in the graph. The rest encode good
practice without a corresponding policy artefact — stated plainly rather than
implied.

| Rule | Severity | Policy node |
|---|---|---|
| `governance.hub_table_without_dq_rule` | block | `graph:Policy/POL_HUB_DQ` ✅ pre-existing |
| `governance.incremental_load_without_key` | block | `graph:Policy/POL_INCREMENTAL_KEY` ✅ pre-existing |
| `governance.ard_below_dq_threshold` | warn | `graph:Policy/POL_HUB_DQ` ✅ pre-existing |
| `consent.withdrawn_patient_data` | block | `bio:DataGovernancePolicy/POL001` ⬅ loaded |
| `consent.status_missing` | block | `bio:DataGovernancePolicy/POL001` ⬅ loaded |
| `safety.severe_adverse_event_requires_disclosure` | warn | `bio:DataGovernancePolicy/POL003` ⬅ loaded |
| `governance.uncertified_ard_in_use` | warn | — no policy node |
| `governance.ard_without_owner` | warn | — no policy node |
| `safety.investigational_drug_not_approved` | block | — no policy node |
| `safety.trial_still_recruiting` | warn | — no policy node |
| `safety.underpowered_trial` | inform | — no policy node |
| `safety.drug_without_external_identifier` | warn | — no policy node |
| `consent.unlinked_patient` | warn | — no policy node |

**The two rules that block the headline demo questions both cite pre-existing or
loaded policy artefacts, not invented ones.**

---

## Verifying any of this yourself

```cypher
// Anything this project created carries provenance. This returns pre-existing nodes.
MATCH (n) WHERE n.dataset IS NULL AND n.derived_from IS NULL
  AND NOT any(l IN labels(n) WHERE l STARTS WITH 'Ctx_')
RETURN labels(n)[0] AS label, count(*) ORDER BY count(*) DESC;

// Every derived edge, with its method and confidence
MATCH ()-[r]->() WHERE r.derivation IS NOT NULL
RETURN type(r), r.derivation, r.evidence, r.confidence;

// The policy behind the headline block, unmodified
MATCH (p:Policy {policy_id:'POL_HUB_DQ'}) RETURN properties(p);

// Why it fires: no DQRule instances exist
MATCH (n:DQRule) RETURN count(n);
```
