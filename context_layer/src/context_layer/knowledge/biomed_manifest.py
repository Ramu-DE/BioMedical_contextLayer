"""Load manifest for the biomedical dataset. Task 0.4 / 1.6.

Source: ramu-de/BioMedical_KnowledgeGraph_ontology_MCP
        neo4j-neptune-mcp-platform/sampledata/nodes/*.csv  (32 files)
        neo4j-neptune-mcp-platform/relationships/*.csv     (37 files)

Kept as data rather than code so the mapping is auditable at a glance and a
schema change is a one-line edit.

Label-collision policy: the target Aura database already holds an enterprise
data-governance graph. Two biomedical files would collide, so they are renamed:

    data_governance_policies -> :DataGovernancePolicy  (not :Policy, which
        already has 6 enterprise governance nodes)
    entities                 -> :BioEntity             (not :Entity, which the
        enterprise graph uses for Job-[:CONFIGURES]->Entity)

Every loaded node also carries ``dataset='biomed'`` so the two subgraphs stay
distinguishable in every query.
"""

from __future__ import annotations

from typing import NamedTuple


class NodeSpec(NamedTuple):
    csv: str
    label: str
    key: str


class RelSpec(NamedTuple):
    csv: str
    src_label: str
    src_col: str
    rel_type: str
    dst_label: str
    dst_col: str
    # Property to match on the target node. Defaults to dst_col, which is right
    # whenever the CSV's foreign-key column is named like the target's key.
    # For derived edges it often is not: patients.primary_disease holds a value
    # that must be matched against Disease.disease_id.
    dst_key: str | None = None

    @property
    def target_key(self) -> str:
        return self.dst_key or self.dst_col


# ── Nodes: csv stem -> label, natural key column ──────────────────────────────

NODES: tuple[NodeSpec, ...] = (
    NodeSpec("drugs", "Drug", "drug_id"),
    NodeSpec("diseases", "Disease", "disease_id"),
    NodeSpec("genes", "Gene", "gene_id"),
    NodeSpec("proteins", "Protein", "protein_id"),
    NodeSpec("patients", "Patient", "patient_id"),
    NodeSpec("clinical_trials", "ClinicalTrial", "trial_id"),
    NodeSpec("adverse_events", "AdverseEvent", "event_id"),
    NodeSpec("biomarkers", "Biomarker", "biomarker_id"),
    NodeSpec("phenotypes", "Phenotype", "phenotype_id"),
    NodeSpec("pathways", "Pathway", "pathway_id"),
    NodeSpec("molecular_functions", "MolecularFunction", "function_id"),
    NodeSpec("biological_processes", "BiologicalProcess", "process_id"),
    NodeSpec("cell_types", "CellType", "cell_type_id"),
    NodeSpec("anatomy", "Anatomy", "anatomy_id"),
    NodeSpec("exposures", "Exposure", "exposure_id"),
    NodeSpec("researchers", "Researcher", "researcher_id"),
    NodeSpec("research_papers", "ResearchPaper", "paper_id"),
    NodeSpec("institutions", "Institution", "institution_id"),
    NodeSpec("advisory_boards", "AdvisoryBoard", "board_id"),
    NodeSpec("manufacturing_sites", "ManufacturingSite", "site_id"),
    NodeSpec("drug_batches", "DrugBatch", "batch_id"),
    NodeSpec("quality_events", "QualityEvent", "event_id"),
    NodeSpec("regulatory_submissions", "RegulatorySubmission", "submission_id"),
    NodeSpec("patient_outcomes", "PatientOutcome", "outcome_id"),
    NodeSpec("patient_reported_outcomes", "PatientReportedOutcome", "pro_id"),
    NodeSpec("medical_information_requests", "MedicalInformationRequest", "request_id"),
    NodeSpec("clusters", "Cluster", "cluster_id"),
    NodeSpec("cluster_summaries", "ClusterSummary", "summary_id"),
    NodeSpec("compliance_records", "ComplianceRecord", "record_id"),
    # Renamed to avoid colliding with the enterprise governance graph
    NodeSpec("data_governance_policies", "DataGovernancePolicy", "policy_id"),
    NodeSpec("entities", "BioEntity", "entity_id"),
)

# external_mappings is handled separately: it becomes the CURIE crosswalk.

# ── Relationships ─────────────────────────────────────────────────────────────

RELS: tuple[RelSpec, ...] = (
    RelSpec("drug_treats_disease", "Drug", "drug_id", "TREATS", "Disease", "disease_id"),
    RelSpec("drug_targets_protein", "Drug", "drug_id", "TARGETS", "Protein", "protein_id"),
    RelSpec("drug_manufactured_at", "Drug", "drug_id", "MANUFACTURED_AT", "ManufacturingSite", "site_id"),
    RelSpec("gene_associated_with_disease", "Gene", "gene_id", "ASSOCIATED_WITH", "Disease", "disease_id"),
    RelSpec("gene_has_molecular_function", "Gene", "gene_id", "HAS_FUNCTION", "MolecularFunction", "function_id"),
    RelSpec("gene_involved_in_biological_process", "Gene", "gene_id", "INVOLVED_IN", "BiologicalProcess", "process_id"),
    RelSpec("gene_participates_in_pathway", "Gene", "gene_id", "PARTICIPATES_IN", "Pathway", "pathway_id"),
    RelSpec("protein_expressed_in_anatomy", "Protein", "protein_id", "EXPRESSED_IN", "Anatomy", "anatomy_id"),
    RelSpec("protein_has_molecular_function", "Protein", "protein_id", "HAS_FUNCTION", "MolecularFunction", "function_id"),
    RelSpec("protein_involved_in_biological_process", "Protein", "protein_id", "INVOLVED_IN", "BiologicalProcess", "process_id"),
    RelSpec("protein_involved_in_pathway", "Protein", "protein_id", "INVOLVED_IN_PATHWAY", "Pathway", "pathway_id"),
    RelSpec("disease_has_phenotype", "Disease", "disease_id", "HAS_PHENOTYPE", "Phenotype", "phenotype_id"),
    RelSpec("disease_affects_anatomy", "Disease", "disease_id", "AFFECTS", "Anatomy", "anatomy_id"),
    RelSpec("disease_involves_cell_type", "Disease", "disease_id", "INVOLVES", "CellType", "cell_type_id"),
    RelSpec("phenotype_associated_with_gene", "Phenotype", "phenotype_id", "ASSOCIATED_WITH", "Gene", "gene_id"),
    RelSpec("cell_type_found_in_anatomy", "CellType", "cell_type_id", "FOUND_IN", "Anatomy", "anatomy_id"),
    RelSpec("pathway_involves_biological_process", "Pathway", "pathway_id", "INVOLVES", "BiologicalProcess", "process_id"),
    RelSpec("exposure_affects_gene", "Exposure", "exposure_id", "AFFECTS", "Gene", "gene_id"),
    RelSpec("exposure_increases_risk_disease", "Exposure", "exposure_id", "INCREASES_RISK_OF", "Disease", "disease_id"),
    RelSpec("biomarker_predicts_response", "Biomarker", "biomarker_id", "PREDICTS_RESPONSE_TO", "Drug", "drug_id"),
    RelSpec("trial_investigates_drug", "ClinicalTrial", "trial_id", "INVESTIGATES", "Drug", "drug_id"),
    RelSpec("trial_studies_disease", "ClinicalTrial", "trial_id", "STUDIES", "Disease", "disease_id"),
    RelSpec("trial_reports_adverse_event", "ClinicalTrial", "trial_id", "REPORTS", "AdverseEvent", "event_id"),
    RelSpec("institution_sponsors_trial", "Institution", "institution_id", "SPONSORS", "ClinicalTrial", "trial_id"),
    RelSpec("patient_enrolled_in_trial", "Patient", "patient_id", "ENROLLED_IN", "ClinicalTrial", "trial_id"),
    RelSpec("patient_has_outcome", "Patient", "patient_id", "HAS_OUTCOME", "PatientOutcome", "outcome_id"),
    RelSpec("paper_authored_by", "ResearchPaper", "paper_id", "AUTHORED_BY", "Researcher", "researcher_id"),
    RelSpec("paper_mentions_disease", "ResearchPaper", "paper_id", "MENTIONS_DISEASE", "Disease", "disease_id"),
    RelSpec("paper_mentions_drug", "ResearchPaper", "paper_id", "MENTIONS_DRUG", "Drug", "drug_id"),
    RelSpec("researcher_advises_board", "Researcher", "researcher_id", "ADVISES", "AdvisoryBoard", "board_id"),
    RelSpec("researcher_affiliated_with", "Researcher", "researcher_id", "AFFILIATED_WITH", "Institution", "institution_id"),
    RelSpec("submission_for_drug", "RegulatorySubmission", "submission_id", "FOR_DRUG", "Drug", "drug_id"),
    RelSpec("batch_produced_for_drug", "DrugBatch", "batch_id", "PRODUCED_FOR", "Drug", "drug_id"),
    RelSpec("entity_associated_with_disease", "BioEntity", "entity_id", "ASSOCIATED_WITH", "Disease", "disease_id"),
    RelSpec("cluster_has_summary", "Cluster", "cluster_id", "HAS_SUMMARY", "ClusterSummary", "summary_id"),
)

# Polymorphic edges: the target label comes from a column, not the filename.
# node_belongs_to_cluster: node_id + node_type -> Cluster
# policy_governs_entity:   policy_id -> entity_type + entity_id
POLYMORPHIC_RELS = ("node_belongs_to_cluster", "policy_governs_entity")

# quality_events.batch_id -> DrugBatch, derived from the node CSV itself.
# dst_key is given explicitly wherever the CSV column name differs from the
# target label's natural key — otherwise the MATCH silently finds nothing.
DERIVED_RELS = (
    RelSpec("quality_events", "QualityEvent", "event_id", "AFFECTS_BATCH", "DrugBatch", "batch_id", "batch_id"),
    RelSpec("drug_batches", "DrugBatch", "batch_id", "OF_DRUG", "Drug", "drug_id", "drug_id"),
    RelSpec("drug_batches", "DrugBatch", "batch_id", "PRODUCED_AT", "ManufacturingSite", "manufacturing_site_id", "site_id"),
    RelSpec("patients", "Patient", "patient_id", "HAS_PRIMARY_DISEASE", "Disease", "primary_disease", "disease_id"),
    RelSpec("patient_reported_outcomes", "PatientReportedOutcome", "pro_id", "REPORTED_BY", "Patient", "patient_id", "patient_id"),
    RelSpec("medical_information_requests", "MedicalInformationRequest", "request_id", "ABOUT_DRUG", "Drug", "drug_id", "drug_id"),
    RelSpec("compliance_records", "ComplianceRecord", "record_id", "AGAINST_POLICY", "DataGovernancePolicy", "policy_id", "policy_id"),
)

# ── CURIE crosswalk from external_mappings.csv ────────────────────────────────
# standard -> CURIE prefix. Drives entity linking (Requirement 4.1, 4.3).
CURIE_PREFIX = {
    "RxNorm": "RXNORM",
    "SNOMED": "SNOMED",
    "SNOMEDCT": "SNOMED",
    "MeSH": "MESH",
    "ICD10": "ICD10",
    "HPO": "HP",
    "GO": "GO",
    "UniProt": "UNIPROT",
    "HGNC": "HGNC",
    "KEGG": "KEGG",
    "UBERON": "UBERON",
    "MONDO": "MONDO",
    "ChEBI": "CHEBI",
    "OMOP": "OMOP",
    "MedDRA": "MEDDRA",
    "NCBI Taxonomy": "NCBITAXON",
    "CellOntology": "CL",
    "ORCID": "ORCID",
    "ROR": "ROR",
    "GS1_GTIN": "GS1",
}

# entity_type in external_mappings -> node label
MAPPING_ENTITY_LABEL = {
    "Drug": "Drug",
    "Disease": "Disease",
    "Gene": "Gene",
    "Protein": "Protein",
    "Phenotype": "Phenotype",
    "Pathway": "Pathway",
    "Anatomy": "Anatomy",
    "CellType": "CellType",
    "AdverseEvent": "AdverseEvent",
    "Biomarker": "Biomarker",
    "Organism": "BioEntity",
    "Researcher": "Researcher",
    "Institution": "Institution",
    "BioEntity": "BioEntity",
}
