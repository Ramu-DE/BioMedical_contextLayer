"""Enterprise Knowledge: ontologies, SHACL, provenance. Tasks 1.2-1.5.

Diagram: [EK-1] Concepts & Policies, [EK-2] Data & Metadata,
         [EK-4] Graph-Based Ontologies. Hub node: Ontologies.

Contracts:
    ontology_loader.load_modules(paths) -> OntologyBundle
        Aborts on parse error naming module and line (R1.2).
    shacl_gate.validate(node_rdf, shape) -> ValidationReport
        Rejects non-conforming nodes naming the constraint (R1.3).
    provenance.attach(assertion, source, ingested_at, confidence)
    provenance.chain_for(answer_id) -> ProvenanceChain   (R2.4)
    catalog.emit_dcat(dataset) -> DcatRecord             (R2.1)
"""
