"""Enterprise Knowledge: ontology, SHACL, IRIs, provenance.

Requirements 1.1-1.5, 2.1-2.4. Covers three integration defects found against
the vendored assets: camelCase/snake_case divergence between ontology and
shapes, rdflib treating a typed xsd:string literal as distinct from a plain one
(which broke every sh:in constraint), and shape identifier properties named
after the class rather than the source column.
"""

from __future__ import annotations

import pytest
from rdflib import Literal
from rdflib.namespace import XSD

from context_layer.knowledge.ontology_loader import (
    EXPECTED_MODULES,
    OntologyError,
    camel_to_snake,
    load_modules,
    snake_to_camel,
)
from context_layer.knowledge.provenance import (
    DcatCatalog,
    IriMinter,
    PolicyError,
    PolicyInstance,
    ProvenanceChain,
    chain_for,
    flag_incomplete,
    is_complete,
)
from context_layer.knowledge.shacl_gate import (
    KNOWN_SHAPE_DATA_CONFLICTS,
    PROPERTY_ALIASES,
    ShaclGate,
)
from context_layer.types import ContextElement, ContextPackage, Provenance, utcnow


# ── name conversion ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "snake,camel",
    [
        ("drug_id", "drugId"),
        ("generic_name", "genericName"),
        ("approval_status", "approvalStatus"),
        ("icd10_code", "icd10Code"),
        ("name", "name"),
    ],
)
def test_snake_to_camel(snake, camel):
    assert snake_to_camel(snake) == camel


def test_camel_to_snake_round_trips():
    for snake in ("drug_id", "generic_name", "approval_status"):
        assert camel_to_snake(snake_to_camel(snake)) == snake


# ── ontology loading, R1.1-1.2 ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bundle():
    return load_modules()


def test_loads_every_expected_module(bundle):
    names = {m.name for m in bundle.modules}
    assert set(EXPECTED_MODULES) <= names
    assert bundle.missing_modules == ()


def test_reports_per_module_triple_counts(bundle):
    assert all(m.triples > 0 for m in bundle.modules)
    assert bundle.triples > 3000
    # An RDF graph is a set, so merging modules deduplicates any triple declared
    # in more than one of them. The combined count is therefore <= the sum.
    per_module_total = sum(m.triples for m in bundle.modules)
    assert bundle.triples <= per_module_total
    assert bundle.duplicate_declarations == per_module_total - bundle.triples


def test_duplicate_declarations_are_reported(bundle):
    """Shared property declarations across modules are visible, not hidden."""
    assert bundle.duplicate_declarations >= 0
    if bundle.duplicate_declarations:
        assert "declared in more than one module" in bundle.report()


def test_indexes_the_declared_vocabulary(bundle):
    for cls in ("Drug", "Disease", "Patient", "DataGovernancePolicy"):
        assert bundle.has_class(cls), cls
    assert bundle.has_property("drug_id")
    assert not bundle.has_class("NoSuchClass")


def test_unknown_properties_tolerate_both_cases(bundle):
    # declared as drug_id; both spellings must be recognised
    assert bundle.unknown_properties(["drug_id", "drugId"]) == ()
    assert "totally_invented" in bundle.unknown_properties(["totally_invented"])


def test_missing_directory_raises(tmp_path):
    with pytest.raises(OntologyError, match="not found"):
        load_modules(tmp_path / "nope")


def test_empty_directory_raises(tmp_path):
    with pytest.raises(OntologyError, match="no .ttl modules"):
        load_modules(tmp_path)


def test_malformed_module_names_the_file(tmp_path):
    """R1.2: abort and identify the offending module, never load partially."""
    (tmp_path / "broken.ttl").write_text("@prefix bad <missing-angle-brackets .\n")
    with pytest.raises(OntologyError, match="broken.ttl"):
        load_modules(tmp_path)


def test_report_is_human_readable(bundle):
    text = bundle.report()
    assert "triples across" in text
    assert "foundation" in text


# ── SHACL gate, R1.3 ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gate():
    return ShaclGate()


def test_loads_all_shape_files(gate):
    assert len(gate.shape_files) == 10
    assert len(gate.covered_classes) == 10


def test_case_insensitive_shape_resolution(gate):
    """Upstream shapes declare Adverseevent / Clinicaltrial."""
    assert gate.has_shape("AdverseEvent")
    assert gate.has_shape("ClinicalTrial")
    assert gate.has_shape("adverseevent")


def test_miscased_targets_are_reported_not_hidden(gate):
    mis = dict(gate.miscased_targets())
    assert mis.get("Adverseevent") == "AdverseEvent"
    assert mis.get("Clinicaltrial") == "ClinicalTrial"


def test_absent_shape_does_not_block_a_write(gate):
    report = gate.validate("Patient", {"patient_id": "PAT001"})
    assert report.conforms
    assert "no shape" in report.shape_name


def test_valid_drug_conforms(gate):
    report = gate.validate(
        "Drug",
        {
            "drug_id": "D001",
            "name": "Pembrolizumab",
            "generic_name": "pembrolizumab",
            "drug_type": "Monoclonal Antibody",
            "approval_status": "Approved",
            "approval_year": 2014,
        },
    )
    assert report.conforms, report.summary()


def test_plain_literal_required_for_sh_in_matching(gate):
    """rdflib: Literal('x', datatype=xsd:string) != Literal('x'), so an
    explicit datatype made every sh:in constraint fail."""
    assert Literal("Approved") != Literal("Approved", datatype=XSD.string)
    assert gate._literal("Approved") == Literal("Approved")


def test_missing_required_property_is_rejected(gate):
    report = gate.validate("Drug", {"drug_id": "D001", "name": "X"})
    assert not report.conforms
    assert any(v.constraint == "MinCountConstraintComponent" for v in report.violations)


def test_value_outside_sh_in_is_rejected(gate):
    report = gate.validate(
        "Drug",
        {
            "drug_id": "D001", "name": "X", "generic_name": "x",
            "drug_type": "Monoclonal Antibody",
            "approval_status": "Fabricated Status",
        },
    )
    assert not report.conforms
    assert any(v.path == "approvalStatus" for v in report.violations)


def test_violation_names_the_constraint(gate):
    report = gate.validate("Drug", {"drug_id": "D001"})
    assert report.violations
    assert all(v.constraint for v in report.violations)
    assert "violates" in report.violations[0].describe()


def test_aliases_bridge_shape_and_graph_naming(gate):
    """ClinicalTrial shape wants clinicaltrialId and name; graph has trial_id, title."""
    report = gate.validate(
        "ClinicalTrial", {"trial_id": "CT001", "title": "Pembrolizumab in NSCLC"}
    )
    assert report.conforms, report.summary()


def test_alias_map_is_declared_not_inferred():
    assert PROPERTY_ALIASES["ClinicalTrial"]["trial_id"] == "clinicaltrialId"
    assert PROPERTY_ALIASES["AdverseEvent"]["event_id"] == "adverseeventId"


def test_known_conflict_is_surfaced(gate):
    """Disease.prevalence is categorical in the data, xsd:float in the shape."""
    report = gate.validate(
        "Disease",
        {"disease_id": "DIS001", "name": "NSCLC", "prevalence": "High"},
    )
    assert not report.conforms
    assert report.known_conflicts
    assert "categorical" in report.known_conflicts[0]
    assert ("Disease", "prevalence") in KNOWN_SHAPE_DATA_CONFLICTS


def test_unmapped_properties_are_reported(gate):
    report = gate.validate(
        "Drug",
        {
            "drug_id": "D001", "name": "X", "generic_name": "x",
            "drug_type": "Biologic", "mechanism": "PD-1 inhibitor",
        },
    )
    assert "mechanism" in report.unmapped_properties


def test_provenance_properties_are_not_treated_as_domain_data(gate):
    report = gate.validate(
        "Drug",
        {
            "drug_id": "D001", "name": "X", "generic_name": "x",
            "drug_type": "Biologic", "dataset": "biomed",
            "source": "github:...", "ingested_at": "2026-01-01", "confidence": 1.0,
        },
    )
    assert not set(report.unmapped_properties) & {
        "dataset", "source", "ingested_at", "confidence"
    }


# ── IRI minting, R1.5 ─────────────────────────────────────────────────────────


def test_minting_is_deterministic():
    m = IriMinter()
    assert m.mint("Drug", "D001") == m.mint("Drug", "D001")


def test_minting_distinguishes_class_and_key():
    m = IriMinter()
    assert m.mint("Drug", "D001") != m.mint("Disease", "D001")
    assert m.mint("Drug", "D001") != m.mint("Drug", "D002")


def test_minting_slugifies_unsafe_characters():
    iri = IriMinter().mint("Drug", "Drug/With Spaces & Slashes")
    assert " " not in iri and "&" not in iri
    assert iri.count("/") == IriMinter().mint("Drug", "x").count("/")


def test_hashed_minting_is_stable_for_composite_keys():
    m = IriMinter()
    assert m.mint_hashed("Batch", ["D001", "SITE1"]) == m.mint_hashed(
        "Batch", ["D001", "SITE1"]
    )
    assert m.mint_hashed("Batch", ["D001"]) != m.mint_hashed("Batch", ["D002"])


def test_minting_requires_inputs():
    with pytest.raises(ValueError):
        IriMinter().mint("Drug", "   ")
    with pytest.raises(ValueError):
        IriMinter().mint_hashed("Drug", [])


def test_is_minted_recognises_own_iris():
    m = IriMinter()
    assert m.is_minted(m.mint("Drug", "D001"))
    assert not m.is_minted("https://example.org/other")


# ── policy instances, R1.4 ────────────────────────────────────────────────────


def test_valid_policy_instance():
    p = PolicyInstance("POL001", "Patient Data Privacy", "Mandatory", "EU")
    assert p.iri(IriMinter()).endswith("DataGovernancePolicy/POL001")
    assert len(p.to_rdf(IriMinter())) >= 5


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(policy_id="", policy_name="n", enforcement_level="Mandatory", jurisdiction="EU"), "policy_id"),
        (dict(policy_id="P", policy_name="", enforcement_level="Mandatory", jurisdiction="EU"), "policy_name"),
        (dict(policy_id="P", policy_name="n", enforcement_level="Sometimes", jurisdiction="EU"), "enforcement_level"),
        (dict(policy_id="P", policy_name="n", enforcement_level="Mandatory", jurisdiction=""), "jurisdiction"),
    ],
)
def test_policy_requires_all_governance_fields(kwargs, match):
    with pytest.raises(PolicyError, match=match):
        PolicyInstance(**kwargs)


# ── DCAT and provenance, R2.1-2.4 ─────────────────────────────────────────────


def test_dcat_records_capture_source_and_count():
    cat = DcatCatalog()
    rec = cat.record("ds1", "Drugs", "neo4j:biomed", 10)
    assert rec.record_count == 10
    graph = cat.to_rdf()
    assert len(graph) >= 5
    assert "neo4j:biomed" in graph.serialize(format="turtle")


def test_dcat_serialises_to_turtle(tmp_path):
    cat = DcatCatalog()
    cat.record("ds1", "Drugs", "neo4j", 5)
    out = tmp_path / "c.ttl"
    cat.serialize(str(out))
    assert out.is_file() and out.stat().st_size > 0


def test_incomplete_provenance_is_detected():
    assert is_complete(Provenance("src", utcnow(), 1.0))
    assert not is_complete(Provenance("", utcnow(), 1.0))
    assert not is_complete(Provenance("src", utcnow(), 0.0))
    assert not is_complete(None)


def test_flag_incomplete_lists_offending_elements():
    good = ContextElement("g", "chunk", "t", Provenance("s", utcnow(), 1.0))
    bad = ContextElement("b", "chunk", "t", Provenance("", utcnow(), 1.0))
    assert flag_incomplete([good, bad]) == ("b",)


def test_chain_for_assembles_full_lineage():
    pkg = ContextPackage(
        context_package_id="c1",
        question="q",
        elements=(
            ContextElement(
                "e1", "chunk", "text",
                Provenance("neo4j:Drug", utcnow(), 0.9, "D001"),
                curies=("RXNORM:1",),
            ),
        ),
    )
    chain = chain_for("ans1", pkg)
    assert not chain.is_empty
    assert chain.sources() == ("neo4j:Drug",)
    assert chain.weakest_confidence() == 0.9
    d = chain.to_dict()
    assert d["links"][0]["curies"] == ["RXNORM:1"]
    assert d["links"][0]["record_id"] == "D001"


def test_chain_weakest_confidence_governs():
    chain = ProvenanceChain("a")
    chain.add("e1", "s", utcnow(), 0.9)
    chain.add("e2", "s", utcnow(), 0.4)
    assert chain.weakest_confidence() == 0.4


def test_empty_chain_reports_zero_confidence():
    assert ProvenanceChain("a").weakest_confidence() == 0.0
