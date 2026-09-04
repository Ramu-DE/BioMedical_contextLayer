"""Pipeline relevance gating.

Regression cover for a real defect: the pipeline originally scanned data-asset
types on every question, so a data-quality policy about sales tables blocked a
question about PD-1 inhibitors. Rules must only bind to entities that are in
scope for the question.
"""

from __future__ import annotations

import pytest

from context_layer.agent.pipeline import ContextLayerAgent, _label_for_id
from context_layer.rules.models import Fact, FactSet
from context_layer.types import ContextPackage


class FakeFactBuilder:
    """Records what was requested so scoping can be asserted."""

    def __init__(self):
        self.entity_calls: list[dict] = []
        self.type_calls: list[str] = []
        self.curie_calls: list[list[str]] = []

    def for_entities(self, wanted):
        self.entity_calls.append(dict(wanted))
        facts = tuple(
            Fact(entity_id=i, entity_type=label, properties={})
            for label, ids in wanted.items()
            for i in ids
        )
        return FactSet(facts=facts)

    def for_type(self, label, limit=500):
        self.type_calls.append(label)
        return FactSet(facts=(Fact(entity_id=f"{label}-1", entity_type=label),))

    def for_curies(self, curies, limit=200):
        self.curie_calls.append(list(curies))
        return FactSet()

    def close(self):
        pass


class FakeAssembler:
    def __init__(self, package):
        self.package = package

    def assemble(self, question, k=6):
        return self.package

    def close(self):
        pass


def package(question, curies=()):
    return ContextPackage(
        context_package_id="ctx_x", question=question, elements=(), degradations=()
    )


def agent(cfg, question, builder=None):
    builder = builder or FakeFactBuilder()
    return (
        ContextLayerAgent(
            cfg,
            llm=object(),
            assembler=FakeAssembler(package(question)),
            fact_builder=builder,
        ),
        builder,
    )


# ── the fix ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "Which drugs target PD-1 and what diseases do they treat?",
        "What is the mechanism of pembrolizumab?",
        "Which trials reported severe adverse events?",
    ],
)
def test_clinical_questions_do_not_pull_in_data_asset_facts(cfg, question):
    """The bug: these questions were blocked by unrelated table policies."""
    a, builder = agent(cfg, question)
    a.gather_facts(question, package(question))
    assert builder.type_calls == [], (
        f"data-asset scan leaked into a clinical question: {builder.type_calls}"
    )


@pytest.mark.parametrize(
    "question",
    [
        "What policy applies to hub layer tables and are we compliant?",
        "Are all our datasets certified?",
        "Which incremental load jobs are missing keys?",
        "Show data quality coverage for the warehouse.",
    ],
)
def test_data_asset_questions_do_pull_in_governance_facts(cfg, question):
    a, builder = agent(cfg, question)
    a.gather_facts(question, package(question))
    assert set(builder.type_calls) == {"Table", "ARD", "Entity"}


def test_relevance_predicate_directly():
    yes = ContextLayerAgent.question_is_about_data_assets
    assert yes("What policy applies to hub layer tables?")
    assert yes("Are our datasets certified?")
    assert yes("Which jobs use incremental load?")
    assert not yes("Which drugs target PD-1?")
    assert not yes("What is the mechanism of pembrolizumab?")
    assert not yes("")


# ── id scoping ────────────────────────────────────────────────────────────────


def test_named_patient_id_is_brought_into_scope(cfg):
    q = "Can I include patient PAT015 in the analysis?"
    a, builder = agent(cfg, q)
    facts = a.gather_facts(q, package(q))
    assert any(f.entity_type == "Patient" and f.entity_id == "PAT015" for f in facts.facts)


def test_multiple_ids_of_different_types(cfg):
    q = "Compare D001 and D002 for disease DIS001 in trial CT001"
    a, builder = agent(cfg, q)
    facts = a.gather_facts(q, package(q))
    types = {f.entity_type for f in facts.facts}
    assert {"Drug", "Disease", "ClinicalTrial"} <= types


def test_no_ids_and_no_asset_terms_yields_no_facts(cfg):
    q = "Tell me something interesting"
    a, builder = agent(cfg, q)
    assert len(a.gather_facts(q, package(q))) == 0


@pytest.mark.parametrize(
    "token,expected",
    [
        ("PAT001", "Patient"),
        ("D001", "Drug"),
        ("DIS001", "Disease"),
        ("CT010", "ClinicalTrial"),
        ("AE001", "AdverseEvent"),
        ("B001", "Biomarker"),
        ("POL001", "DataGovernancePolicy"),
        ("XYZ999", None),
    ],
)
def test_id_prefix_maps_to_label(token, expected):
    assert _label_for_id(token) == expected


def test_facts_are_deduplicated_and_ordered(cfg):
    q = "Patient PAT015 and PAT015 again"
    a, builder = agent(cfg, q)
    facts = a.gather_facts(q, package(q))
    ids = [f.entity_id for f in facts.facts]
    assert ids == sorted(set(ids))
