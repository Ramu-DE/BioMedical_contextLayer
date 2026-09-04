"""SHACL write gate. Task 1.3, Requirement 1.3.

Validates a node against its shape before it is written, and reports the
violated constraint by name rather than returning a bare boolean.

**Naming bridge.** The vendored assets disagree with each other: the ontology
declares ``bio:drug_id`` (snake_case) while the SHACL shapes target
``biomedkg:drugId`` (camelCase), both in the same namespace. Neo4j properties are
snake_case, matching the source CSVs. Without translation every ``sh:minCount``
constraint fails on a perfectly valid node. This module converts snake_case graph
properties to the camelCase paths the shapes expect, and reports any property the
shape does not describe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

from context_layer.knowledge.ontology_loader import snake_to_camel

BIO = Namespace("https://biomedkg.org/ontology/")
SH = Namespace("http://www.w3.org/ns/shacl#")
SHAPES_DIR = Path(__file__).parent / "shapes"
INSTANCE_BASE = Namespace("https://biomedkg.org/instance/")

# Explicit graph-property to shape-path aliases.
#
# Auto-derived snake_to_camel handles most properties (drug_id -> drugId), but
# some shapes name their identifier after the class rather than after the source
# column. The ClinicalTrial shape requires `clinicaltrialId` while the graph key
# is `trial_id`; AdverseEvent requires `adverseeventId` against `event_id`.
# These are declared rather than inferred so the mapping is reviewable — a
# heuristic here would silently mis-bind properties.
PROPERTY_ALIASES: dict[str, dict[str, str]] = {
    # clinical_trials.csv uses `title`; the shape requires `name`.
    "Clinicaltrial": {"trial_id": "clinicaltrialId", "title": "name"},
    "ClinicalTrial": {"trial_id": "clinicaltrialId", "title": "name"},
    "Adverseevent": {"event_id": "adverseeventId"},
    "AdverseEvent": {"event_id": "adverseeventId"},
    "Institution": {"institution_id": "institutionId"},
    "Researcher": {"researcher_id": "researcherId"},
    "Pathway": {"pathway_id": "pathwayId"},
}

# Properties where the shape and the loaded data genuinely disagree about type.
# Recorded so the gate can report them as a modelling conflict rather than
# presenting them as ordinary data errors. Resolving these requires a decision
# from a data owner: either the shape relaxes or the data is normalised.
KNOWN_SHAPE_DATA_CONFLICTS: dict[tuple[str, str], str] = {
    ("Disease", "prevalence"): (
        "shape requires xsd:float but the loaded dataset stores categorical "
        "values ('High', 'Medium', 'Low')"
    ),
}


@dataclass(frozen=True)
class Violation:
    focus_node: str
    path: str
    constraint: str
    message: str
    severity: str = "Violation"

    def describe(self) -> str:
        return f"{self.path or '(node)'} violates {self.constraint}: {self.message}"


@dataclass
class ValidationReport:
    conforms: bool
    violations: tuple[Violation, ...] = ()
    shape_name: str = ""
    unmapped_properties: tuple[str, ...] = ()
    known_conflicts: tuple[str, ...] = ()
    text: str = ""

    def summary(self) -> str:
        if self.conforms:
            extra = (
                f" (properties not in shape: {list(self.unmapped_properties)})"
                if self.unmapped_properties
                else ""
            )
            return f"conforms to {self.shape_name}{extra}"
        detail = "; ".join(v.describe() for v in self.violations[:4])
        if self.known_conflicts:
            detail += f"  [known shape/data conflict: {'; '.join(self.known_conflicts)}]"
        return f"{len(self.violations)} violation(s) of {self.shape_name}: {detail}"


class ShaclGate:
    """Loads the shape graph once, then validates nodes against it."""

    def __init__(self, shapes_dir: str | Path = SHAPES_DIR) -> None:
        self.shapes_dir = Path(shapes_dir)
        self.shape_graph = Graph()
        self.shape_files: list[str] = []
        self._targets: dict[str, URIRef] = {}
        self._paths: dict[str, set[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.shapes_dir.is_dir():
            raise FileNotFoundError(f"shapes directory not found: {self.shapes_dir}")
        for path in sorted(self.shapes_dir.glob("*.ttl")):
            self.shape_graph.parse(path, format="turtle")
            self.shape_files.append(path.name)

        # Index target class -> shape, and the property paths each shape declares.
        for shape, target in self.shape_graph.subject_objects(SH.targetClass):
            local = str(target).rsplit("/", 1)[-1]
            self._targets[local] = shape
            paths: set[str] = set()
            for _, prop in self.shape_graph.subject_objects(SH.property):
                pass
            for prop in self.shape_graph.objects(shape, SH.property):
                for p in self.shape_graph.objects(prop, SH.path):
                    paths.add(str(p).rsplit("/", 1)[-1])
            self._paths[local] = paths

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def covered_classes(self) -> tuple[str, ...]:
        return tuple(sorted(self._targets))

    def shape_paths(self, class_name: str) -> tuple[str, ...]:
        return tuple(sorted(self._paths.get(self._resolve(class_name) or "", ())))

    def _resolve(self, class_name: str) -> str | None:
        """Find the shape target for a class, tolerating casing differences.

        The vendored shapes declare ``biomedkg:Adverseevent`` and
        ``biomedkg:Clinicaltrial`` while the ontology declares ``AdverseEvent``
        and ``ClinicalTrial``. Exact matching silently skips those classes, so
        validation reports 'no shape' for entities that do have one.
        """
        if class_name in self._targets:
            return class_name
        folded = class_name.casefold()
        for target in self._targets:
            if target.casefold() == folded:
                return target
        return None

    def has_shape(self, class_name: str) -> bool:
        return self._resolve(class_name) is not None

    def miscased_targets(self) -> tuple[tuple[str, str], ...]:
        """Shape targets whose casing differs from the conventional class name.

        Reported rather than silently normalised, because the upstream shapes
        should be corrected.
        """
        expected = {
            "Adverseevent": "AdverseEvent",
            "Clinicaltrial": "ClinicalTrial",
        }
        return tuple(
            (actual, expected[actual]) for actual in self._targets if actual in expected
        )

    # ── node -> RDF ───────────────────────────────────────────────────────────

    @staticmethod
    def _literal(value: Any) -> Literal:
        """Build a literal that SHACL can match.

        Strings are emitted *without* an explicit xsd:string datatype. In RDF 1.1
        a plain literal is xsd:string, but rdflib treats ``Literal("x")`` and
        ``Literal("x", datatype=xsd:string)`` as distinct terms — so an explicit
        datatype makes every ``sh:in`` list fail, since those lists are written
        with plain literals.
        """
        if isinstance(value, bool):
            return Literal(value, datatype=XSD.boolean)
        if isinstance(value, int):
            return Literal(value, datatype=XSD.integer)
        if isinstance(value, float):
            return Literal(value, datatype=XSD.decimal)
        return Literal(str(value))

    def to_rdf(
        self, class_name: str, node: Mapping[str, Any], iri: str | None = None
    ) -> tuple[Graph, tuple[str, ...]]:
        """Build a data graph for one node. Returns (graph, unmapped_properties)."""
        target = self._resolve(class_name) or class_name
        g = Graph()
        g.bind("biomedkg", BIO)
        subject = URIRef(iri or f"{INSTANCE_BASE}{class_name}/unknown")
        g.add((subject, RDF.type, BIO[target]))

        declared = self._paths.get(target, set())
        aliases = PROPERTY_ALIASES.get(target, {}) | PROPERTY_ALIASES.get(class_name, {})
        unmapped: list[str] = []
        for key, value in node.items():
            if value in (None, ""):
                continue
            if key in ("dataset", "source", "ingested_at", "confidence"):
                continue  # provenance, not modelled by the domain shapes
            candidates = (aliases.get(key), snake_to_camel(key), key)
            path = next((c for c in candidates if c and c in declared), None)
            if path is None:
                unmapped.append(key)
                continue
            g.add((subject, BIO[path], self._literal(value)))
        return g, tuple(sorted(unmapped))

    # ── validation ────────────────────────────────────────────────────────────

    def validate(
        self, class_name: str, node: Mapping[str, Any], iri: str | None = None
    ) -> ValidationReport:
        """Requirement 1.3: reject non-conforming nodes, naming the constraint."""
        target = self._resolve(class_name)
        if target is None:
            return ValidationReport(
                conforms=True,
                shape_name=f"(no shape for {class_name})",
                text="no shape declared; nothing to enforce",
            )

        data, unmapped = self.to_rdf(class_name, node, iri)
        import pyshacl

        conforms, results_graph, results_text = pyshacl.validate(
            data_graph=data,
            shacl_graph=self.shape_graph,
            inference="none",
            abort_on_first=False,
            meta_shacl=False,
            advanced=False,
        )
        violations = self._parse(results_graph)
        conflicts = tuple(
            f"{prop}: {reason}"
            for (cls, prop), reason in KNOWN_SHAPE_DATA_CONFLICTS.items()
            if cls.casefold() in (class_name.casefold(), target.casefold())
            and any(v.path == prop for v in violations)
        )
        return ValidationReport(
            conforms=bool(conforms),
            violations=violations,
            shape_name=f"{target}Shape",
            unmapped_properties=unmapped,
            known_conflicts=conflicts,
            text=str(results_text),
        )

    @staticmethod
    def _parse(results_graph: Graph) -> tuple[Violation, ...]:
        out: list[Violation] = []
        for result in results_graph.subjects(RDF.type, SH.ValidationResult):
            def one(predicate):
                for o in results_graph.objects(result, predicate):
                    return str(o)
                return ""

            constraint = one(SH.sourceConstraintComponent)
            out.append(
                Violation(
                    focus_node=one(SH.focusNode),
                    path=one(SH.resultPath).rsplit("/", 1)[-1],
                    constraint=constraint.rsplit("#", 1)[-1] or "Unknown",
                    message=one(SH.resultMessage) or "constraint not satisfied",
                    severity=one(SH.resultSeverity).rsplit("#", 1)[-1] or "Violation",
                )
            )
        return tuple(sorted(out, key=lambda v: (v.path, v.constraint)))
