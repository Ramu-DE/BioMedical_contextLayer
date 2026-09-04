"""Ontology loading. Task 1.2, Requirements 1.1, 1.2.

Loads the vendored OWL/TTL modules into a single RDF graph and indexes the
classes and properties they declare, so the rest of the layer can check that a
concept it is about to assert is actually part of the agreed vocabulary.

Parse failures abort with the offending module named (Requirement 1.2): a
partially loaded ontology is worse than none, because downstream validation
would silently accept terms the ontology never defined.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import SKOS

BIO = Namespace("https://biomedkg.org/ontology/")
ONTOLOGY_DIR = Path(__file__).resolve().parents[3] / "ontology"

# Expected module set. A missing module is reported rather than ignored.
EXPECTED_MODULES = (
    "foundation",
    "clinical",
    "patient",
    "governance",
    "commercial",
    "medical_affairs",
    "supply_quality",
    "biomedkg-data",
)


class OntologyError(RuntimeError):
    """Ontology could not be loaded. Raised at startup, never mid-request."""


def snake_to_camel(name: str) -> str:
    """drug_id -> drugId. The ontology uses snake_case, SHACL uses camelCase."""
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest if part)


def camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


@dataclass(frozen=True)
class ModuleStats:
    name: str
    path: str
    triples: int
    classes: int
    object_properties: int
    datatype_properties: int


@dataclass
class OntologyBundle:
    graph: Graph
    modules: tuple[ModuleStats, ...] = ()
    classes: frozenset[str] = frozenset()
    object_properties: frozenset[str] = frozenset()
    datatype_properties: frozenset[str] = frozenset()
    labels: dict[str, str] = field(default_factory=dict)
    missing_modules: tuple[str, ...] = ()

    @property
    def triples(self) -> int:
        return len(self.graph)

    @property
    def duplicate_declarations(self) -> int:
        """Triples declared in more than one module.

        An RDF graph is a set, so merging collapses repeats. A non-zero count
        means modules restate each other — usually shared property declarations,
        which is worth knowing when reasoning about module boundaries.
        """
        return max(0, sum(m.triples for m in self.modules) - self.triples)

    def has_class(self, local_name: str) -> bool:
        return local_name in self.classes

    def has_property(self, local_name: str) -> bool:
        return (
            local_name in self.datatype_properties
            or local_name in self.object_properties
        )

    def label_for(self, local_name: str) -> str:
        return self.labels.get(local_name, local_name)

    def unknown_properties(self, names: Iterable[str]) -> tuple[str, ...]:
        """Property names not declared by any module. Supports both cases."""
        out = []
        for n in names:
            if self.has_property(n):
                continue
            if self.has_property(camel_to_snake(n)) or self.has_property(
                snake_to_camel(n)
            ):
                continue
            out.append(n)
        return tuple(sorted(out))

    def report(self) -> str:
        lines = [f"ontology: {self.triples} triples across {len(self.modules)} modules"]
        for m in self.modules:
            lines.append(
                f"  {m.triples:>6} triples  {m.name:<16} "
                f"classes={m.classes:<3} obj={m.object_properties:<3} "
                f"data={m.datatype_properties}"
            )
        lines.append(
            f"  vocabulary: {len(self.classes)} classes, "
            f"{len(self.object_properties)} object properties, "
            f"{len(self.datatype_properties)} datatype properties"
        )
        if self.duplicate_declarations:
            lines.append(
                f"  {self.duplicate_declarations} triple(s) declared in more than "
                "one module (deduplicated on merge)"
            )
        if self.missing_modules:
            lines.append(f"  MISSING modules: {list(self.missing_modules)}")
        return "\n".join(lines)


def _local(uri) -> str:
    text = str(uri)
    return text.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def load_modules(directory: str | Path = ONTOLOGY_DIR) -> OntologyBundle:
    """Load every TTL module. Aborts on the first parse failure."""
    directory = Path(directory)
    if not directory.is_dir():
        raise OntologyError(f"ontology directory not found: {directory}")
    paths = sorted(directory.glob("*.ttl"))
    if not paths:
        raise OntologyError(f"no .ttl modules found in {directory}")

    combined = Graph()
    combined.bind("bio", BIO)
    combined.bind("skos", SKOS)
    stats: list[ModuleStats] = []

    for path in paths:
        module = Graph()
        try:
            module.parse(path, format="turtle")
        except Exception as e:  # noqa: BLE001
            # Requirement 1.2: name the module, and the line when rdflib gives it.
            line = ""
            match = re.search(r"line[: ]+(\d+)", str(e), re.IGNORECASE)
            if match:
                line = f" at line {match.group(1)}"
            raise OntologyError(
                f"failed to parse ontology module {path.name}{line}: {e}"
            ) from e

        stats.append(
            ModuleStats(
                name=path.stem,
                path=str(path),
                triples=len(module),
                classes=len(set(module.subjects(RDF.type, OWL.Class))),
                object_properties=len(set(module.subjects(RDF.type, OWL.ObjectProperty))),
                datatype_properties=len(
                    set(module.subjects(RDF.type, OWL.DatatypeProperty))
                ),
            )
        )
        combined += module

    classes = {_local(s) for s in combined.subjects(RDF.type, OWL.Class)}
    obj_props = {_local(s) for s in combined.subjects(RDF.type, OWL.ObjectProperty)}
    data_props = {_local(s) for s in combined.subjects(RDF.type, OWL.DatatypeProperty)}
    labels = {
        _local(s): str(o) for s, o in combined.subject_objects(RDFS.label)
        if isinstance(s, URIRef)
    }

    loaded = {m.name for m in stats}
    missing = tuple(m for m in EXPECTED_MODULES if m not in loaded)

    return OntologyBundle(
        graph=combined,
        modules=tuple(stats),
        classes=frozenset(classes),
        object_properties=frozenset(obj_props),
        datatype_properties=frozenset(data_props),
        labels=labels,
        missing_modules=missing,
    )
