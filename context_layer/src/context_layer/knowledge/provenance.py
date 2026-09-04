"""IRI minting and DCAT provenance. Tasks 1.4, 1.5, Requirements 1.5, 2.1-2.4.

Two small pieces that make assertions citable:

* **IRI minting** gives every entity a stable, deterministic identifier derived
  from its class and natural key, so the same entity always mints the same IRI
  regardless of load order or machine.
* **DCAT cataloguing** records where a dataset came from and when, and assembles
  the provenance chain behind an answer (Requirement 2.4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import DCTERMS, XSD

from context_layer.types import Provenance, utcnow

if TYPE_CHECKING:
    from config import Config

BIO = Namespace("https://biomedkg.org/ontology/")
INSTANCE = Namespace("https://biomedkg.org/instance/")
DCAT = Namespace("http://www.w3.org/ns/dcat#")
PROV = Namespace("http://www.w3.org/ns/prov#")


# ── IRI minting, R1.5 ─────────────────────────────────────────────────────────


class IriMinter:
    """Deterministic IRIs. Same inputs always yield the same IRI."""

    def __init__(self, base: str = str(INSTANCE)) -> None:
        self.base = base.rstrip("/") + "/"

    @staticmethod
    def _slug(value: str) -> str:
        cleaned = "".join(
            ch if ch.isalnum() or ch in "-_." else "-" for ch in str(value).strip()
        )
        return "-".join(filter(None, cleaned.split("-")))[:96]

    def mint(self, class_name: str, natural_key: str) -> str:
        if not class_name or not str(natural_key).strip():
            raise ValueError("minting an IRI requires a class name and a natural key")
        return f"{self.base}{self._slug(class_name)}/{self._slug(natural_key)}"

    def mint_hashed(self, class_name: str, parts: Sequence[str]) -> str:
        """For composite keys: a stable hash keeps the IRI short and opaque."""
        if not parts:
            raise ValueError("minting a hashed IRI requires at least one part")
        digest = hashlib.sha256(
            "|".join(str(p) for p in parts).encode()
        ).hexdigest()[:20]
        return f"{self.base}{self._slug(class_name)}/{digest}"

    def is_minted(self, iri: str) -> bool:
        return str(iri).startswith(self.base)


# ── governance policy instances, R1.4 ─────────────────────────────────────────

ENFORCEMENT_LEVELS = ("Mandatory", "Advisory", "Informational")


class PolicyError(ValueError):
    """A governance policy instance is incomplete."""


@dataclass(frozen=True)
class PolicyInstance:
    policy_id: str
    policy_name: str
    enforcement_level: str
    jurisdiction: str
    description: str = ""

    def __post_init__(self) -> None:
        # Requirement 1.4: governance.ttl requires all three.
        if not self.policy_id:
            raise PolicyError("policy_id is required")
        if not self.policy_name:
            raise PolicyError(f"{self.policy_id}: policy_name is required")
        if self.enforcement_level not in ENFORCEMENT_LEVELS:
            raise PolicyError(
                f"{self.policy_id}: enforcement_level {self.enforcement_level!r} "
                f"must be one of {ENFORCEMENT_LEVELS}"
            )
        if not self.jurisdiction:
            raise PolicyError(f"{self.policy_id}: jurisdiction is required")

    def iri(self, minter: IriMinter) -> str:
        return minter.mint("DataGovernancePolicy", self.policy_id)

    def to_rdf(self, minter: IriMinter) -> Graph:
        g = Graph()
        g.bind("bio", BIO)
        s = URIRef(self.iri(minter))
        g.add((s, RDF.type, BIO.DataGovernancePolicy))
        g.add((s, BIO.policy_id, Literal(self.policy_id)))
        g.add((s, BIO.policy_name, Literal(self.policy_name)))
        g.add((s, BIO.enforcement_level, Literal(self.enforcement_level)))
        g.add((s, BIO.jurisdiction, Literal(self.jurisdiction)))
        if self.description:
            g.add((s, DCTERMS.description, Literal(self.description)))
        return g


# ── DCAT catalogue, R2.1 ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    title: str
    source: str
    ingested_at: datetime
    record_count: int
    description: str = ""

    def to_rdf(self, minter: IriMinter) -> Graph:
        g = Graph()
        g.bind("dcat", DCAT)
        g.bind("dcterms", DCTERMS)
        g.bind("prov", PROV)
        s = URIRef(minter.mint("Dataset", self.dataset_id))
        g.add((s, RDF.type, DCAT.Dataset))
        g.add((s, DCTERMS.title, Literal(self.title)))
        g.add((s, DCTERMS.source, Literal(self.source)))
        g.add((s, DCTERMS.issued, Literal(self.ingested_at.isoformat(), datatype=XSD.dateTime)))
        g.add((s, DCAT.byteSize, Literal(self.record_count, datatype=XSD.integer)))
        if self.description:
            g.add((s, DCTERMS.description, Literal(self.description)))
        g.add((s, PROV.wasGeneratedBy, Literal("context-layer-loader")))
        return g


class DcatCatalog:
    """Accumulates dataset records and emits them as RDF."""

    def __init__(self, minter: IriMinter | None = None) -> None:
        self.minter = minter or IriMinter()
        self.records: list[DatasetRecord] = []

    def record(
        self,
        dataset_id: str,
        title: str,
        source: str,
        record_count: int,
        description: str = "",
        ingested_at: datetime | None = None,
    ) -> DatasetRecord:
        rec = DatasetRecord(
            dataset_id=dataset_id,
            title=title,
            source=source,
            ingested_at=ingested_at or utcnow(),
            record_count=record_count,
            description=description,
        )
        self.records.append(rec)
        return rec

    def to_rdf(self) -> Graph:
        g = Graph()
        for rec in self.records:
            g += rec.to_rdf(self.minter)
        return g

    def serialize(self, path: str, fmt: str = "turtle") -> str:
        self.to_rdf().serialize(destination=path, format=fmt)
        return path


# ── provenance chain, R2.3, R2.4 ──────────────────────────────────────────────


PROVENANCE_MISSING = "provenance_missing"


def is_complete(provenance: Provenance | None) -> bool:
    return provenance is not None and provenance.is_complete


def flag_incomplete(elements: Iterable[Any]) -> tuple[str, ...]:
    """Element ids whose provenance is unusable. R2.3 — exclude from grounding."""
    return tuple(
        e.element_id
        for e in elements
        if not is_complete(getattr(e, "provenance", None))
    )


@dataclass
class ProvenanceChain:
    """Answer -> context element -> graph concept -> source record. R2.4."""

    answer_id: str
    links: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        element_id: str,
        source: str,
        ingested_at: datetime,
        confidence: float,
        curies: Sequence[str] = (),
        record_id: str | None = None,
    ) -> None:
        self.links.append(
            {
                "element_id": element_id,
                "source": source,
                "ingested_at": ingested_at.isoformat(),
                "confidence": confidence,
                "curies": list(curies),
                "record_id": record_id,
            }
        )

    @property
    def is_empty(self) -> bool:
        return not self.links

    def sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(link["source"] for link in self.links))

    def weakest_confidence(self) -> float:
        return min((link["confidence"] for link in self.links), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "sources": list(self.sources()),
            "weakest_confidence": self.weakest_confidence(),
            "links": self.links,
        }


def chain_for(answer_id: str, package) -> ProvenanceChain:
    """Assemble the chain behind an answer from its context package."""
    chain = ProvenanceChain(answer_id=answer_id)
    for element in package.elements:
        p = element.provenance
        chain.add(
            element_id=element.element_id,
            source=p.source,
            ingested_at=p.ingested_at,
            confidence=p.confidence,
            curies=element.curies,
            record_id=p.record_id,
        )
    return chain
