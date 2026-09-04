"""Biomedical Context Layer.

Implements the reference architecture in
../../context_Layer/context_layer_linkedin.png over Neo4j + OpenSearch.

Spec: .kiro/specs/context-layer/
"""

__version__ = "0.1.0"

from context_layer.types import (  # noqa: F401
    Action,
    Citation,
    ContextElement,
    ContextPackage,
    Decision,
    EntityMention,
    Event,
    FiredRule,
    GovernedResponse,
    Outcome,
    Provenance,
    RuleVerdict,
    TextChunk,
)
