"""Governance gate. Task 6.1-6.2, Requirements 9.2, 10.1-10.4.

This is where the neurosymbolic override happens. Precedence is fixed in code,
not suggested in a prompt:

    1. clinical advice guard   -> refuse before anything else
    2. symbolic block verdict  -> suppress the model's answer, return the rule
    3. grounding threshold     -> refuse rather than guess
    4. otherwise              -> release the answer with citations and warnings

Because step 2 sits above step 4, a rule can veto fluent, confident prose. The
model's proposal is still published in ``neural_proposal`` so the override is
visible and auditable rather than implied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from context_layer.governance import grounding
from context_layer.types import (
    ContextPackage,
    GovernedResponse,
    RuleVerdict,
)

if TYPE_CHECKING:
    from config import Config

# Phrasings that request individual treatment guidance. Research questions
# about mechanism, evidence, or policy are unaffected.
ADVICE_PATTERNS = (
    r"\bshould (?:i|he|she|they|we|the patient|my)\b.*\b(take|use|start|stop|switch|dose|prescrib)",
    r"\bwhat (?:dose|dosage)\b.*\b(should|do|for me|for him|for her|for my)\b",
    r"\bis it safe for (?:me|him|her|my|the patient)\b",
    r"\bprescrib(?:e|ing) (?:for|to) (?:me|him|her|my|the patient)\b",
    r"\bhow much .* should (?:i|he|she|they|the patient) (?:take|receive)\b",
    r"\b(?:diagnose|treat) (?:me|my|him|her)\b",
    r"\bwhat treatment (?:should|do) (?:i|we|they) (?:choose|use|give)\b.*\bpatient\b",
)
_ADVICE_RE = tuple(re.compile(p, re.IGNORECASE) for p in ADVICE_PATTERNS)

ADVICE_REFUSAL = (
    "This asks for individual treatment guidance, which is clinical decision "
    "support and outside this system's scope. I can instead summarise the "
    "evidence, mechanism, trial results, or governing policy for the same "
    "subject."
)


@dataclass(frozen=True)
class GuardResult:
    triggered: bool
    reason: str = ""
    pattern: str = ""


def check_clinical_advice(question: str, enabled: bool = True) -> GuardResult:
    """Requirement 10.3."""
    if not enabled or not question:
        return GuardResult(False)
    for rx in _ADVICE_RE:
        if rx.search(question):
            return GuardResult(True, ADVICE_REFUSAL, rx.pattern)
    return GuardResult(False)


class GovernanceGate:
    """Turns a model proposal plus a rule verdict into a governed response."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        question: str,
        package: ContextPackage,
        verdict: RuleVerdict,
        neural_proposal: str | None,
    ) -> GovernedResponse:
        common = {
            "context_package_id": package.context_package_id,
            "rules_fired": verdict.fired,
            "degradations": package.degradations,
        }

        # 1. Clinical advice guard, before any generation is released.
        guard = check_clinical_advice(question, self.cfg.clinical_advice_guard)
        if guard.triggered:
            return GovernedResponse.refusal(
                reason=guard.reason,
                symbolic_verdict=verdict.summary(),
                grounding_confidence=0.0,
                **common,
            )

        # 2. Symbolic block overrides the neural proposal. Requirement 9.2.
        if verdict.blocked:
            return GovernedResponse(
                answer=(
                    "Blocked by policy. "
                    f"{verdict.block_rationale}"
                ),
                refused=False,
                symbolic_verdict=verdict.summary(),
                grounding_confidence=grounding.score(neural_proposal or "", package),
                citations=grounding.extract_citations(neural_proposal or "", package),
                neural_proposal=neural_proposal,
                **common,
            )

        # 3. Grounding threshold. Requirement 10.2.
        confidence = grounding.score(neural_proposal or "", package)
        if not neural_proposal or confidence < self.cfg.min_grounding_confidence:
            return GovernedResponse.refusal(
                reason=(
                    f"Insufficient grounding ({confidence:.2f} < "
                    f"{self.cfg.min_grounding_confidence:.2f}): "
                    f"{grounding.missing_context_note(package)}."
                ),
                symbolic_verdict=verdict.summary(),
                grounding_confidence=confidence,
                **common,
            )

        # 4. Release, carrying any non-blocking rules as advisories.
        answer = neural_proposal
        advisories = [f.rationale for f in verdict.fired if f.severity == "warn"]
        if advisories:
            answer += "\n\nAdvisories: " + " ".join(advisories)

        return GovernedResponse(
            answer=answer,
            refused=False,
            symbolic_verdict=verdict.summary(),
            grounding_confidence=confidence,
            citations=grounding.extract_citations(neural_proposal, package),
            neural_proposal=neural_proposal,
            **common,
        )
