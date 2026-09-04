"""Governed AI Responses. Tasks 6.1-6.2.

Diagram: [AI-5] Governed AI Responses. Hub node: Governance.

Contracts:
    grounding.score(answer, package) -> float
    grounding.citation_coverage(answer, citations) -> float   (R10.4)
    guards.check_clinical_advice(question) -> GuardResult     (R10.3)
    contract.GovernedResponse                                 (R10.1)

Refuses below MIN_GROUNDING_CONFIDENCE, stating what context was missing
(R10.2). Refusal is a first-class outcome, not an error.
"""
