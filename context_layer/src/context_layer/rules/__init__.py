"""Processes & Rules: the symbolic half of neurosymbolic. Tasks 4.1-4.3.

Diagram: [EK-3] Processes & Rules, [AI-1] Neurosymbolic Reasoning.

Verified absent from all surveyed repos. SHACL validates shape conformance;
it does not decide. This module decides.

Contracts:
    loader.load_pack(path) -> RulePack
        Validates id/description/severity/when/rationale. Fails at startup,
        never at request time (R3.1).
    engine.evaluate(package) -> RuleVerdict
        Pure function. No network, no LLM. Identical input yields identical
        output (R3.5, R3.6).
"""
