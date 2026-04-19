"""
agents/compliance.py — Compliance Agent verdict logic.

Verifies substitute candidates against regulatory and quality criteria
retrieved from the cognee knowledge graph via the FastAPI /search endpoint.
Returns PASS / FAIL / NEEDS_REVIEW verdicts with evidence citations.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPLIANCE_TIMEOUT_SECONDS = 30
VALID_VERDICTS = frozenset({"PASS", "FAIL", "NEEDS_REVIEW"})


# ---------------------------------------------------------------------------
# 1. Assessment — Requirements 5.1, 5.2
# ---------------------------------------------------------------------------


def assess_compliance(
    candidate: dict[str, Any],
    evidence_nodes: list[dict[str, Any]],
    llm_client: Any,
) -> str:
    """Assess a substitute candidate against regulatory/quality evidence.

    Returns exactly one of ``PASS``, ``FAIL``, or ``NEEDS_REVIEW``.

    Requirement 5.1 — verify against criteria from cognee.
    Requirement 5.2 — verdict in {PASS, FAIL, NEEDS_REVIEW}.
    """
    if not evidence_nodes:
        return "NEEDS_REVIEW"

    candidate_sku = candidate.get("substitute_sku", "unknown")
    candidate_id = candidate.get("substitute_id", "unknown")
    evidence_text = _format_evidence_for_prompt(evidence_nodes)

    prompt = (
        "You are a regulatory compliance assessor for food ingredients.\n\n"
        f"Candidate ingredient: {candidate_sku} (ID: {candidate_id})\n\n"
        f"Evidence from knowledge graph:\n{evidence_text}\n\n"
        "Based on the evidence, determine if this ingredient meets "
        "regulatory and quality standards.\n\n"
        "Respond with exactly one JSON object:\n"
        '{"verdict": "PASS" or "FAIL" or "NEEDS_REVIEW", '
        '"reason": "brief explanation"}\n\n'
        "JSON only, no other text."
    )

    try:
        import json
        response = llm_client.generate_content(prompt)
        parsed = json.loads(response.text)
        verdict = parsed.get("verdict", "NEEDS_REVIEW")
        if verdict not in VALID_VERDICTS:
            logger.warning(
                "LLM returned invalid verdict %r for candidate %s; "
                "defaulting to NEEDS_REVIEW", verdict, candidate_id,
            )
            return "NEEDS_REVIEW"
        return verdict
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "assess_compliance: LLM call failed for candidate %s — %s: %s",
            candidate_id, type(exc).__name__, exc,
        )
        return "NEEDS_REVIEW"


# ---------------------------------------------------------------------------
# 2. Result assembly — Requirements 5.3, 5.4, 5.6
# ---------------------------------------------------------------------------


def build_compliance_result(
    candidate_id: int | str,
    verdict: str,
    evidence_nodes: list[dict[str, Any]],
    fail_reason: str | None = None,
    missing_evidence: str | None = None,
) -> dict[str, Any]:
    """Assemble a ComplianceResult dict with evidence citations.

    Enforces:
      - FAIL -> fail_reason must be non-null and non-empty (Req 5.6).
      - NEEDS_REVIEW -> missing_evidence must be non-null and non-empty (Req 5.4).
      - Every result includes >= 1 evidence citation (Req 5.3).
    """
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"Invalid verdict {verdict!r}; must be one of {sorted(VALID_VERDICTS)}"
        )

    if verdict == "FAIL" and not fail_reason:
        fail_reason = "Compliance check failed based on available evidence."

    if verdict == "NEEDS_REVIEW" and not missing_evidence:
        missing_evidence = "Insufficient evidence to make a definitive determination."

    citations = _build_citations(evidence_nodes)

    return {
        "substitute_id": candidate_id,
        "verdict": verdict,
        "fail_reason": fail_reason if verdict == "FAIL" else None,
        "missing_evidence": missing_evidence if verdict == "NEEDS_REVIEW" else None,
        "evidence_citations": citations,
    }


# ---------------------------------------------------------------------------
# 3. Orchestration — Requirement 5.5
# ---------------------------------------------------------------------------


def run_compliance(
    candidate: dict[str, Any],
    search_client: Any,
    llm_client: Any,
) -> dict[str, Any]:
    """Orchestrate compliance verification for a single substitute candidate.

    Must complete within COMPLIANCE_TIMEOUT_SECONDS (30 s).
    Requirement 5.5 — single-candidate verification within 30 seconds.
    """
    start = time.monotonic()
    candidate_id = candidate.get("substitute_id", 0)
    candidate_sku = candidate.get("substitute_sku", "")

    evidence_nodes = _query_evidence(search_client, candidate_sku, candidate_id)
    _check_timeout(start)

    verdict = assess_compliance(candidate, evidence_nodes, llm_client)
    _check_timeout(start)

    fail_reason: str | None = None
    missing_evidence: str | None = None

    if verdict == "FAIL":
        fail_reason = _extract_fail_reason(candidate, evidence_nodes, llm_client)
    elif verdict == "NEEDS_REVIEW":
        missing_evidence = _describe_missing_evidence(candidate, evidence_nodes)

    result = build_compliance_result(
        candidate_id=candidate_id,
        verdict=verdict,
        evidence_nodes=evidence_nodes,
        fail_reason=fail_reason,
        missing_evidence=missing_evidence,
    )

    elapsed = time.monotonic() - start
    logger.info(
        "Compliance for candidate %s completed in %.2f s — verdict: %s",
        candidate_id, elapsed, verdict,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_evidence_for_prompt(evidence_nodes: list[dict[str, Any]]) -> str:
    """Format evidence nodes into a readable string for the LLM prompt."""
    parts: list[str] = []
    for i, node in enumerate(evidence_nodes, 1):
        content = node.get("content", "")
        node_type = node.get("node_type", "")
        confidence = node.get("confidence_score", 0.0)
        parts.append(f"[{i}] ({node_type}, confidence={confidence:.2f}) {content}")
    return "\n".join(parts) if parts else "(no evidence available)"


def _build_citations(evidence_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build evidence citation dicts from search result nodes."""
    return [
        {
            "source_url": node.get("metadata", {}).get("source_url", ""),
            "extracted_field": node.get("content", ""),
            "confidence_score": node.get("confidence_score", 0.0),
            "node_id": str(node.get("node_id", "")),
        }
        for node in evidence_nodes
    ]


def _query_evidence(
    search_client: Any,
    candidate_sku: str,
    candidate_id: int | str,
) -> list[dict[str, Any]]:
    """Query the FastAPI /search endpoint for compliance evidence."""
    try:
        response = search_client.post(
            "/search",
            json={
                "query": (
                    f"regulatory compliance certifications quality "
                    f"standards for {candidate_sku}"
                ),
                "k": 10,
                "node_types": ["Evidence", "RawMaterial"],
            },
        )
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception:
        logger.exception(
            "Evidence query failed for candidate %s (%s)",
            candidate_id, candidate_sku,
        )
        return []


def _extract_fail_reason(
    candidate: dict[str, Any],
    evidence_nodes: list[dict[str, Any]],
    llm_client: Any,
) -> str:
    """Ask the LLM to produce a human-readable fail reason."""
    candidate_sku = candidate.get("substitute_sku", "unknown")
    evidence_text = _format_evidence_for_prompt(evidence_nodes)
    prompt = (
        f"The ingredient '{candidate_sku}' FAILED compliance verification.\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Provide a concise, human-readable explanation of why it failed. "
        "One or two sentences only."
    )
    try:
        response = llm_client.generate_content(prompt)
        reason = response.text.strip()
        return reason if reason else "Compliance check failed based on available evidence."
    except Exception as exc:  # noqa: BLE001
        logger.error("_extract_fail_reason: LLM call failed — %s: %s", type(exc).__name__, exc)
        return "Compliance check failed based on available evidence."


def _describe_missing_evidence(
    candidate: dict[str, Any],
    evidence_nodes: list[dict[str, Any]],
) -> str:
    """Describe what evidence is missing for a NEEDS_REVIEW verdict."""
    candidate_sku = candidate.get("substitute_sku", "unknown")
    if not evidence_nodes:
        return (
            f"No regulatory or quality evidence found for '{candidate_sku}'. "
            "Manual review of certifications, allergen status, and "
            "regulatory approvals is required."
        )
    return (
        f"Insufficient evidence to definitively assess compliance for "
        f"'{candidate_sku}'. Additional regulatory documentation or "
        "certification records may be needed."
    )


def _check_timeout(start: float) -> None:
    """Raise TimeoutError if the compliance budget is exhausted."""
    elapsed = time.monotonic() - start
    if elapsed >= COMPLIANCE_TIMEOUT_SECONDS:
        raise TimeoutError(
            f"Compliance exceeded {COMPLIANCE_TIMEOUT_SECONDS}s timeout "
            f"(elapsed: {elapsed:.1f}s)"
        )
