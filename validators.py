"""
validators.py — SourcingProposal JSON schema validation and citation enforcement.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import jsonschema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SchemaValidationError(Exception):
    """Raised when a SourcingProposal fails JSON schema validation."""

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__(f"Schema validation failed with {len(errors)} error(s)")


# ---------------------------------------------------------------------------
# SourcingProposal JSON Schema (from design doc)
# ---------------------------------------------------------------------------

SOURCING_PROPOSAL_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SourcingProposal",
    "type": "object",
    "required": [
        "consolidation_actions",
        "compliance_verdicts",
        "evidence_citations",
        "summary",
    ],
    "properties": {
        "consolidation_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "ingredient_id",
                    "recommended_supplier",
                    "compliance_verdict",
                    "evidence_citations",
                ],
                "properties": {
                    "ingredient_id": {"type": "integer"},
                    "recommended_supplier": {"type": "string"},
                    "substitute_ingredient": {"type": ["integer", "null"]},
                    "similarity_score": {
                        "type": ["number", "null"],
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "compliance_verdict": {
                        "type": "string",
                        "enum": ["PASS", "FAIL", "NEEDS_REVIEW"],
                    },
                    "evidence_citations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/definitions/EvidenceCitation"},
                    },
                },
            },
        },
        "compliance_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["ingredient_id", "verdict"],
                "properties": {
                    "ingredient_id": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": ["PASS", "FAIL", "NEEDS_REVIEW"],
                    },
                    "fail_reason": {"type": ["string", "null"]},
                    "missing_evidence": {"type": ["string", "null"]},
                },
            },
        },
        "evidence_citations": {
            "type": "array",
            "items": {"$ref": "#/definitions/EvidenceCitation"},
        },
        "summary": {"type": "string", "maxLength": 500},
        "solver_status": {
            "type": "string",
            "description": "Present only when PuLP solver is invoked",
        },
    },
    "definitions": {
        "EvidenceCitation": {
            "type": "object",
            "required": [
                "source_url",
                "extracted_field",
                "confidence_score",
                "node_id",
            ],
            "properties": {
                "source_url": {"type": "string"},
                "extracted_field": {"type": "string"},
                "confidence_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "node_id": {"type": "string"},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# 1. Schema validation — Requirements 8.1, 8.2, 8.3
# ---------------------------------------------------------------------------


def validate_proposal(proposal_dict: dict[str, Any]) -> None:
    """Validate a SourcingProposal dict against the JSON schema.

    Raises SchemaValidationError with a structured error list on failure.
    Requirements 8.2, 8.3.
    """
    validator = jsonschema.Draft7Validator(SOURCING_PROPOSAL_SCHEMA)
    errors = sorted(validator.iter_errors(proposal_dict), key=lambda e: list(e.path))

    if errors:
        structured = [
            {
                "path": list(e.absolute_path),
                "message": e.message,
                "validator": e.validator,
            }
            for e in errors
        ]
        raise SchemaValidationError(structured)


def validate_summary_length(summary: str) -> None:
    """Assert the summary has at most 500 words.

    Requirement 8.4.
    """
    word_count = len(summary.split())
    if word_count > 500:
        raise ValueError(
            f"Summary exceeds 500-word limit: {word_count} words"
        )


# ---------------------------------------------------------------------------
# 2. Citation validation — Requirements 7.1, 7.2, 7.3
# ---------------------------------------------------------------------------


def validate_citations(
    proposal_dict: dict[str, Any],
    cognee_client: Any,
) -> list[dict[str, Any]]:
    """Check every node_id in every evidence_citation exists in cognee.

    Returns a list of invalid citation dicts (empty if all valid).
    Requirements 7.2, 7.3.
    """
    invalid: list[dict[str, Any]] = []

    all_citations = _collect_all_citations(proposal_dict)

    for citation in all_citations:
        node_id = citation.get("node_id", "")
        if not node_id:
            invalid.append({"citation": citation, "reason": "empty node_id"})
            continue

        if not _node_exists(cognee_client, node_id):
            invalid.append({
                "citation": citation,
                "reason": f"node_id '{node_id}' not found in cognee",
            })

    return invalid


def enforce_citations(
    proposal_dict: dict[str, Any],
    agent_fn: Callable[[], dict[str, Any]],
    cognee_client: Any,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Validate citations; retry agent_fn on failure; mark unverified after exhaustion.

    Requirements 7.4, 7.5.
    """
    current = proposal_dict

    for attempt in range(1, max_retries + 2):  # 1 initial + max_retries
        invalid = validate_citations(current, cognee_client)
        if not invalid:
            return current

        if attempt <= max_retries:
            logger.warning(
                "Citation validation failed (attempt %d/%d): %d invalid citations",
                attempt, max_retries, len(invalid),
            )
            try:
                current = agent_fn()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Agent regeneration failed on attempt %d: %s", attempt, exc
                )
                break
        else:
            break

    # All retries exhausted — mark affected claims as unverified
    logger.warning("All citation retries exhausted; marking claims as unverified")
    return _mark_unverified(current, invalid)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_all_citations(proposal_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect all evidence citations from a SourcingProposal."""
    citations: list[dict[str, Any]] = []

    for action in proposal_dict.get("consolidation_actions", []):
        citations.extend(action.get("evidence_citations", []))

    citations.extend(proposal_dict.get("evidence_citations", []))

    return citations


def _node_exists(cognee_client: Any, node_id: str) -> bool:
    """Check if a node exists in the cognee knowledge graph."""
    try:
        search_fn = getattr(cognee_client, "search", None)
        if search_fn is None:
            return False
        result = search_fn(query=node_id, k=1)
        if result is None:
            return False
        results = list(result)
        return any(str(r.get("node_id", "")) == node_id for r in results)
    except Exception:
        logger.exception("Failed to check node existence for %s", node_id)
        return False


def _mark_unverified(
    proposal_dict: dict[str, Any],
    invalid_citations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mark claims with invalid citations as unverified."""
    invalid_node_ids = {
        c["citation"].get("node_id", "") for c in invalid_citations
    }

    for action in proposal_dict.get("consolidation_actions", []):
        for citation in action.get("evidence_citations", []):
            if citation.get("node_id", "") in invalid_node_ids:
                action["unverified"] = True
                break

    return proposal_dict
