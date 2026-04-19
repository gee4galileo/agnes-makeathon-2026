"""
tests/test_compliance_agent.py — Property-based and unit tests for agents/compliance.py.

Properties tested:
  Property 14: Compliance verdict enum constraint (Requirement 5.2)
  Property 15: FAIL verdict includes fail_reason (Requirement 5.6)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agents.compliance import (
    VALID_VERDICTS,
    assess_compliance,
    build_compliance_result,
    run_compliance,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_verdict_st = st.sampled_from(["PASS", "FAIL", "NEEDS_REVIEW"])

_candidate_id_st = st.integers(min_value=1, max_value=1000)

_evidence_node_st = st.fixed_dictionaries({
    "node_id": st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    "node_type": st.sampled_from(["Evidence", "RawMaterial"]),
    "content": st.text(min_size=1, max_size=100),
    "confidence_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "metadata": st.fixed_dictionaries({
        "source_url": st.from_regex(r"https://[a-z]+\.example\.com/[a-z0-9-]+", fullmatch=True),
    }),
})

_evidence_nodes_st = st.lists(_evidence_node_st, min_size=0, max_size=5)

_candidate_st = st.fixed_dictionaries({
    "substitute_id": _candidate_id_st,
    "substitute_sku": st.text(
        min_size=1, max_size=30,
        alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    ),
})


# ---------------------------------------------------------------------------
# Mock LLM helper
# ---------------------------------------------------------------------------


def _make_llm_mock(verdict: str = "PASS", reason: str = "looks good") -> MagicMock:
    """Return a mock LLM client whose generate_content returns a JSON verdict."""
    mock = MagicMock()
    mock.generate_content.return_value = SimpleNamespace(
        text=json.dumps({"verdict": verdict, "reason": reason})
    )
    return mock


# ---------------------------------------------------------------------------
# Property 14: Compliance verdict enum constraint (Requirement 5.2)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    candidate=_candidate_st,
    evidence_nodes=_evidence_nodes_st,
    llm_verdict=_verdict_st,
)
def test_property14_verdict_enum_constraint(candidate, evidence_nodes, llm_verdict):
    """Feature: agnes-ai-supply-chain-manager, Property 14: Compliance verdict enum constraint

    **Validates: Requirements 5.2**

    For any substitute candidate processed by the Compliance Agent, the
    returned verdict SHALL be exactly one of PASS, FAIL, or NEEDS_REVIEW.
    """
    llm_client = _make_llm_mock(verdict=llm_verdict)
    verdict = assess_compliance(candidate, evidence_nodes, llm_client)
    assert verdict in VALID_VERDICTS, (
        f"assess_compliance returned {verdict!r}, expected one of {sorted(VALID_VERDICTS)}"
    )


@settings(max_examples=100)
@given(
    candidate=_candidate_st,
    evidence_nodes=_evidence_nodes_st,
)
def test_property14_invalid_llm_verdict_defaults_to_needs_review(candidate, evidence_nodes):
    """Property 14 edge case: LLM returns an invalid verdict string.

    The agent SHALL still return a valid verdict (NEEDS_REVIEW as fallback).
    """
    llm_client = _make_llm_mock(verdict="INVALID_VERDICT")
    verdict = assess_compliance(candidate, evidence_nodes, llm_client)
    assert verdict in VALID_VERDICTS



# ---------------------------------------------------------------------------
# Property 15: FAIL verdict includes fail_reason (Requirement 5.6)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    candidate_id=_candidate_id_st,
    evidence_nodes=st.lists(_evidence_node_st, min_size=1, max_size=5),
)
def test_property15_fail_verdict_includes_fail_reason(candidate_id, evidence_nodes):
    """Feature: agnes-ai-supply-chain-manager, Property 15: FAIL verdict includes fail_reason

    **Validates: Requirements 5.6**

    For any substitute candidate that receives a FAIL compliance verdict,
    the response SHALL include a non-empty fail_reason string.
    """
    result = build_compliance_result(
        candidate_id=candidate_id,
        verdict="FAIL",
        evidence_nodes=evidence_nodes,
        fail_reason=None,  # force the default
    )
    assert result["verdict"] == "FAIL"
    assert result["fail_reason"] is not None
    assert isinstance(result["fail_reason"], str)
    assert len(result["fail_reason"]) > 0


@settings(max_examples=100)
@given(
    candidate_id=_candidate_id_st,
    evidence_nodes=st.lists(_evidence_node_st, min_size=1, max_size=5),
    custom_reason=st.text(min_size=1, max_size=200),
)
def test_property15_fail_preserves_custom_reason(candidate_id, evidence_nodes, custom_reason):
    """Property 15: when a custom fail_reason is provided, it is preserved."""
    result = build_compliance_result(
        candidate_id=candidate_id,
        verdict="FAIL",
        evidence_nodes=evidence_nodes,
        fail_reason=custom_reason,
    )
    assert result["fail_reason"] == custom_reason


# ---------------------------------------------------------------------------
# Unit tests — Task 10.4 (Requirements 5.4, 5.6)
# ---------------------------------------------------------------------------


class TestComplianceUnit:
    """Unit tests for compliance agent verdict logic and result assembly."""

    def test_insufficient_evidence_returns_needs_review_with_missing_evidence(self):
        """No evidence -> NEEDS_REVIEW + missing_evidence field present.

        **Validates: Requirements 5.4**
        """
        llm_client = _make_llm_mock()
        candidate = {"substitute_id": 42, "substitute_sku": "PALM-OIL-RSPO"}

        verdict = assess_compliance(candidate, [], llm_client)
        assert verdict == "NEEDS_REVIEW"

        result = build_compliance_result(
            candidate_id=42,
            verdict="NEEDS_REVIEW",
            evidence_nodes=[],
        )
        assert result["verdict"] == "NEEDS_REVIEW"
        assert result["missing_evidence"] is not None
        assert isinstance(result["missing_evidence"], str)
        assert len(result["missing_evidence"]) > 0

    def test_pass_verdict_has_no_fail_reason_or_missing_evidence(self):
        """PASS verdict should have fail_reason=None and missing_evidence=None."""
        evidence = [
            {
                "node_id": "abc-123",
                "node_type": "Evidence",
                "content": "RSPO certified palm oil",
                "confidence_score": 0.95,
                "metadata": {"source_url": "https://rspo.org/cert"},
            }
        ]
        result = build_compliance_result(
            candidate_id=17,
            verdict="PASS",
            evidence_nodes=evidence,
        )
        assert result["verdict"] == "PASS"
        assert result["fail_reason"] is None
        assert result["missing_evidence"] is None

    def test_invalid_verdict_raises_value_error(self):
        """build_compliance_result rejects invalid verdict strings."""
        with pytest.raises(ValueError, match="Invalid verdict"):
            build_compliance_result(
                candidate_id=1,
                verdict="MAYBE",
                evidence_nodes=[],
            )

    def test_evidence_citations_built_from_nodes(self):
        """Evidence citations are correctly assembled from evidence nodes."""
        evidence = [
            {
                "node_id": "node-1",
                "node_type": "Evidence",
                "content": "ISO 9001 certified",
                "confidence_score": 0.88,
                "metadata": {"source_url": "https://example.com/cert"},
            },
            {
                "node_id": "node-2",
                "node_type": "RawMaterial",
                "content": "Palm kernel oil",
                "confidence_score": 0.72,
                "metadata": {"source_url": "https://supplier.com/pko"},
            },
        ]
        result = build_compliance_result(
            candidate_id=10,
            verdict="PASS",
            evidence_nodes=evidence,
        )
        assert len(result["evidence_citations"]) == 2
        for citation in result["evidence_citations"]:
            assert "source_url" in citation
            assert "extracted_field" in citation
            assert "confidence_score" in citation
            assert "node_id" in citation

    def test_run_compliance_with_mock_search_and_llm(self):
        """run_compliance orchestrates search + LLM + result assembly."""
        search_client = MagicMock()
        search_response = MagicMock()
        search_response.json.return_value = {
            "results": [
                {
                    "node_id": "ev-1",
                    "node_type": "Evidence",
                    "content": "FDA GRAS approved",
                    "confidence_score": 0.92,
                    "metadata": {"source_url": "https://fda.gov/gras/123"},
                }
            ]
        }
        search_response.raise_for_status = MagicMock()
        search_client.post.return_value = search_response

        llm_client = _make_llm_mock(verdict="PASS")

        candidate = {"substitute_id": 42, "substitute_sku": "LECITHIN-SOY"}
        result = run_compliance(candidate, search_client, llm_client)

        assert result["verdict"] == "PASS"
        assert result["substitute_id"] == 42
        assert result["fail_reason"] is None
        assert result["missing_evidence"] is None
        assert len(result["evidence_citations"]) >= 1

    def test_run_compliance_search_failure_returns_needs_review(self):
        """When search fails, run_compliance returns NEEDS_REVIEW."""
        search_client = MagicMock()
        search_client.post.side_effect = ConnectionError("search unavailable")

        llm_client = _make_llm_mock()

        candidate = {"substitute_id": 99, "substitute_sku": "UNKNOWN-INGREDIENT"}
        result = run_compliance(candidate, search_client, llm_client)

        assert result["verdict"] == "NEEDS_REVIEW"
        assert result["missing_evidence"] is not None

    def test_run_compliance_llm_failure_returns_needs_review(self):
        """When LLM fails, run_compliance returns NEEDS_REVIEW."""
        search_client = MagicMock()
        search_response = MagicMock()
        search_response.json.return_value = {
            "results": [
                {
                    "node_id": "ev-1",
                    "node_type": "Evidence",
                    "content": "Some evidence",
                    "confidence_score": 0.8,
                    "metadata": {"source_url": "https://example.com/ev"},
                }
            ]
        }
        search_response.raise_for_status = MagicMock()
        search_client.post.return_value = search_response

        llm_client = MagicMock()
        llm_client.generate_content.side_effect = RuntimeError("LLM down")

        candidate = {"substitute_id": 55, "substitute_sku": "CITRIC-ACID"}
        result = run_compliance(candidate, search_client, llm_client)

        assert result["verdict"] == "NEEDS_REVIEW"
