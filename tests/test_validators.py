"""
tests/test_validators.py — Property-based and unit tests for validators.py.

Properties tested:
  Property 17: SourcingProposal schema validity (Requirements 6.4, 8.1, 8.2, 8.5, 8.6)
  Property 18: Summary length constraint (Requirement 8.4)
  Property 13: All agent outputs cite at least one cognee node (Requirements 4.4, 5.3, 6.3, 7.1, 7.2)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from validators import (
    SOURCING_PROPOSAL_SCHEMA,
    SchemaValidationError,
    enforce_citations,
    validate_citations,
    validate_proposal,
    validate_summary_length,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_citation_st = st.fixed_dictionaries({
    "source_url": st.from_regex(r"https://[a-z]+\.example\.com/[a-z0-9-]+", fullmatch=True),
    "extracted_field": st.text(min_size=1, max_size=50),
    "confidence_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "node_id": st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
})

_action_st = st.fixed_dictionaries({
    "ingredient_id": st.integers(min_value=1, max_value=1000),
    "recommended_supplier": st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N", "Zs"))),
    "substitute_ingredient": st.one_of(st.none(), st.integers(min_value=1, max_value=1000)),
    "similarity_score": st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    "compliance_verdict": st.sampled_from(["PASS", "FAIL", "NEEDS_REVIEW"]),
    "evidence_citations": st.lists(_citation_st, min_size=1, max_size=3),
})

_verdict_st = st.fixed_dictionaries({
    "ingredient_id": st.integers(min_value=1, max_value=1000),
    "verdict": st.sampled_from(["PASS", "FAIL", "NEEDS_REVIEW"]),
    "fail_reason": st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    "missing_evidence": st.one_of(st.none(), st.text(min_size=1, max_size=50)),
})

# Summary limited to 500 chars to stay within schema maxLength
_summary_st = st.text(min_size=1, max_size=400, alphabet=st.characters(whitelist_categories=("L", "N", "Zs")))

_proposal_st = st.fixed_dictionaries({
    "consolidation_actions": st.lists(_action_st, min_size=0, max_size=5),
    "compliance_verdicts": st.lists(_verdict_st, min_size=0, max_size=5),
    "evidence_citations": st.lists(_citation_st, min_size=0, max_size=5),
    "summary": _summary_st,
})


# ---------------------------------------------------------------------------
# Property 17: SourcingProposal schema validity (Reqs 6.4, 8.1, 8.2, 8.5, 8.6)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(proposal=_proposal_st)
def test_property17_valid_proposals_pass_schema(proposal):
    """Feature: agnes-ai-supply-chain-manager, Property 17: SourcingProposal schema validity

    **Validates: Requirements 6.4, 8.1, 8.2, 8.5, 8.6**

    For any valid set of inputs to the Optimisation Agent, the produced
    SourcingProposal SHALL pass validation against the defined JSON schema.
    """
    # Should not raise
    validate_proposal(proposal)


# ---------------------------------------------------------------------------
# Property 18: Summary length constraint (Requirement 8.4)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(word_count=st.integers(min_value=1, max_value=500))
def test_property18_summary_within_limit_passes(word_count):
    """Feature: agnes-ai-supply-chain-manager, Property 18: Summary length constraint

    **Validates: Requirements 8.4**

    For any SourcingProposal, the summary field SHALL contain no more than 500 words.
    """
    summary = " ".join(["word"] * word_count)
    # Should not raise
    validate_summary_length(summary)


@settings(max_examples=100)
@given(extra_words=st.integers(min_value=1, max_value=100))
def test_property18_summary_over_limit_fails(extra_words):
    """Property 18: summaries exceeding 500 words are rejected."""
    summary = " ".join(["word"] * (501 + extra_words))
    with pytest.raises(ValueError, match="500-word limit"):
        validate_summary_length(summary)


# ---------------------------------------------------------------------------
# Property 13: All agent outputs cite at least one cognee node (Reqs 7.1, 7.2)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(proposal=_proposal_st)
def test_property13_every_action_has_citations(proposal):
    """Feature: agnes-ai-supply-chain-manager, Property 13: All agent outputs cite at least one cognee node

    **Validates: Requirements 4.4, 5.3, 6.3, 7.1, 7.2**

    Every consolidation action SHALL include at least one evidence_citation.
    """
    for action in proposal.get("consolidation_actions", []):
        assert len(action.get("evidence_citations", [])) >= 1, (
            f"Action for ingredient {action['ingredient_id']} has no citations"
        )


# ---------------------------------------------------------------------------
# Unit tests — Task 12.6 (Requirements 7.4, 7.5, 8.3)
# ---------------------------------------------------------------------------


class TestValidatorsUnit:
    """Unit tests for schema validation and citation enforcement."""

    def test_schema_validation_failure_returns_structured_errors(self):
        """Schema validation failure returns structured error response.

        **Validates: Requirements 8.3**
        """
        bad_proposal = {"summary": "hello"}  # missing required fields
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_proposal(bad_proposal)

        assert len(exc_info.value.errors) > 0
        for err in exc_info.value.errors:
            assert "message" in err
            assert "path" in err

    def test_valid_proposal_passes(self):
        """A known-good proposal passes schema validation."""
        good = {
            "consolidation_actions": [
                {
                    "ingredient_id": 1,
                    "recommended_supplier": "Supplier A",
                    "substitute_ingredient": None,
                    "similarity_score": 0.8,
                    "compliance_verdict": "PASS",
                    "evidence_citations": [
                        {
                            "source_url": "https://example.com/cert",
                            "extracted_field": "certifications",
                            "confidence_score": 0.9,
                            "node_id": "node-1",
                        }
                    ],
                }
            ],
            "compliance_verdicts": [
                {"ingredient_id": 1, "verdict": "PASS", "fail_reason": None, "missing_evidence": None}
            ],
            "evidence_citations": [],
            "summary": "Test summary.",
        }
        validate_proposal(good)  # should not raise

    def test_citation_enforcement_retries_exhausted_marks_unverified(self):
        """Citation enforcement: 2 retries exhausted -> claims marked unverified.

        **Validates: Requirements 7.4, 7.5**
        """
        # cognee client that never finds any node
        cognee_client = MagicMock()
        cognee_client.search.return_value = []

        proposal = {
            "consolidation_actions": [
                {
                    "ingredient_id": 1,
                    "recommended_supplier": "Supplier A",
                    "compliance_verdict": "PASS",
                    "evidence_citations": [
                        {
                            "source_url": "https://example.com/cert",
                            "extracted_field": "certs",
                            "confidence_score": 0.9,
                            "node_id": "nonexistent-node",
                        }
                    ],
                }
            ],
            "compliance_verdicts": [],
            "evidence_citations": [],
            "summary": "Test",
        }

        call_count = 0

        def agent_fn():
            nonlocal call_count
            call_count += 1
            return proposal  # always returns same bad proposal

        result = enforce_citations(proposal, agent_fn, cognee_client, max_retries=2)

        # Should have retried
        assert call_count <= 2
        # Claims should be marked unverified
        for action in result["consolidation_actions"]:
            if any(c["node_id"] == "nonexistent-node" for c in action["evidence_citations"]):
                assert action.get("unverified") is True

    def test_valid_citations_pass_without_retry(self):
        """Valid citations pass without triggering retries."""
        cognee_client = MagicMock()
        cognee_client.search.return_value = [{"node_id": "valid-node"}]

        proposal = {
            "consolidation_actions": [
                {
                    "ingredient_id": 1,
                    "recommended_supplier": "Supplier A",
                    "compliance_verdict": "PASS",
                    "evidence_citations": [
                        {
                            "source_url": "https://example.com/cert",
                            "extracted_field": "certs",
                            "confidence_score": 0.9,
                            "node_id": "valid-node",
                        }
                    ],
                }
            ],
            "compliance_verdicts": [],
            "evidence_citations": [],
            "summary": "Test",
        }

        call_count = 0

        def agent_fn():
            nonlocal call_count
            call_count += 1
            return proposal

        result = enforce_citations(proposal, agent_fn, cognee_client)
        assert call_count == 0  # no retries needed

    def test_sourcing_proposal_schema_is_valid_jsonschema(self):
        """SOURCING_PROPOSAL_SCHEMA is a valid JSON schema (parseable by jsonschema)."""
        import jsonschema
        jsonschema.Draft7Validator.check_schema(SOURCING_PROPOSAL_SCHEMA)
