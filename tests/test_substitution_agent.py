"""
tests/test_substitution_agent.py — Property-based and unit tests for agents/substitution.py.

Properties tested:
  Property 11: Similarity score range invariant (Requirement 4.2)
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agents.substitution import score_candidate


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Small 3-element embedding vectors for speed.
_embedding_st = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=3,
)

# LLM scores intentionally outside [0, 1] to exercise clamping.
_llm_score_st = st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 11: Similarity score range invariant (Requirement 4.2)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    ingredient_embedding=_embedding_st,
    candidate_embedding=_embedding_st,
    llm_score=_llm_score_st,
)
def test_property11_similarity_score_range(
    ingredient_embedding,
    candidate_embedding,
    llm_score,
):
    """Feature: agnes-ai-supply-chain-manager, Property 11: Similarity score range invariant

    **Validates: Requirements 4.2**

    For any raw-material product submitted to the Substitution Agent, every
    candidate substitute in the response SHALL have a similarity_score in the
    closed interval [0, 1].
    """
    score = score_candidate(ingredient_embedding, candidate_embedding, llm_score)
    assert 0.0 <= score <= 1.0, (
        f"score_candidate returned {score}, expected value in [0, 1]. "
        f"ingredient_embedding={ingredient_embedding}, "
        f"candidate_embedding={candidate_embedding}, "
        f"llm_score={llm_score}"
    )


# ---------------------------------------------------------------------------
# Strategies for Property 12
# ---------------------------------------------------------------------------

_similarity_score_st = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False,
)

_candidate_st = st.fixed_dictionaries(
    {"similarity_score": _similarity_score_st},
)

_candidates_list_st = st.lists(_candidate_st, min_size=0, max_size=30)


# ---------------------------------------------------------------------------
# Property 12: Similarity threshold filtering (Requirement 4.3)
# ---------------------------------------------------------------------------

from agents.substitution import filter_candidates, SIMILARITY_THRESHOLD


@settings(max_examples=100)
@given(candidates=_candidates_list_st)
def test_property12_similarity_threshold_filtering(candidates):
    """Feature: agnes-ai-supply-chain-manager, Property 12: Similarity threshold filtering

    **Validates: Requirements 4.3**

    For any set of substitute candidates produced by the Substitution Agent,
    only candidates with similarity_score >= 0.6 SHALL be forwarded to the
    Compliance Agent; all candidates with similarity_score < 0.6 SHALL be
    excluded.
    """
    result = filter_candidates(candidates)

    # 1. Every candidate in the result has similarity_score >= 0.6
    for c in result:
        assert c["similarity_score"] >= SIMILARITY_THRESHOLD, (
            f"Candidate with similarity_score={c['similarity_score']} "
            f"should have been excluded (threshold={SIMILARITY_THRESHOLD})"
        )

    # 2. No candidate with similarity_score < 0.6 appears in the result
    result_ids = [id(c) for c in result]
    for c in candidates:
        if c["similarity_score"] < SIMILARITY_THRESHOLD:
            assert id(c) not in result_ids, (
                f"Candidate with similarity_score={c['similarity_score']} "
                f"should NOT appear in filtered results"
            )

    # 3. All candidates with similarity_score >= 0.6 from the input ARE in the result
    expected_above = [c for c in candidates if c["similarity_score"] >= SIMILARITY_THRESHOLD]
    assert len(result) == len(expected_above), (
        f"Expected {len(expected_above)} candidates above threshold, "
        f"got {len(result)}"
    )
    for c in expected_above:
        assert c in result, (
            f"Candidate with similarity_score={c['similarity_score']} "
            f"should appear in filtered results"
        )


# ---------------------------------------------------------------------------
# Unit tests — Task 9.4 (Requirements 4.3, 4.5)
# ---------------------------------------------------------------------------

from agents.substitution import build_substitute_response


class TestSubstitutionUnit:
    """Unit tests for substitution agent scoring, filtering, and response assembly."""

    def test_no_candidates_returns_empty_list_and_reason(self):
        """No candidates → empty list + no_candidates_reason field present.

        **Validates: Requirements 4.5**
        """
        response = build_substitute_response(
            ingredient_id=42, candidates=[], search_results=[]
        )
        assert response["candidates"] == []
        assert response["no_candidates_reason"] is not None
        assert isinstance(response["no_candidates_reason"], str)
        assert len(response["no_candidates_reason"]) > 0


