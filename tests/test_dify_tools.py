"""
tests/test_dify_tools.py — Tests for Dify custom tools.

Covers:
- PuLP solver correctness (optimal, infeasible, edge cases)
- Property tests for solver invariants

Requirements: 6.1, 10.1
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dify_setup.pulp_solver_tool import run_consolidation_solver, TOOL_SCHEMA as SOLVER_SCHEMA


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

supplier_names = st.sampled_from(["SupplierA", "SupplierB", "SupplierC", "SupplierD", "SupplierE"])
verdicts = st.sampled_from(["PASS", "NEEDS_REVIEW", "FAIL"])
confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

candidate_st = st.fixed_dictionaries({
    "ingredient_id": st.integers(min_value=1, max_value=50),
    "supplier": supplier_names,
    "compliance_verdict": verdicts,
    "confidence_score": confidence,
})

candidates_list_st = st.lists(candidate_st, min_size=0, max_size=30)


# ---------------------------------------------------------------------------
# PuLP Solver — Unit Tests
# ---------------------------------------------------------------------------

class TestPuLPSolverUnit:
    """Unit tests for the PuLP consolidation solver."""

    def test_empty_candidates_returns_infeasible(self):
        result = run_consolidation_solver([])
        assert result["status"] == "infeasible"
        assert result["assignments"] == []

    def test_all_fail_returns_infeasible(self):
        candidates = [
            {"ingredient_id": 1, "supplier": "A", "compliance_verdict": "FAIL", "confidence_score": 0.9},
            {"ingredient_id": 2, "supplier": "B", "compliance_verdict": "FAIL", "confidence_score": 0.8},
        ]
        result = run_consolidation_solver(candidates)
        assert result["status"] == "infeasible"
        assert result["assignments"] == []

    def test_single_ingredient_single_supplier_optimal(self):
        candidates = [
            {"ingredient_id": 1, "supplier": "A", "compliance_verdict": "PASS", "confidence_score": 0.9},
        ]
        result = run_consolidation_solver(candidates)
        assert result["status"] == "optimal"
        assert len(result["assignments"]) == 1
        assert result["assignments"][0]["ingredient_id"] == 1
        assert result["assignments"][0]["supplier"] == "A"

    def test_minimises_supplier_count(self):
        """When one supplier can cover all ingredients, solver should pick them."""
        candidates = [
            {"ingredient_id": 1, "supplier": "A", "compliance_verdict": "PASS", "confidence_score": 0.9},
            {"ingredient_id": 2, "supplier": "A", "compliance_verdict": "PASS", "confidence_score": 0.8},
            {"ingredient_id": 1, "supplier": "B", "compliance_verdict": "PASS", "confidence_score": 0.7},
            {"ingredient_id": 2, "supplier": "B", "compliance_verdict": "PASS", "confidence_score": 0.6},
        ]
        result = run_consolidation_solver(candidates)
        assert result["status"] == "optimal"
        assert len(result["assignments"]) == 2
        # All assignments should use the same supplier (minimise count)
        suppliers_used = {a["supplier"] for a in result["assignments"]}
        assert len(suppliers_used) == 1

    def test_needs_review_candidates_are_eligible(self):
        candidates = [
            {"ingredient_id": 1, "supplier": "A", "compliance_verdict": "NEEDS_REVIEW", "confidence_score": 0.5},
        ]
        result = run_consolidation_solver(candidates)
        assert result["status"] == "optimal"
        assert len(result["assignments"]) == 1

    def test_fail_candidates_excluded_from_solver(self):
        """Ingredient 2 only has FAIL candidates → infeasible."""
        candidates = [
            {"ingredient_id": 1, "supplier": "A", "compliance_verdict": "PASS", "confidence_score": 0.9},
            {"ingredient_id": 2, "supplier": "A", "compliance_verdict": "FAIL", "confidence_score": 0.8},
            {"ingredient_id": 2, "supplier": "B", "compliance_verdict": "FAIL", "confidence_score": 0.7},
        ]
        result = run_consolidation_solver(candidates)
        # Ingredient 2 has no eligible supplier → infeasible
        assert result["status"] == "infeasible"

    def test_assignments_sorted_by_ingredient_id(self):
        candidates = [
            {"ingredient_id": 3, "supplier": "A", "compliance_verdict": "PASS", "confidence_score": 0.9},
            {"ingredient_id": 1, "supplier": "A", "compliance_verdict": "PASS", "confidence_score": 0.8},
            {"ingredient_id": 2, "supplier": "A", "compliance_verdict": "PASS", "confidence_score": 0.7},
        ]
        result = run_consolidation_solver(candidates)
        assert result["status"] == "optimal"
        ids = [a["ingredient_id"] for a in result["assignments"]]
        assert ids == sorted(ids)

    def test_duplicate_candidates_keeps_highest_confidence(self):
        candidates = [
            {"ingredient_id": 1, "supplier": "A", "compliance_verdict": "PASS", "confidence_score": 0.3},
            {"ingredient_id": 1, "supplier": "A", "compliance_verdict": "PASS", "confidence_score": 0.9},
        ]
        result = run_consolidation_solver(candidates)
        assert result["status"] == "optimal"
        assert result["assignments"][0]["confidence_score"] == 0.9


# ---------------------------------------------------------------------------
# PuLP Solver — Property Tests
# ---------------------------------------------------------------------------

class TestPuLPSolverProperties:
    """Property-based tests for the PuLP solver."""

    @settings(max_examples=100)
    @given(candidates=candidates_list_st)
    def test_status_is_always_valid_string(self, candidates):
        """Solver always returns a recognisable status string."""
        result = run_consolidation_solver(candidates)
        assert isinstance(result["status"], str)
        assert result["status"] in ("optimal", "infeasible", "not_solved", "unbounded", "undefined") or result["status"].startswith("pulp_status_")

    @settings(max_examples=100)
    @given(candidates=candidates_list_st)
    def test_optimal_covers_all_eligible_ingredients(self, candidates):
        """If status is optimal, every ingredient with ≥1 eligible supplier is assigned."""
        result = run_consolidation_solver(candidates)
        if result["status"] != "optimal":
            return
        eligible_ingredients = {
            c["ingredient_id"] for c in candidates
            if c["compliance_verdict"] in ("PASS", "NEEDS_REVIEW")
        }
        assigned_ingredients = {a["ingredient_id"] for a in result["assignments"]}
        assert assigned_ingredients == eligible_ingredients

    @settings(max_examples=100)
    @given(candidates=candidates_list_st)
    def test_each_ingredient_assigned_exactly_once(self, candidates):
        """No ingredient appears twice in the assignments."""
        result = run_consolidation_solver(candidates)
        if result["status"] != "optimal":
            return
        ids = [a["ingredient_id"] for a in result["assignments"]]
        assert len(ids) == len(set(ids))

    @settings(max_examples=100)
    @given(candidates=candidates_list_st)
    def test_no_fail_candidates_in_assignments(self, candidates):
        """FAIL candidates must never appear in assignments."""
        result = run_consolidation_solver(candidates)
        fail_pairs = {
            (c["ingredient_id"], c["supplier"])
            for c in candidates
            if c["compliance_verdict"] == "FAIL"
        }
        # Check that no assignment matches a FAIL-only pair
        # (a pair could also have a PASS entry, so we check the eligible lookup)
        for a in result["assignments"]:
            pair = (a["ingredient_id"], a["supplier"])
            # The pair must have had at least one eligible candidate
            eligible_for_pair = [
                c for c in candidates
                if c["ingredient_id"] == pair[0]
                and c["supplier"] == pair[1]
                and c["compliance_verdict"] in ("PASS", "NEEDS_REVIEW")
            ]
            assert len(eligible_for_pair) > 0


# ---------------------------------------------------------------------------
# Tool Schema Tests
# ---------------------------------------------------------------------------

class TestToolSchemas:
    """Verify Dify tool schemas are well-formed."""

    def test_solver_schema_has_required_fields(self):
        assert SOLVER_SCHEMA["name"] == "pulp_solver"
        assert "description" in SOLVER_SCHEMA
        assert "parameters" in SOLVER_SCHEMA
        assert "candidates" in SOLVER_SCHEMA["parameters"]["properties"]
        assert SOLVER_SCHEMA["parameters"]["required"] == ["candidates"]
