"""
dify_setup/pulp_solver_tool.py — Dify custom tool implementing supplier consolidation via PuLP.

Solves a set-cover-style LP: assign each ingredient to exactly one supplier
while minimising the total number of active suppliers.

Requirements: 6.1 (PuLP solver), 10.1 (Dify orchestration)
"""
from __future__ import annotations

import logging
from typing import Any

from pulp import (  # type: ignore[import-untyped]
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatusOptimal,
    LpVariable,
    lpSum,
    value,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dify tool schema
# ---------------------------------------------------------------------------

TOOL_SCHEMA: dict[str, Any] = {
    "name": "pulp_solver",
    "description": (
        "Run the PuLP supplier-consolidation optimiser. Accepts a list of "
        "candidate assignments (ingredient → supplier) with compliance "
        "verdicts and returns the optimal assignment that minimises the "
        "total number of active suppliers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "description": (
                    "List of candidate dicts, each with keys: "
                    "ingredient_id (int), supplier (str), "
                    "compliance_verdict (PASS | NEEDS_REVIEW | FAIL), "
                    "confidence_score (float 0-1)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "ingredient_id": {"type": "integer"},
                        "supplier": {"type": "string"},
                        "compliance_verdict": {
                            "type": "string",
                            "enum": ["PASS", "NEEDS_REVIEW", "FAIL"],
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": [
                        "ingredient_id",
                        "supplier",
                        "compliance_verdict",
                        "confidence_score",
                    ],
                },
            },
        },
        "required": ["candidates"],
    },
}


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

def run_consolidation_solver(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Solve the supplier-consolidation LP.

    Parameters
    ----------
    candidates:
        Each dict must contain:
        - ``ingredient_id`` (int)
        - ``supplier`` (str)
        - ``compliance_verdict`` ("PASS" | "NEEDS_REVIEW" | "FAIL")
        - ``confidence_score`` (float in [0, 1])

        Candidates with verdict ``FAIL`` are excluded automatically.

    Returns
    -------
    dict with:
        ``status``      — "optimal", "infeasible", or the raw PuLP status string
        ``assignments`` — list of ``{ingredient_id, supplier, confidence_score}``
                          dicts representing the chosen assignments
    """
    # Filter out FAIL candidates — only PASS / NEEDS_REVIEW are eligible (Req 6.1)
    eligible = [
        c for c in candidates
        if c.get("compliance_verdict") in ("PASS", "NEEDS_REVIEW")
    ]

    if not eligible:
        return {"status": "infeasible", "assignments": []}

    # Collect ALL ingredients from the input (including those with only FAIL candidates)
    all_ingredients: set[int] = {c["ingredient_id"] for c in candidates}
    # Collect ingredients that have at least one eligible candidate
    eligible_ingredients: set[int] = {c["ingredient_id"] for c in eligible}

    # If any ingredient has zero eligible candidates, the problem is infeasible
    uncoverable = all_ingredients - eligible_ingredients
    if uncoverable:
        return {"status": "infeasible", "assignments": []}

    ingredients = eligible_ingredients
    suppliers: set[str] = {c["supplier"] for c in eligible}

    # Build lookup: (ingredient, supplier) → candidate
    lookup: dict[tuple[int, str], dict] = {}
    for c in eligible:
        key = (c["ingredient_id"], c["supplier"])
        # Keep the candidate with the highest confidence if duplicates exist
        if key not in lookup or c["confidence_score"] > lookup[key]["confidence_score"]:
            lookup[key] = c

    # --- LP formulation ---
    prob = LpProblem("supplier_consolidation", LpMinimize)

    # Binary decision variables: x[i][s] = 1 iff ingredient i is assigned to supplier s
    x: dict[tuple[int, str], Any] = {
        (i, s): LpVariable(f"x_{i}_{s}", cat=LpBinary)
        for (i, s) in lookup
    }

    # Binary indicator: y[s] = 1 iff supplier s is used for any ingredient
    y: dict[str, Any] = {
        s: LpVariable(f"y_{s}", cat=LpBinary)
        for s in suppliers
    }

    # Objective: minimise total number of active suppliers
    prob += lpSum(y[s] for s in suppliers), "minimise_supplier_count"

    # Constraint 1: each ingredient must be covered by exactly one supplier
    for i in ingredients:
        assigned_vars = [x[(i, s)] for s in suppliers if (i, s) in x]
        if not assigned_vars:
            # Ingredient has no eligible supplier — problem is infeasible
            return {"status": "infeasible", "assignments": []}
        prob += lpSum(assigned_vars) == 1, f"cover_{i}"

    # Constraint 2: if any ingredient is assigned to supplier s, y[s] must be 1
    for s in suppliers:
        for i in ingredients:
            if (i, s) in x:
                prob += x[(i, s)] <= y[s], f"link_{i}_{s}"

    # Solve
    prob.solve()

    if prob.status != LpStatusOptimal:
        # Map common PuLP statuses to a readable string
        status_map = {-1: "infeasible", 0: "not_solved", -2: "unbounded", -3: "undefined"}
        status_str = status_map.get(prob.status, f"pulp_status_{prob.status}")
        return {"status": status_str, "assignments": []}

    # Extract assignments
    assignments: list[dict[str, Any]] = []
    for (i, s), var in x.items():
        if value(var) is not None and value(var) > 0.5:
            assignments.append({
                "ingredient_id": i,
                "supplier": s,
                "confidence_score": lookup[(i, s)]["confidence_score"],
            })

    assignments.sort(key=lambda a: a["ingredient_id"])

    return {"status": "optimal", "assignments": assignments}
