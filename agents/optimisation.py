"""
agents/optimisation.py — Optimisation Agent: PuLP solver and proposal ranking.

Consumes compliance-verified substitute candidates, runs a PuLP LP to
minimise supplier count, ranks consolidation actions by composite score,
and assembles the final SourcingProposal JSON.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTIMISATION_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# 1. LP solver — Requirement 6.1
# ---------------------------------------------------------------------------


def build_lp_problem(
    compliance_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build and solve a PuLP LP for supplier consolidation.

    Only candidates with verdict PASS or NEEDS_REVIEW are eligible.
    Delegates to the shared solver in ``dify_setup.pulp_solver_tool``.

    Returns the solver output dict with ``status`` and ``assignments``.
    Requirement 6.1.
    """
    from dify_setup.pulp_solver_tool import run_consolidation_solver

    candidates: list[dict[str, Any]] = []
    for cr in compliance_results:
        sub_id = cr.get("substitute_id")
        verdict = cr.get("verdict", "FAIL")
        citations = cr.get("evidence_citations", [])
        avg_conf = _avg_confidence(citations)
        supplier = _extract_supplier(cr)

        candidates.append({
            "ingredient_id": sub_id,
            "supplier": supplier,
            "compliance_verdict": verdict,
            "confidence_score": avg_conf,
        })

    return run_consolidation_solver(candidates)


# ---------------------------------------------------------------------------
# 2. Composite scoring — Requirement 6.2
# ---------------------------------------------------------------------------


def compute_composite_score(
    coverage_ratio: float,
    compliance_weight: float,
    avg_confidence: float,
) -> float:
    """Compute a composite ranking score for a consolidation action.

    Formula: 0.4 * coverage_ratio + 0.3 * compliance_weight + 0.3 * avg_confidence

    All inputs should be in [0, 1]. The result is clamped to [0, 1].
    Requirement 6.2.
    """
    raw = 0.4 * coverage_ratio + 0.3 * compliance_weight + 0.3 * avg_confidence
    return max(0.0, min(1.0, raw))


def rank_consolidation_actions(
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort consolidation actions by composite_score descending.

    Requirement 6.2 — actions ranked by composite score.
    """
    return sorted(
        actions,
        key=lambda a: a.get("composite_score", 0.0),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# 3. Orchestration — Requirements 6.4, 6.5, 6.6
# ---------------------------------------------------------------------------


def run_optimisation(
    compliance_results: list[dict[str, Any]],
    search_client: Any,
) -> dict[str, Any]:
    """Orchestrate the full optimisation workflow.

    Steps:
      1. Invoke PuLP solver via build_lp_problem.
      2. Handle infeasible result.
      3. Build consolidation actions with evidence citations.
      4. Compute composite scores and rank.
      5. Assemble SourcingProposal.

    Must complete within OPTIMISATION_TIMEOUT_SECONDS (60 s) for up to
    15 raw-material products.

    Requirements 6.4, 6.5, 6.6.
    """
    start = time.monotonic()

    # --- Step 1: solve ---------------------------------------------------
    solver_result = build_lp_problem(compliance_results)
    solver_status = solver_result.get("status", "not_solved")
    assignments = solver_result.get("assignments", [])
    _check_timeout(start)

    # --- Step 2: handle infeasible ---------------------------------------
    if solver_status != "optimal" or not assignments:
        return _build_empty_proposal(solver_status, compliance_results)

    # --- Step 3: build consolidation actions with citations --------------
    cr_lookup = _build_cr_lookup(compliance_results)
    total_ingredients = len({a["ingredient_id"] for a in assignments})

    actions: list[dict[str, Any]] = []
    all_citations: list[dict[str, Any]] = []

    for assignment in assignments:
        ing_id = assignment["ingredient_id"]
        supplier = assignment["supplier"]
        conf = assignment.get("confidence_score", 0.0)

        cr = cr_lookup.get(ing_id, {})
        verdict = cr.get("verdict", "NEEDS_REVIEW")
        citations = cr.get("evidence_citations", [])

        # Guarantee at least one citation (Req 6.3)
        if not citations:
            citations = _fetch_fallback_citation(search_client, ing_id)

        coverage_ratio = 1.0 / total_ingredients if total_ingredients > 0 else 0.0
        compliance_weight = _verdict_weight(verdict)
        avg_conf = _avg_confidence(citations) if citations else conf

        composite = compute_composite_score(coverage_ratio, compliance_weight, avg_conf)

        action = {
            "ingredient_id": int(ing_id) if isinstance(ing_id, (int, float)) else 0,
            "recommended_supplier": supplier,
            "substitute_ingredient": _to_int_or_none(cr.get("substitute_id")),
            "similarity_score": cr.get("similarity_score"),
            "compliance_verdict": verdict,
            "evidence_citations": citations,
            "composite_score": composite,
        }
        actions.append(action)
        all_citations.extend(citations)

    _check_timeout(start)

    # --- Step 4: rank by composite score ---------------------------------
    actions = rank_consolidation_actions(actions)

    # Strip internal composite_score before building the proposal
    for action in actions:
        action.pop("composite_score", None)

    # --- Step 5: assemble SourcingProposal -------------------------------
    compliance_verdicts = _build_compliance_verdicts(compliance_results)
    summary = _build_summary(actions, solver_status)

    proposal: dict[str, Any] = {
        "consolidation_actions": actions,
        "compliance_verdicts": compliance_verdicts,
        "evidence_citations": all_citations,
        "summary": summary,
        "solver_status": solver_status,
    }

    elapsed = time.monotonic() - start
    logger.info(
        "Optimisation completed in %.2f s — %d actions, status=%s",
        elapsed, len(actions), solver_status,
    )
    return proposal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_int_or_none(val: Any) -> int | None:
    """Convert a value to int if possible, otherwise return None."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _avg_confidence(citations: list[dict[str, Any]]) -> float:
    """Average confidence_score across citations; 0.0 if empty."""
    if not citations:
        return 0.0
    scores = [c.get("confidence_score", 0.0) for c in citations]
    return sum(scores) / len(scores)


def _to_int_or_none(value: Any) -> int | None:
    """Coerce a value to int, or return None if not numeric."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _extract_supplier(cr: dict[str, Any]) -> str:
    """Extract a supplier name from a compliance result's citations."""
    for citation in cr.get("evidence_citations", []):
        url = citation.get("source_url", "")
        if url:
            return url.split("/")[2] if len(url.split("/")) > 2 else url
    return f"supplier-for-{cr.get('substitute_id', 'unknown')}"


def _verdict_weight(verdict: str) -> float:
    """Map a compliance verdict to a numeric weight for scoring."""
    return {"PASS": 1.0, "NEEDS_REVIEW": 0.5, "FAIL": 0.0}.get(verdict, 0.0)


def _build_cr_lookup(
    compliance_results: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Build a lookup from substitute_id to compliance result."""
    return {cr["substitute_id"]: cr for cr in compliance_results if "substitute_id" in cr}


def _fetch_fallback_citation(
    search_client: Any,
    ingredient_id: int,
) -> list[dict[str, Any]]:
    """Fetch a fallback citation from cognee when none exist."""
    try:
        response = search_client.post(
            "/search",
            json={
                "query": f"supplier evidence for ingredient {ingredient_id}",
                "k": 1,
                "node_types": ["Evidence", "RawMaterial"],
            },
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            r = results[0]
            return [{
                "source_url": r.get("metadata", {}).get("source_url", ""),
                "extracted_field": r.get("content", ""),
                "confidence_score": r.get("confidence_score", 0.0),
                "node_id": str(r.get("node_id", "")),
            }]
    except Exception:
        logger.exception("Fallback citation fetch failed for ingredient %s", ingredient_id)
    return [{"source_url": "", "extracted_field": "no evidence available",
             "confidence_score": 0.0, "node_id": ""}]


def _build_compliance_verdicts(
    compliance_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the compliance_verdicts array for the SourcingProposal."""
    return [
        {
            "ingredient_id": int(cr.get("substitute_id", 0)) if isinstance(cr.get("substitute_id"), (int, float)) else 0,
            "verdict": cr.get("verdict", "NEEDS_REVIEW"),
            "fail_reason": cr.get("fail_reason"),
            "missing_evidence": cr.get("missing_evidence"),
        }
        for cr in compliance_results
    ]


def _build_summary(actions: list[dict[str, Any]], solver_status: str) -> str:
    """Build a natural-language summary for the SourcingProposal.

    Kept under 500 words per Requirement 8.4.
    """
    if not actions:
        return f"No consolidation actions could be produced. Solver status: {solver_status}."

    suppliers = {a["recommended_supplier"] for a in actions}
    n_actions = len(actions)
    n_suppliers = len(suppliers)
    pass_count = sum(1 for a in actions if a.get("compliance_verdict") == "PASS")
    review_count = sum(1 for a in actions if a.get("compliance_verdict") == "NEEDS_REVIEW")

    parts = [
        f"Agnes recommends {n_actions} sourcing actions across {n_suppliers} suppliers.",
    ]
    if pass_count:
        parts.append(f"{pass_count} actions have full compliance clearance.")
    if review_count:
        parts.append(f"{review_count} actions require additional review.")
    parts.append(f"Solver status: {solver_status}.")

    return " ".join(parts)


def _build_empty_proposal(
    solver_status: str,
    compliance_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a SourcingProposal when the solver returns infeasible/empty."""
    return {
        "consolidation_actions": [],
        "compliance_verdicts": _build_compliance_verdicts(compliance_results),
        "evidence_citations": [],
        "summary": f"No consolidation actions could be produced. Solver status: {solver_status}.",
        "solver_status": solver_status,
    }


def _check_timeout(start: float) -> None:
    """Raise TimeoutError if the optimisation budget is exhausted."""
    elapsed = time.monotonic() - start
    if elapsed >= OPTIMISATION_TIMEOUT_SECONDS:
        raise TimeoutError(
            f"Optimisation exceeded {OPTIMISATION_TIMEOUT_SECONDS}s timeout "
            f"(elapsed: {elapsed:.1f}s)"
        )
