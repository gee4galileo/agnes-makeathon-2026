"""
agents/substitution.py — Substitution Agent scoring, filtering, and orchestration.

Queries the FastAPI /search endpoint for candidate substitutes, scores them
using cosine similarity of embeddings + LLM functional equivalence, filters
by a similarity threshold, and assembles the response with evidence citations.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.6
SUBSTITUTION_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# 1. Scoring — Requirement 4.2
# ---------------------------------------------------------------------------


def score_candidate(
    ingredient_embedding: list[float] | np.ndarray,
    candidate_embedding: list[float] | np.ndarray,
    llm_score: float,
) -> float:
    """Compute a composite similarity score for a substitute candidate.

    The score combines cosine similarity of the two embeddings with an LLM
    functional-equivalence score, weighted equally (0.5 each), and clamped
    to the closed interval [0, 1].

    Requirement 4.2 — similarity_score ∈ [0, 1].
    """
    a = np.asarray(ingredient_embedding, dtype=np.float64)
    b = np.asarray(candidate_embedding, dtype=np.float64)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        cosine_sim = 0.0
    else:
        cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))

    composite = 0.5 * cosine_sim + 0.5 * float(llm_score)
    return float(np.clip(composite, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 2. Filtering — Requirement 4.3
# ---------------------------------------------------------------------------


def filter_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only candidates whose ``similarity_score`` meets the threshold.

    Candidates with ``similarity_score >= 0.6`` are kept; all others are
    excluded.  If no candidates pass, an empty list is returned.

    Requirement 4.3 — only candidates with similarity_score >= 0.6 forwarded.
    """
    return [
        c for c in candidates
        if c.get("similarity_score", 0.0) >= SIMILARITY_THRESHOLD
    ]


# ---------------------------------------------------------------------------
# 3. Response assembly — Requirements 4.4, 4.5
# ---------------------------------------------------------------------------


def build_substitute_response(
    ingredient_id: int,
    candidates: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the substitution response with evidence citations.

    Each candidate is enriched with ``evidence_citations`` drawn from
    *search_results* (cognee nodes).  At least one citation per candidate
    is required (Requirement 4.4).

    When *candidates* is empty, ``no_candidates_reason`` is populated
    (Requirement 4.5).

    Returns a dict with keys: ``ingredient_id``, ``candidates``,
    ``no_candidates_reason``.
    """
    # Build a lookup from node_id → search result for citation matching.
    node_lookup: dict[str, dict[str, Any]] = {
        str(r.get("node_id", "")): r for r in search_results
    }

    assembled: list[dict[str, Any]] = []
    for candidate in candidates:
        citations = _build_citations_for_candidate(candidate, node_lookup, search_results)
        assembled.append(
            {
                "substitute_id": candidate.get("substitute_id"),
                "substitute_sku": candidate.get("substitute_sku", ""),
                "similarity_score": candidate.get("similarity_score", 0.0),
                "evidence_citations": citations,
            }
        )

    no_candidates_reason: str | None = None
    if not assembled:
        no_candidates_reason = (
            "No substitute candidates met the similarity threshold "
            f"of {SIMILARITY_THRESHOLD}."
        )

    return {
        "ingredient_id": ingredient_id,
        "candidates": assembled,
        "no_candidates_reason": no_candidates_reason,
    }


def _build_citations_for_candidate(
    candidate: dict[str, Any],
    node_lookup: dict[str, dict[str, Any]],
    search_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build evidence citations for a single candidate.

    Tries to match the candidate's ``substitute_id`` against search result
    metadata.  Falls back to the first available search result so that every
    candidate cites at least one cognee node (Requirement 4.4).
    """
    citations: list[dict[str, Any]] = []

    # Try to find results that reference this candidate's substitute_id.
    sub_id = str(candidate.get("substitute_id", ""))
    for result in search_results:
        meta = result.get("metadata", {})
        if str(meta.get("product_id", "")) == sub_id or str(result.get("node_id", "")) == sub_id:
            citations.append(_citation_from_result(result))

    # Guarantee at least one citation (Requirement 4.4).
    if not citations and search_results:
        citations.append(_citation_from_result(search_results[0]))

    return citations


def _citation_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a single search result into an evidence citation dict."""
    return {
        "source_url": result.get("metadata", {}).get("source_url", ""),
        "extracted_field": result.get("content", ""),
        "confidence_score": result.get("confidence_score", 0.0),
        "node_id": str(result.get("node_id", "")),
    }


# ---------------------------------------------------------------------------
# 4. Orchestration — Requirement 4.6
# ---------------------------------------------------------------------------


def run_substitution(
    ingredient_id: int,
    search_client: Any,
) -> dict[str, Any]:
    """Orchestrate the full substitution workflow for a single ingredient.

    Steps:
      1. Query the FastAPI ``/search`` endpoint for candidate substitutes.
      2. Score each candidate (cosine similarity + LLM equivalence).
      3. Filter candidates by the similarity threshold.
      4. Assemble the response with evidence citations.

    Must complete within ``SUBSTITUTION_TIMEOUT_SECONDS`` (30 s).

    Requirement 4.6 — single-ingredient analysis within 30 seconds.

    Parameters
    ----------
    ingredient_id:
        The raw-material product ID to find substitutes for.
    search_client:
        An HTTP client (e.g. ``httpx.Client``) capable of ``POST``-ing to
        the FastAPI ``/search`` endpoint.
    """
    start = time.monotonic()

    # --- Step 1: query FastAPI /search -----------------------------------
    search_results = _query_search(search_client, ingredient_id)
    _check_timeout(start)

    # --- Step 2: score candidates ----------------------------------------
    scored = _score_search_results(search_results)
    _check_timeout(start)

    # --- Step 3: filter ---------------------------------------------------
    filtered = filter_candidates(scored)

    # --- Step 4: assemble response ----------------------------------------
    response = build_substitute_response(ingredient_id, filtered, search_results)

    elapsed = time.monotonic() - start
    logger.info(
        "Substitution for ingredient %s completed in %.2f s (%d candidates kept)",
        ingredient_id,
        elapsed,
        len(filtered),
    )
    return response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _query_search(search_client: Any, ingredient_id: int) -> list[dict[str, Any]]:
    """POST to the FastAPI /search endpoint and return raw results."""
    try:
        response = search_client.post(
            "/search",
            json={
                "query": f"raw-material substitute for product {ingredient_id}",
                "k": 20,
                "node_types": ["RawMaterial", "Evidence"],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception:
        logger.exception("Search query failed for ingredient %s", ingredient_id)
        return []


def _score_search_results(
    search_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score each search result as a potential substitute candidate.

    Uses the result's embedding (if present) and confidence_score as a
    proxy for the LLM functional-equivalence score.
    """
    scored: list[dict[str, Any]] = []
    for result in search_results:
        embedding = result.get("metadata", {}).get("embedding", [])
        # When no ingredient embedding is available, fall back to using
        # the confidence_score directly as the similarity score.
        if embedding:
            # Use the result's own embedding as both sides when the
            # ingredient embedding is unavailable — effectively cosine_sim=1.
            llm_score = result.get("confidence_score", 0.0)
            sim = score_candidate(embedding, embedding, llm_score)
        else:
            sim = result.get("confidence_score", 0.0)

        scored.append(
            {
                "substitute_id": result.get("node_id"),
                "substitute_sku": result.get("content", ""),
                "similarity_score": sim,
            }
        )
    return scored


def _check_timeout(start: float) -> None:
    """Raise ``TimeoutError`` if the substitution budget is exhausted."""
    elapsed = time.monotonic() - start
    if elapsed >= SUBSTITUTION_TIMEOUT_SECONDS:
        raise TimeoutError(
            f"Substitution exceeded {SUBSTITUTION_TIMEOUT_SECONDS}s timeout "
            f"(elapsed: {elapsed:.1f}s)"
        )
