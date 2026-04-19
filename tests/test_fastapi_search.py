"""
tests/test_fastapi_search.py — Property 10: Search returns at most K results with confidence scores

Feature: agnes-ai-supply-chain-manager, Property 10: Search returns at most K results with confidence scores

Validates: Requirements 3.8, 3.10

For any natural language query and any value of K (1-50), the FastAPI `/search`
endpoint SHALL return at most K results, and every result item SHALL contain a
`confidence_score` field in [0, 1].
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from api.main import app


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

query_strategy = st.text(min_size=1, max_size=100)
k_strategy = st.integers(min_value=1, max_value=50)
confidence_strategy = st.floats(min_value=0, max_value=1, allow_nan=False)


@st.composite
def search_scenario(draw):
    """Generate a (query, k, mock_results) triple.

    The mock cognee client returns cognee v1.0 format: list of dicts with
    search_result lists. We generate between 0 and K+5 chunk items.
    """
    query = draw(query_strategy)
    k = draw(k_strategy)
    n_results = draw(st.integers(min_value=0, max_value=k + 5))

    chunks = [
        {"id": f"chunk-{i}", "text": f"content-{i}", "type": "IndexSchema"}
        for i in range(n_results)
    ]
    # cognee v1.0 wraps results in a dataset envelope
    results = [
        {
            "dataset_id": "test-id",
            "dataset_name": "agnes_bom",
            "dataset_tenant_id": "test-tenant",
            "search_result": chunks,
        }
    ] if chunks else []
    return query, k, results


# ---------------------------------------------------------------------------
# Property 10: Search returns at most K results with confidence scores
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(data=search_scenario())
def test_search_k_results_and_confidence_scores_property10(data):
    """
    Feature: agnes-ai-supply-chain-manager, Property 10: Search returns at most K results with confidence scores

    **Validates: Requirements 3.8, 3.10**

    For any natural language query and any value of K (1-50):
    1. The `/search` endpoint returns at most K results.
    2. Every result item contains a `confidence_score` in [0, 1].
    """
    query, k, mock_results = data

    async def _run():
        mock_cognee = MagicMock()

        async def mock_search(**kwargs):
            return mock_results

        mock_cognee.search = mock_search
        app.state.cognee_client = mock_cognee

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/search",
                json={"query": query, "k": k},
            )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        body = response.json()
        results = body["results"]

        # --- Assertion 1: at most K results ---
        assert len(results) <= k, (
            f"Expected at most {k} results, got {len(results)}"
        )

        # --- Assertion 2: every confidence_score in [0, 1] ---
        for item in results:
            score = item["confidence_score"]
            assert isinstance(score, (int, float)), (
                f"confidence_score is not numeric: {score!r}"
            )
            assert 0 <= score <= 1, (
                f"confidence_score {score} is outside [0, 1]"
            )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Unit tests for FastAPI endpoints (Task 6.3)
# Validates: Requirements 3.8, 3.9, 3.10
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cognee_client():
    """Return a mock cognee client whose search returns cognee v1.0 format."""
    mock = MagicMock()
    # cognee v1.0 returns list of dicts with search_result lists
    mock.search = MagicMock(
        return_value=[
            {
                "dataset_id": "test-dataset-id",
                "dataset_name": "agnes_bom",
                "dataset_tenant_id": "test-tenant",
                "search_result": [
                    {"id": f"chunk-{i}", "text": f"Raw Material: RM-TEST-{i}", "type": "IndexSchema"}
                    for i in range(5)
                ],
            }
        ]
    )
    return mock


@pytest.mark.asyncio
async def test_health_returns_200():
    """GET /health returns HTTP 200 with {"status": "ok"}."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_search_valid_query_returns_results_le_k(mock_cognee_client):
    """POST /search with a valid query returns results where len(results) <= k."""
    app.state.cognee_client = mock_cognee_client

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/search",
            json={"query": "palm oil substitutes", "k": 10},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) <= 10
    # The mock returns 5 results, so we should get exactly 5
    assert len(body["results"]) == 5


@pytest.mark.asyncio
async def test_search_invalid_body_returns_422():
    """POST /search with an invalid body (missing query field) returns HTTP 422."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/search",
            json={"k": 10},  # missing required 'query' field
        )

    assert response.status_code == 422
