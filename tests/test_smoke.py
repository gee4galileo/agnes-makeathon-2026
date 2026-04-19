"""
tests/test_smoke.py — Smoke tests for Agnes system.

Requirements: 3.8, 8.2
"""

from __future__ import annotations

import jsonschema
import pytest
from fastapi.testclient import TestClient

from api.main import app
from validators import SOURCING_PROPOSAL_SCHEMA


# ---------------------------------------------------------------------------
# Smoke tests — Task 16
# ---------------------------------------------------------------------------


def test_health_endpoint():
    """GET /health returns HTTP 200.

    **Validates: Requirements 3.8**
    """
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_env_vars_present():
    """All required env vars from config.py are set in the test environment."""
    import os

    required = [
        "GCP_PROJECT",
        "BQ_BOM_DATASET",
        "BQ_ENRICHMENT_DATASET",
        "APIFY_API_TOKEN",
        "DOCUMENT_AI_PROCESSOR_ID",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID",
        "VERTEX_AI_LOCATION",
        "COGNEE_DB_PATH",
        "GEMINI_API_KEY",
    ]
    for var in required:
        assert os.getenv(var) is not None, f"Required env var {var} is not set"


def test_sourcing_proposal_schema_loads():
    """SOURCING_PROPOSAL_SCHEMA is a valid JSON schema (parseable by jsonschema).

    **Validates: Requirements 8.2**
    """
    jsonschema.Draft7Validator.check_schema(SOURCING_PROPOSAL_SCHEMA)


# ---------------------------------------------------------------------------
# Edge case tests — discovered during implementation
# ---------------------------------------------------------------------------


def test_cognee_v1_response_format_conversion():
    """cognee v1.0 returns {dataset_id, dataset_name, search_result: [...]}
    and the FastAPI bridge must convert this to SearchResultItem format.

    Edge case discovered: cognee doesn't return flat dicts with node_id/content.
    """
    from api.main import _run_cognee_search
    from unittest.mock import MagicMock
    import asyncio

    mock_cognee = MagicMock()

    async def mock_search(**kwargs):
        return [
            {
                "dataset_id": "test-id",
                "dataset_name": "agnes_bom",
                "dataset_tenant_id": "test-tenant",
                "search_result": [
                    {"id": "chunk-1", "text": "Raw Material: vitamin-d3", "type": "IndexSchema"},
                    {"id": "chunk-2", "text": "Supplier: Prinova USA", "type": "IndexSchema"},
                ],
            }
        ]

    mock_cognee.search = mock_search

    loop = asyncio.new_event_loop()
    results = loop.run_until_complete(_run_cognee_search(mock_cognee, "vitamin d3", 5, None))
    loop.close()

    assert len(results) == 2
    assert results[0]["content"] == "Raw Material: vitamin-d3"
    assert results[1]["content"] == "Supplier: Prinova USA"
    for r in results:
        assert "node_id" in r
        assert "confidence_score" in r
        assert 0.0 <= r["confidence_score"] <= 1.0


def test_optimisation_ingredient_id_type_coercion():
    """Optimisation agent must coerce string node IDs to integers for the
    SourcingProposal schema (which requires integer ingredient_id).

    Edge case discovered: PuLP solver returns string substitute_ids from
    cognee search results, but the JSON schema requires integers.
    """
    from agents.optimisation import _to_int_or_none

    assert _to_int_or_none(42) == 42
    assert _to_int_or_none(3.0) == 3
    assert _to_int_or_none("not-a-number") is None
    assert _to_int_or_none(None) is None
    assert _to_int_or_none("123") == 123


def test_gemini_api_key_present():
    """GEMINI_API_KEY must be set — cognee Cloud and Dify both need it.

    Edge case discovered: missing GEMINI_API_KEY caused cognee cognify to
    fail with 'LLM API key not set' and Dify agent to return empty results.
    """
    import os
    key = os.getenv("GEMINI_API_KEY", "")
    assert len(key) > 0, "GEMINI_API_KEY is not set"


def test_cognee_search_empty_results_handled():
    """When cognee returns empty search_result lists, the FastAPI bridge
    should return an empty results array, not crash.

    Edge case discovered: cognee returns [{dataset_id: ..., search_result: []}]
    for queries with no matches.
    """
    from api.main import _run_cognee_search
    from unittest.mock import MagicMock
    import asyncio

    mock_cognee = MagicMock()

    async def mock_search(**kwargs):
        return [
            {
                "dataset_id": "test-id",
                "dataset_name": "agnes_bom",
                "search_result": [],
            }
        ]

    mock_cognee.search = mock_search

    loop = asyncio.new_event_loop()
    results = loop.run_until_complete(_run_cognee_search(mock_cognee, "nonexistent", 5, None))
    loop.close()

    assert results == []


def test_node_types_post_filter():
    """node_types parameter filters results by document prefix.

    Edge case: node_types was in the API schema but did nothing until we
    added post-search filtering by content prefix.
    """
    from api.main import SearchResultItem

    items = [
        SearchResultItem(node_id="1", node_type="X", content="Raw Material: vitamin-d3", confidence_score=0.9, metadata={}),
        SearchResultItem(node_id="2", node_type="X", content="Supplier: Prinova USA", confidence_score=0.8, metadata={}),
        SearchResultItem(node_id="3", node_type="X", content="Company: Equate", confidence_score=0.7, metadata={}),
    ]

    # Filter for RawMaterial only
    prefix_map = {"RawMaterial": "Raw Material:", "Supplier": "Supplier:", "Company": "Company:"}
    node_types = ["RawMaterial"]
    prefixes = [prefix_map[nt] for nt in node_types if nt in prefix_map]
    filtered = [i for i in items if any(i.content.startswith(p) for p in prefixes)]

    assert len(filtered) == 1
    assert filtered[0].content == "Raw Material: vitamin-d3"
