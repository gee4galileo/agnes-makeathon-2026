"""
conftest.py — Shared pytest fixtures for the Agnes test suite.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def set_required_env_vars(monkeypatch):
    """
    Ensure all required environment variables are set for every test.
    Tests that exercise real external services should override individual vars.
    """
    defaults = {
        "GCP_PROJECT": "test-project",
        "BQ_BOM_DATASET": "agnes_bom",
        "BQ_ENRICHMENT_DATASET": "agnes_enrichment",
        "APIFY_API_TOKEN": "test-apify-token",
        "DOCUMENT_AI_PROCESSOR_ID": "test-processor-id",
        "ELEVENLABS_API_KEY": "test-elevenlabs-key",
        "ELEVENLABS_VOICE_ID": "test-voice-id",
        "VERTEX_AI_LOCATION": "us-central1",
        "COGNEE_DB_PATH": "/tmp/cognee_test_db",
        "COGNEE_API_KEY": "test-cognee-api-key",
        "GEMINI_API_KEY": "test-gemini-api-key",
    }
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def sample_bom_rows():
    """Minimal valid BOM dataset for unit tests."""
    return {
        "company": [{"id": 1, "name": "Acme Corp"}],
        "product": [
            {"id": 10, "sku": "FG-001", "company_id": 1, "type": "finished-good", "canonical_category": "snacks"},
            {"id": 20, "sku": "RM-001", "company_id": None, "type": "raw-material", "canonical_category": "oils"},
            {"id": 21, "sku": "RM-002", "company_id": None, "type": "raw-material", "canonical_category": "sugars"},
        ],
        "bom": [{"id": 100, "produced_product_id": 10}],
        "bom_component": [
            {"bom_id": 100, "consumed_product_id": 20},
            {"bom_id": 100, "consumed_product_id": 21},
        ],
        "supplier": [{"id": 200, "name": "Supplier A"}],
        "supplier_product": [{"supplier_id": 200, "product_id": 20}],
    }


@pytest.fixture
def sample_evidence_record():
    """Minimal valid evidence record for unit tests."""
    return {
        "id": "00000000-0000-0000-0000-000000000001",
        "supplier_url": "https://example-supplier.com/palm-oil",
        "supplier_name": "Example Supplier",
        "ingredient_name": "Palm Oil",
        "certifications": ["RSPO", "ISO 9001"],
        "price_indicators": "$1.20/kg",
        "confidence_score": 0.85,
        "field_confidences": {
            "supplier_name": 0.92,
            "ingredient_name": 0.88,
            "certifications": 0.75,
            "price_indicators": 0.65,
        },
        "low_confidence_fields": [],
        "scraped_at": "2024-01-01T00:00:00Z",
    }
