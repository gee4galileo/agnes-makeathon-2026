"""
tests/test_cognee_pipeline.py — Unit tests for cognee_pipeline.py (Tasks 5.1, 5.2).
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, call

import pytest

from knowledge.pipeline import (
    embed_text_fields,
    ingest_boms,
    ingest_companies,
    ingest_evidence,
    ingest_products,
    ingest_suppliers,
    run_pipeline,
    upsert_node,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cognee():
    """Fake cognee client that tracks calls."""
    client = MagicMock()
    # By default get_node raises (node not found) so upsert creates.
    client.get_node.side_effect = Exception("not found")
    return client


@pytest.fixture
def vertex():
    """Fake Vertex AI client that returns a dummy embedding."""
    client = MagicMock()
    client.get_embeddings.return_value = [0.1, 0.2, 0.3]
    return client


# ---------------------------------------------------------------------------
# upsert_node tests (Requirement 3.7)
# ---------------------------------------------------------------------------

class TestUpsertNode:
    def test_creates_node_when_not_found(self, cognee):
        upsert_node(cognee, "Company", 1, {"name": "Acme"})

        cognee.get_node.assert_called_once_with("Company:1")
        cognee.create_node.assert_called_once_with("Company:1", "Company", {"name": "Acme"})
        cognee.update_node.assert_not_called()

    def test_updates_node_when_exists(self, cognee):
        cognee.get_node.side_effect = None
        cognee.get_node.return_value = {"name": "Old"}

        upsert_node(cognee, "Company", 1, {"name": "New"})

        cognee.update_node.assert_called_once_with("Company:1", {"name": "New"})
        cognee.create_node.assert_not_called()

    def test_node_id_format(self, cognee):
        upsert_node(cognee, "RawMaterial", 42, {"sku": "RM-042"})
        cognee.create_node.assert_called_once()
        assert cognee.create_node.call_args[0][0] == "RawMaterial:42"


# ---------------------------------------------------------------------------
# ingest_companies tests
# ---------------------------------------------------------------------------

class TestIngestCompanies:
    def test_ingests_all_companies(self, cognee):
        rows = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
        ingest_companies(cognee, rows)
        assert cognee.create_node.call_count == 2


# ---------------------------------------------------------------------------
# ingest_products tests (Requirements 3.1, 3.2)
# ---------------------------------------------------------------------------

class TestIngestProducts:
    def test_finished_good_creates_node_and_owns_edge(self, cognee):
        rows = [{"id": 10, "sku": "FG-1", "company_id": 1, "type": "finished-good", "canonical_category": "snacks"}]
        ingest_products(cognee, rows)

        cognee.create_node.assert_called_once()
        node_id = cognee.create_node.call_args[0][0]
        assert node_id == "FinishedGood:10"

        cognee.add_edge.assert_called_once_with("Company:1", "FinishedGood:10", "OWNS")

    def test_raw_material_creates_node_no_owns_edge(self, cognee):
        rows = [{"id": 20, "sku": "RM-1", "type": "raw-material", "canonical_category": "oils"}]
        ingest_products(cognee, rows)

        cognee.create_node.assert_called_once()
        node_id = cognee.create_node.call_args[0][0]
        assert node_id == "RawMaterial:20"
        cognee.add_edge.assert_not_called()

    def test_mixed_products(self, cognee):
        rows = [
            {"id": 10, "sku": "FG-1", "company_id": 1, "type": "finished-good", "canonical_category": "snacks"},
            {"id": 20, "sku": "RM-1", "type": "raw-material", "canonical_category": "oils"},
        ]
        ingest_products(cognee, rows)
        assert cognee.create_node.call_count == 2
        assert cognee.add_edge.call_count == 1  # only OWNS for finished-good


# ---------------------------------------------------------------------------
# ingest_boms tests (Requirement 3.3)
# ---------------------------------------------------------------------------

class TestIngestBoms:
    def test_creates_bom_nodes_and_edges(self, cognee):
        bom_rows = [{"id": 100, "produced_product_id": 10}]
        comp_rows = [
            {"bom_id": 100, "consumed_product_id": 20},
            {"bom_id": 100, "consumed_product_id": 21},
        ]
        products = {10: {"type": "finished-good"}, 20: {"type": "raw-material"}, 21: {"type": "raw-material"}}

        ingest_boms(cognee, bom_rows, comp_rows, products)

        # 1 BOM node
        cognee.create_node.assert_called_once()
        # 1 HAS_BOM + 2 CONTAINS = 3 edges
        assert cognee.add_edge.call_count == 3
        edge_calls = [c[0] for c in cognee.add_edge.call_args_list]
        assert ("FinishedGood:10", "BOM:100", "HAS_BOM") in edge_calls
        assert ("BOM:100", "RawMaterial:20", "CONTAINS") in edge_calls
        assert ("BOM:100", "RawMaterial:21", "CONTAINS") in edge_calls


# ---------------------------------------------------------------------------
# ingest_suppliers tests (Requirement 3.4)
# ---------------------------------------------------------------------------

class TestIngestSuppliers:
    def test_creates_supplier_nodes_and_supplies_edges(self, cognee):
        supplier_rows = [{"id": 200, "name": "Supplier A"}]
        sp_rows = [{"supplier_id": 200, "product_id": 20}]
        products = {20: {"type": "raw-material"}}

        ingest_suppliers(cognee, supplier_rows, sp_rows, products)

        cognee.create_node.assert_called_once()
        cognee.add_edge.assert_called_once_with("Supplier:200", "RawMaterial:20", "SUPPLIES")

    def test_skips_supplies_edge_for_finished_good(self, cognee):
        supplier_rows = [{"id": 200, "name": "Supplier A"}]
        sp_rows = [{"supplier_id": 200, "product_id": 10}]
        products = {10: {"type": "finished-good"}}

        ingest_suppliers(cognee, supplier_rows, sp_rows, products)

        cognee.create_node.assert_called_once()
        cognee.add_edge.assert_not_called()

    def test_skips_supplies_edge_for_unknown_product(self, cognee):
        supplier_rows = [{"id": 200, "name": "Supplier A"}]
        sp_rows = [{"supplier_id": 200, "product_id": 999}]
        products = {}

        ingest_suppliers(cognee, supplier_rows, sp_rows, products)

        cognee.add_edge.assert_not_called()


# ---------------------------------------------------------------------------
# ingest_evidence tests (Requirement 3.5)
# ---------------------------------------------------------------------------

class TestIngestEvidence:
    def test_creates_evidence_nodes_and_evidences_edges(self, cognee):
        rows = [
            {
                "id": "ev-1",
                "supplier_url": "https://example.com",
                "ingredient_name": "Palm Oil",
                "certifications": ["RSPO"],
                "confidence_score": 0.9,
                "product_id": 20,
            }
        ]
        ingest_evidence(cognee, rows)

        cognee.create_node.assert_called_once()
        cognee.add_edge.assert_called_once_with("Evidence:ev-1", "RawMaterial:20", "EVIDENCES")

    def test_missing_product_id_uses_unknown(self, cognee):
        rows = [
            {
                "id": "ev-2",
                "supplier_url": "https://example.com",
                "ingredient_name": "Sugar",
                "certifications": [],
                "confidence_score": 0.5,
            }
        ]
        ingest_evidence(cognee, rows)

        cognee.add_edge.assert_called_once_with("Evidence:ev-2", "RawMaterial:unknown", "EVIDENCES")


# ---------------------------------------------------------------------------
# embed_text_fields tests (Requirement 3.6)
# ---------------------------------------------------------------------------

class TestEmbedTextFields:
    def test_embeds_text_fields_and_stores(self, cognee, vertex):
        node = {"id": "Company:1", "_node_id": "Company:1", "name": "Acme Corp"}
        result = embed_text_fields(cognee, vertex, node)

        vertex.get_embeddings.assert_called_once_with("Acme Corp")
        cognee.store_embedding.assert_called_once_with("Company:1", [0.1, 0.2, 0.3])
        assert result["_embedding"] == [0.1, 0.2, 0.3]

    def test_concatenates_multiple_fields(self, cognee, vertex):
        node = {"id": "RM:1", "_node_id": "RM:1", "name": "Palm Oil", "sku": "PO-001", "canonical_category": "oils"}
        embed_text_fields(cognee, vertex, node)

        call_text = vertex.get_embeddings.call_args[0][0]
        assert "Palm Oil" in call_text
        assert "PO-001" in call_text
        assert "oils" in call_text

    def test_handles_list_certifications(self, cognee, vertex):
        node = {"id": "E:1", "_node_id": "E:1", "certifications": ["RSPO", "ISO"]}
        embed_text_fields(cognee, vertex, node)

        call_text = vertex.get_embeddings.call_args[0][0]
        assert "RSPO" in call_text
        assert "ISO" in call_text

    def test_skips_empty_node(self, cognee, vertex):
        node = {"id": "X:1"}
        result = embed_text_fields(cognee, vertex, node)

        vertex.get_embeddings.assert_not_called()
        assert "_embedding" not in result

    def test_returns_node_unchanged_on_error(self, cognee, vertex):
        vertex.get_embeddings.side_effect = RuntimeError("API down")
        node = {"id": "C:1", "_node_id": "C:1", "name": "Acme"}
        result = embed_text_fields(cognee, vertex, node)

        assert "_embedding" not in result
        assert result["name"] == "Acme"


# ---------------------------------------------------------------------------
# run_pipeline integration test (Requirements 3.1 – 3.7)
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_calls_ingestion_in_order(self, cognee, vertex):
        """Verify run_pipeline reads BQ, ingests, and embeds."""
        bq = MagicMock()

        # Simulate BQ query results — each .result() returns an iterable of dicts
        company_data = [{"id": 1, "name": "Acme"}]
        product_data = [
            {"id": 10, "sku": "FG-1", "company_id": 1, "type": "finished-good", "canonical_category": "snacks"},
            {"id": 20, "sku": "RM-1", "type": "raw-material", "canonical_category": "oils"},
        ]
        bom_data = [{"id": 100, "produced_product_id": 10}]
        comp_data = [{"bom_id": 100, "consumed_product_id": 20}]
        supplier_data = [{"id": 200, "name": "Supplier A"}]
        sp_data = [{"supplier_id": 200, "product_id": 20}]
        evidence_data = [
            {
                "id": "ev-1",
                "supplier_url": "https://example.com",
                "ingredient_name": "Oil",
                "certifications": ["RSPO"],
                "confidence_score": 0.9,
                "product_id": 20,
            }
        ]

        query_results = iter([
            company_data,
            product_data,
            bom_data,
            comp_data,
            supplier_data,
            sp_data,
            evidence_data,
        ])

        def fake_query(sql):
            mock_result = MagicMock()
            mock_result.result.return_value = next(query_results)
            return mock_result

        bq.query.side_effect = fake_query

        run_pipeline(bq, cognee, vertex)

        # 7 queries issued
        assert bq.query.call_count == 7

        # Nodes created: 1 company + 1 FG + 1 RM + 1 BOM + 1 supplier + 1 evidence = 6
        assert cognee.create_node.call_count == 6

        # Edges: 1 OWNS + 1 HAS_BOM + 1 CONTAINS + 1 SUPPLIES + 1 EVIDENCES = 5
        assert cognee.add_edge.call_count == 5

        # Embeddings: company(1) + products(2) + suppliers(1) + evidence(1) = 5
        assert vertex.get_embeddings.call_count == 5
