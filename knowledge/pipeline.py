"""
cognee_pipeline.py — Ingest BOM data and enrichment evidence into the cognee knowledge graph.

Reads from BigQuery datasets ``agnes_bom`` and ``agnes_enrichment``, creates
typed nodes and edges in cognee (LanceDB vectors + Kuzu graph), embeds text
fields with Vertex AI text-embedding-004, and deduplicates on re-ingestion.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEXT_FIELDS = ("name", "sku", "canonical_category", "certifications", "ingredient_name")


def _row_to_dict(row: Any) -> dict:
    """Convert a BigQuery Row (or plain dict) to a regular dict."""
    if isinstance(row, dict):
        return row
    return dict(row)


# ---------------------------------------------------------------------------
# Upsert (Requirement 3.7)
# ---------------------------------------------------------------------------

def upsert_node(cognee_client: Any, node_type: str, source_id: Any, fields: dict) -> None:
    """Create or update a node in the cognee knowledge graph.

    Nodes are keyed by ``{node_type}:{source_id}``.  If a node with that id
    already exists it is updated; otherwise a new node is created.

    Requirement 3.7 — deduplication on re-ingestion.
    """
    node_id = f"{node_type}:{source_id}"
    try:
        existing = cognee_client.get_node(node_id)
    except Exception:
        existing = None

    if existing:
        cognee_client.update_node(node_id, fields)
        logger.debug("Updated existing node %s", node_id)
    else:
        cognee_client.create_node(node_id, node_type, fields)
        logger.debug("Created new node %s", node_id)


# ---------------------------------------------------------------------------
# Node & edge ingestion (Requirements 3.1 – 3.5)
# ---------------------------------------------------------------------------

def ingest_companies(cognee_client: Any, rows: list[dict]) -> None:
    """Ingest ``Company`` nodes from the ``company`` table."""
    for row in rows:
        upsert_node(cognee_client, "Company", row["id"], {"name": row["name"]})
    logger.info("Ingested %d Company node(s)", len(rows))


def ingest_products(cognee_client: Any, rows: list[dict]) -> None:
    """Ingest ``FinishedGood`` and ``RawMaterial`` nodes from the ``product`` table.

    Also creates ``OWNS`` edges from Company → FinishedGood.

    Requirements 3.1, 3.2
    """
    fg_count = 0
    rm_count = 0
    for row in rows:
        if row["type"] == "finished-good":
            upsert_node(
                cognee_client,
                "FinishedGood",
                row["id"],
                {
                    "sku": row.get("sku"),
                    "company_id": row.get("company_id"),
                    "canonical_category": row.get("canonical_category"),
                },
            )
            cognee_client.add_edge(
                f"Company:{row['company_id']}",
                f"FinishedGood:{row['id']}",
                "OWNS",
            )
            fg_count += 1
        elif row["type"] == "raw-material":
            upsert_node(
                cognee_client,
                "RawMaterial",
                row["id"],
                {
                    "sku": row.get("sku"),
                    "canonical_category": row.get("canonical_category"),
                },
            )
            rm_count += 1
    logger.info(
        "Ingested %d FinishedGood and %d RawMaterial node(s)", fg_count, rm_count
    )


def ingest_boms(
    cognee_client: Any,
    bom_rows: list[dict],
    component_rows: list[dict],
    products: dict[int, dict],
) -> None:
    """Ingest ``BOM`` nodes, ``HAS_BOM`` and ``CONTAINS`` edges.

    Requirement 3.3
    """
    for bom_row in bom_rows:
        upsert_node(
            cognee_client,
            "BOM",
            bom_row["id"],
            {"produced_product_id": bom_row["produced_product_id"]},
        )
        cognee_client.add_edge(
            f"FinishedGood:{bom_row['produced_product_id']}",
            f"BOM:{bom_row['id']}",
            "HAS_BOM",
        )
    logger.info("Ingested %d BOM node(s)", len(bom_rows))

    for comp in component_rows:
        cognee_client.add_edge(
            f"BOM:{comp['bom_id']}",
            f"RawMaterial:{comp['consumed_product_id']}",
            "CONTAINS",
        )
    logger.info("Ingested %d CONTAINS edge(s)", len(component_rows))


def ingest_suppliers(
    cognee_client: Any,
    supplier_rows: list[dict],
    sp_rows: list[dict],
    products: dict[int, dict],
) -> None:
    """Ingest ``Supplier`` nodes and ``SUPPLIES`` edges (raw-materials only).

    Requirement 3.4
    """
    for sup in supplier_rows:
        upsert_node(cognee_client, "Supplier", sup["id"], {"name": sup["name"]})
    logger.info("Ingested %d Supplier node(s)", len(supplier_rows))

    supplies_count = 0
    for sp in sp_rows:
        product = products.get(sp["product_id"])
        if product and product.get("type") == "raw-material":
            cognee_client.add_edge(
                f"Supplier:{sp['supplier_id']}",
                f"RawMaterial:{sp['product_id']}",
                "SUPPLIES",
            )
            supplies_count += 1
    logger.info("Ingested %d SUPPLIES edge(s)", supplies_count)


def ingest_evidence(cognee_client: Any, evidence_rows: list[dict]) -> None:
    """Ingest ``Evidence`` nodes and ``EVIDENCES`` edges to RawMaterial.

    Requirement 3.5
    """
    for row in evidence_rows:
        upsert_node(
            cognee_client,
            "Evidence",
            row["id"],
            {
                "supplier_url": row.get("supplier_url"),
                "ingredient_name": row.get("ingredient_name"),
                "certifications": row.get("certifications"),
                "confidence_score": row.get("confidence_score"),
            },
        )
        cognee_client.add_edge(
            f"Evidence:{row['id']}",
            f"RawMaterial:{row.get('product_id', 'unknown')}",
            "EVIDENCES",
        )
    logger.info("Ingested %d Evidence node(s)", len(evidence_rows))


# ---------------------------------------------------------------------------
# Embedding (Requirement 3.6)
# ---------------------------------------------------------------------------

def embed_text_fields(
    cognee_client: Any, vertex_client: Any, node: dict
) -> dict:
    """Embed all text fields of *node* using Vertex AI text-embedding-004.

    Collects non-empty values from the standard text fields, concatenates them,
    obtains an embedding vector from *vertex_client*, and stores it via
    *cognee_client*.  Returns the node dict with an added ``_embedding`` key.

    On any exception the error is logged and the node is returned unchanged.

    Requirement 3.6
    """
    try:
        parts: list[str] = []
        for field in TEXT_FIELDS:
            value = node.get(field)
            if value:
                if isinstance(value, list):
                    parts.append(" ".join(str(v) for v in value))
                else:
                    parts.append(str(value))

        if not parts:
            return node

        text = " ".join(parts)
        embedding = vertex_client.get_embeddings(text)

        node_id = node.get("_node_id") or node.get("id")
        if node_id is not None:
            cognee_client.store_embedding(node_id, embedding)

        node["_embedding"] = embedding
        return node
    except Exception:
        logger.exception("Failed to embed node %s", node.get("id"))
        return node


# ---------------------------------------------------------------------------
# cognee cloud configuration
# ---------------------------------------------------------------------------

def _configure_cognee_cloud() -> None:
    """Set the cognee Cloud API key so the SDK talks to app.cognee.ai.

    Reads ``COGNEE_API_KEY`` from the environment (validated at import time
    by ``config.py``).  If ``cognee.config`` exposes a ``set_llm_api_key``
    helper we use it; otherwise we fall back to setting the key attribute
    directly.  Any import/config failure is logged but not fatal — the
    pipeline can still operate with a locally-configured cognee instance.
    """
    api_key = os.getenv("COGNEE_API_KEY")
    if not api_key:
        logger.debug("COGNEE_API_KEY not set — skipping cloud configuration")
        return

    try:
        import cognee  # type: ignore[import-untyped]

        if hasattr(cognee, "config"):
            cfg = cognee.config
            if callable(getattr(cfg, "set_llm_api_key", None)):
                cfg.set_llm_api_key(api_key)
            elif hasattr(cfg, "llm_api_key"):
                cfg.llm_api_key = api_key

        logger.info("cognee cloud configured — using API key from COGNEE_API_KEY")
    except Exception:
        logger.exception("Failed to configure cognee cloud API key")


# ---------------------------------------------------------------------------
# Pipeline orchestrator (Requirements 3.1 – 3.7)
# ---------------------------------------------------------------------------

def run_pipeline(
    bq_client: Any, cognee_client: Any, vertex_client: Any
) -> None:
    """Read all BigQuery tables and run the full cognee ingestion pipeline.

    Steps executed in dependency order:
    1. Companies
    2. Products (FinishedGood / RawMaterial + OWNS edges)
    3. BOMs (BOM nodes + HAS_BOM + CONTAINS edges)
    4. Suppliers (Supplier nodes + SUPPLIES edges)
    5. Evidence (Evidence nodes + EVIDENCES edges)
    6. Embed all nodes
    """
    # -- Configure cognee cloud ------------------------------------------------
    _configure_cognee_cloud()

    logger.info("Starting cognee ingestion pipeline")

    # ---- Read BigQuery tables ------------------------------------------------
    company_rows = [
        _row_to_dict(r)
        for r in bq_client.query("SELECT * FROM agnes_bom.company").result()
    ]
    product_rows = [
        _row_to_dict(r)
        for r in bq_client.query("SELECT * FROM agnes_bom.product").result()
    ]
    bom_rows = [
        _row_to_dict(r)
        for r in bq_client.query("SELECT * FROM agnes_bom.bom").result()
    ]
    bom_component_rows = [
        _row_to_dict(r)
        for r in bq_client.query("SELECT * FROM agnes_bom.bom_component").result()
    ]
    supplier_rows = [
        _row_to_dict(r)
        for r in bq_client.query("SELECT * FROM agnes_bom.supplier").result()
    ]
    supplier_product_rows = [
        _row_to_dict(r)
        for r in bq_client.query("SELECT * FROM agnes_bom.supplier_product").result()
    ]
    evidence_rows = [
        _row_to_dict(r)
        for r in bq_client.query(
            "SELECT * FROM agnes_enrichment.evidence WHERE is_active = TRUE"
        ).result()
    ]

    # ---- Build lookup --------------------------------------------------------
    products_by_id: dict[int, dict] = {p["id"]: p for p in product_rows}

    # ---- Ingest in dependency order ------------------------------------------
    ingest_companies(cognee_client, company_rows)
    ingest_products(cognee_client, product_rows)
    ingest_boms(cognee_client, bom_rows, bom_component_rows, products_by_id)
    ingest_suppliers(cognee_client, supplier_rows, supplier_product_rows, products_by_id)
    ingest_evidence(cognee_client, evidence_rows)

    # ---- Embed all nodes -----------------------------------------------------
    all_rows = company_rows + product_rows + supplier_rows + evidence_rows
    embedded_count = 0
    for row in all_rows:
        embed_text_fields(cognee_client, vertex_client, row)
        embedded_count += 1

    logger.info(
        "cognee ingestion pipeline complete — "
        "%d companies, %d products, %d BOMs, %d suppliers, %d evidence records, "
        "%d nodes embedded",
        len(company_rows),
        len(product_rows),
        len(bom_rows),
        len(supplier_rows),
        len(evidence_rows),
        embedded_count,
    )
