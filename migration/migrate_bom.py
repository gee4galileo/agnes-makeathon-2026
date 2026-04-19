"""
migrate_bom.py — Migrate BOM data from SQLite to BigQuery.

Usage:
    python migrate_bom.py --project my-gcp-project [options]
"""

import argparse
import logging
import sqlite3
import sys
from typing import Any

from google.cloud import bigquery
from google.cloud.exceptions import Conflict

from enrichment.pipeline import create_enrichment_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_datasets(
    client: bigquery.Client,
    project: str,
    bom_dataset: str,
    enrichment_dataset: str,
) -> None:
    """Create agnes_bom and agnes_enrichment datasets (idempotent). Req 1.9"""
    for dataset_id in (bom_dataset, enrichment_dataset):
        full_id = f"{project}.{dataset_id}"
        dataset = bigquery.Dataset(full_id)
        dataset.location = "US"
        try:
            client.create_dataset(dataset, timeout=30)
            logger.info("Created dataset %s", full_id)
        except Conflict:
            logger.info("Dataset %s already exists — skipping creation", full_id)


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------

def load_table(
    client: bigquery.Client,
    dataset: str,
    table_name: str,
    rows: list[dict],
) -> int:
    """
    Insert rows into BigQuery via streaming inserts.
    Logs {row_id, error_reason} for each failed row and continues. Req 1.10
    Returns count of successfully inserted rows.
    """
    if not rows:
        logger.info("No rows to insert into %s.%s", dataset, table_name)
        return 0

    table_ref = f"{dataset}.{table_name}"
    errors = client.insert_rows_json(table_ref, rows)

    failed_indices: set[int] = set()
    if errors:
        for error_item in errors:
            idx = error_item.get("index", -1)
            reasons = error_item.get("errors", [])
            reason_str = "; ".join(e.get("message", str(e)) for e in reasons)
            row = rows[idx] if 0 <= idx < len(rows) else {}
            row_id = row.get("id", row.get("bom_id", row.get("supplier_id", idx)))
            logger.error(
                "Failed row in %s — row_id=%s error=%s", table_ref, row_id, reason_str
            )
            failed_indices.add(idx)

    success_count = len(rows) - len(failed_indices)
    logger.info(
        "Loaded %d/%d rows into %s", success_count, len(rows), table_ref
    )
    return success_count


# ---------------------------------------------------------------------------
# SQLite reader
# ---------------------------------------------------------------------------

def read_sqlite(sqlite_path: str) -> dict[str, list[dict]]:
    """
    Read all six tables from SQLite and return snake_case row dicts.
    Raises SystemExit(1) if the file is not found.
    """
    try:
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        logger.critical("SQLite file not found or unreadable: %s — %s", sqlite_path, exc)
        sys.exit(1)

    conn.row_factory = sqlite3.Row

    def fetch(table: str) -> list[dict]:
        cur = conn.execute(f"SELECT * FROM {table}")
        return [dict(row) for row in cur.fetchall()]

    raw: dict[str, list[dict]] = {
        "Company": fetch("Company"),
        "Product": fetch("Product"),
        "BOM": fetch("BOM"),
        "BOM_Component": fetch("BOM_Component"),
        "Supplier": fetch("Supplier"),
        "Supplier_Product": fetch("Supplier_Product"),
    }
    conn.close()

    # Map PascalCase SQLite columns → snake_case BigQuery columns
    company_rows = [
        {"id": r["Id"], "name": r["Name"]}
        for r in raw["Company"]
    ]

    product_rows = [
        {
            "id": r["Id"],
            "sku": r["SKU"],
            "company_id": r["CompanyId"],
            "type": r["Type"],
            "canonical_category": None,
        }
        for r in raw["Product"]
    ]

    bom_rows = [
        {"id": r["Id"], "produced_product_id": r["ProducedProductId"]}
        for r in raw["BOM"]
    ]

    bom_component_rows = [
        {"bom_id": r["BOMId"], "consumed_product_id": r["ConsumedProductId"]}
        for r in raw["BOM_Component"]
    ]

    supplier_rows = [
        {"id": r["Id"], "name": r["Name"]}
        for r in raw["Supplier"]
    ]

    supplier_product_rows = [
        {"supplier_id": r["SupplierId"], "product_id": r["ProductId"]}
        for r in raw["Supplier_Product"]
    ]

    return {
        "company": company_rows,
        "product": product_rows,
        "bom": bom_rows,
        "bom_component": bom_component_rows,
        "supplier": supplier_rows,
        "supplier_product": supplier_product_rows,
    }


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

VALID_PRODUCT_TYPES = {"finished-good", "raw-material"}


def validate_product_type(row: dict) -> tuple[bool, Any]:
    """Req 1.4 — type must be 'finished-good' or 'raw-material'."""
    t = row.get("type")
    if t in VALID_PRODUCT_TYPES:
        return True, None
    return False, f"invalid product type '{t}' for product id={row.get('id')}"


def validate_bom_product_type(
    bom_row: dict, products_by_id: dict[int, dict]
) -> tuple[bool, Any]:
    """Req 1.5 — BOM must reference a finished-good product."""
    pid = bom_row.get("produced_product_id")
    product = products_by_id.get(pid)
    if product is None:
        return False, f"bom id={bom_row.get('id')} references unknown product id={pid}"
    if product.get("type") == "finished-good":
        return True, None
    return (
        False,
        f"bom id={bom_row.get('id')} references raw-material product id={pid}",
    )


def validate_bom_component_type(
    component_row: dict, products_by_id: dict[int, dict]
) -> tuple[bool, Any]:
    """Req 1.6 — bom_component must reference a raw-material product."""
    pid = component_row.get("consumed_product_id")
    product = products_by_id.get(pid)
    if product is None:
        return (
            False,
            f"bom_component bom_id={component_row.get('bom_id')} references unknown product id={pid}",
        )
    if product.get("type") == "raw-material":
        return True, None
    return (
        False,
        f"bom_component bom_id={component_row.get('bom_id')} references finished-good product id={pid}",
    )


def validate_supplier_product_type(
    sp_row: dict, products_by_id: dict[int, dict]
) -> tuple[bool, Any]:
    """Req 1.8 — supplier_product must reference a raw-material product."""
    pid = sp_row.get("product_id")
    product = products_by_id.get(pid)
    if product is None:
        return (
            False,
            f"supplier_product supplier_id={sp_row.get('supplier_id')} references unknown product id={pid}",
        )
    if product.get("type") == "raw-material":
        return True, None
    return (
        False,
        f"supplier_product supplier_id={sp_row.get('supplier_id')} references finished-good product id={pid}",
    )


def warn_bom_component_count(bom_id: Any, component_rows: list[dict]) -> None:
    """Req 1.7 — warn (do not reject) when a BOM has fewer than 2 components."""
    if len(component_rows) < 2:
        logger.warning(
            "BOM id=%s has only %d component(s) — expected at least 2",
            bom_id,
            len(component_rows),
        )


# ---------------------------------------------------------------------------
# Table truncation (prevents duplicates on re-run)
# ---------------------------------------------------------------------------

def _truncate_table(client: bigquery.Client, dataset_ref: str, table_name: str) -> None:
    """Delete all rows from a BigQuery table before re-loading.

    Uses DML DELETE rather than TRUNCATE for compatibility with streaming
    buffer constraints.  Silently skips if the table does not exist yet.
    """
    table_ref = f"{dataset_ref}.{table_name}"
    try:
        client.query(f"DELETE FROM `{table_ref}` WHERE TRUE").result()
        logger.info("Truncated table %s", table_ref)
    except Exception as exc:
        # Table may not exist on first run — that's fine
        logger.debug("Could not truncate %s (may not exist yet): %s", table_ref, exc)


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _filter_rows(
    rows: list[dict],
    validator,
    table_label: str,
    *extra_args,
) -> list[dict]:
    """Apply a validator to each row; log and drop rejected rows."""
    accepted, rejected = [], 0
    for row in rows:
        ok, reason = validator(row, *extra_args)
        if ok:
            accepted.append(row)
        else:
            rejected += 1
            logger.warning("Rejected %s row: %s", table_label, reason)
    if rejected:
        logger.info(
            "%s: accepted %d, rejected %d", table_label, len(accepted), rejected
        )
    return accepted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate BOM data from SQLite to BigQuery."
    )
    parser.add_argument(
        "--sqlite-path", default="./assets/db.sqlite", help="Path to source SQLite file"
    )
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--bom-dataset", default="agnes_bom", help="BOM dataset name")
    parser.add_argument(
        "--enrichment-dataset",
        default="agnes_enrichment",
        help="Enrichment dataset name",
    )
    args = parser.parse_args()

    # 1. Read SQLite
    logger.info("Reading SQLite from %s", args.sqlite_path)
    tables = read_sqlite(args.sqlite_path)

    # 2. Create BigQuery datasets (idempotent)
    client = bigquery.Client(project=args.project)
    create_datasets(client, args.project, args.bom_dataset, args.enrichment_dataset)
    create_enrichment_schema(client, args.project, args.enrichment_dataset)

    # 3. Validate and filter rows
    # Products first — needed as lookup for dependent tables
    product_rows = _filter_rows(
        tables["product"], validate_product_type, "product"
    )
    products_by_id: dict[int, dict] = {r["id"]: r for r in product_rows}

    bom_rows = _filter_rows(
        tables["bom"], validate_bom_product_type, "bom", products_by_id
    )

    bom_component_rows = _filter_rows(
        tables["bom_component"],
        validate_bom_component_type,
        "bom_component",
        products_by_id,
    )

    # Warn on BOMs with < 2 components
    from collections import defaultdict
    components_by_bom: dict[Any, list[dict]] = defaultdict(list)
    for comp in bom_component_rows:
        components_by_bom[comp["bom_id"]].append(comp)
    for bom in bom_rows:
        warn_bom_component_count(bom["id"], components_by_bom[bom["id"]])

    supplier_product_rows = _filter_rows(
        tables["supplier_product"],
        validate_supplier_product_type,
        "supplier_product",
        products_by_id,
    )

    # 4. Truncate existing data to prevent duplicates on re-run, then load
    bom_dataset_ref = f"{args.project}.{args.bom_dataset}"
    summary: dict[str, dict] = {}

    load_order = [
        ("company", tables["company"]),
        ("product", product_rows),
        ("bom", bom_rows),
        ("bom_component", bom_component_rows),
        ("supplier", tables["supplier"]),
        ("supplier_product", supplier_product_rows),
    ]

    for table_name, _rows in load_order:
        _truncate_table(client, bom_dataset_ref, table_name)

    for table_name, rows in load_order:
        original_count = len(tables.get(table_name, rows))
        loaded = load_table(client, bom_dataset_ref, table_name, rows)
        rejected = original_count - len(rows)  # validation rejects
        summary[table_name] = {
            "original": original_count,
            "loaded": loaded,
            "rejected": rejected,
        }

    # 5. Print summary
    print("\n=== Migration Summary ===")
    for table_name, stats in summary.items():
        print(
            f"  {table_name}: {stats['loaded']} loaded, "
            f"{stats['rejected']} rejected (validation), "
            f"{stats['original']} total in source"
        )
    print("=========================\n")


if __name__ == "__main__":
    main()
