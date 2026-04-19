"""
tests/test_migration_fidelity.py — Property 1: Migration fidelity

Feature: agnes-ai-supply-chain-manager, Property 1: Migration fidelity — row counts and schema preservation

Validates: Requirements 1.2, 1.3
"""

import sqlite3
import tempfile
import os
from collections import defaultdict
from unittest.mock import MagicMock, call, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from migration.migrate_bom import (
    read_sqlite,
    load_table,
    validate_product_type,
    validate_bom_product_type,
    validate_bom_component_type,
    validate_supplier_product_type,
    _filter_rows,
)


# ---------------------------------------------------------------------------
# Strategies — generate valid BOM datasets
# ---------------------------------------------------------------------------

# Valid product types
VALID_TYPES = ["finished-good", "raw-material"]

# Short non-empty text for names/SKUs
short_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
)


@st.composite
def valid_bom_dataset(draw):
    """
    Generate a valid BOM dataset where all rows pass type-constraint validators.

    Structure:
    - 1..3 companies
    - 1..3 finished-good products (one per company)
    - 2..6 raw-material products
    - 1 BOM per finished-good (linked to that finished-good)
    - Each BOM has exactly 2 bom_component rows (both raw-material)
    - 1..3 suppliers
    - Each raw-material has at least one supplier_product entry
    """
    # Companies
    n_companies = draw(st.integers(min_value=1, max_value=3))
    company_ids = list(range(1, n_companies + 1))
    companies = [{"id": cid, "name": draw(short_text)} for cid in company_ids]

    # Finished-good products — one per company
    fg_ids = list(range(100, 100 + n_companies))
    fg_products = [
        {
            "id": fg_ids[i],
            "sku": draw(short_text),
            "company_id": company_ids[i],
            "type": "finished-good",
            "canonical_category": None,
        }
        for i in range(n_companies)
    ]

    # Raw-material products — at least 2 (so each BOM can have 2 components)
    n_raw = draw(st.integers(min_value=2, max_value=6))
    rm_ids = list(range(200, 200 + n_raw))
    rm_products = [
        {
            "id": rm_ids[i],
            "sku": draw(short_text),
            "company_id": None,
            "type": "raw-material",
            "canonical_category": None,
        }
        for i in range(n_raw)
    ]

    products = fg_products + rm_products

    # BOMs — one per finished-good
    bom_ids = list(range(300, 300 + n_companies))
    boms = [
        {"id": bom_ids[i], "produced_product_id": fg_ids[i]}
        for i in range(n_companies)
    ]

    # BOM components — exactly 2 raw-material components per BOM
    # Pick 2 distinct raw-material IDs for each BOM
    bom_components = []
    for bom_id in bom_ids:
        # Draw 2 distinct indices into rm_ids
        idx_a = draw(st.integers(min_value=0, max_value=n_raw - 1))
        idx_b = draw(st.integers(min_value=0, max_value=n_raw - 1))
        # Ensure distinct
        if idx_b == idx_a:
            idx_b = (idx_a + 1) % n_raw
        bom_components.append({"bom_id": bom_id, "consumed_product_id": rm_ids[idx_a]})
        bom_components.append({"bom_id": bom_id, "consumed_product_id": rm_ids[idx_b]})

    # Suppliers
    n_suppliers = draw(st.integers(min_value=1, max_value=3))
    supplier_ids = list(range(400, 400 + n_suppliers))
    suppliers = [{"id": sid, "name": draw(short_text)} for sid in supplier_ids]

    # Supplier products — each raw-material gets at least one supplier
    supplier_products = []
    for rm_id in rm_ids:
        sid = draw(st.sampled_from(supplier_ids))
        supplier_products.append({"supplier_id": sid, "product_id": rm_id})

    return {
        "company": companies,
        "product": products,
        "bom": boms,
        "bom_component": bom_components,
        "supplier": suppliers,
        "supplier_product": supplier_products,
    }


# ---------------------------------------------------------------------------
# Helper — write dataset to an in-memory (temp file) SQLite DB
# ---------------------------------------------------------------------------

def write_sqlite(dataset: dict, path: str) -> None:
    """
    Write a BOM dataset dict (snake_case keys) into a SQLite file using the
    PascalCase schema that read_sqlite expects.
    """
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute(
        "CREATE TABLE Company (Id INTEGER PRIMARY KEY, Name TEXT NOT NULL)"
    )
    cur.execute(
        """CREATE TABLE Product (
            Id INTEGER PRIMARY KEY,
            SKU TEXT NOT NULL,
            CompanyId INTEGER,
            Type TEXT NOT NULL,
            CanonicalCategory TEXT
        )"""
    )
    cur.execute(
        """CREATE TABLE BOM (
            Id INTEGER PRIMARY KEY,
            ProducedProductId INTEGER NOT NULL
        )"""
    )
    cur.execute(
        """CREATE TABLE BOM_Component (
            BOMId INTEGER NOT NULL,
            ConsumedProductId INTEGER NOT NULL
        )"""
    )
    cur.execute(
        "CREATE TABLE Supplier (Id INTEGER PRIMARY KEY, Name TEXT NOT NULL)"
    )
    cur.execute(
        """CREATE TABLE Supplier_Product (
            SupplierId INTEGER NOT NULL,
            ProductId INTEGER NOT NULL
        )"""
    )

    for row in dataset["company"]:
        cur.execute("INSERT INTO Company VALUES (?, ?)", (row["id"], row["name"]))

    for row in dataset["product"]:
        cur.execute(
            "INSERT INTO Product VALUES (?, ?, ?, ?, ?)",
            (row["id"], row["sku"], row["company_id"], row["type"], row["canonical_category"]),
        )

    for row in dataset["bom"]:
        cur.execute("INSERT INTO BOM VALUES (?, ?)", (row["id"], row["produced_product_id"]))

    for row in dataset["bom_component"]:
        cur.execute(
            "INSERT INTO BOM_Component VALUES (?, ?)",
            (row["bom_id"], row["consumed_product_id"]),
        )

    for row in dataset["supplier"]:
        cur.execute("INSERT INTO Supplier VALUES (?, ?)", (row["id"], row["name"]))

    for row in dataset["supplier_product"]:
        cur.execute(
            "INSERT INTO Supplier_Product VALUES (?, ?)",
            (row["supplier_id"], row["product_id"]),
        )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Helper — run the migration flow (validation + load) against a mock BQ client
# ---------------------------------------------------------------------------

def run_migration_flow(tables: dict) -> dict[str, list[dict]]:
    """
    Replicate the validation + load_table calls from migrate_bom.main(),
    using a mock BigQuery client.  Returns a dict of table_name → rows that
    were passed to load_table (i.e. what would be inserted into BigQuery).
    """
    mock_client = MagicMock()
    # insert_rows_json returns [] (no errors) for all calls
    mock_client.insert_rows_json.return_value = []

    dataset_ref = "test-project.agnes_bom"

    # Validate products first
    product_rows = _filter_rows(tables["product"], validate_product_type, "product")
    products_by_id = {r["id"]: r for r in product_rows}

    bom_rows = _filter_rows(
        tables["bom"], validate_bom_product_type, "bom", products_by_id
    )
    bom_component_rows = _filter_rows(
        tables["bom_component"],
        validate_bom_component_type,
        "bom_component",
        products_by_id,
    )
    supplier_product_rows = _filter_rows(
        tables["supplier_product"],
        validate_supplier_product_type,
        "supplier_product",
        products_by_id,
    )

    load_order = [
        ("company", tables["company"]),
        ("product", product_rows),
        ("bom", bom_rows),
        ("bom_component", bom_component_rows),
        ("supplier", tables["supplier"]),
        ("supplier_product", supplier_product_rows),
    ]

    loaded: dict[str, list[dict]] = {}
    for table_name, rows in load_order:
        load_table(mock_client, dataset_ref, table_name, rows)
        loaded[table_name] = rows

    return loaded


# ---------------------------------------------------------------------------
# Property 1: Migration fidelity — row counts and schema preservation
# ---------------------------------------------------------------------------

ALL_TABLES = ["company", "product", "bom", "bom_component", "supplier", "supplier_product"]


@settings(max_examples=100)
@given(dataset=valid_bom_dataset())
def test_migration_fidelity_property1(dataset):
    """
    Feature: agnes-ai-supply-chain-manager, Property 1: Migration fidelity — row counts and schema preservation

    Validates: Requirements 1.2, 1.3

    For any valid SQLite BOM file, running the migration script SHALL produce
    BigQuery tables whose row counts and column schemas exactly match the source
    tables across all six tables.

    Because all generated rows are valid (pass all type-constraint validators),
    no rows should be dropped during the migration flow.
    """
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        sqlite_path = f.name

    try:
        # Write the generated dataset to a real SQLite file
        write_sqlite(dataset, sqlite_path)

        # Read it back via the production read_sqlite function
        tables = read_sqlite(sqlite_path)

        # Run the migration flow (with mocked BigQuery)
        loaded = run_migration_flow(tables)

        # --- Assertion 1: load_table is called once per table (all 6 tables) ---
        assert set(loaded.keys()) == set(ALL_TABLES), (
            f"Expected all 6 tables to be loaded, got: {set(loaded.keys())}"
        )

        # --- Assertion 2: Row counts match source exactly ---
        for table_name in ALL_TABLES:
            source_count = len(tables[table_name])
            loaded_count = len(loaded[table_name])
            assert loaded_count == source_count, (
                f"Table '{table_name}': source has {source_count} rows but "
                f"{loaded_count} rows were passed to load_table"
            )

        # --- Assertion 3: Column schemas (key sets) match source exactly ---
        for table_name in ALL_TABLES:
            source_rows = tables[table_name]
            loaded_rows = loaded[table_name]
            if not source_rows:
                continue  # empty table — nothing to compare
            source_keys = set(source_rows[0].keys())
            loaded_keys = set(loaded_rows[0].keys())
            assert loaded_keys == source_keys, (
                f"Table '{table_name}': source columns {source_keys} != "
                f"loaded columns {loaded_keys}"
            )
            # Verify every row has the same column set
            for i, row in enumerate(loaded_rows):
                assert set(row.keys()) == source_keys, (
                    f"Table '{table_name}' row {i}: column mismatch — "
                    f"expected {source_keys}, got {set(row.keys())}"
                )

        # --- Assertion 4: No rows are dropped or added ---
        for table_name in ALL_TABLES:
            source_rows = tables[table_name]
            loaded_rows = loaded[table_name]
            # Compare as multisets of frozen dicts (order-independent, None-safe).
            # Use repr(v) so all values are strings and tuples are sortable.
            def row_to_key(r):
                return tuple(sorted((k, repr(v)) for k, v in r.items()))

            source_set = sorted(row_to_key(r) for r in source_rows)
            loaded_set = sorted(row_to_key(r) for r in loaded_rows)
            assert source_set == loaded_set, (
                f"Table '{table_name}': row data mismatch between source and loaded rows"
            )

    finally:
        os.unlink(sqlite_path)


# ---------------------------------------------------------------------------
# Property 3: BOM component count warning
# ---------------------------------------------------------------------------

from unittest.mock import patch, call
import logging as _logging

from migration.migrate_bom import warn_bom_component_count


@settings(max_examples=100)
@given(
    bom_id=st.integers(min_value=1, max_value=10_000),
    n_components=st.integers(min_value=0, max_value=5),
)
def test_bom_component_count_warning_property3(bom_id, n_components):
    """
    Feature: agnes-ai-supply-chain-manager, Property 3: BOM component count warning

    Validates: Requirements 1.7

    For any BOM record with fewer than 2 associated bom_component rows, the
    migration script SHALL emit at least one warning log entry referencing that
    BOM's identifier.  When the component count is >= 2, no warning is emitted.
    """
    component_rows = [
        {"bom_id": bom_id, "consumed_product_id": i} for i in range(n_components)
    ]

    with patch("migration.migrate_bom.logger") as mock_logger:
        warn_bom_component_count(bom_id, component_rows)

    if n_components < 2:
        # At least one warning call must reference the BOM's identifier
        warning_calls = mock_logger.warning.call_args_list
        assert warning_calls, (
            f"Expected logger.warning to be called for bom_id={bom_id} "
            f"with n_components={n_components}, but it was not called."
        )
        # The warning message args should contain the bom_id somewhere
        assert any(
            str(bom_id) in str(c) for c in warning_calls
        ), (
            f"Expected a warning referencing bom_id={bom_id}, "
            f"but warning calls were: {warning_calls}"
        )
    else:
        # No warning should be emitted
        mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# Strategy — generate rows with a subset of failing indices
# ---------------------------------------------------------------------------

@st.composite
def rows_with_failures(draw):
    n_rows = draw(st.integers(min_value=2, max_value=20))
    rows = [{"id": i + 1, "name": draw(short_text)} for i in range(n_rows)]
    # Pick at least 1 failing index, leave at least 1 succeeding
    n_fail = draw(st.integers(min_value=1, max_value=n_rows - 1))
    fail_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_rows - 1),
            min_size=n_fail,
            max_size=n_fail,
            unique=True,
        )
    )
    return rows, set(fail_indices)


# ---------------------------------------------------------------------------
# Property 4: Migration error logging and continuation
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(data=rows_with_failures())
def test_migration_error_logging_and_continuation_property4(data):
    """
    Feature: agnes-ai-supply-chain-manager, Property 4: Migration error logging and continuation

    **Validates: Requirements 1.10**

    For any batch of rows where a subset fail to insert, the migration script
    SHALL log each failed row's identifier and error reason, and SHALL
    successfully insert all non-failing rows.
    """
    rows, fail_indices = data

    # Ensure the assumption: at least 1 failing and at least 1 succeeding
    assume(len(fail_indices) >= 1)
    assume(len(fail_indices) < len(rows))

    # Build the BigQuery-style error response: list of {index, errors} for failing rows
    bq_errors = [
        {"index": idx, "errors": [{"message": f"simulated error for row {idx}"}]}
        for idx in sorted(fail_indices)
    ]

    mock_client = MagicMock()
    mock_client.insert_rows_json.return_value = bq_errors

    with patch("migration.migrate_bom.logger") as mock_logger:
        result = load_table(mock_client, "test-project.agnes_bom", "company", rows)

    # --- Assertion 1: Returns count of successfully inserted rows ---
    expected_success = len(rows) - len(fail_indices)
    assert result == expected_success, (
        f"Expected {expected_success} successful rows, got {result}"
    )

    # --- Assertion 2: logger.error called at least once per failing row ---
    error_calls = mock_logger.error.call_args_list
    for idx in fail_indices:
        row_id = rows[idx]["id"]
        matching = [
            c for c in error_calls
            if str(row_id) in str(c) and "simulated error" in str(c)
        ]
        assert matching, (
            f"Expected logger.error to reference row_id={row_id} and error reason, "
            f"but error calls were: {error_calls}"
        )

    # --- Assertion 3: logger.error NOT called for rows that succeeded ---
    # Check by inspecting the row_id positional argument (index 2) in each error call.
    # The log format is: logger.error("Failed row in %s — row_id=%s error=%s", table_ref, row_id, reason_str)
    fail_ids = {rows[i]["id"] for i in fail_indices}
    success_indices = set(range(len(rows))) - fail_indices
    for idx in success_indices:
        row_id = rows[idx]["id"]
        # Only check row_ids that are not also a failing row id (to avoid ambiguity)
        if row_id in fail_ids:
            continue
        spurious = [
            c for c in error_calls
            if len(c.args) >= 3 and c.args[2] == row_id
        ]
        assert not spurious, (
            f"logger.error was unexpectedly called for successful row_id={row_id}: {spurious}"
        )


# ---------------------------------------------------------------------------
# Strategies — generate datasets with BOTH valid and invalid rows
# ---------------------------------------------------------------------------

# Invalid product types — anything not in {finished-good, raw-material}
INVALID_TYPES = ["finished", "raw", "FG", "RM", "", "unknown", "semi-finished"]


@st.composite
def mixed_validity_dataset(draw):
    """
    Generate a BOM dataset containing a mix of valid and invalid rows across
    product, bom, bom_component, and supplier_product tables.

    Returns (dataset_dict, expected_invalids) where expected_invalids is a dict
    mapping table name → set of row identifiers that should be rejected.
    """
    # --- Products: mix of valid and invalid types ---
    n_fg = draw(st.integers(min_value=1, max_value=3))
    n_rm = draw(st.integers(min_value=2, max_value=4))
    n_bad_products = draw(st.integers(min_value=1, max_value=3))

    next_id = 1

    fg_products = []
    for _ in range(n_fg):
        fg_products.append({
            "id": next_id, "sku": draw(short_text),
            "company_id": 1, "type": "finished-good", "canonical_category": None,
        })
        next_id += 1

    rm_products = []
    for _ in range(n_rm):
        rm_products.append({
            "id": next_id, "sku": draw(short_text),
            "company_id": 1, "type": "raw-material", "canonical_category": None,
        })
        next_id += 1

    bad_products = []
    for _ in range(n_bad_products):
        bad_type = draw(st.sampled_from(INVALID_TYPES))
        bad_products.append({
            "id": next_id, "sku": draw(short_text),
            "company_id": 1, "type": bad_type, "canonical_category": None,
        })
        next_id += 1

    all_products = fg_products + rm_products + bad_products
    invalid_product_ids = {p["id"] for p in bad_products}

    # Build lookup of VALID products only (what the pipeline will have after
    # filtering products). Bad-type products are rejected before dependent
    # tables are validated.
    valid_products_by_id = {p["id"]: p for p in fg_products + rm_products}

    # --- BOMs: some reference finished-good (valid), some reference raw-material (invalid) ---
    bom_next_id = 500
    valid_boms = []
    invalid_bom_ids = set()

    # Valid BOMs — reference finished-good products
    for fg in fg_products:
        valid_boms.append({"id": bom_next_id, "produced_product_id": fg["id"]})
        bom_next_id += 1

    # Invalid BOMs — reference raw-material products (violates Req 1.5)
    n_bad_boms = draw(st.integers(min_value=1, max_value=2))
    invalid_boms = []
    for i in range(n_bad_boms):
        rm = rm_products[i % len(rm_products)]
        invalid_boms.append({"id": bom_next_id, "produced_product_id": rm["id"]})
        invalid_bom_ids.add(bom_next_id)
        bom_next_id += 1

    all_boms = valid_boms + invalid_boms

    # --- BOM Components: some reference raw-material (valid), some reference finished-good (invalid) ---
    valid_components = []
    invalid_component_keys = set()

    # Valid components — reference raw-material products
    for bom in valid_boms:
        for j in range(2):
            rm = rm_products[j % len(rm_products)]
            valid_components.append({"bom_id": bom["id"], "consumed_product_id": rm["id"]})

    # Invalid components — reference finished-good products (violates Req 1.6)
    n_bad_components = draw(st.integers(min_value=1, max_value=2))
    invalid_components = []
    for i in range(n_bad_components):
        fg = fg_products[i % len(fg_products)]
        bom = valid_boms[i % len(valid_boms)]
        comp = {"bom_id": bom["id"], "consumed_product_id": fg["id"]}
        invalid_components.append(comp)
        invalid_component_keys.add((comp["bom_id"], comp["consumed_product_id"]))

    all_components = valid_components + invalid_components

    # --- Supplier Products: some reference raw-material (valid), some reference finished-good (invalid) ---
    valid_sp = []
    invalid_sp_keys = set()

    # Valid supplier_products — reference raw-material products
    for rm in rm_products:
        valid_sp.append({"supplier_id": 1, "product_id": rm["id"]})

    # Invalid supplier_products — reference finished-good products (violates Req 1.8)
    n_bad_sp = draw(st.integers(min_value=1, max_value=2))
    invalid_sps = []
    for i in range(n_bad_sp):
        fg = fg_products[i % len(fg_products)]
        sp = {"supplier_id": 1, "product_id": fg["id"]}
        invalid_sps.append(sp)
        invalid_sp_keys.add((sp["supplier_id"], sp["product_id"]))

    all_sp = valid_sp + invalid_sps

    dataset = {
        "company": [{"id": 1, "name": "TestCo"}],
        "product": all_products,
        "bom": all_boms,
        "bom_component": all_components,
        "supplier": [{"id": 1, "name": "SupplierCo"}],
        "supplier_product": all_sp,
    }

    expected_invalids = {
        "product_ids": invalid_product_ids,
        "bom_ids": invalid_bom_ids,
        "component_keys": invalid_component_keys,
        "sp_keys": invalid_sp_keys,
    }

    return dataset, expected_invalids


# ---------------------------------------------------------------------------
# Property 2: Migration rejects type-constraint violations
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=mixed_validity_dataset())
def test_migration_rejects_type_constraint_violations_property2(data):
    """
    Feature: agnes-ai-supply-chain-manager, Property 2: Migration rejects type-constraint violations

    **Validates: Requirements 1.4, 1.5, 1.6, 1.8**
    """
    dataset, expected_invalids = data

    # --- 1. Validate individual validator return values ---

    # Req 1.4: validate_product_type
    for product in dataset["product"]:
        ok, reason = validate_product_type(product)
        if product["id"] in expected_invalids["product_ids"]:
            assert ok is False, (
                f"Expected product id={product['id']} type='{product['type']}' to be rejected"
            )
            assert isinstance(reason, str) and len(reason) > 0, (
                "Rejected product must have a non-empty reason string"
            )
        else:
            assert ok is True, (
                f"Expected product id={product['id']} type='{product['type']}' to be accepted"
            )
            assert reason is None, "Accepted product must have reason=None"

    # Build products_by_id from VALID products only (mirrors pipeline behaviour)
    valid_product_rows = _filter_rows(
        dataset["product"], validate_product_type, "product"
    )
    products_by_id = {r["id"]: r for r in valid_product_rows}

    # Req 1.5: validate_bom_product_type
    for bom in dataset["bom"]:
        ok, reason = validate_bom_product_type(bom, products_by_id)
        if bom["id"] in expected_invalids["bom_ids"]:
            assert ok is False, (
                f"Expected bom id={bom['id']} to be rejected (references raw-material)"
            )
            assert isinstance(reason, str) and len(reason) > 0
        else:
            assert ok is True, (
                f"Expected bom id={bom['id']} to be accepted"
            )
            assert reason is None

    # Req 1.6: validate_bom_component_type
    for comp in dataset["bom_component"]:
        ok, reason = validate_bom_component_type(comp, products_by_id)
        key = (comp["bom_id"], comp["consumed_product_id"])
        if key in expected_invalids["component_keys"]:
            assert ok is False, (
                f"Expected bom_component {key} to be rejected (references finished-good)"
            )
            assert isinstance(reason, str) and len(reason) > 0
        else:
            assert ok is True, (
                f"Expected bom_component {key} to be accepted"
            )
            assert reason is None

    # Req 1.8: validate_supplier_product_type
    for sp in dataset["supplier_product"]:
        ok, reason = validate_supplier_product_type(sp, products_by_id)
        key = (sp["supplier_id"], sp["product_id"])
        if key in expected_invalids["sp_keys"]:
            assert ok is False, (
                f"Expected supplier_product {key} to be rejected (references finished-good)"
            )
            assert isinstance(reason, str) and len(reason) > 0
        else:
            assert ok is True, (
                f"Expected supplier_product {key} to be accepted"
            )
            assert reason is None

    # --- 2. Validate _filter_rows rejects invalid and accepts valid ---

    filtered_products = _filter_rows(
        dataset["product"], validate_product_type, "product"
    )
    filtered_boms = _filter_rows(
        dataset["bom"], validate_bom_product_type, "bom", products_by_id
    )
    filtered_components = _filter_rows(
        dataset["bom_component"], validate_bom_component_type, "bom_component",
        products_by_id,
    )
    filtered_sp = _filter_rows(
        dataset["supplier_product"], validate_supplier_product_type,
        "supplier_product", products_by_id,
    )

    # Every invalid product is absent from filtered output
    filtered_product_ids = {p["id"] for p in filtered_products}
    for bad_id in expected_invalids["product_ids"]:
        assert bad_id not in filtered_product_ids, (
            f"Invalid product id={bad_id} should have been rejected"
        )

    # Every valid product is present
    valid_product_ids = {
        p["id"] for p in dataset["product"]
        if p["id"] not in expected_invalids["product_ids"]
    }
    for good_id in valid_product_ids:
        assert good_id in filtered_product_ids, (
            f"Valid product id={good_id} should have been accepted"
        )

    # Every invalid BOM is absent
    filtered_bom_ids = {b["id"] for b in filtered_boms}
    for bad_id in expected_invalids["bom_ids"]:
        assert bad_id not in filtered_bom_ids, (
            f"Invalid bom id={bad_id} should have been rejected"
        )

    # Every valid BOM is present
    valid_bom_ids = {
        b["id"] for b in dataset["bom"]
        if b["id"] not in expected_invalids["bom_ids"]
    }
    for good_id in valid_bom_ids:
        assert good_id in filtered_bom_ids, (
            f"Valid bom id={good_id} should have been accepted"
        )

    # Every invalid bom_component is absent
    filtered_comp_keys = {
        (c["bom_id"], c["consumed_product_id"]) for c in filtered_components
    }
    for bad_key in expected_invalids["component_keys"]:
        assert bad_key not in filtered_comp_keys, (
            f"Invalid bom_component {bad_key} should have been rejected"
        )

    # Every valid bom_component is present
    valid_comp_keys = {
        (c["bom_id"], c["consumed_product_id"]) for c in dataset["bom_component"]
        if (c["bom_id"], c["consumed_product_id"]) not in expected_invalids["component_keys"]
    }
    for good_key in valid_comp_keys:
        assert good_key in filtered_comp_keys, (
            f"Valid bom_component {good_key} should have been accepted"
        )

    # Every invalid supplier_product is absent
    filtered_sp_keys = {
        (s["supplier_id"], s["product_id"]) for s in filtered_sp
    }
    for bad_key in expected_invalids["sp_keys"]:
        assert bad_key not in filtered_sp_keys, (
            f"Invalid supplier_product {bad_key} should have been rejected"
        )

    # Every valid supplier_product is present
    valid_sp_keys = {
        (s["supplier_id"], s["product_id"]) for s in dataset["supplier_product"]
        if (s["supplier_id"], s["product_id"]) not in expected_invalids["sp_keys"]
    }
    for good_key in valid_sp_keys:
        assert good_key in filtered_sp_keys, (
            f"Valid supplier_product {good_key} should have been accepted"
        )
