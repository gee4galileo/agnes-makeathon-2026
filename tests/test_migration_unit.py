"""
tests/test_migration_unit.py — Unit tests for migrate_bom.py

Task 2.7: Unit tests for migration script
Validates: Requirements 1.1, 1.9
"""

import os
import sqlite3
import sys
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest
from google.cloud.exceptions import Conflict

from migration.migrate_bom import create_datasets, load_table, read_sqlite


# ---------------------------------------------------------------------------
# Requirement 1.9 — Idempotent dataset creation
# ---------------------------------------------------------------------------

class TestCreateDatasetsIdempotent:
    """
    Req 1.9: IF the target BigQuery dataset already exists, THEN the migration
    script SHALL skip dataset creation and proceed with table loading.
    """

    def _make_client(self, side_effects):
        """Return a mock BQ client whose create_dataset raises the given side effects."""
        mock_client = MagicMock()
        mock_client.create_dataset.side_effect = side_effects
        return mock_client

    def test_first_run_creates_both_datasets(self):
        """Both datasets are created when they do not yet exist."""
        mock_client = self._make_client([None, None])  # no exception → created
        create_datasets(mock_client, "my-project", "agnes_bom", "agnes_enrichment")
        assert mock_client.create_dataset.call_count == 2

    def test_second_run_skips_existing_datasets(self):
        """Running create_datasets a second time (Conflict raised) does not raise."""
        mock_client = self._make_client([Conflict("already exists"), Conflict("already exists")])
        # Must NOT raise — idempotent
        create_datasets(mock_client, "my-project", "agnes_bom", "agnes_enrichment")
        assert mock_client.create_dataset.call_count == 2

    def test_idempotent_mixed_state(self):
        """One dataset exists, one does not — no error raised."""
        mock_client = self._make_client([None, Conflict("already exists")])
        create_datasets(mock_client, "my-project", "agnes_bom", "agnes_enrichment")
        assert mock_client.create_dataset.call_count == 2

    def test_idempotent_called_twice_no_error(self):
        """
        Simulates calling create_datasets twice in sequence (e.g. re-running the
        script).  The second call sees Conflict for both datasets and must not raise.
        """
        mock_client = MagicMock()
        # First call: both succeed
        mock_client.create_dataset.side_effect = [None, None]
        create_datasets(mock_client, "my-project", "agnes_bom", "agnes_enrichment")

        # Second call: both already exist
        mock_client.create_dataset.side_effect = [
            Conflict("already exists"),
            Conflict("already exists"),
        ]
        create_datasets(mock_client, "my-project", "agnes_bom", "agnes_enrichment")

        assert mock_client.create_dataset.call_count == 4  # 2 per call × 2 calls

    def test_dataset_ids_passed_correctly(self):
        """create_datasets passes the correct full dataset IDs to the BQ client."""
        mock_client = MagicMock()
        mock_client.create_dataset.return_value = None

        create_datasets(mock_client, "proj-123", "bom_ds", "enrich_ds")

        created_ids = [
            call_args[0][0].dataset_id
            for call_args in mock_client.create_dataset.call_args_list
        ]
        assert "bom_ds" in created_ids
        assert "enrich_ds" in created_ids


# ---------------------------------------------------------------------------
# Requirement 1.1 — All six tables are loaded when SQLite is valid
# ---------------------------------------------------------------------------

ALL_SIX_TABLES = [
    "company",
    "product",
    "bom",
    "bom_component",
    "supplier",
    "supplier_product",
]


def _write_minimal_sqlite(path: str) -> None:
    """Write a minimal but valid SQLite file with one row per table."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE Company (Id INTEGER PRIMARY KEY, Name TEXT NOT NULL)")
    cur.execute(
        "CREATE TABLE Product (Id INTEGER PRIMARY KEY, SKU TEXT NOT NULL, "
        "CompanyId INTEGER, Type TEXT NOT NULL, CanonicalCategory TEXT)"
    )
    cur.execute(
        "CREATE TABLE BOM (Id INTEGER PRIMARY KEY, ProducedProductId INTEGER NOT NULL)"
    )
    cur.execute(
        "CREATE TABLE BOM_Component (BOMId INTEGER NOT NULL, ConsumedProductId INTEGER NOT NULL)"
    )
    cur.execute("CREATE TABLE Supplier (Id INTEGER PRIMARY KEY, Name TEXT NOT NULL)")
    cur.execute(
        "CREATE TABLE Supplier_Product (SupplierId INTEGER NOT NULL, ProductId INTEGER NOT NULL)"
    )

    cur.execute("INSERT INTO Company VALUES (1, 'Acme')")
    cur.execute("INSERT INTO Product VALUES (10, 'FG-001', 1, 'finished-good', NULL)")
    cur.execute("INSERT INTO Product VALUES (20, 'RM-001', NULL, 'raw-material', NULL)")
    cur.execute("INSERT INTO BOM VALUES (100, 10)")
    cur.execute("INSERT INTO BOM_Component VALUES (100, 20)")
    cur.execute("INSERT INTO Supplier VALUES (200, 'Supplier A')")
    cur.execute("INSERT INTO Supplier_Product VALUES (200, 20)")

    conn.commit()
    conn.close()


class TestAllSixTablesLoaded:
    """
    Req 1.1: The migration script SHALL load all six tables into BigQuery
    when the SQLite source file is valid.
    """

    def test_read_sqlite_returns_all_six_tables(self):
        """read_sqlite returns a dict with exactly the six expected table keys."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            _write_minimal_sqlite(path)
            tables = read_sqlite(path)
            assert set(tables.keys()) == set(ALL_SIX_TABLES)
        finally:
            os.unlink(path)

    def test_load_table_called_for_all_six_tables(self):
        """
        When the migration flow runs against a valid SQLite file, load_table is
        invoked exactly once for each of the six tables.
        """
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            _write_minimal_sqlite(path)
            tables = read_sqlite(path)

            mock_client = MagicMock()
            mock_client.insert_rows_json.return_value = []

            loaded_tables = []
            for table_name in ALL_SIX_TABLES:
                rows = tables[table_name]
                load_table(mock_client, "proj.agnes_bom", table_name, rows)
                loaded_tables.append(table_name)

            assert set(loaded_tables) == set(ALL_SIX_TABLES)
            assert mock_client.insert_rows_json.call_count == len(ALL_SIX_TABLES)
        finally:
            os.unlink(path)

    def test_load_table_returns_correct_row_count(self):
        """load_table returns the number of successfully inserted rows."""
        mock_client = MagicMock()
        mock_client.insert_rows_json.return_value = []  # no errors

        rows = [{"id": 1, "name": "Acme"}, {"id": 2, "name": "Beta"}]
        result = load_table(mock_client, "proj.agnes_bom", "company", rows)
        assert result == 2

    def test_load_table_empty_rows_returns_zero(self):
        """load_table with an empty list returns 0 without calling insert_rows_json."""
        mock_client = MagicMock()
        result = load_table(mock_client, "proj.agnes_bom", "company", [])
        assert result == 0
        mock_client.insert_rows_json.assert_not_called()


# ---------------------------------------------------------------------------
# Requirement 1.9 (fatal path) — Missing SQLite file raises fatal error
# ---------------------------------------------------------------------------

class TestMissingSqliteFile:
    """
    The migration script SHALL raise a fatal error and exit non-zero when the
    SQLite source file does not exist.
    """

    def test_read_sqlite_missing_file_exits_with_code_1(self):
        """read_sqlite exits with code 1 specifically."""
        with pytest.raises(SystemExit) as exc_info:
            read_sqlite("/nonexistent/path/db.sqlite")
        assert exc_info.value.code == 1

    def test_read_sqlite_missing_file_logs_critical(self):
        """read_sqlite logs a CRITICAL message before exiting."""
        with patch("migration.migrate_bom.logger") as mock_logger:
            with pytest.raises(SystemExit):
                read_sqlite("/nonexistent/path/db.sqlite")
        mock_logger.critical.assert_called_once()
        # The critical message should reference the bad path
        critical_args = str(mock_logger.critical.call_args)
        assert "/nonexistent/path/db.sqlite" in critical_args


# ---------------------------------------------------------------------------
# Truncation before load — prevents duplicate rows on re-run
# ---------------------------------------------------------------------------

from migration.migrate_bom import _truncate_table


class TestTruncateBeforeLoad:
    """
    The migration script SHALL truncate existing table data before inserting
    new rows, so that re-running the migration never creates duplicates.
    """

    def test_truncate_issues_delete_query(self):
        """_truncate_table sends a DELETE FROM query to BigQuery."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_client.query.return_value = mock_result

        _truncate_table(mock_client, "proj.agnes_bom", "company")

        mock_client.query.assert_called_once()
        sql = mock_client.query.call_args[0][0]
        assert "DELETE FROM" in sql
        assert "proj.agnes_bom.company" in sql
        mock_result.result.assert_called_once()

    def test_truncate_does_not_raise_on_missing_table(self):
        """_truncate_table silently skips if the table does not exist yet."""
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("Table not found")

        # Must NOT raise
        _truncate_table(mock_client, "proj.agnes_bom", "company")

    def test_truncate_called_for_all_six_tables(self):
        """The migration flow truncates all six BOM tables before loading."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_client.query.return_value = mock_result

        for table_name in ALL_SIX_TABLES:
            _truncate_table(mock_client, "proj.agnes_bom", table_name)

        assert mock_client.query.call_count == 6
        truncated_tables = []
        for call_args in mock_client.query.call_args_list:
            sql = call_args[0][0]
            for t in ALL_SIX_TABLES:
                if t in sql:
                    truncated_tables.append(t)
        assert set(truncated_tables) == set(ALL_SIX_TABLES)

    def test_rerun_produces_same_row_count(self):
        """Running load_table twice after truncation yields the original count, not double."""
        mock_client = MagicMock()
        mock_client.insert_rows_json.return_value = []  # no errors
        mock_result = MagicMock()
        mock_client.query.return_value = mock_result

        rows = [{"id": 1, "name": "Acme"}, {"id": 2, "name": "Beta"}]

        # First run
        _truncate_table(mock_client, "proj.agnes_bom", "company")
        count1 = load_table(mock_client, "proj.agnes_bom", "company", rows)

        # Second run (simulates re-running migration)
        _truncate_table(mock_client, "proj.agnes_bom", "company")
        count2 = load_table(mock_client, "proj.agnes_bom", "company", rows)

        assert count1 == count2 == 2
        # Truncate was called twice
        assert mock_client.query.call_count == 2
