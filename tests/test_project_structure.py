"""
tests/test_project_structure.py — Structural tests for project organization.

Ensures the package layout, public imports, and module locations stay
consistent as the codebase evolves. If a file is moved or an import path
breaks, these tests catch it immediately.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Expected packages and modules
# ---------------------------------------------------------------------------

EXPECTED_PACKAGES = [
    "migration",
    "enrichment",
    "knowledge",
    "agents",
    "api",
    "dify_setup",
]

EXPECTED_MODULES = {
    "migration.migrate_bom": [
        "create_datasets",
        "load_table",
        "read_sqlite",
        "validate_product_type",
        "validate_bom_product_type",
        "validate_bom_component_type",
        "validate_supplier_product_type",
        "warn_bom_component_count",
        "_truncate_table",
    ],
    "enrichment.pipeline": [
        "scrape_url",
        "extract_evidence",
        "assign_confidence_scores",
        "build_evidence_record",
        "write_evidence_to_bq",
        "run_enrichment",
        "create_enrichment_schema",
        "construct_deterministic_url",
        "build_search_query",
        "passes_keyword_check",
        "classify_with_gemini_flash",
        "validate_relevance",
        "log_rejection",
        "run_net_new_diff",
        "run_ghost_diff",
        "retire_ghost_record",
        "run_temporal_diff",
    ],
    "knowledge.pipeline": [
        "upsert_node",
        "ingest_companies",
        "ingest_products",
        "ingest_boms",
        "ingest_suppliers",
        "ingest_evidence",
        "embed_text_fields",
        "run_pipeline",
    ],
    "api.main": [
        "app",
        "SearchRequest",
        "SearchResponse",
        "SearchResultItem",
    ],
    "dify_setup.pulp_solver_tool": [
        "run_consolidation_solver",
        "TOOL_SCHEMA",
    ],
}

# Modules that must NOT exist at the root (old locations)
FORBIDDEN_ROOT_MODULES = [
    "migrate_bom",
    "enrichment_agent",
    "cognee_pipeline",
]


# ---------------------------------------------------------------------------
# Package existence tests
# ---------------------------------------------------------------------------

class TestPackageStructure:
    """Verify all expected packages exist with __init__.py files."""

    @pytest.mark.parametrize("package", EXPECTED_PACKAGES)
    def test_package_directory_exists(self, package):
        pkg_path = Path(package)
        assert pkg_path.is_dir(), f"Package directory '{package}/' is missing"

    @pytest.mark.parametrize("package", EXPECTED_PACKAGES)
    def test_package_has_init(self, package):
        init_path = Path(package) / "__init__.py"
        assert init_path.is_file(), f"'{package}/__init__.py' is missing"

    @pytest.mark.parametrize("package", EXPECTED_PACKAGES)
    def test_package_is_importable(self, package):
        mod = importlib.import_module(package)
        assert mod is not None


# ---------------------------------------------------------------------------
# Module import tests
# ---------------------------------------------------------------------------

class TestModuleImports:
    """Verify all expected modules are importable and export the right symbols."""

    @pytest.mark.parametrize("module_path", EXPECTED_MODULES.keys())
    def test_module_is_importable(self, module_path):
        mod = importlib.import_module(module_path)
        assert mod is not None

    @pytest.mark.parametrize(
        "module_path,symbol",
        [
            (mod, sym)
            for mod, syms in EXPECTED_MODULES.items()
            for sym in syms
        ],
    )
    def test_symbol_exists_in_module(self, module_path, symbol):
        mod = importlib.import_module(module_path)
        assert hasattr(mod, symbol), (
            f"'{module_path}' is missing expected symbol '{symbol}'. "
            f"Was it renamed or moved?"
        )


# ---------------------------------------------------------------------------
# Old root-level modules must not exist
# ---------------------------------------------------------------------------

class TestNoStaleRootModules:
    """Ensure old root-level module files have been removed after reorganization."""

    @pytest.mark.parametrize("module_name", FORBIDDEN_ROOT_MODULES)
    def test_old_module_file_not_at_root(self, module_name):
        root_file = Path(f"{module_name}.py")
        assert not root_file.exists(), (
            f"Stale root-level module '{module_name}.py' still exists. "
            f"It should have been moved to its package directory."
        )


# ---------------------------------------------------------------------------
# config.py stays at root
# ---------------------------------------------------------------------------

class TestConfigAtRoot:
    """config.py is the one module that intentionally stays at the project root."""

    def test_config_exists_at_root(self):
        assert Path("config.py").is_file(), "config.py must remain at the project root"

    def test_config_is_importable(self):
        mod = importlib.import_module("config")
        assert hasattr(mod, "GCP_PROJECT")
        assert hasattr(mod, "COGNEE_API_KEY")
