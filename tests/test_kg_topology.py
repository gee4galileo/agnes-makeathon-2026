"""
tests/test_kg_topology.py — Property 8: Knowledge graph topology correctness

Feature: agnes-ai-supply-chain-manager, Property 8

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5

For any BOM dataset ingested into cognee, the resulting graph SHALL contain:
one node per company, one node per product (typed correctly as FinishedGood or
RawMaterial), one BOM node per finished-good with a HAS_BOM edge, CONTAINS
edges from each BOM to its raw-material components, and SUPPLIES edges from
each supplier to its raw-material products only.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from knowledge.pipeline import (
    ingest_companies,
    ingest_products,
    ingest_boms,
    ingest_suppliers,
    ingest_evidence,
    upsert_node,
)


# ---------------------------------------------------------------------------
# Strategies — generate valid BOM datasets for cognee ingestion
# ---------------------------------------------------------------------------

short_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
)


@st.composite
def bom_dataset(draw):
    """Generate a valid BOM dataset suitable for cognee ingestion."""
    # Companies: 1–3
    n_companies = draw(st.integers(min_value=1, max_value=3))
    companies = [{"id": i + 1, "name": draw(short_text)} for i in range(n_companies)]

    # Finished-good products: one per company
    fg_products = [
        {
            "id": 100 + i,
            "sku": draw(short_text),
            "company_id": companies[i]["id"],
            "type": "finished-good",
            "canonical_category": draw(st.one_of(st.none(), short_text)),
        }
        for i in range(n_companies)
    ]

    # Raw-material products: 2–6
    n_raw = draw(st.integers(min_value=2, max_value=6))
    rm_products = [
        {
            "id": 200 + i,
            "sku": draw(short_text),
            "company_id": None,
            "type": "raw-material",
            "canonical_category": draw(st.one_of(st.none(), short_text)),
        }
        for i in range(n_raw)
    ]

    products = fg_products + rm_products
    rm_ids = [p["id"] for p in rm_products]

    # BOMs: one per finished-good
    boms = [
        {"id": 300 + i, "produced_product_id": fg_products[i]["id"]}
        for i in range(n_companies)
    ]

    # BOM components: 2 distinct raw-materials per BOM
    bom_components = []
    for bom in boms:
        idx_a = draw(st.integers(min_value=0, max_value=n_raw - 1))
        idx_b = draw(st.integers(min_value=0, max_value=n_raw - 1))
        if idx_b == idx_a:
            idx_b = (idx_a + 1) % n_raw
        bom_components.append({"bom_id": bom["id"], "consumed_product_id": rm_ids[idx_a]})
        bom_components.append({"bom_id": bom["id"], "consumed_product_id": rm_ids[idx_b]})

    # Suppliers: 1–3
    n_suppliers = draw(st.integers(min_value=1, max_value=3))
    suppliers = [{"id": 400 + i, "name": draw(short_text)} for i in range(n_suppliers)]
    supplier_ids = [s["id"] for s in suppliers]

    # Supplier-product links: each raw-material gets one supplier
    supplier_products = [
        {"supplier_id": draw(st.sampled_from(supplier_ids)), "product_id": rm_id}
        for rm_id in rm_ids
    ]

    # Evidence records: 0–3 linked to random raw-materials
    n_evidence = draw(st.integers(min_value=0, max_value=3))
    evidence = [
        {
            "id": f"ev-{i}",
            "supplier_url": f"https://example.com/{i}",
            "ingredient_name": draw(short_text),
            "certifications": [],
            "confidence_score": 0.8,
            "product_id": draw(st.sampled_from(rm_ids)),
        }
        for i in range(n_evidence)
    ]

    return {
        "companies": companies,
        "products": products,
        "fg_products": fg_products,
        "rm_products": rm_products,
        "boms": boms,
        "bom_components": bom_components,
        "suppliers": suppliers,
        "supplier_products": supplier_products,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Helpers — tracking cognee client
# ---------------------------------------------------------------------------

def make_tracking_cognee():
    """Return a mock cognee client that records created nodes and edges."""
    client = MagicMock()
    client.get_node.side_effect = Exception("not found")

    nodes: dict[str, dict] = {}
    edges: list[tuple[str, str, str]] = []

    def _create_node(node_id, node_type, fields):
        nodes[node_id] = {"type": node_type, **fields}

    def _add_edge(src, dst, label):
        edges.append((src, dst, label))

    client.create_node.side_effect = _create_node
    client.add_edge.side_effect = _add_edge

    return client, nodes, edges


# ---------------------------------------------------------------------------
# Property 8: Knowledge graph topology correctness
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(data=bom_dataset())
def test_kg_topology_correctness_property8(data):
    """
    Feature: agnes-ai-supply-chain-manager, Property 8

    Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5

    For any BOM dataset ingested into cognee, the resulting graph SHALL contain:
    - one Company node per company row
    - one FinishedGood node per finished-good product
    - one RawMaterial node per raw-material product
    - one OWNS edge per finished-good (Company → FinishedGood)
    - one BOM node per BOM row with a HAS_BOM edge (FinishedGood → BOM)
    - one CONTAINS edge per bom_component (BOM → RawMaterial)
    - one Supplier node per supplier row
    - SUPPLIES edges only to raw-material products
    - one Evidence node per evidence row with an EVIDENCES edge to RawMaterial
    """
    client, nodes, edges = make_tracking_cognee()
    products_by_id = {p["id"]: p for p in data["products"]}

    # Run all ingestion functions in dependency order
    ingest_companies(client, data["companies"])
    ingest_products(client, data["products"])
    ingest_boms(client, data["boms"], data["bom_components"], products_by_id)
    ingest_suppliers(client, data["suppliers"], data["supplier_products"], products_by_id)
    ingest_evidence(client, data["evidence"])

    # --- Assertion 1: One Company node per company row ---
    company_nodes = {nid for nid, n in nodes.items() if n["type"] == "Company"}
    expected_company_ids = {f"Company:{c['id']}" for c in data["companies"]}
    assert company_nodes == expected_company_ids, (
        f"Company nodes mismatch: got {company_nodes}, expected {expected_company_ids}"
    )

    # --- Assertion 2: Correct FinishedGood and RawMaterial nodes (Req 3.1, 3.2) ---
    fg_nodes = {nid for nid, n in nodes.items() if n["type"] == "FinishedGood"}
    rm_nodes = {nid for nid, n in nodes.items() if n["type"] == "RawMaterial"}
    expected_fg = {f"FinishedGood:{p['id']}" for p in data["fg_products"]}
    expected_rm = {f"RawMaterial:{p['id']}" for p in data["rm_products"]}
    assert fg_nodes == expected_fg, (
        f"FinishedGood nodes mismatch: got {fg_nodes}, expected {expected_fg}"
    )
    assert rm_nodes == expected_rm, (
        f"RawMaterial nodes mismatch: got {rm_nodes}, expected {expected_rm}"
    )

    # --- Assertion 3: One OWNS edge per finished-good (Req 3.2) ---
    owns_edges = [(s, d) for s, d, l in edges if l == "OWNS"]
    assert len(owns_edges) == len(data["fg_products"]), (
        f"Expected {len(data['fg_products'])} OWNS edges, got {len(owns_edges)}"
    )
    for fg in data["fg_products"]:
        expected_edge = (f"Company:{fg['company_id']}", f"FinishedGood:{fg['id']}")
        assert expected_edge in owns_edges, (
            f"Missing OWNS edge: {expected_edge}"
        )

    # --- Assertion 4: One BOM node + HAS_BOM edge per BOM (Req 3.3) ---
    bom_nodes = {nid for nid, n in nodes.items() if n["type"] == "BOM"}
    expected_bom = {f"BOM:{b['id']}" for b in data["boms"]}
    assert bom_nodes == expected_bom, (
        f"BOM nodes mismatch: got {bom_nodes}, expected {expected_bom}"
    )

    has_bom_edges = [(s, d) for s, d, l in edges if l == "HAS_BOM"]
    assert len(has_bom_edges) == len(data["boms"]), (
        f"Expected {len(data['boms'])} HAS_BOM edges, got {len(has_bom_edges)}"
    )
    for bom in data["boms"]:
        expected_edge = (f"FinishedGood:{bom['produced_product_id']}", f"BOM:{bom['id']}")
        assert expected_edge in has_bom_edges, (
            f"Missing HAS_BOM edge: {expected_edge}"
        )

    # --- Assertion 5: CONTAINS edges from BOM → RawMaterial (Req 3.3) ---
    contains_edges = [(s, d) for s, d, l in edges if l == "CONTAINS"]
    assert len(contains_edges) == len(data["bom_components"]), (
        f"Expected {len(data['bom_components'])} CONTAINS edges, got {len(contains_edges)}"
    )
    for comp in data["bom_components"]:
        expected_edge = (f"BOM:{comp['bom_id']}", f"RawMaterial:{comp['consumed_product_id']}")
        assert expected_edge in contains_edges, (
            f"Missing CONTAINS edge: {expected_edge}"
        )

    # --- Assertion 6: Supplier nodes (Req 3.4) ---
    supplier_nodes = {nid for nid, n in nodes.items() if n["type"] == "Supplier"}
    expected_suppliers = {f"Supplier:{s['id']}" for s in data["suppliers"]}
    assert supplier_nodes == expected_suppliers, (
        f"Supplier nodes mismatch: got {supplier_nodes}, expected {expected_suppliers}"
    )

    # --- Assertion 7: SUPPLIES edges only to raw-materials (Req 3.4) ---
    supplies_edges = [(s, d) for s, d, l in edges if l == "SUPPLIES"]
    for src, dst in supplies_edges:
        assert dst.startswith("RawMaterial:"), (
            f"SUPPLIES edge targets non-RawMaterial node: {dst}"
        )

    # Verify expected SUPPLIES edges exist for raw-material supplier_product rows
    expected_supplies = set()
    for sp in data["supplier_products"]:
        product = products_by_id.get(sp["product_id"])
        if product and product["type"] == "raw-material":
            expected_supplies.add(
                (f"Supplier:{sp['supplier_id']}", f"RawMaterial:{sp['product_id']}")
            )
    actual_supplies = {(s, d) for s, d in supplies_edges}
    assert actual_supplies == expected_supplies, (
        f"SUPPLIES edges mismatch: got {actual_supplies}, expected {expected_supplies}"
    )

    # --- Assertion 8: Evidence nodes + EVIDENCES edges (Req 3.5) ---
    evidence_nodes = {nid for nid, n in nodes.items() if n["type"] == "Evidence"}
    expected_evidence = {f"Evidence:{e['id']}" for e in data["evidence"]}
    assert evidence_nodes == expected_evidence, (
        f"Evidence nodes mismatch: got {evidence_nodes}, expected {expected_evidence}"
    )

    evidences_edges = [(s, d) for s, d, l in edges if l == "EVIDENCES"]
    assert len(evidences_edges) == len(data["evidence"]), (
        f"Expected {len(data['evidence'])} EVIDENCES edges, got {len(evidences_edges)}"
    )
    for ev in data["evidence"]:
        expected_edge = (f"Evidence:{ev['id']}", f"RawMaterial:{ev['product_id']}")
        assert expected_edge in evidences_edges, (
            f"Missing EVIDENCES edge: {expected_edge}"
        )

    # --- Assertion 9: Total node count matches expectations ---
    expected_total = (
        len(data["companies"])
        + len(data["fg_products"])
        + len(data["rm_products"])
        + len(data["boms"])
        + len(data["suppliers"])
        + len(data["evidence"])
    )
    assert len(nodes) == expected_total, (
        f"Total node count mismatch: got {len(nodes)}, expected {expected_total}"
    )


# ---------------------------------------------------------------------------
# Helpers — stateful cognee client for idempotence testing
# ---------------------------------------------------------------------------

def make_stateful_cognee():
    """Return a mock cognee client that supports upsert semantics.

    - Nodes are stored in a dict keyed by node_id.
    - get_node() raises Exception on first call for an id (node doesn't exist),
      returns the stored node on subsequent calls.
    - create_node() stores a new node; update_node() overwrites fields in-place.
    - Edges are stored in a set of (src, dst, label) tuples so duplicates are
      automatically ignored, matching real cognee deduplication behaviour.
    """
    client = MagicMock()

    nodes: dict[str, dict] = {}
    edges: set[tuple[str, str, str]] = set()

    def _get_node(node_id):
        if node_id in nodes:
            return nodes[node_id]
        raise Exception("not found")

    def _create_node(node_id, node_type, fields):
        nodes[node_id] = {"type": node_type, **fields}

    def _update_node(node_id, fields):
        nodes[node_id].update(fields)

    def _add_edge(src, dst, label):
        edges.add((src, dst, label))

    client.get_node.side_effect = _get_node
    client.create_node.side_effect = _create_node
    client.update_node.side_effect = _update_node
    client.add_edge.side_effect = _add_edge

    return client, nodes, edges


# ---------------------------------------------------------------------------
# Property 9: Ingestion idempotence
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(data=bom_dataset())
def test_ingestion_idempotence_property9(data):
    """
    Feature: agnes-ai-supply-chain-manager, Property 9

    Validates: Requirements 3.7

    For any dataset ingested into cognee, ingesting the same dataset a second
    time SHALL produce the same node count and edge count as after the first
    ingestion (no duplicates created).
    """
    client, nodes, edges = make_stateful_cognee()
    products_by_id = {p["id"]: p for p in data["products"]}

    # --- First ingestion ---
    ingest_companies(client, data["companies"])
    ingest_products(client, data["products"])
    ingest_boms(client, data["boms"], data["bom_components"], products_by_id)
    ingest_suppliers(client, data["suppliers"], data["supplier_products"], products_by_id)
    ingest_evidence(client, data["evidence"])

    node_count_after_first = len(nodes)
    edge_count_after_first = len(edges)

    # --- Second ingestion (identical data) ---
    ingest_companies(client, data["companies"])
    ingest_products(client, data["products"])
    ingest_boms(client, data["boms"], data["bom_components"], products_by_id)
    ingest_suppliers(client, data["suppliers"], data["supplier_products"], products_by_id)
    ingest_evidence(client, data["evidence"])

    node_count_after_second = len(nodes)
    edge_count_after_second = len(edges)

    # --- Assertions: counts must be identical ---
    assert node_count_after_second == node_count_after_first, (
        f"Node count changed after second ingestion: "
        f"{node_count_after_first} → {node_count_after_second} (duplicates created)"
    )
    assert edge_count_after_second == edge_count_after_first, (
        f"Edge count changed after second ingestion: "
        f"{edge_count_after_first} → {edge_count_after_second} (duplicate edges created)"
    )
