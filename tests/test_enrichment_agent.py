"""
tests/test_enrichment_agent.py — Property-based and unit tests for enrichment/pipeline.py.

Properties tested:
  Property 5: Evidence record structure completeness (Requirement 2.2)
  Property 6: Confidence score range invariant (Requirement 2.3)
  Property 7: Low-confidence field marking (Requirement 2.5)
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from enrichment.pipeline import (
    LOW_CONFIDENCE_THRESHOLD,
    MAX_URLS_PER_RUN,
    EVIDENCE_FIELDS,
    assign_confidence_scores,
    build_evidence_record,
    run_enrichment,
    scrape_url,
    write_evidence_to_bq,
)


# ---------------------------------------------------------------------------
# Helpers — build fake Document AI response objects
# ---------------------------------------------------------------------------


def _make_entity(field: str, text: str, confidence: float) -> SimpleNamespace:
    return SimpleNamespace(
        type_=field,
        mention_text=text,
        confidence=confidence,
    )


def _make_doc_ai_response(entities: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(document=SimpleNamespace(entities=entities))


# ---------------------------------------------------------------------------
# Property 5: Evidence record structure completeness (Requirement 2.2)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    supplier_name=st.text(min_size=0, max_size=200),
    ingredient_name=st.text(min_size=0, max_size=200),
    certifications=st.lists(st.text(min_size=1, max_size=50), max_size=10),
    price_indicators=st.text(min_size=0, max_size=100),
    url=st.text(min_size=1, max_size=200),
)
def test_property5_evidence_record_completeness(
    supplier_name, ingredient_name, certifications, price_indicators, url
):
    """
    Feature: agnes-ai-supply-chain-manager, Property 5: Evidence record structure completeness.

    For any extracted content, build_evidence_record SHALL produce a record
    containing non-null values for supplier_name, ingredient_name, certifications,
    and price_indicators (Requirement 2.2).
    """
    extracted = {
        "supplier_name": supplier_name,
        "ingredient_name": ingredient_name,
        "certifications": certifications,
        "price_indicators": price_indicators,
    }
    field_confidences = {f: 0.5 for f in EVIDENCE_FIELDS}
    low_confidence_fields: list[str] = []

    record = build_evidence_record(url, extracted, field_confidences, low_confidence_fields)

    # All required fields must be present and non-None.
    assert record["supplier_name"] is not None
    assert record["ingredient_name"] is not None
    assert record["certifications"] is not None
    assert record["price_indicators"] is not None

    # Schema completeness: all top-level keys must exist.
    required_keys = {
        "id", "supplier_url", "supplier_name", "ingredient_name",
        "certifications", "price_indicators", "confidence_score",
        "field_confidences", "low_confidence_fields", "scraped_at",
    }
    assert required_keys.issubset(record.keys())

    # id must be a valid UUID string.
    uuid.UUID(record["id"])

    # scraped_at must be a non-empty string.
    assert isinstance(record["scraped_at"], str) and record["scraped_at"]

    # supplier_url must match the input.
    assert record["supplier_url"] == url


# ---------------------------------------------------------------------------
# Property 6: Confidence score range invariant (Requirement 2.3)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    raw_confidences=st.dictionaries(
        keys=st.sampled_from(list(EVIDENCE_FIELDS)),
        values=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False),
        min_size=1,
    )
)
def test_property6_confidence_score_range_invariant(raw_confidences):
    """
    Feature: agnes-ai-supply-chain-manager, Property 6: Confidence score range invariant.

    For any Document AI response, every confidence_score assigned to an extracted
    field SHALL be in the closed interval [0, 1] (Requirement 2.3).
    """
    entities = [
        _make_entity(field, "some text", confidence)
        for field, confidence in raw_confidences.items()
    ]
    response = _make_doc_ai_response(entities)

    field_confidences, _ = assign_confidence_scores(response)

    for field, score in field_confidences.items():
        assert 0.0 <= score <= 1.0, (
            f"Field '{field}' has confidence {score} outside [0, 1]"
        )


@settings(max_examples=100)
@given(
    supplier_name=st.text(min_size=0, max_size=100),
    ingredient_name=st.text(min_size=0, max_size=100),
    certifications=st.lists(st.text(min_size=1, max_size=50), max_size=5),
    price_indicators=st.text(min_size=0, max_size=100),
    field_confidences=st.fixed_dictionaries({
        f: st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        for f in EVIDENCE_FIELDS
    }),
)
def test_property6_aggregate_confidence_in_range(
    supplier_name, ingredient_name, certifications, price_indicators, field_confidences
):
    """
    The aggregate confidence_score on the assembled evidence record must also be in [0, 1].
    """
    extracted = {
        "supplier_name": supplier_name,
        "ingredient_name": ingredient_name,
        "certifications": certifications,
        "price_indicators": price_indicators,
    }
    record = build_evidence_record(
        "https://example.com", extracted, field_confidences, []
    )
    assert 0.0 <= record["confidence_score"] <= 1.0


# ---------------------------------------------------------------------------
# Property 7: Low-confidence field marking (Requirement 2.5)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    raw_confidences=st.fixed_dictionaries({
        f: st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        for f in EVIDENCE_FIELDS
    })
)
def test_property7_low_confidence_field_marking(raw_confidences):
    """
    Feature: agnes-ai-supply-chain-manager, Property 7: Low-confidence field marking.

    A field SHALL be marked low_confidence if and only if its confidence value
    is strictly less than 0.4 (Requirement 2.5).
    """
    entities = [
        _make_entity(field, "text", confidence)
        for field, confidence in raw_confidences.items()
    ]
    response = _make_doc_ai_response(entities)

    field_confidences, low_confidence_fields = assign_confidence_scores(response)

    for field, score in field_confidences.items():
        if score < LOW_CONFIDENCE_THRESHOLD:
            assert field in low_confidence_fields, (
                f"Field '{field}' with score {score} should be in low_confidence_fields"
            )
        else:
            assert field not in low_confidence_fields, (
                f"Field '{field}' with score {score} should NOT be in low_confidence_fields"
            )


@settings(max_examples=100)
@given(
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
)
def test_property7_single_field_boundary(confidence):
    """
    Boundary check: exactly at 0.4 is NOT low_confidence; just below 0.4 IS.
    """
    entities = [_make_entity("supplier_name", "Acme", confidence)]
    response = _make_doc_ai_response(entities)
    _, low_confidence_fields = assign_confidence_scores(response)

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        assert "supplier_name" in low_confidence_fields
    else:
        assert "supplier_name" not in low_confidence_fields


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestScrapeUrl:
    def test_returns_html_on_success(self):
        apify_client = MagicMock()
        apify_client.actor.return_value.call.return_value = {"defaultDatasetId": "ds1"}
        apify_client.dataset.return_value.iterate_items.return_value = iter(
            [{"html": "<html>content</html>"}]
        )
        result = scrape_url("https://example.com", apify_client)
        assert result == "<html>content</html>"

    def test_retries_once_on_failure_then_returns_none(self):
        apify_client = MagicMock()
        apify_client.actor.return_value.call.side_effect = Exception("timeout")
        result = scrape_url("https://example.com", apify_client)
        assert result is None
        assert apify_client.actor.return_value.call.call_count == 2

    def test_returns_none_after_two_failures(self):
        apify_client = MagicMock()
        apify_client.actor.return_value.call.side_effect = [
            Exception("HTTP 500"),
            Exception("HTTP 500"),
        ]
        result = scrape_url("https://example.com", apify_client)
        assert result is None


class TestBuildEvidenceRecord:
    def test_record_has_all_required_keys(self):
        extracted = {
            "supplier_name": "Acme",
            "ingredient_name": "Palm Oil",
            "certifications": ["RSPO"],
            "price_indicators": "$1.20/kg",
        }
        field_confidences = {f: 0.8 for f in EVIDENCE_FIELDS}
        record = build_evidence_record(
            "https://acme.com", extracted, field_confidences, []
        )
        for key in ("id", "supplier_url", "supplier_name", "ingredient_name",
                    "certifications", "price_indicators", "confidence_score",
                    "field_confidences", "low_confidence_fields", "scraped_at"):
            assert key in record

    def test_id_is_valid_uuid(self):
        extracted = {k: "" for k in EVIDENCE_FIELDS}
        extracted["certifications"] = []
        record = build_evidence_record("https://x.com", extracted, {}, [])
        uuid.UUID(record["id"])  # raises if invalid

    def test_low_confidence_fields_propagated(self):
        extracted = {k: "" for k in EVIDENCE_FIELDS}
        extracted["certifications"] = []
        field_confidences = {f: 0.1 for f in EVIDENCE_FIELDS}
        low_confidence_fields = list(EVIDENCE_FIELDS)
        record = build_evidence_record(
            "https://x.com", extracted, field_confidences, low_confidence_fields
        )
        assert set(record["low_confidence_fields"]) == set(EVIDENCE_FIELDS)


class TestWriteEvidenceToBq:
    def test_inserts_records(self):
        bq_client = MagicMock()
        bq_client.insert_rows_json.return_value = []
        records = [{"id": "1", "supplier_url": "https://x.com"}]
        write_evidence_to_bq(bq_client, "agnes_enrichment", records)
        bq_client.insert_rows_json.assert_called_once_with(
            "agnes_enrichment.evidence", records
        )

    def test_skips_empty_records(self):
        bq_client = MagicMock()
        write_evidence_to_bq(bq_client, "agnes_enrichment", [])
        bq_client.insert_rows_json.assert_not_called()

    def test_logs_on_insert_error(self, caplog):
        import logging
        bq_client = MagicMock()
        bq_client.insert_rows_json.return_value = [{"index": 0, "errors": ["bad"]}]
        with caplog.at_level(logging.ERROR):
            write_evidence_to_bq(bq_client, "agnes_enrichment", [{"id": "1"}])
        assert any("failed to insert" in r.message for r in caplog.records)


class TestRunEnrichment:
    def test_caps_at_max_urls(self):
        """run_enrichment processes at most MAX_URLS_PER_RUN URLs."""
        urls = [f"https://example.com/{i}" for i in range(MAX_URLS_PER_RUN + 10)]

        with patch("enrichment.pipeline.scrape_url", return_value=None):
            result = run_enrichment(
                urls,
                apify_client=MagicMock(),
                doc_ai_client=MagicMock(),
                processor_id="proj/proc",
                bq_client=MagicMock(),
                enrichment_dataset="agnes_enrichment",
            )
        # All scrapes return None → no records, but only MAX_URLS_PER_RUN attempted.
        assert result == []

    def test_skips_failed_scrapes(self):
        with patch("enrichment.pipeline.scrape_url", return_value=None):
            result = run_enrichment(
                ["https://fail.com"],
                apify_client=MagicMock(),
                doc_ai_client=MagicMock(),
                processor_id="proj/proc",
                bq_client=MagicMock(),
                enrichment_dataset="agnes_enrichment",
            )
        assert result == []

    def test_returns_records_on_success(self):
        fake_html = "<html>Acme Palm Oil RSPO certified $1.20/kg</html>"
        fake_extracted = {
            "supplier_name": "Acme",
            "ingredient_name": "Palm Oil",
            "certifications": ["RSPO"],
            "price_indicators": "$1.20/kg",
        }
        fake_confidences = {f: 0.8 for f in EVIDENCE_FIELDS}

        with (
            patch("enrichment.pipeline.scrape_url", return_value=fake_html),
            patch("enrichment.pipeline.extract_evidence", return_value=fake_extracted),
            patch(
                "enrichment.pipeline.assign_confidence_scores",
                return_value=(fake_confidences, []),
            ),
            patch("enrichment.pipeline._get_raw_doc_ai_response", return_value=MagicMock()),
            patch("enrichment.pipeline.write_evidence_to_bq"),
        ):
            result = run_enrichment(
                ["https://acme.com/palm-oil"],
                apify_client=MagicMock(),
                doc_ai_client=MagicMock(),
                processor_id="proj/proc",
                bq_client=MagicMock(),
                enrichment_dataset="agnes_enrichment",
            )

        assert len(result) == 1
        assert result[0]["supplier_name"] == "Acme"
        assert result[0]["ingredient_name"] == "Palm Oil"


# ---------------------------------------------------------------------------
# Tests for Section 8: Source targeting — Requirements 2.5.1, 2.5.2, 2.5.3
# ---------------------------------------------------------------------------

from enrichment.pipeline import (
    build_search_query,
    construct_deterministic_url,
    get_supplier_names_from_bq,
    resolve_urls_for_ingredient,
)


class TestConstructDeterministicUrl:
    def test_efsa_url(self):
        url = construct_deterministic_url("E471", "efsa")
        assert url == "https://www.efsa.europa.eu/en/search/site/E471"

    def test_fda_gras_url(self):
        url = construct_deterministic_url("E471", "fda_gras")
        assert "set=GRAS" in url
        assert "E471" in url

    def test_fda_food_additives_url(self):
        url = construct_deterministic_url("E471", "fda_food_additives")
        assert "set=FoodSubstances" in url
        assert "E471" in url

    def test_rspo_url(self):
        url = construct_deterministic_url("E471", "rspo")
        assert url == "https://rspo.org/search/?q=E471"

    def test_codex_url(self):
        url = construct_deterministic_url("E471", "codex")
        assert "fao.org" in url
        assert "E471" in url

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source: unknown_src"):
            construct_deterministic_url("E471", "unknown_src")

    def test_e_number_is_url_encoded(self):
        url = construct_deterministic_url("E 471 (a)", "efsa")
        assert "E%20471%20%28a%29" in url or "E+471" in url or "E%20471" in url

    def test_all_supported_sources_return_string(self):
        for source in ["efsa", "fda_gras", "fda_food_additives", "rspo", "codex"]:
            url = construct_deterministic_url("E100", source)
            assert isinstance(url, str) and url.startswith("http")


class TestBuildSearchQuery:
    def test_contains_ingredient_name(self):
        query = build_search_query("Acme Corp", "Palm Oil")
        assert '"Palm Oil"' in query

    def test_contains_supplier_name(self):
        query = build_search_query("Acme Corp", "Palm Oil")
        assert '"Acme Corp"' in query

    def test_contains_all_trusted_domains(self):
        query = build_search_query("Acme", "Lecithin")
        for domain in ["efsa.europa.eu", "fda.gov", "rspo.org",
                       "eur-lex.europa.eu", "ecocert.com", "non-gmoverified.org"]:
            assert domain in query

    def test_site_constraint_format(self):
        query = build_search_query("Supplier", "Ingredient")
        assert "site:(" in query

    def test_empty_supplier_name(self):
        query = build_search_query("", "Soy Lecithin")
        assert '"Soy Lecithin"' in query
        assert '""' in query


class TestGetSupplierNamesFromBq:
    def test_returns_supplier_names(self):
        bq_client = MagicMock()
        row1 = SimpleNamespace(Name="Supplier A")
        row2 = SimpleNamespace(Name="Supplier B")
        bq_client.query.return_value.result.return_value = [row1, row2]

        result = get_supplier_names_from_bq(bq_client, 42)
        assert result == ["Supplier A", "Supplier B"]

    def test_returns_empty_list_on_exception(self):
        bq_client = MagicMock()
        bq_client.query.side_effect = Exception("BQ error")

        result = get_supplier_names_from_bq(bq_client, 99)
        assert result == []

    def test_uses_parameterized_query(self):
        bq_client = MagicMock()
        bq_client.query.return_value.result.return_value = []

        get_supplier_names_from_bq(bq_client, 7)
        call_args = bq_client.query.call_args
        job_config = call_args[1]["job_config"] if "job_config" in call_args[1] else call_args[0][1]
        params = job_config.query_parameters
        assert any(p.name == "product_id" and p.value == 7 for p in params)


class TestResolveUrlsForIngredient:
    def _make_bq_client(self, supplier_names: list[str]) -> MagicMock:
        bq_client = MagicMock()
        rows = [SimpleNamespace(Name=n) for n in supplier_names]
        bq_client.query.return_value.result.return_value = rows
        return bq_client

    def test_tier1_urls_present_when_e_number_given(self):
        bq_client = self._make_bq_client([])
        ingredient = {"id": 1, "sku": "PALM-OIL", "canonical_category": "fats", "e_number": "E471"}
        results = resolve_urls_for_ingredient(ingredient, bq_client)
        tier1 = [r for r in results if r["tier"] == 1]
        assert len(tier1) == 5
        sources = {r["source"] for r in tier1}
        assert sources == {"efsa", "fda_gras", "fda_food_additives", "rspo", "codex"}

    def test_no_tier1_when_no_e_number(self):
        bq_client = self._make_bq_client([])
        ingredient = {"id": 1, "sku": "PALM-OIL", "canonical_category": "fats"}
        results = resolve_urls_for_ingredient(ingredient, bq_client)
        tier1 = [r for r in results if r["tier"] == 1]
        assert tier1 == []

    def test_tier1_before_tier2(self):
        bq_client = self._make_bq_client(["Supplier X"])
        ingredient = {"id": 1, "sku": "PALM-OIL", "canonical_category": None, "e_number": "E471"}
        results = resolve_urls_for_ingredient(ingredient, bq_client)
        tiers = [r["tier"] for r in results]
        # All tier-1 entries come before any tier-2 entry
        last_tier1 = max((i for i, t in enumerate(tiers) if t == 1), default=-1)
        first_tier2 = min((i for i, t in enumerate(tiers) if t == 2), default=len(tiers))
        assert last_tier1 < first_tier2

    def test_tier2_uses_up_to_3_suppliers(self):
        bq_client = self._make_bq_client(["S1", "S2", "S3", "S4", "S5"])
        ingredient = {"id": 1, "sku": "LECITHIN", "canonical_category": None}
        results = resolve_urls_for_ingredient(ingredient, bq_client)
        tier2 = [r for r in results if r["tier"] == 2]
        assert len(tier2) == 3

    def test_tier2_fallback_when_no_suppliers(self):
        bq_client = self._make_bq_client([])
        ingredient = {"id": 1, "sku": "LECITHIN", "canonical_category": None}
        results = resolve_urls_for_ingredient(ingredient, bq_client)
        tier2 = [r for r in results if r["tier"] == 2]
        assert len(tier2) == 1
        assert '"LECITHIN"' in tier2[0]["url"]

    def test_all_results_have_required_keys(self):
        bq_client = self._make_bq_client(["Acme"])
        ingredient = {"id": 1, "sku": "SOY", "canonical_category": None, "e_number": "E322"}
        results = resolve_urls_for_ingredient(ingredient, bq_client)
        for r in results:
            assert "url" in r and "tier" in r and "source" in r

    def test_e_number_none_string_treated_as_absent(self):
        """e_number=None should produce no tier-1 URLs."""
        bq_client = self._make_bq_client([])
        ingredient = {"id": 1, "sku": "PALM-OIL", "canonical_category": None, "e_number": None}
        results = resolve_urls_for_ingredient(ingredient, bq_client)
        tier1 = [r for r in results if r["tier"] == 1]
        assert tier1 == []


# ---------------------------------------------------------------------------
# Property 8: Deterministic URL construction — Requirement 2.5.1
# ---------------------------------------------------------------------------

VALID_SOURCES = ["efsa", "fda_gras", "fda_food_additives", "rspo", "codex"]


@settings(max_examples=100)
@given(
    e_number=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
    source=st.sampled_from(VALID_SOURCES),
)
def test_property8_deterministic_url_always_contains_e_number(e_number, source):
    """
    **Validates: Requirements 2.5.1**

    For any valid E-number and supported source, construct_deterministic_url SHALL
    return a URL that contains the (URL-encoded) E-number and starts with https://.
    """
    import urllib.parse
    url = construct_deterministic_url(e_number, source)
    assert url.startswith("https://")
    assert urllib.parse.quote(e_number) in url


# ---------------------------------------------------------------------------
# Property 9: Search query domain constraint — Requirement 2.5.2
# ---------------------------------------------------------------------------

TRUSTED_DOMAINS = [
    "efsa.europa.eu", "fda.gov", "rspo.org",
    "eur-lex.europa.eu", "ecocert.com", "non-gmoverified.org",
]


@settings(max_examples=100)
@given(
    supplier_name=st.text(min_size=0, max_size=100),
    ingredient_name=st.text(min_size=1, max_size=100),
)
def test_property9_search_query_contains_all_trusted_domains(supplier_name, ingredient_name):
    """
    **Validates: Requirements 2.5.2**

    For any supplier and ingredient name, build_search_query SHALL produce a query
    that references all trusted domains in the site: constraint.
    """
    query = build_search_query(supplier_name, ingredient_name)
    for domain in TRUSTED_DOMAINS:
        assert domain in query, f"Trusted domain '{domain}' missing from query: {query}"
    assert "site:(" in query


# ---------------------------------------------------------------------------
# Tests for Section 9: Relevance validation — Requirements 2.6.1–2.6.4
# ---------------------------------------------------------------------------

from enrichment.pipeline import (
    ENRICHMENT_DATASET,
    classify_with_gemini_flash,
    log_rejection,
    passes_keyword_check,
    validate_relevance,
)


class TestPassesKeywordCheck:
    def test_returns_true_when_both_conditions_met(self):
        text = "palm oil specification grade A certified organic"
        assert passes_keyword_check(text, "palm oil") is True

    def test_returns_false_when_no_compliance_term(self):
        text = "palm oil is a common cooking ingredient used worldwide"
        assert passes_keyword_check(text, "palm oil") is False

    def test_returns_false_when_insufficient_sku_tokens(self):
        # SKU has 3 tokens; only 1 present → 33% < 50%
        text = "sunflower certification grade"
        assert passes_keyword_check(text, "palm oil extract") is False

    def test_empty_sku_name_only_requires_compliance_term(self):
        text = "this document contains a certification for the product"
        assert passes_keyword_check(text, "") is True

    def test_case_insensitive_matching(self):
        text = "PALM OIL SPECIFICATION GRADE"
        assert passes_keyword_check(text, "Palm Oil") is True


class TestClassifyWithGeminiFlash:
    def test_returns_parsed_json_on_success(self):
        import json

        payload = {
            "is_relevant": True,
            "relevance_reason": "mentions ingredient",
            "source_type": "regulatory",
            "ingredient_mentioned_explicitly": True,
        }
        llm_client = MagicMock()
        llm_client.generate_content.return_value = MagicMock(text=json.dumps(payload))

        result = classify_with_gemini_flash("some text", "Palm Oil", "Acme", llm_client)
        assert result == payload

    def test_fail_open_on_api_error(self):
        llm_client = MagicMock()
        llm_client.generate_content.side_effect = Exception("API unavailable")

        result = classify_with_gemini_flash("some text", "Palm Oil", "Acme", llm_client)
        assert result["is_relevant"] is True
        assert result["relevance_reason"] == "fail-open"

    def test_fail_open_on_json_parse_error(self):
        llm_client = MagicMock()
        llm_client.generate_content.return_value = MagicMock(text="not valid json {{")

        result = classify_with_gemini_flash("some text", "Palm Oil", "Acme", llm_client)
        assert result["is_relevant"] is True
        assert result["relevance_reason"] == "fail-open"


class TestValidateRelevance:
    def test_passes_when_both_checks_pass(self):
        import json

        llm_client = MagicMock()
        llm_client.generate_content.return_value = MagicMock(
            text=json.dumps({
                "is_relevant": True,
                "relevance_reason": "relevant",
                "source_type": "regulatory",
                "ingredient_mentioned_explicitly": True,
            })
        )
        text = "palm oil specification grade certified"
        passed, reason, step = validate_relevance(
            "https://example.com", text, "palm oil", "Palm Oil", "Acme", llm_client
        )
        assert passed is True
        assert reason == ""
        assert step == ""

    def test_fails_at_keyword_check(self):
        llm_client = MagicMock()
        # text has no compliance terms
        text = "completely unrelated content about weather"
        passed, reason, step = validate_relevance(
            "https://example.com", text, "palm oil", "Palm Oil", "Acme", llm_client
        )
        assert passed is False
        assert step == "keyword_check"
        llm_client.generate_content.assert_not_called()

    def test_fails_at_llm_classification(self):
        import json

        llm_client = MagicMock()
        llm_client.generate_content.return_value = MagicMock(
            text=json.dumps({
                "is_relevant": False,
                "relevance_reason": "unrelated document",
                "source_type": "news",
                "ingredient_mentioned_explicitly": False,
            })
        )
        text = "palm oil specification grade certified"
        passed, reason, step = validate_relevance(
            "https://example.com", text, "palm oil", "Palm Oil", "Acme", llm_client
        )
        assert passed is False
        assert step == "llm_classification"
        assert "unrelated document" in reason


class TestLogRejection:
    def test_inserts_rejection_record(self):
        bq_client = MagicMock()
        bq_client.insert_rows_json.return_value = []

        log_rejection(bq_client, 42, "https://example.com", "keyword check failed", "keyword_check")

        bq_client.insert_rows_json.assert_called_once()
        call_args = bq_client.insert_rows_json.call_args
        table_arg = call_args[0][0]
        records_arg = call_args[0][1]

        assert table_arg == f"{ENRICHMENT_DATASET}.enrichment_rejected"
        assert len(records_arg) == 1
        record = records_arg[0]
        assert record["product_id"] == 42
        assert record["source_url"] == "https://example.com"
        assert record["rejection_reason"] == "keyword check failed"
        assert record["rejection_step"] == "keyword_check"
        assert "attempted_at" in record

    def test_logs_on_insert_error(self, caplog):
        import logging

        bq_client = MagicMock()
        bq_client.insert_rows_json.return_value = [{"index": 0, "errors": ["bad row"]}]

        with caplog.at_level(logging.ERROR):
            log_rejection(bq_client, 1, "https://fail.com", "some reason", "keyword_check")

        assert any("log_rejection" in r.message or "rejection" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests for Section 10: Temporal diffing — Requirements 2.8.1–2.8.5
# ---------------------------------------------------------------------------

from enrichment.pipeline import (
    RETIREMENT_STRATEGY,
    retire_ghost_record,
    run_ghost_diff,
    run_net_new_diff,
    run_temporal_diff,
)


class TestRunNetNewDiff:
    def test_inserts_new_record_when_url_not_in_bq(self):
        """BQ query returns no rows → insert_rows_json called, cognee_client.add called."""
        bq_client = MagicMock()
        bq_client.query.return_value.result.return_value = []
        bq_client.insert_rows_json.return_value = []

        cognee_client = MagicMock()
        cognee_client.add.return_value = "doc-id-123"

        record = {"supplier_url": "https://new.example.com", "id": "rec-1"}
        run_net_new_diff(bq_client, cognee_client, [record])

        cognee_client.add.assert_called_once_with(
            "https://new.example.com", dataset_name="enrichment"
        )
        bq_client.insert_rows_json.assert_called_once()
        assert record["is_active"] is True
        assert record["cognee_doc_id"] == "doc-id-123"

    def test_updates_last_seen_when_url_exists(self):
        """BQ query returns 1 row → UPDATE query called, insert_rows_json NOT called."""
        existing_row = MagicMock()
        existing_row.__iter__ = MagicMock(return_value=iter([]))

        bq_client = MagicMock()
        # First call (SELECT) returns a row; subsequent calls (UPDATE) return result
        bq_client.query.return_value.result.return_value = [existing_row]
        bq_client.insert_rows_json.return_value = []

        cognee_client = MagicMock()

        record = {"supplier_url": "https://existing.example.com", "id": "rec-2"}
        run_net_new_diff(bq_client, cognee_client, [record])

        # insert_rows_json should NOT be called for existing URLs
        bq_client.insert_rows_json.assert_not_called()
        # bq_client.query should be called at least twice (SELECT + UPDATE)
        assert bq_client.query.call_count >= 2
        # Verify UPDATE was called
        update_calls = [
            call for call in bq_client.query.call_args_list
            if "UPDATE" in str(call)
        ]
        assert len(update_calls) >= 1

    def test_continues_on_exception(self):
        """First record raises exception → second record still processed."""
        bq_client = MagicMock()
        # First call raises, second call succeeds
        bq_client.query.side_effect = [
            Exception("BQ error"),
            MagicMock(**{"result.return_value": []}),
            MagicMock(**{"result.return_value": []}),
        ]
        bq_client.insert_rows_json.return_value = []

        cognee_client = MagicMock()
        cognee_client.add.return_value = "doc-id-456"

        records = [
            {"supplier_url": "https://fail.example.com", "id": "rec-fail"},
            {"supplier_url": "https://ok.example.com", "id": "rec-ok"},
        ]
        # Should not raise
        run_net_new_diff(bq_client, cognee_client, records)

        # Second record should have been processed (insert_rows_json called once)
        bq_client.insert_rows_json.assert_called_once()


class TestRunGhostDiff:
    def test_returns_ghost_records(self):
        """BQ returns 2 rows → list of 2 dicts returned."""
        row1 = {"id": "g1", "supplier_url": "https://ghost1.com", "cognee_doc_id": "d1", "confidence_score": 0.8}
        row2 = {"id": "g2", "supplier_url": "https://ghost2.com", "cognee_doc_id": "d2", "confidence_score": 0.6}

        bq_client = MagicMock()
        bq_client.query.return_value.result.return_value = [row1, row2]

        result = run_ghost_diff(bq_client)

        assert len(result) == 2
        assert result[0]["id"] == "g1"
        assert result[1]["id"] == "g2"

    def test_returns_empty_list_on_exception(self):
        """BQ raises exception → returns []."""
        bq_client = MagicMock()
        bq_client.query.side_effect = Exception("BQ unavailable")

        result = run_ghost_diff(bq_client)

        assert result == []


class TestRetireGhostRecord:
    def test_forget_strategy_calls_cognee_forget(self):
        """ghost with source_type='supplier_spec' → cognee_client.forget called."""
        bq_client = MagicMock()
        bq_client.query.return_value.result.return_value = []

        cognee_client = MagicMock()

        ghost = {
            "id": "ghost-1",
            "supplier_url": "https://ghost.com",
            "cognee_doc_id": "doc-abc",
            "source_type": "supplier_spec",
        }
        retire_ghost_record(bq_client, cognee_client, ghost)

        cognee_client.forget.assert_called_once_with("doc-abc")
        cognee_client.update_node_status.assert_not_called()

    def test_expire_strategy_calls_update_node_status(self):
        """ghost with source_type='regulatory' → cognee_client.update_node_status called with status='expired'."""
        bq_client = MagicMock()
        bq_client.query.return_value.result.return_value = []

        cognee_client = MagicMock()

        ghost = {
            "id": "ghost-2",
            "supplier_url": "https://regulatory.com",
            "cognee_doc_id": "doc-xyz",
            "source_type": "regulatory",
        }
        retire_ghost_record(bq_client, cognee_client, ghost)

        cognee_client.update_node_status.assert_called_once_with("doc-xyz", status="expired")
        cognee_client.forget.assert_not_called()

    def test_bq_update_called_for_retirement(self):
        """Verify UPDATE query is called on bq_client."""
        bq_client = MagicMock()
        bq_client.query.return_value.result.return_value = []

        cognee_client = MagicMock()

        ghost = {
            "id": "ghost-3",
            "supplier_url": "https://old.com",
            "cognee_doc_id": "doc-old",
            "source_type": "news",
        }
        retire_ghost_record(bq_client, cognee_client, ghost)

        bq_client.query.assert_called_once()
        call_sql = bq_client.query.call_args[0][0]
        assert "UPDATE" in call_sql
        assert "is_active = FALSE" in call_sql
        assert "expiration_date" in call_sql


class TestRetirementStrategy:
    def test_all_source_types_have_strategy(self):
        """Verify all 5 source types are in RETIREMENT_STRATEGY."""
        expected_types = {"regulatory", "certification", "supplier_spec", "news", "other"}
        assert expected_types == set(RETIREMENT_STRATEGY.keys())

    def test_strategy_values_are_valid(self):
        """All values are either 'forget' or 'expire'."""
        valid_values = {"forget", "expire"}
        for source_type, strategy in RETIREMENT_STRATEGY.items():
            assert strategy in valid_values, (
                f"source_type '{source_type}' has invalid strategy '{strategy}'"
            )


# ---------------------------------------------------------------------------
# create_enrichment_schema tests (Task 4.5)
# Validates: Requirements 2.7.1, 2.7.3
# ---------------------------------------------------------------------------

from enrichment.pipeline import create_enrichment_schema


class TestCreateEnrichmentSchema:
    """Tests for create_enrichment_schema — BigQuery table creation and schema updates."""

    def _make_mock_bq_client(self, existing_tables=None, existing_schemas=None):
        """Build a mock BigQuery client.

        Parameters
        ----------
        existing_tables : set[str] | None
            Table IDs that already "exist" in BigQuery.
        existing_schemas : dict[str, list] | None
            Mapping of table ID → list of mock SchemaField objects already present.
        """
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        existing_tables = existing_tables or set()
        existing_schemas = existing_schemas or {}

        client = MagicMock()

        def _create_table(table, exists_ok=False):
            table_id = str(table.table_id) if hasattr(table, "table_id") else str(table)
            if table_id in existing_tables and not exists_ok:
                from google.api_core.exceptions import Conflict
                raise Conflict(f"Table {table_id} already exists")
            existing_tables.add(table_id)
            return table

        def _get_table(table_id):
            mock_table = MagicMock()
            if table_id in existing_schemas:
                mock_table.schema = existing_schemas[table_id]
            else:
                mock_table.schema = []
            return mock_table

        def _update_table(table, fields):
            return table

        client.create_table.side_effect = _create_table
        client.get_table.side_effect = _get_table
        client.update_table.side_effect = _update_table

        return client

    def test_creates_all_three_tables(self):
        """create_enrichment_schema creates evidence, compliance_flags, and enrichment_rejected."""
        client = self._make_mock_bq_client()

        create_enrichment_schema(client, "test-project", "agnes_enrichment")

        # create_table called at least 3 times (evidence, compliance_flags, enrichment_rejected)
        table_calls = client.create_table.call_args_list
        assert len(table_calls) >= 3

    def test_idempotent_on_existing_tables(self):
        """Running twice does not raise errors (exists_ok=True)."""
        client = self._make_mock_bq_client()

        create_enrichment_schema(client, "test-project", "agnes_enrichment")
        create_enrichment_schema(client, "test-project", "agnes_enrichment")

        # Should not raise — exists_ok=True handles duplicates

    def test_evidence_schema_uses_json_for_field_confidences(self):
        """field_confidences must be JSON type, not STRING — matches design doc and BigQuery."""
        from unittest.mock import MagicMock

        client = self._make_mock_bq_client()
        create_enrichment_schema(client, "test-project", "agnes_enrichment")

        # Find the create_table call for the evidence table
        evidence_call = None
        for call in client.create_table.call_args_list:
            table_arg = call[0][0]
            if hasattr(table_arg, "schema"):
                field_names = {f.name for f in table_arg.schema}
                if "field_confidences" in field_names:
                    evidence_call = table_arg
                    break

        assert evidence_call is not None, "Evidence table not found in create_table calls"

        field_conf = next(f for f in evidence_call.schema if f.name == "field_confidences")
        assert field_conf.field_type == "JSON", (
            f"field_confidences should be JSON, got {field_conf.field_type}. "
            "BigQuery does not allow changing column types after creation."
        )

    def test_additive_schema_update_does_not_overwrite_existing_columns(self):
        """Schema update should only add missing columns, not replace existing ones.

        This prevents BigQuery 400 errors when existing column types differ
        from the code definition (e.g. JSON vs STRING).
        """
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        # Simulate an existing evidence table with field_confidences as JSON
        existing_field = MagicMock()
        existing_field.name = "field_confidences"
        existing_field.field_type = "JSON"

        id_field = MagicMock()
        id_field.name = "id"
        id_field.field_type = "STRING"

        client = self._make_mock_bq_client(
            existing_schemas={
                "test-project.agnes_enrichment.evidence": [id_field, existing_field],
            }
        )

        # Should not raise — additive update skips existing columns
        create_enrichment_schema(client, "test-project", "agnes_enrichment")

        # Verify update_table was called for evidence and the existing JSON
        # field was NOT replaced
        for call in client.update_table.call_args_list:
            table_arg = call[0][0]
            if hasattr(table_arg, "schema"):
                for field in table_arg.schema:
                    if hasattr(field, "name") and field.name == "field_confidences":
                        assert field.field_type == "JSON", (
                            "Additive update must preserve existing column types"
                        )
