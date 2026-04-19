"""
enrichment/pipeline.py — Web Enrichment Pipeline for Agnes AI Supply Chain Manager.

Scrapes supplier URLs via Apify + Playwright, extracts structured evidence
using Google Document AI, assigns confidence scores, and writes records to
BigQuery `agnes_enrichment.evidence`.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_URLS_PER_RUN = 50
SCRAPE_TIMEOUT_SECONDS = 30
LOW_CONFIDENCE_THRESHOLD = 0.4
ENRICHMENT_DATASET = "agnes_enrichment"

# Apify actor that uses Playwright for dynamic page rendering.
APIFY_ACTOR_ID = "apify/playwright-scraper"

# Fields extracted from Document AI that we care about.
EVIDENCE_FIELDS = ("supplier_name", "ingredient_name", "certifications", "price_indicators")


# ---------------------------------------------------------------------------
# 1. Scraping — Requirement 2.1, 2.4
# ---------------------------------------------------------------------------


def scrape_url(url: str, apify_client: Any, timeout: int = SCRAPE_TIMEOUT_SECONDS) -> str | None:
    """
    Trigger an Apify Playwright actor to render and extract page content.

    Retries once on HTTP error or timeout (Requirement 2.4).
    Returns the extracted HTML/text content, or None if both attempts fail.
    """
    for attempt in range(1, 3):  # attempts 1 and 2
        try:
            run = apify_client.actor(APIFY_ACTOR_ID).call(
                run_input={"startUrls": [{"url": url}], "pageFunction": _page_function()},
                timeout_secs=timeout,
            )
            dataset_items = list(
                apify_client.dataset(run["defaultDatasetId"]).iterate_items()
            )
            if dataset_items:
                return dataset_items[0].get("html") or dataset_items[0].get("text", "")
            logger.warning("scrape_url: no items returned for %s (attempt %d)", url, attempt)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "scrape_url: attempt %d failed for %s — %s: %s",
                attempt,
                url,
                type(exc).__name__,
                exc,
            )
        if attempt == 2:
            logger.error("scrape_url: marking URL as failed after 2 attempts — %s", url)
            return None
    return None  # unreachable, but satisfies type checker


def _page_function() -> str:
    """Minimal Apify page function that returns the full page HTML."""
    return (
        "async function pageFunction(context) {"
        "  return { html: document.documentElement.outerHTML };"
        "}"
    )


# ---------------------------------------------------------------------------
# 2. Extraction — Requirement 2.2
# ---------------------------------------------------------------------------


def extract_evidence(
    html_content: str,
    doc_ai_client: Any,
    processor_id: str,
) -> dict[str, Any]:
    """
    Submit page content to Google Document AI and return a structured dict.

    Guaranteed keys: supplier_name, ingredient_name, certifications, price_indicators.
    Each key maps to the extracted value (str / list[str]) or an empty default.
    """
    from google.cloud import documentai  # type: ignore[import]

    document = documentai.RawDocument(
        content=html_content.encode("utf-8"),
        mime_type="text/html",
    )
    request = documentai.ProcessRequest(name=processor_id, raw_document=document)
    result = doc_ai_client.process_document(request=request)
    doc = result.document

    extracted: dict[str, Any] = {
        "supplier_name": "",
        "ingredient_name": "",
        "certifications": [],
        "price_indicators": "",
    }

    for entity in doc.entities:
        field = entity.type_.lower().replace(" ", "_")
        if field == "supplier_name":
            extracted["supplier_name"] = entity.mention_text or ""
        elif field == "ingredient_name":
            extracted["ingredient_name"] = entity.mention_text or ""
        elif field == "certifications":
            certs = extracted["certifications"]
            if entity.mention_text:
                certs.append(entity.mention_text)
        elif field == "price_indicators":
            extracted["price_indicators"] = entity.mention_text or ""

    return extracted


# ---------------------------------------------------------------------------
# 3. Confidence scoring — Requirements 2.3, 2.5
# ---------------------------------------------------------------------------


def assign_confidence_scores(
    doc_ai_response: Any,
) -> tuple[dict[str, float], list[str]]:
    """
    Map Document AI entity confidence values to per-field scores in [0, 1].

    Returns:
        field_confidences: dict mapping field name → confidence float in [0, 1].
        low_confidence_fields: list of field names whose score < 0.4.

    Requirement 2.3: every score is in [0, 1].
    Requirement 2.5: fields with score < 0.4 are marked low_confidence.
    """
    field_confidences: dict[str, float] = {f: 0.0 for f in EVIDENCE_FIELDS}

    for entity in doc_ai_response.document.entities:
        field = entity.type_.lower().replace(" ", "_")
        if field in field_confidences:
            raw = float(entity.confidence) if entity.confidence is not None else 0.0
            # Clamp to [0, 1] to satisfy Property 6.
            field_confidences[field] = max(0.0, min(1.0, raw))

    low_confidence_fields = [
        f for f, score in field_confidences.items()
        if score < LOW_CONFIDENCE_THRESHOLD
    ]

    return field_confidences, low_confidence_fields


# ---------------------------------------------------------------------------
# 4. Evidence record assembly — Requirement 2.2 (schema)
# ---------------------------------------------------------------------------


def build_evidence_record(
    url: str,
    extracted: dict[str, Any],
    field_confidences: dict[str, float],
    low_confidence_fields: list[str],
) -> dict[str, Any]:
    """
    Assemble a full evidence record matching the BigQuery agnes_enrichment.evidence schema.

    Includes a UUID `id` and `scraped_at` UTC timestamp.
    """
    aggregate_confidence = (
        sum(field_confidences.values()) / len(field_confidences)
        if field_confidences
        else 0.0
    )

    return {
        "id": str(uuid.uuid4()),
        "supplier_url": url,
        "supplier_name": extracted.get("supplier_name", ""),
        "ingredient_name": extracted.get("ingredient_name", ""),
        "certifications": extracted.get("certifications", []),
        "price_indicators": extracted.get("price_indicators", ""),
        "confidence_score": round(aggregate_confidence, 6),
        "field_confidences": field_confidences,
        "low_confidence_fields": low_confidence_fields,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# 5. BigQuery write — Requirement 2.1
# ---------------------------------------------------------------------------


def write_evidence_to_bq(
    client: Any,
    dataset: str,
    records: list[dict[str, Any]],
) -> None:
    """
    Insert evidence records into `{dataset}.evidence` in BigQuery.

    Logs and skips rows that fail to insert; does not raise on partial failure.
    """
    if not records:
        return

    table_ref = f"{dataset}.evidence"
    errors = client.insert_rows_json(table_ref, records)
    if errors:
        for error_info in errors:
            logger.error(
                "write_evidence_to_bq: failed to insert row — %s", error_info
            )


# ---------------------------------------------------------------------------
# 6. Orchestration — Requirement 2.6
# ---------------------------------------------------------------------------


def run_enrichment(
    urls: list[str],
    apify_client: Any,
    doc_ai_client: Any,
    processor_id: str,
    bq_client: Any,
    enrichment_dataset: str,
) -> list[dict[str, Any]]:
    """
    Orchestrate enrichment for up to MAX_URLS_PER_RUN (50) supplier URLs.

    For each URL:
      1. Scrape with Apify + Playwright (1 retry on failure).
      2. Extract structured evidence via Document AI.
      3. Assign confidence scores; mark low-confidence fields.
      4. Build evidence record.
      5. Write to BigQuery.

    Returns the list of successfully built evidence records.
    Requirement 2.6: processes up to 50 URLs per run.
    """
    if len(urls) > MAX_URLS_PER_RUN:
        logger.warning(
            "run_enrichment: received %d URLs; capping at %d per run.",
            len(urls),
            MAX_URLS_PER_RUN,
        )
        urls = urls[:MAX_URLS_PER_RUN]

    records: list[dict[str, Any]] = []

    for url in urls:
        logger.info("run_enrichment: processing %s", url)

        html_content = scrape_url(url, apify_client)
        if html_content is None:
            logger.error("run_enrichment: skipping %s — scrape failed", url)
            continue

        try:
            extracted = extract_evidence(html_content, doc_ai_client, processor_id)
            doc_ai_response = _get_raw_doc_ai_response(
                html_content, doc_ai_client, processor_id
            )
            field_confidences, low_confidence_fields = assign_confidence_scores(
                doc_ai_response
            )
            record = build_evidence_record(
                url, extracted, field_confidences, low_confidence_fields
            )
            records.append(record)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "run_enrichment: error processing %s — %s: %s",
                url,
                type(exc).__name__,
                exc,
            )
            continue

    if records:
        write_evidence_to_bq(bq_client, enrichment_dataset, records)

    logger.info(
        "run_enrichment: completed — %d/%d URLs produced evidence records.",
        len(records),
        len(urls),
    )
    return records


def _get_raw_doc_ai_response(
    html_content: str,
    doc_ai_client: Any,
    processor_id: str,
) -> Any:
    """
    Re-call Document AI to obtain the raw response object needed for confidence scoring.

    In production this would be refactored to avoid the double call; kept separate
    here to preserve the clean public interface of extract_evidence and assign_confidence_scores.
    """
    from google.cloud import documentai  # type: ignore[import]

    document = documentai.RawDocument(
        content=html_content.encode("utf-8"),
        mime_type="text/html",
    )
    request = documentai.ProcessRequest(name=processor_id, raw_document=document)
    return doc_ai_client.process_document(request=request)


# ---------------------------------------------------------------------------
# 7. BigQuery schema creation — Requirements 2.7.1, 2.7.3
# ---------------------------------------------------------------------------


def create_enrichment_schema(
    client: Any,
    project: str,
    enrichment_dataset: str,
) -> None:
    """
    Create or update BigQuery tables for the enrichment pipeline.

    Idempotent — safe to call multiple times.
    Requirements: 2.7.1, 2.7.3
    """
    from google.cloud import bigquery  # type: ignore[import]

    bom_dataset = "agnes_bom"

    # ------------------------------------------------------------------
    # 1. Extend evidence table (agnes_enrichment.evidence)
    # ------------------------------------------------------------------
    evidence_schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("supplier_url", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("supplier_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ingredient_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("certifications", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("price_indicators", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("confidence_score", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("field_confidences", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("low_confidence_fields", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("scraped_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("is_active", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("last_seen_date", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("expiration_date", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("cognee_doc_id", "STRING", mode="NULLABLE"),
    ]

    evidence_table_id = f"{project}.{enrichment_dataset}.evidence"
    evidence_table = bigquery.Table(evidence_table_id, schema=evidence_schema)
    client.create_table(evidence_table, exists_ok=True)
    logger.info("Created/verified table %s", evidence_table_id)

    # Additive schema update — only append columns that don't already exist.
    # BigQuery does not allow changing a column's type, so we never overwrite
    # existing fields.
    existing_evidence = client.get_table(evidence_table_id)
    existing_field_names = {f.name for f in existing_evidence.schema}
    new_fields = [f for f in evidence_schema if f.name not in existing_field_names]
    if new_fields:
        merged_schema = list(existing_evidence.schema) + new_fields
        existing_evidence.schema = merged_schema
        client.update_table(existing_evidence, ["schema"])
        logger.info("Added %d new column(s) to %s", len(new_fields), evidence_table_id)
    else:
        logger.info("Schema for %s is up to date — no changes needed", evidence_table_id)

    # ------------------------------------------------------------------
    # 2. Create compliance_flags table (agnes_enrichment.compliance_flags)
    # ------------------------------------------------------------------
    compliance_schema = [
        bigquery.SchemaField("id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("product_id", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("requirement_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("value", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("source_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("confidence", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("is_active", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("last_seen_date", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("expiration_date", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("cognee_doc_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    ]

    compliance_table_id = f"{project}.{enrichment_dataset}.compliance_flags"
    compliance_table = bigquery.Table(compliance_table_id, schema=compliance_schema)
    client.create_table(compliance_table, exists_ok=True)
    logger.info("Created/verified table %s", compliance_table_id)

    # ------------------------------------------------------------------
    # 3. Create enrichment_rejected table (agnes_enrichment.enrichment_rejected)
    # ------------------------------------------------------------------
    rejected_schema = [
        bigquery.SchemaField("product_id", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("source_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("rejection_reason", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("rejection_step", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("attempted_at", "TIMESTAMP", mode="NULLABLE"),
    ]

    rejected_table_id = f"{project}.{enrichment_dataset}.enrichment_rejected"
    rejected_table = bigquery.Table(rejected_table_id, schema=rejected_schema)
    client.create_table(rejected_table, exists_ok=True)
    logger.info("Created/verified table %s", rejected_table_id)

    # ------------------------------------------------------------------
    # 4. Extend supplier table (agnes_bom.supplier) with flag_for_review
    # ------------------------------------------------------------------
    supplier_table_id = f"{project}.{bom_dataset}.supplier"
    try:
        supplier_table = client.get_table(supplier_table_id)
        existing_field_names = {field.name for field in supplier_table.schema}
        if "flag_for_review" not in existing_field_names:
            new_schema = list(supplier_table.schema) + [
                bigquery.SchemaField("flag_for_review", "BOOL", mode="NULLABLE"),
            ]
            supplier_table.schema = new_schema
            client.update_table(supplier_table, ["schema"])
            logger.info(
                "Added flag_for_review column to table %s", supplier_table_id
            )
        else:
            logger.info(
                "Column flag_for_review already exists in %s — skipping", supplier_table_id
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not update supplier table %s — %s: %s",
            supplier_table_id,
            type(exc).__name__,
            exc,
        )


# ---------------------------------------------------------------------------
# 8. Source targeting — Requirements 2.5.1, 2.5.2, 2.5.3
# ---------------------------------------------------------------------------

import urllib.parse


def construct_deterministic_url(ingredient_e_number: str, source: str) -> str:
    """
    Construct an authoritative regulatory URL from an E-number and source name.

    Supported sources: efsa, fda_gras, fda_food_additives, rspo, codex.
    Raises ValueError for unknown sources.

    Requirements: 2.5.1
    """
    e_number = urllib.parse.quote(ingredient_e_number)

    url_patterns: dict[str, str] = {
        "efsa": f"https://www.efsa.europa.eu/en/search/site/{e_number}",
        "fda_gras": (
            f"https://www.accessdata.fda.gov/scripts/fdcc/?set=GRAS&sort=Substance"
            f"&order=ASC&startrow=1&type=basic&search={e_number}"
        ),
        "fda_food_additives": (
            f"https://www.accessdata.fda.gov/scripts/fdcc/?set=FoodSubstances&sort=Substance"
            f"&order=ASC&startrow=1&type=basic&search={e_number}"
        ),
        "rspo": f"https://rspo.org/search/?q={e_number}",
        "codex": (
            f"https://www.fao.org/fao-who-codexalimentarius/codex-texts/dbs/gsfa/en/"
            f"?lang=en&q={e_number}"
        ),
    }

    if source not in url_patterns:
        raise ValueError(f"Unknown source: {source}")

    return url_patterns[source]


def build_search_query(supplier_name: str, ingredient_name: str) -> str:
    """
    Build a domain-constrained Google-style search query against a trusted allowlist.

    Trusted domains: efsa.europa.eu, fda.gov, rspo.org, eur-lex.europa.eu,
    ecocert.com, non-gmoverified.org

    Requirements: 2.5.2
    """
    trusted_domains = (
        "efsa.europa.eu OR fda.gov OR rspo.org OR eur-lex.europa.eu"
        " OR ecocert.com OR non-gmoverified.org"
    )
    return (
        f'"{ingredient_name}" "{supplier_name}" site:({trusted_domains})'
    )


def get_supplier_names_from_bq(bq_client: Any, product_id: int) -> list[str]:
    """
    Pull supplier names from BigQuery for a given ingredient product_id.

    Returns list of supplier name strings.
    On any exception, logs the error and returns an empty list.

    Requirements: 2.5.3
    """
    from google.cloud import bigquery  # type: ignore[import]

    query = """
        SELECT s.Name
        FROM `agnes_bom.supplier_product` sp
        JOIN `agnes_bom.supplier` s ON sp.SupplierId = s.Id
        WHERE sp.ProductId = @product_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("product_id", "INT64", product_id)
        ]
    )

    try:
        results = bq_client.query(query, job_config=job_config).result()
        return [row.Name for row in results]
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "get_supplier_names_from_bq: failed for product_id=%d — %s: %s",
            product_id,
            type(exc).__name__,
            exc,
        )
        return []


def resolve_urls_for_ingredient(ingredient: dict, bq_client: Any) -> list[dict]:
    """
    Return an ordered list of URL dicts to fetch for an ingredient.

    Tier 1 (deterministic, regulatory) URLs come before Tier 2 (search-driven) URLs.

    ingredient keys: id (int), sku (str), canonical_category (str|None),
                     e_number (str|None, optional).

    Returns list of dicts with keys: url (str), tier (int), source (str).

    Requirements: 2.5.1, 2.5.2, 2.5.3
    """
    results: list[dict] = []

    # Tier 1 — deterministic regulatory URLs (only when e_number is present)
    e_number = ingredient.get("e_number") or None
    if e_number:
        for source in ["efsa", "fda_gras", "fda_food_additives", "rspo", "codex"]:
            url = construct_deterministic_url(e_number, source)
            results.append({"url": url, "tier": 1, "source": source})

    # Tier 2 — search-driven URLs
    supplier_names = get_supplier_names_from_bq(bq_client, ingredient["id"])
    if supplier_names:
        for supplier_name in supplier_names[:3]:
            query = build_search_query(supplier_name, ingredient["sku"])
            results.append({"url": query, "tier": 2, "source": "search"})
    else:
        query = build_search_query("", ingredient["sku"])
        results.append({"url": query, "tier": 2, "source": "search"})

    return results


# ---------------------------------------------------------------------------
# 9. Relevance validation — Requirements 2.6.1, 2.6.2, 2.6.3, 2.6.4
# ---------------------------------------------------------------------------

_COMPLIANCE_TERMS = {
    "specification", "certification", "allergen", "additive", "e-number",
    "gras", "organic", "non-gmo", "purity", "grade", "regulatory",
    "approved", "permitted", "restriction",
}


def passes_keyword_check(extracted_text: str, sku_name: str) -> bool:
    """
    Returns True if BOTH conditions are met:
    1. At least 50% of SKU tokens are present in extracted_text (case-insensitive).
       If sku_name is empty or has no tokens, this condition is considered met.
    2. At least 1 compliance term from the known set is present in extracted_text
       (case-insensitive).

    Requirements: 2.6.1
    """
    text_lower = extracted_text.lower()

    # Condition 1: SKU token coverage
    sku_tokens = sku_name.lower().split()
    if sku_tokens:
        matched = sum(1 for token in sku_tokens if token in text_lower)
        if matched / len(sku_tokens) < 0.5:
            return False

    # Condition 2: at least one compliance term present
    if not any(term in text_lower for term in _COMPLIANCE_TERMS):
        return False

    return True


def classify_with_gemini_flash(
    extracted_text: str,
    ingredient_name: str,
    supplier_name: str,
    llm_client: Any,
) -> dict:
    """
    Invoke an LLM client with a binary classification prompt.

    On any exception (API unavailable, JSON parse error, etc.) returns a
    fail-open result so we don't miss important data.

    Requirements: 2.6.2
    """
    import json

    prompt = (
        f'You are a compliance document classifier. Determine if the following text is '
        f'relevant to the ingredient "{ingredient_name}" from supplier "{supplier_name}" '
        f'for regulatory/compliance purposes.\n\n'
        f'Text: {extracted_text[:2000]}\n\n'
        f'Respond with a JSON object with these exact keys:\n'
        f'- is_relevant: boolean\n'
        f'- relevance_reason: string (brief explanation)\n'
        f'- source_type: string (one of: "regulatory", "certification", "supplier_spec", "news", "other")\n'
        f'- ingredient_mentioned_explicitly: boolean\n\n'
        f'JSON response only, no other text.'
    )

    _fail_open = {
        "is_relevant": True,
        "relevance_reason": "fail-open",
        "source_type": "other",
        "ingredient_mentioned_explicitly": False,
    }

    try:
        response = llm_client.generate_content(prompt)
        return json.loads(response.text)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "classify_with_gemini_flash: error — %s: %s", type(exc).__name__, exc
        )
        return _fail_open


def validate_relevance(
    url: str,
    extracted_text: str,
    sku_name: str,
    ingredient_name: str,
    supplier_name: str,
    llm_client: Any,
) -> tuple[bool, str, str]:
    """
    Run both validation steps in sequence.

    Returns (passed, rejection_reason, rejection_step).

    Requirements: 2.6.3
    """
    # Step 1: keyword check
    if not passes_keyword_check(extracted_text, sku_name):
        return (
            False,
            "keyword check failed: insufficient SKU tokens or compliance terms",
            "keyword_check",
        )

    # Step 2: LLM classification
    result = classify_with_gemini_flash(extracted_text, ingredient_name, supplier_name, llm_client)
    if not result.get("is_relevant", True):
        return (
            False,
            result.get("relevance_reason", "LLM classified as not relevant"),
            "llm_classification",
        )

    return (True, "", "")


def log_rejection(
    bq_client: Any,
    product_id: int,
    url: str,
    rejection_reason: str,
    rejection_step: str,
) -> None:
    """
    Write a rejection record to the enrichment_rejected BigQuery table.

    On any insert error, logs and continues without raising.

    Requirements: 2.6.4
    """
    record = {
        "product_id": product_id,
        "source_url": url,
        "rejection_reason": rejection_reason,
        "rejection_step": rejection_step,
        "attempted_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        errors = bq_client.insert_rows_json(
            f"{ENRICHMENT_DATASET}.enrichment_rejected", [record]
        )
        if errors:
            logger.error(
                "log_rejection: failed to insert rejection record — %s", errors
            )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "log_rejection: exception inserting rejection record — %s: %s",
            type(exc).__name__,
            exc,
        )


# ---------------------------------------------------------------------------
# 10. Temporal diffing — Requirements 2.8.1, 2.8.2, 2.8.3, 2.8.4, 2.8.5
# ---------------------------------------------------------------------------

RETIREMENT_STRATEGY: dict[str, str] = {
    "regulatory": "expire",
    "certification": "expire",
    "supplier_spec": "forget",
    "news": "forget",
    "other": "forget",
}

RED_FLAG_KEYWORDS: frozenset[str] = frozenset({
    "recall", "withdrawn", "suspended", "enforcement", "plant closure",
    "capacity reduction", "force majeure", "violation", "fda warning", "lawsuit",
})


def run_net_new_diff(bq_client: Any, cognee_client: Any, scraped_records: list[dict]) -> None:
    """
    For each scraped record, insert it into BQ if the URL is new, or update
    last_seen_date if it already exists.

    Requirements: 2.8.1, 2.8.2
    """
    from google.cloud import bigquery  # type: ignore[import]

    for record in scraped_records:
        url = record.get("supplier_url", "")
        try:
            select_sql = (
                "SELECT id, cognee_doc_id FROM agnes_enrichment.evidence "
                "WHERE supplier_url = @url AND is_active = TRUE LIMIT 1"
            )
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("url", "STRING", url)
                ]
            )
            rows = list(bq_client.query(select_sql, job_config=job_config).result())

            if not rows:
                # URL not in BQ — insert new record
                record["is_active"] = True
                record["last_seen_date"] = datetime.now(timezone.utc).isoformat()
                try:
                    doc_id = cognee_client.add(url, dataset_name="enrichment")
                    record["cognee_doc_id"] = doc_id if doc_id is not None else None
                except Exception as cognee_exc:  # noqa: BLE001
                    logger.error(
                        "run_net_new_diff: cognee.add failed for %s — %s: %s",
                        url,
                        type(cognee_exc).__name__,
                        cognee_exc,
                    )
                    record["cognee_doc_id"] = None
                bq_client.insert_rows_json(f"{ENRICHMENT_DATASET}.evidence", [record])
                logger.info("run_net_new_diff: INSERT new record for %s", url)
            else:
                # URL already in BQ — update last_seen_date
                update_sql = (
                    "UPDATE agnes_enrichment.evidence "
                    "SET last_seen_date = CURRENT_TIMESTAMP() "
                    "WHERE supplier_url = @url AND is_active = TRUE"
                )
                update_job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("url", "STRING", url)
                    ]
                )
                bq_client.query(update_sql, job_config=update_job_config).result()
                logger.info("run_net_new_diff: UPDATE last_seen_date for %s", url)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "run_net_new_diff: error processing %s — %s: %s",
                url,
                type(exc).__name__,
                exc,
            )


def run_ghost_diff(bq_client: Any) -> list[dict]:
    """
    Query BQ for records that were NOT seen in the current scrape run (ghost records).

    Returns list of row dicts. On exception, logs error and returns empty list.

    Requirements: 2.8.3
    """
    query = """
        SELECT id, supplier_url, cognee_doc_id, confidence_score
        FROM agnes_enrichment.evidence
        WHERE is_active = TRUE
          AND DATE(last_seen_date) < CURRENT_DATE()
    """
    try:
        rows = list(bq_client.query(query).result())
        return [dict(row) for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "run_ghost_diff: error querying ghost records — %s: %s",
            type(exc).__name__,
            exc,
        )
        return []


def retire_ghost_record(bq_client: Any, cognee_client: Any, ghost: dict) -> None:
    """
    Retire a ghost record by marking it inactive in BQ and applying the
    appropriate cognee retirement strategy.

    Requirements: 2.8.4, 2.8.5
    """
    from google.cloud import bigquery  # type: ignore[import]

    try:
        # Step 1: mark inactive in BQ
        update_sql = (
            "UPDATE agnes_enrichment.evidence "
            "SET is_active = FALSE, expiration_date = CURRENT_TIMESTAMP() "
            "WHERE id = @id"
        )
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("id", "STRING", ghost.get("id"))
            ]
        )
        bq_client.query(update_sql, job_config=job_config).result()

        # Step 2: determine retirement strategy
        source_type = ghost.get("source_type", "other")
        strategy = RETIREMENT_STRATEGY.get(source_type, "forget")
        cognee_doc_id = ghost.get("cognee_doc_id")

        # Step 3: apply strategy
        if strategy == "forget":
            if cognee_doc_id is None:
                logger.warning(
                    "retire_ghost_record: cognee_doc_id is None for ghost id=%s — skipping forget",
                    ghost.get("id"),
                )
            else:
                cognee_client.forget(cognee_doc_id)
        elif strategy == "expire":
            if cognee_doc_id is None:
                logger.warning(
                    "retire_ghost_record: cognee_doc_id is None for ghost id=%s — skipping expire",
                    ghost.get("id"),
                )
            else:
                cognee_client.update_node_status(cognee_doc_id, status="expired")

        logger.info(
            "retire_ghost_record: retired ghost id=%s via strategy=%s",
            ghost.get("id"),
            strategy,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "retire_ghost_record: error retiring ghost id=%s — %s: %s",
            ghost.get("id"),
            type(exc).__name__,
            exc,
        )


def run_temporal_diff(
    bq_client: Any, cognee_client: Any, scraped_records: list[dict]
) -> None:
    """
    Orchestrate the full temporal diff: insert/update new records, then retire ghosts.

    Requirements: 2.8.1–2.8.5
    """
    run_net_new_diff(bq_client, cognee_client, scraped_records)

    ghosts = run_ghost_diff(bq_client)

    for ghost in ghosts:
        retire_ghost_record(bq_client, cognee_client, ghost)

    logger.info(
        "run_temporal_diff: processed %d scraped records; retired %d ghost records.",
        len(scraped_records),
        len(ghosts),
    )

# ---------------------------------------------------------------------------
# 11. Three-tier scraping schedule — Requirements 2.9.1–2.9.5
# ---------------------------------------------------------------------------


def run_tier1_heavy_scrape(
    bq_client: Any,
    apify_client: Any,
    doc_ai_client: Any,
    cognee_client: Any,
) -> None:
    """
    Full re-scrape of all supplier pages + PDFs + Document AI.

    Queries all active supplier URLs from BQ, runs enrichment, then rebuilds
    the cognee knowledge graph.

    Requirements: 2.9.1, 2.9.2
    """
    from google.cloud import bigquery  # type: ignore[import]

    logger.info("run_tier1_heavy_scrape: starting full heavy scrape")

    try:
        query = "SELECT DISTINCT supplier_url FROM agnes_enrichment.evidence WHERE is_active = TRUE"
        rows = list(bq_client.query(query).result())
        urls = [row.supplier_url for row in rows]

        logger.info("run_tier1_heavy_scrape: found %d active supplier URLs", len(urls))

        run_enrichment(
            urls,
            apify_client,
            doc_ai_client,
            processor_id="",
            bq_client=bq_client,
            enrichment_dataset=ENRICHMENT_DATASET,
        )

        cognee_client.cognify()

        logger.info("run_tier1_heavy_scrape: completed successfully")

    except Exception as exc:
        logger.error(
            "run_tier1_heavy_scrape: failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        raise


def run_tier2_expiration_check(
    bq_client: Any,
    apify_client: Any,
    doc_ai_client: Any,
    cognee_client: Any,
) -> None:
    """
    Targeted re-scrape of compliance flags expiring within 14 days.

    Requirements: 2.9.3
    """
    logger.info("run_tier2_expiration_check: starting expiration check")

    try:
        query = """
            SELECT DISTINCT source_url
            FROM agnes_enrichment.compliance_flags
            WHERE is_active = TRUE
              AND expiration_date IS NOT NULL
              AND DATE(expiration_date) <= DATE_ADD(CURRENT_DATE(), INTERVAL 14 DAY)
        """
        rows = list(bq_client.query(query).result())
        urls = [row.source_url for row in rows]

        if not urls:
            logger.info("run_tier2_expiration_check: no expiring compliance flags")
            return

        logger.info("run_tier2_expiration_check: processing %d expiring URLs", len(urls))

        run_enrichment(
            urls,
            apify_client,
            doc_ai_client,
            processor_id="",
            bq_client=bq_client,
            enrichment_dataset=ENRICHMENT_DATASET,
        )

        logger.info(
            "run_tier2_expiration_check: completed — %d URLs processed", len(urls)
        )

    except Exception as exc:
        logger.error(
            "run_tier2_expiration_check: failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        raise


def run_tier3_pulse_check(
    bq_client: Any,
    apify_client: Any,
) -> None:
    """
    Checks News/Updates/Press pages of tracked suppliers for red flag keywords.

    Sets flag_for_review = TRUE on affected supplier records in BigQuery.

    Requirements: 2.9.4, 2.9.5
    """
    from google.cloud import bigquery  # type: ignore[import]

    logger.info("run_tier3_pulse_check: starting pulse check")

    try:
        query = """
            SELECT DISTINCT s.id as supplier_id, s.Name as supplier_name
            FROM agnes_bom.supplier s
            WHERE s.flag_for_review IS NULL OR s.flag_for_review = FALSE
        """
        rows = list(bq_client.query(query).result())
        suppliers = [{"supplier_id": row.supplier_id, "supplier_name": row.supplier_name} for row in rows]

        total_checked = len(suppliers)
        total_flagged = 0

        for supplier in suppliers:
            supplier_id = supplier["supplier_id"]
            supplier_name = supplier["supplier_name"]

            news_url = (
                f"https://www.google.com/search?q="
                f"{urllib.parse.quote(supplier_name)}+news+recall+violation"
            )

            content = scrape_url(news_url, apify_client)
            if content is None:
                continue

            content_lower = content.lower()
            if any(keyword in content_lower for keyword in RED_FLAG_KEYWORDS):
                update_sql = (
                    "UPDATE agnes_bom.supplier SET flag_for_review = TRUE WHERE id = @supplier_id"
                )
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("supplier_id", "INT64", supplier_id)
                    ]
                )
                bq_client.query(update_sql, job_config=job_config).result()
                total_flagged += 1
                logger.warning(
                    "Tier 3 pulse check: red flag detected for supplier %s (id=%s)",
                    supplier_name,
                    supplier_id,
                )

        logger.info(
            "run_tier3_pulse_check: completed — %d suppliers checked, %d flagged",
            total_checked,
            total_flagged,
        )

    except Exception as exc:
        logger.error(
            "run_tier3_pulse_check: failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        raise
