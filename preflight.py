#!/usr/bin/env python3
"""
preflight.py — Validate all external service connections before running Agnes.

Checks:
  1. Environment variables loaded
  2. Gemini LLM model responds (gemini-2.5-flash)
  3. Gemini embedding model responds (gemini-embedding-001)
  4. BigQuery connection works
  5. cognee initialises

Usage:
    python3 preflight.py
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("preflight")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

failures = 0


def check(name: str, ok: bool, detail: str = ""):
    global failures
    if ok:
        logger.info("%s %s %s", PASS, name, detail)
    else:
        logger.error("%s %s %s", FAIL, name, detail)
        failures += 1


def main():
    global failures

    logger.info("=" * 50)
    logger.info("Agnes Preflight Check")
    logger.info("=" * 50)

    # 1. Env vars
    required = [
        "GCP_PROJECT", "BQ_BOM_DATASET", "BQ_ENRICHMENT_DATASET",
        "GEMINI_API_KEY", "COGNEE_API_KEY", "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID", "LLM_MODEL",
    ]
    missing = [v for v in required if not os.getenv(v)]
    check("Environment variables", not missing,
          f"missing: {', '.join(missing)}" if missing else "all set")

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    llm_model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")

    # 2. LLM model
    try:
        import litellm
        response = litellm.completion(
            model=llm_model,
            messages=[{"role": "user", "content": "Say OK"}],
            api_key=gemini_key,
            timeout=15,
        )
        text = response.choices[0].message.content.strip()[:50]
        check("LLM model", True, f"({llm_model}) responded: {text}")
    except Exception as e:
        err = str(e)[:150]
        check("LLM model", False, f"({llm_model}) {err}")

    # 3. Embedding model
    try:
        response = litellm.embedding(
            model="gemini/gemini-embedding-001",
            input=["test embedding"],
            api_key=gemini_key,
            timeout=15,
        )
        dim = len(response.data[0]["embedding"])
        check("Embedding model", True, f"(gemini-embedding-001) {dim} dimensions")
    except Exception as e:
        err = str(e)[:150]
        check("Embedding model", False, f"(gemini-embedding-001) {err}")

    # 4. BigQuery
    try:
        from google.cloud import bigquery
        project = os.getenv("GCP_PROJECT", "")
        client = bigquery.Client(project=project)
        ds = os.getenv("BQ_BOM_DATASET", "agnes_bom")
        rows = list(client.query(f"SELECT COUNT(*) as cnt FROM {ds}.company").result())
        cnt = rows[0].cnt if rows else 0
        check("BigQuery", True, f"({project}) {cnt} companies in {ds}")
    except Exception as e:
        check("BigQuery", False, str(e)[:150])

    # 5. cognee
    try:
        import cognee  # noqa: F401
        check("cognee import", True, f"v{cognee.__version__}")
    except Exception as e:
        check("cognee import", False, str(e)[:150])

    # Summary
    logger.info("=" * 50)
    if failures:
        logger.error("%d check(s) failed. Fix the issues above before running Agnes.", failures)
        sys.exit(1)
    else:
        logger.info("All checks passed! You're good to go.")
        logger.info("Next steps:")
        logger.info("  python3 run_cloud_ingestion.py  # populate cognee Cloud")
        logger.info("  Open Dify Agent app to search")
        sys.exit(0)


if __name__ == "__main__":
    main()
