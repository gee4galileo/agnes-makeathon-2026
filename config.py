"""
config.py — Load and validate required environment variables at import time.

Raises EnvironmentError immediately if any required variable is missing,
so misconfiguration is caught early rather than at runtime.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_REQUIRED_VARS = [
    "GCP_PROJECT",
    "BQ_BOM_DATASET",
    "BQ_ENRICHMENT_DATASET",
    "APIFY_API_TOKEN",
    "DOCUMENT_AI_PROCESSOR_ID",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "VERTEX_AI_LOCATION",
    "COGNEE_DB_PATH",
    "COGNEE_API_KEY",
    "GEMINI_API_KEY",
]

_missing = [var for var in _REQUIRED_VARS if not os.getenv(var)]
if _missing:
    raise EnvironmentError(
        f"Agnes: missing required environment variable(s): {', '.join(_missing)}. "
        "Copy .env.example to .env and fill in all values."
    )

# Typed accessors
GCP_PROJECT: str = os.environ["GCP_PROJECT"]
BQ_BOM_DATASET: str = os.environ["BQ_BOM_DATASET"]
BQ_ENRICHMENT_DATASET: str = os.environ["BQ_ENRICHMENT_DATASET"]
APIFY_API_TOKEN: str = os.environ["APIFY_API_TOKEN"]
DOCUMENT_AI_PROCESSOR_ID: str = os.environ["DOCUMENT_AI_PROCESSOR_ID"]
ELEVENLABS_API_KEY: str = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID: str = os.environ["ELEVENLABS_VOICE_ID"]
VERTEX_AI_LOCATION: str = os.environ["VERTEX_AI_LOCATION"]
COGNEE_DB_PATH: str = os.environ["COGNEE_DB_PATH"]
COGNEE_API_KEY: str = os.environ["COGNEE_API_KEY"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
AGNES_FASTAPI_URL: str = os.environ.get("AGNES_FASTAPI_URL", "http://localhost:8000")
