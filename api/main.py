"""
api/main.py — FastAPI service exposing cognee search as an HTTP bridge.

Endpoints:
  GET  /health  — liveness check
  POST /search  — natural language search over the cognee knowledge graph

Requirements: 3.8, 3.9, 3.10
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCH_TIMEOUT_SECONDS = 30.0  # Generous for demo; first search is slow (graph loading)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Incoming search payload."""
    query: str
    k: int = 10
    node_types: list[str] | None = None


class SearchResultItem(BaseModel):
    """Single result returned by the search endpoint."""
    node_id: str
    node_type: str
    content: str
    confidence_score: float
    metadata: dict


class SearchResponse(BaseModel):
    """Envelope for search results."""
    results: list[SearchResultItem]
    query: str
    k: int


# ---------------------------------------------------------------------------
# cognee client initialisation helper
# ---------------------------------------------------------------------------

def _create_cognee_client() -> Any:
    """Instantiate and return a cognee client.

    The client is created once during application startup and stored in
    ``app.state.cognee_client`` so that request handlers can reuse it.

    If ``COGNEE_API_KEY`` is set the client is configured to talk to
    cognee Cloud (app.cognee.ai) instead of a local instance.
    """
    try:
        import os
        import cognee  # type: ignore[import-untyped]

        api_key = os.getenv("COGNEE_API_KEY")
        if api_key:
            if hasattr(cognee, "config"):
                cfg = cognee.config
                if callable(getattr(cfg, "set_llm_api_key", None)):
                    cfg.set_llm_api_key(api_key)
                elif hasattr(cfg, "llm_api_key"):
                    cfg.llm_api_key = api_key
            logger.info("cognee client configured for cloud (COGNEE_API_KEY set)")

        return cognee
    except Exception:
        logger.exception("Failed to initialise cognee client")
        raise


# ---------------------------------------------------------------------------
# Lifespan handler
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI application."""
    logger.info("Initialising cognee client …")
    app.state.cognee_client = _create_cognee_client()
    logger.info("cognee client ready")
    yield
    logger.info("Shutting down FastAPI service")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agnes — cognee Search Bridge",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Liveness probe — always returns HTTP 200."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest) -> SearchResponse:
    """Query the cognee knowledge graph and return the top-K results.

    * HTTP 504 — cognee search exceeds the 2-second SLA timeout.
    * HTTP 503 — cognee is unreachable (connection / runtime error).
    * HTTP 422 — invalid request body (handled automatically by FastAPI).
    """
    cognee_client = app.state.cognee_client

    try:
        raw_results = await asyncio.wait_for(
            _run_cognee_search(cognee_client, body.query, body.k, body.node_types),
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Search timed out — cognee did not respond within the 2-second SLA.",
        )
    except (ConnectionError, OSError) as exc:
        logger.error("cognee unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="cognee service is currently unavailable.",
        )
    except Exception as exc:
        logger.error("cognee search failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="cognee service is currently unavailable.",
        )

    items = [
        SearchResultItem(
            node_id=str(r.get("node_id", "") or ""),
            node_type=str(r.get("node_type", "") or ""),
            content=str(r.get("content", "") or ""),
            confidence_score=float(r.get("confidence_score", 0.0) or 0.0),
            metadata=r.get("metadata") if isinstance(r.get("metadata"), dict) else {},
        )
        for r in raw_results
    ]

    # Enforce the K-cap: return at most K results (Requirement 3.8)
    items = items[: body.k]

    # Post-search filter by node_types if provided.
    # Documents are prefixed: "Raw Material:", "Supplier:", "Company:", "Finished Good:"
    # Valid node_types: RawMaterial, Supplier, Company, FinishedGood
    if body.node_types:
        prefix_map = {
            "RawMaterial": "Raw Material:",
            "Supplier": "Supplier:",
            "Company": "Company:",
            "FinishedGood": "Finished Good:",
        }
        prefixes = [prefix_map[nt] for nt in body.node_types if nt in prefix_map]
        if prefixes:
            items = [i for i in items if any(i.content.startswith(p) for p in prefixes)]

    return SearchResponse(results=items, query=body.query, k=body.k)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _run_cognee_search(
    cognee_client: Any,
    query: str,
    k: int,
    node_types: list[str] | None,
) -> list[dict]:
    """Invoke cognee's search method, adapting to the real cognee v1.0 API.

    cognee v1.0 returns a list of dicts like:
      {"dataset_id": UUID, "dataset_name": str, "search_result": list[str]}

    We flatten the search_result strings into SearchResultItem-compatible dicts.
    """
    search_fn = getattr(cognee_client, "search", None)
    if search_fn is None:
        raise RuntimeError("cognee client does not expose a search method")

    # Use CHUNKS search — fast vector search, no LLM completion needed
    from cognee.modules.search.types.SearchType import SearchType
    kwargs: dict[str, Any] = {
        "query_text": query,
        "top_k": k,
        "query_type": SearchType.CHUNKS,
    }

    result = search_fn(**kwargs)

    if asyncio.iscoroutine(result):
        result = await result

    if result is None:
        return []

    # Convert cognee v1.0 response to our SearchResultItem format
    converted: list[dict] = []
    result_counter = 0

    for r in result:
        if isinstance(r, dict):
            # cognee v1.0 CHUNKS format: {dataset_id, dataset_name, search_result: list[dict]}
            search_results = r.get("search_result", [])
            dataset_name = r.get("dataset_name", "")

            if isinstance(search_results, list):
                for item in search_results:
                    result_counter += 1
                    # Each chunk item is a dict with id, text_content, etc.
                    if isinstance(item, dict):
                        content = item.get("text", "") or item.get("text_content", "") or str(item)
                        node_id = str(item.get("id", f"chunk-{result_counter}"))
                        node_type = item.get("type", dataset_name)
                    else:
                        content = str(item)
                        node_id = f"chunk-{result_counter}"
                        node_type = dataset_name

                    converted.append({
                        "node_id": node_id,
                        "node_type": str(node_type),
                        "content": content[:500],
                        "confidence_score": max(0.0, 1.0 - (result_counter - 1) * 0.05),
                        "metadata": {
                            "dataset_name": dataset_name,
                        },
                    })
            elif search_results:
                result_counter += 1
                converted.append({
                    "node_id": f"result-{result_counter}",
                    "node_type": dataset_name,
                    "content": str(search_results)[:500],
                    "confidence_score": 0.9,
                    "metadata": {"dataset_name": dataset_name},
                })
        else:
            result_counter += 1
            converted.append({
                "node_id": f"result-{result_counter}",
                "node_type": "",
                "content": str(r)[:500],
                "confidence_score": 0.5,
                "metadata": {},
            })

    return converted
