# Agnes AI Supply Chain Manager

Built for the **tum.ai x Spherecast Makeathon 2026**.

## The Problem

CPG companies regularly overpay because sourcing is fragmented. The same ingredient may be purchased by multiple companies, plants, or product lines without anyone having full visibility into the combined demand. Suppliers don't see the true buying volume, orders aren't consolidated, and buyers lose leverage on price, lead time, and service levels.

Consolidation is only valuable if the components are actually substitutable and still compliant in the context of the end product. This is where AI creates value: it can connect fragmented purchasing data, infer which materials are functionally equivalent, verify whether quality and compliance requirements are still met, and recommend sourcing decisions that are cheaper, more scalable, and operationally realistic.

## What Agnes Does

Agnes is an AI Supply Chain Manager that helps procurement teams make better sourcing decisions by reasoning across fragmented supply chain data. Given multiple normalized Bills of Materials (BOMs), existing supplier relationships, and historical procurement decisions across several companies, Agnes determines which components are genuinely substitutable and which sourcing decisions can be consolidated.

The system works with real data: 61 CPG companies, 1025 products (149 finished goods, 876 raw materials), 40 suppliers, 1633 supplier-product relationships, and 149 BOMs. Evidence is enriched from regulatory sources (FDA, EFSA) via Apify web scraping.

## Architecture

| Layer | Service |
|---|---|
| UI + Orchestration | Dify Cloud (Agent app with chat UI) |
| Knowledge Graph | cognee Cloud (managed graph + vector store) |
| LLM | Gemini 2.5 Flash via Google AI Studio |
| Embeddings | gemini-embedding-001 (3072 dims) via cognee Cloud |
| Structured Data | Google BigQuery |
| Web Enrichment | Apify (website content crawler for FDA, supplier pages) |
| Voice | ElevenLabs (TTS in Dify) |

## Prerequisites

- Python 3.11+
- Google Cloud account with BigQuery enabled
- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install)
- [Gemini API key](https://aistudio.google.com/apikey) (free)
- [cognee Cloud account](https://platform.cognee.ai) with API key
- [Dify Cloud account](https://cloud.dify.ai) (Team plan)
- [ElevenLabs account](https://elevenlabs.io) with API key

## Setup

### 1. Clone and create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development/testing dependencies:

```bash
pip install -e ".[dev]"
```

### 2. Google Cloud authentication

Install the gcloud CLI if you don't have it:

```bash
# macOS (Homebrew)
brew install --cask google-cloud-sdk

# Or download directly from https://cloud.google.com/sdk/docs/install
```

Authenticate with Application Default Credentials:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_GCP_PROJECT_ID
```

This opens a browser for OAuth login and stores credentials locally. The migration script and BigQuery operations use these credentials automatically.

If you prefer a service account (CI/CD, shared environments):

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### 3. cognee cloud API key

1. Sign in at [app.cognee.ai](https://app.cognee.ai)
2. Go to Settings → API Keys
3. Create a new key — name it something like `agnes-supply-chain`
4. Copy the key for the next step

### 4. Environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in all values. See `.env.example` for descriptions of each variable.

### 5. Playwright browsers (for web scraping)

```bash
playwright install chromium
```

## Running

### 1. Preflight check

```bash
python3 preflight.py
```

Validates Gemini API key, BigQuery connection, and cognee availability.

### 2. Migrate BOM data to BigQuery

```bash
python -m migration.migrate_bom \
  --sqlite-path ./assets/db.sqlite \
  --project YOUR_GCP_PROJECT_ID \
  --bom-dataset agnes_bom \
  --enrichment-dataset agnes_enrichment
```

### 3. Push data to cognee Cloud

```bash
python3 run_cloud_ingestion.py           # upload + cognify
python3 run_cloud_ingestion.py --cognify # retry cognify only
python3 run_cloud_ingestion.py --search  # test search
```

This reads from BigQuery, generates structured text documents, uploads them to cognee Cloud, and runs cognify to build the knowledge graph. The Dify cognee plugin then searches this data.

| Command | What it does | When to use |
|---|---|---|
| `python3 run_cloud_ingestion.py` | Full: generate docs → upload → cognify → test | First time |
| `python3 run_cloud_ingestion.py --cognify` | Trigger cognify only | Upload succeeded, cognify failed |
| `python3 run_cloud_ingestion.py --search` | Test search only | Verify data is searchable |

### 4. Demo via Dify

The demo runs entirely in Dify Cloud — no local server needed:
- Agnes Agent app with chat UI
- cognee plugin for knowledge graph search
- Gemini 2.5 Pro for reasoning
- ElevenLabs for voice narration

See `dify_setup/README.md` for Dify configuration steps.

### Local development (optional)

The `api/` directory contains a FastAPI search bridge for local development and testing only — the production demo uses Dify + cognee Cloud exclusively.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_cognee_pipeline.py -v
pytest tests/test_fastapi_search.py -v
pytest tests/test_kg_topology.py -v

# Run with short output on failure
pytest tests/ -v --tb=short
```

## Dify Setup

Agnes runs as a Dify Agent app with the cognee plugin for knowledge graph search.

### Quick Setup

1. **Install cognee plugin** — Dify Plugins → Marketplace → search "cognee" → Install → Configure with your tenant Base URL and API key
2. **Add Gemini provider** — Settings → Model Providers → Google/Gemini → add your `GEMINI_API_KEY`
3. **Add ElevenLabs** — Settings → Model Providers → ElevenLabs → add API key, configure as TTS
4. **Create Agent app** — Studio → Create App → Agent → add cognee Search tool → set system prompt

See `dify_setup/README.md` for detailed step-by-step instructions.

## Demo Queries

### cognee Knowledge Graph Queries (validate data + graph structure)

Test these via `python3 run_cloud_ingestion.py --search` or the Dify cognee Search tool directly:

1. **Supplier lookup** — "Which suppliers can deliver Vitamin D3?"
   Tests SUPPLIES edges. Expected: Prinova USA, PureBulk.

2. **Functional substitutes** — "What raw materials are functionally similar to soy lecithin?"
   Tests functional category retrieval (emulsifiers). Expected: sunflower lecithin, other lecithin variants.

3. **Evidence retrieval** — "What certifications does Cargill hold for palm oil?"
   Tests Evidence node retrieval. Expected: RSPO, ISCC, Rainforest Alliance from enrichment evidence.

### Dify Agent Queries (validate end-to-end reasoning)

Type these in the Agnes Dify chat app:

1. **Simple ingredient analysis** — "Find me all suppliers for vitamin D3 cholecalciferol and tell me which certifications each one holds."
   Tests: cognee search → evidence retrieval → structured answer with citations. Shows evidence trails.

2. **Compliance-heavy substitution** — "Can soy lecithin be replaced with sunflower lecithin in finished goods? Check allergen and certification compliance."
   Tests: substitution reasoning + compliance inference. Agnes should flag soy as an EU 1169/2011 allergen and note the labeling change required. Shows trustworthiness — Agnes doesn't over-approve.

3. **Multi-ingredient consolidation** — "The Equate multivitamin (product ID 42) has 48 ingredients across many suppliers. Which suppliers could we consolidate to reduce our supplier count while keeping compliance?"
   Tests: full Substitution → Compliance → Optimisation chain. Agnes should identify high-coverage suppliers (Prinova USA covers 408 products, Cargill covers 52) and recommend consolidation. Shows business value.

## Project Structure

```
agnes-makeathon-2026/
├── agents/                    # AI agent modules (Dify workflow logic)
│   └── __init__.py            #   substitution, compliance, optimisation (tasks 9-11)
├── api/                       # FastAPI search bridge (LOCAL DEV ONLY — demo uses Dify)
│   ├── __init__.py
│   └── main.py                #   /health and /search endpoints
├── assets/                    # Enrichment evidence data
│   ├── enrichment_evidence.txt #  Pre-collected supplier evidence (certifications, FDA, RSPO)
│   └── apify_scraped_evidence.txt # Real Apify scrape of FDA food additives list
├── dify_setup/                # Dify configuration and custom tools
│   ├── __init__.py
│   ├── vertex_provider.json   #   Vertex AI provider config (Gemini 2.5 Pro + Flash)
│   ├── cognee_search_tool.py  #   Custom tool wrapping FastAPI /search
│   ├── pulp_solver_tool.py    #   PuLP supplier-consolidation solver tool
│   └── README.md              #   Step-by-step Dify UI setup instructions
├── enrichment/                # Web enrichment pipeline (data infrastructure, not a Dify agent)
│   ├── __init__.py
│   └── pipeline.py            #   Apify + Document AI scraping, relevance validation,
│                              #   temporal diffing, 3-tier scheduling, BQ schema extensions
├── knowledge/                 # Knowledge graph layer (data infrastructure, not a Dify agent)
│   ├── __init__.py
│   └── pipeline.py            #   cognee ingestion (nodes, edges, embeddings, dedup)
├── migration/                 # Data migration
│   ├── __init__.py
│   └── migrate_bom.py         #   SQLite → BigQuery migration with validation
├── tests/                     # Test suite
│   ├── conftest.py            #   Shared fixtures and env var setup
│   ├── test_cognee_pipeline.py
│   ├── test_dify_tools.py     #   Dify tool tests (solver, search tool schema)
│   ├── test_enrichment_agent.py
│   ├── test_fastapi_search.py
│   ├── test_kg_topology.py    #   Property tests: topology + idempotence
│   ├── test_migration_fidelity.py  # Property tests: fidelity, warnings, errors
│   ├── test_migration_unit.py #   Unit tests: datasets, tables, truncation
│   └── test_project_structure.py   # Structural tests: imports, package layout
├── config.py                  # Environment variable loader and validation
├── preflight.py               # Pre-run validation of all external services
├── run_cloud_ingestion.py     # Push data to cognee Cloud (production path)
├── validators.py              # SourcingProposal schema + citation enforcement (used by tests)
├── .env.example               # Template for environment variables
├── .gitignore
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt           # Pip dependencies
└── README.md
```

### Package guidelines

- `migration/` — Data movement scripts. Each script is a standalone CLI tool.
- `enrichment/` — Web scraping, evidence extraction, relevance validation, temporal diffing. This is data pipeline infrastructure, not a Dify agent.
- `knowledge/` — cognee graph ingestion, embedding, deduplication. This is data pipeline infrastructure, not a Dify agent.
- `agents/` — The three Dify AI agent workflows (substitution, compliance, optimisation). Each agent gets its own module. Only Dify agent logic belongs here.
- `api/` — HTTP endpoints. Thin bridge between cognee and Dify tools.
- `dify_setup/` — Dify provider config, custom tool definitions, and setup instructions. See `dify_setup/README.md`.
- `tests/` — All tests live here. Property-based tests use Hypothesis. Test files mirror the package they test.
- `config.py` stays at the root — it's the single entry point for environment configuration.
