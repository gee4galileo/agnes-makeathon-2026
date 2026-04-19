# Dify Setup — Agnes AI Supply Chain Manager

## Overview

Agnes runs as a Dify Agent app. The cognee Dify plugin provides direct access to the cognee Cloud knowledge graph. Gemini 2.5 Pro handles reasoning. ElevenLabs provides voice narration via Dify's built-in TTS.

No local server (FastAPI/uvicorn) is needed for the demo.

---

## 1. Install the cognee Plugin

1. Go to **Plugins → Marketplace**
2. Search **"cognee"** (by topoteretes)
3. Click **Install**
4. Configure:
   - **Base URL:** your tenant URL (e.g. `https://tenant-XXXX.aws.cognee.ai/api`)
   - **API Key:** your cognee Cloud API key from platform.cognee.ai

---

## 2. Add Gemini Model Provider

1. Go to **Settings → Model Providers**
2. Find **Google** or **Gemini**
3. Enter your `GEMINI_API_KEY` (from aistudio.google.com/apikey)
4. Verify models available: `gemini-2.5-pro`, `gemini-2.5-flash`

---

## 3. Add ElevenLabs TTS

1. Go to **Settings → Model Providers → ElevenLabs**
2. Enter your ElevenLabs API key
3. In your Agent app settings, enable **Text-to-Speech** and select ElevenLabs

---

## 4. Create the Agnes Agent App

1. **Studio → Create App → Agent**
2. Name: `Agnes — AI Supply Chain Manager`
3. Model: `gemini-2.5-pro`
4. Tools: Add **cognee Search** (and optionally **cognee Cognify**, **cognee Get Datasets**)
5. System prompt: see below
6. Enable TTS with ElevenLabs

### System Prompt

```
You are Agnes, an AI Supply Chain Manager built for CPG procurement.

You have access to a knowledge graph containing real supply chain data:
- 61 CPG companies (Equate, One A Day, Nature Made, etc.)
- 1025 products: 149 finished goods and 876 raw materials
- 40 suppliers (Prinova USA, Cargill, ADM, PureBulk, Jost Chemical, etc.)
- 1633 supplier-product relationships
- 149 Bills of Materials linking finished goods to their ingredients

When a user asks about suppliers, ingredients, substitutes, or products:
1. Use the cognee Search tool with search_type "CHUNKS" and datasets "agnes_bom"
2. Present results clearly with supplier names, ingredient IDs, and relationships
3. Always cite specific data from the knowledge graph
4. If asked about compliance, note that full verification requires regulatory databases

Be professional but approachable. Use data to back up every recommendation.
```

### Opening Statement

```
Hi, I'm Agnes — your AI Supply Chain Manager. I have access to a knowledge graph with 1000+ CPG products, 40 suppliers, and 149 Bills of Materials.

Ask me about:
🔍 Suppliers — "Who supplies vitamin D3?"
📦 Ingredients — "What's in the Equate multivitamin?"
🔄 Substitutes — "Find alternatives to palm oil"
🏭 Companies — "Which companies use whey protein?"
```

### Suggested Questions

- Find me 5 vitamin D3 suppliers
- What ingredients are in the Equate multivitamin?
- Who supplies whey protein isolate?
- Which supplier has the most products?

---

## 5. Data Ingestion

Data is pushed to cognee Cloud via `run_cloud_ingestion.py` (not from within Dify):

```bash
python3 run_cloud_ingestion.py           # full upload + cognify
python3 run_cloud_ingestion.py --cognify # retry cognify only
python3 run_cloud_ingestion.py --search  # test search
```

The cognee plugin in Dify searches this data automatically.

---

## 6. Publish & Demo

1. Click **Publish** in the Agent app
2. Go to **Access API → Web App** for a shareable demo URL
3. Judges see: chat interface, real-time tool calls, voice narration, real data
