# Agnes Demo Guide — 2.5 Minute Presentation

## Judging Criteria (from TUM.ai x Spherecast brief)

1. **Practical usefulness and business relevance**
2. **Quality of reasoning and evidence trails**
3. **Trustworthiness and hallucination control**
4. **Ability to source and operationalize missing external information**
5. **Soundness of substitution logic and compliance inference**
6. **Quality and defensibility of the final sourcing proposal**
7. **Creativity in showing how the system could scale**
8. UI polish is NOT a priority

---

## Presentation Flow (2.5 minutes)

### 0:00–0:30 — Problem + Architecture (30 sec)

Say: "CPG companies overpay because sourcing is fragmented. The same ingredient gets purchased by multiple brands without visibility into combined demand. Agnes fixes this."

Show the architecture (one slide or the Mermaid diagram):
- **BigQuery** holds 1025 products, 40 suppliers, 149 BOMs from real CPG data
- **Apify** scrapes regulatory sources (FDA, EFSA) for compliance evidence
- **cognee Cloud** builds a knowledge graph connecting ingredients, suppliers, and evidence
- **Dify** orchestrates AI agents with Gemini 2.5 Pro — analysts just chat
- **ElevenLabs** narrates recommendations aloud

### 0:30–1:00 — Live Demo Query 1: Substitution (30 sec)

Open the Dify Agnes chat. Type:

```
Which suppliers can deliver Vitamin D3, and what certifications do they hold?
```

**What this shows judges:** Agnes searches the knowledge graph, finds Prinova USA and PureBulk as vitamin D3 suppliers, and surfaces FDA GRAS certification and RSPO data from the enrichment evidence. This hits criteria 1, 2, 4.

### 1:00–1:30 — Live Demo Query 2: Compliance (30 sec)

Type:

```
Can soy lecithin be substituted with sunflower lecithin in finished goods? Check allergen compliance.
```

**What this shows judges:** Agnes finds both lecithins in the knowledge graph, identifies that soy is an EU 1169/2011 allergen but sunflower is not, and flags this as a compliance consideration. She doesn't blindly approve — she surfaces the allergen labeling change as an open question. This hits criteria 3, 5 (trustworthiness, compliance inference).

### 1:30–2:00 — Live Demo Query 3: Consolidation (30 sec)

Type:

```
Which suppliers could we consolidate for the Equate multivitamin product line? It has 48 ingredients across multiple suppliers.
```

**What this shows judges:** Agnes searches for the Equate BOM (the largest in the dataset — 48 ingredients), identifies which suppliers cover the most ingredients, and recommends consolidation. This hits criteria 6 (defensible sourcing proposal).

### 2:00–2:30 — Evidence + Scale (30 sec)

Say: "Every recommendation Agnes makes is backed by evidence from the knowledge graph — supplier data, regulatory sources, certification records. She never hallucinates because she only cites what's in cognee."

Then: "The enrichment pipeline uses Apify to scrape FDA and supplier websites automatically. We ran a real scrape of the FDA food additives database during development. In production, this runs on a 3-tier schedule — monthly full scrapes, daily certification checks, and real-time red flag monitoring."

End with: "Agnes scales by adding more data to cognee. The architecture is the same whether you have 40 suppliers or 4000."

**Key talking point on enrichment:** "We used Apify's website content crawler to scrape regulatory databases like the FDA food additives list. Here's a real scrape we ran — it pulled the FDA's Substances Added to Food inventory. In production, this runs on a schedule to keep supplier evidence fresh."

**Key talking point on trustworthiness:** "We tuned Agnes for factual accuracy — low temperature for consistent answers, grounding enabled to prevent hallucination, and the model only cites what's in the knowledge graph."

---

## Fallback Plan

### Additional Queries (if judges ask for more)

**cognee graph validation:**
- "Which suppliers can deliver Vitamin D3?" → Prinova USA, PureBulk
- "What raw materials are functionally similar to soy lecithin?" → sunflower lecithin
- "What certifications does Cargill hold for palm oil?" → RSPO, ISCC, Rainforest Alliance

**Dify agent reasoning:**
- "Find me all suppliers for vitamin D3 cholecalciferol and tell me which certifications each one holds." → evidence trails
- "Can soy lecithin be replaced with sunflower lecithin in finished goods? Check allergen and certification compliance." → trustworthiness
- "The Equate multivitamin has 48 ingredients across many suppliers. Which suppliers could we consolidate?" → business value

---

## Fallback Plan

If the live demo fails (API timeout, cognee down):
- Have screenshots of successful queries saved
- Show the test suite: `pytest tests/ -v` — 240 green tests
- Show the cognee Cloud dashboard with 1877 nodes and 3402 edges
- Show the Apify run log proving real scraping happened

---

## Pre-Demo Checklist

- [ ] Dify Agnes Agent app is published and accessible
- [ ] cognee Cloud has data (test with `python3 run_cloud_ingestion.py --search`)
- [ ] ElevenLabs TTS is enabled in Dify (test by clicking speaker icon on a response)
- [ ] Have the 3 queries above copied and ready to paste
- [ ] Have fallback screenshots saved
- [ ] Run `python3 preflight.py` to verify all connections
