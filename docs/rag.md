# IndicationScout RAG Pipeline

## Why RAG Is Needed

PubMed keyword search returns many irrelevant papers. Searching `bupropion AND obesity` returns depression
papers that incidentally mention obesity — burying the papers about bupropion as an actual *treatment* for
obesity.

The RAG pipeline solves this by embedding and reranking retrieved abstracts so the Literature Agent
receives the most relevant papers, not just keyword matches.

**Confirmed empirically with:**
- Bupropion → obesity: papers about immunometabolic depression dominated results; the actual Contrave (naltrexone/bupropion) papers were buried
- Sildenafil → diabetic nephropathy: needed reranking to separate PDE5-specific evidence from noisy non-selective PDE inhibitor literature
- Baricitinib → myelofibrosis: needed reranking to distinguish JAK1/2 inhibition therapeutic context from general oncology mentions

The steps are
- Given a drug+disease , query an llm to get relevant terms
- Convert the papers into embeddings and put into pgvector
- Create a query from drug+disease , embed, and search over pgvector to get most relevant
- Convert these top papers into EvidenceSummary objects

Notes:
e.g. "Metformin + colorectal" should include things like 
"metformin AND colorectal neoplasm"   
"biguanide AND colorectal"

For the query to the LLM, we need to send entire Drug object
Should include - the mechanism, drug class, ATC codes, synonyms

---

## The RAG Loop

**fetch → embed → search → stuff into prompt → generate grounded summary**

1. **Build drug profile** — fetch structured drug data needed for search term expansion. Calls `OpenTargetsClient.get_rich_drug_data()` for targets, mechanisms, indications, and synonyms; then enriches ATC codes with human-readable descriptions via `ChEMBLClient.get_atc_description()`. Returns a `DrugProfile` used in the next step.

   ```python
   # services/retrieval.py
   profile = await build_drug_profile("metformin")
   # profile.synonyms, profile.target_gene_symbols, profile.mechanisms_of_action,
   # profile.atc_codes, profile.atc_descriptions, profile.drug_type
   ```

2. **Expand search terms** — given a drug+disease, query an LLM with the `DrugProfile` (mechanism, drug class, ATC codes, synonyms, target gene symbols) to generate diverse PubMed keyword queries. e.g. "Metformin + colorectal" -> `"metformin AND colorectal neoplasm"`, `"biguanide AND colorectal"`, `"metformin AND AMPK AND colon"`, ...
3. **Fetch & cache** — hit PubMed E-utilities with each query (up to 500 PMIDs), fetch abstracts for any not already stored, embed with BioLORD-2023, store in pgvector
4. **Semantic search** — embed the drug+disease query with BioLORD-2023, run cosine similarity over pgvector, return top 20 abstracts. Finds conceptually relevant papers even without exact keyword matches (e.g. "biguanide antineoplastic mechanisms" matches a metformin/cancer query)
5. **Re-rank** — reduce top 20 → top 5 using a cross-encoder or LLM reranker
6. **Augment + generate** — stuff the top 5 abstracts into a Claude prompt. Claude reads the actual retrieved papers, not training data. Output is a structured `EvidenceSummary` with PMIDs attached to every claim — every finding is traceable to a real paper

```
PubMed E-utilities (keyword search)
       │  up to 500 PMIDs
       ▼
┌──────────────────┐
│  fetch_and_cache  │  Fetch abstracts for new PMIDs only
│                   │  Embed with BioLORD-2023 (768-dim)
│                   │  Store in pgvector (deduplicate by PMID)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  semantic_search  │  Embed query with BioLORD-2023
│                   │  Cosine similarity over pgvector → top 20
│                   │  Re-rank → top 5 (cross-encoder or LLM)
└────────┬─────────┘
         │  top 5 abstracts (title + abstract + PMID)
         ▼
┌──────────────────┐
│  Synthesize       │  
│  EvidenceSummary  │  
│  objects          │
│                   │  Output: EvidenceSummary
│                   │  Every claim is attached to a PMID,
│                   │  traceable back to a real paper.
└──────────────────┘
```

---

## Where RAG Fits in the Full Pipeline

```
┌─────────────┐     ┌─────────────┐
│   Path 1     │     │   Path 2     │
│  (target-    │     │  (drug class │
│   disease    │     │   analogy)   │
│   assoc.)    │     │              │
└──────┬───────┘     └──────┬───────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
          ┌───────────────┐
          │   Merge &      │  Deduplicate by EFO ID,
          │   Rank Top 10  │  flag dual-path candidates
          └───────┬────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │     For each candidate:      │
    │                              │
    │     RAG processing steps     │
    │                              │
    └─────────────┬────────────────┘
                  │
                  ▼
          ┌───────────────┐
          │  Evidence      │  Combine Path scores +
          │  Scoring &     │  literature + trial evidence
          │  Report Gen    │
          └────────────────┘
```

The RAG pipeline sits between raw PubMed retrieval and the Literature Agent. It does not change the
discovery logic (Path 1 / Path 2) — it improves the quality of evidence presented to the LLM for synthesis.

---

## Validated Test Cases

These drug-disease pairs confirmed that PubMed has signal but reranking is needed:

| Drug | Candidate Disease | Issue Without RAG |
|------|-------------------|-------------------|
| Bupropion | Obesity | Depression papers mentioning obesity dominate; Contrave papers buried |
| Bupropion | Narcolepsy | EFO name mismatch (`narcolepsy-cataplexy syndrome`) returned zero results |
| Sildenafil | BPH | Confirmed signal (PMID 40678732, 38448685) but mixed with erectile dysfunction papers |
| Sildenafil | Diabetic nephropathy | Needed reranking to separate PDE5-specific from non-selective PDE literature |
| Baricitinib | Eczematoid dermatitis | EFO name doesn't match PubMed indexing (`atopic dermatitis`) |

---

See [rag_details.md](rag_details.md) for schema, embedding model, stage specifications, infrastructure, and design decisions.
