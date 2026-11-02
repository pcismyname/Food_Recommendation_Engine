# Explainable Food Recommendation Engine
**Group Flower** · AIT DSAI 2026 · Ton (st127004) · Sam (st127012)

> Given a free-text taste query, rank restaurants and show the review sentences that
> justify each position — with a Thai taste case study on Wongnai.

---

## What this system does

A user types a natural-language query such as *"I want a cozy bar with a patio"* or
*"อยากกินอะไรเปรี้ยวๆ"* (Thai: "I want something sour"). The system:

1. **Rewrites the query** using an LLM (Groq / Llama-3.1-8b-instant) to make vague or
   negation-heavy queries concrete — e.g. *"no waitline"* → *"quick seating walk-in no queue"*.
2. **Retrieves** the most relevant review sentences for each restaurant using a multilingual
   dense encoder (`paraphrase-multilingual-mpnet-base-v2`).
3. **Aggregates** sentence scores into a single restaurant score (top-10 mean).
4. **Re-ranks** the evidence sentences using a cross-encoder (`ms-marco-MiniLM-L-6-v2`)
   for higher precision.
5. **Returns** a ranked list of restaurants, each with the review sentences that justify it.

The live demo runs at **http://localhost:8000** (see [Running the app](#running-the-app)).

---

## Full pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: free-text query                                                     │
└───────────────────────┬─────────────────────────────────────────────────────┘
                        │
            ┌───────────▼────────────┐
            │  STAGE 0: Query         │  Groq / Llama-3.1-8b-instant
            │  Rewriting (B3)        │  Converts negations + vague phrases
            │                        │  to reviewer-language keywords
            └───────────┬────────────┘
                        │ rewritten query
            ┌───────────▼────────────┐
            │  STAGE 1: Bi-encoder   │  paraphrase-multilingual-mpnet-base-v2
            │  Sentence Retrieval    │  Cosine similarity: query emb vs
            │                        │  pre-computed sentence embeddings
            │                        │  240,660 RIRD sentences (EN)
            │                        │   93,810 Wongnai sentences (TH)
            └───────────┬────────────┘
                        │ top-k sentences per restaurant
            ┌───────────▼────────────┐
            │  STAGE 2: Score        │  Aggregation: mean of top-10
            │  Aggregation           │  unit scores → one score/restaurant
            │                        │  Evaluated with max, mean, top-3/5/10
            └───────────┬────────────┘
                        │ restaurant score
            ┌───────────▼────────────┐
            │  STAGE 3: Cross-encoder│  ms-marco-MiniLM-L-6-v2
            │  Re-ranking (B3)       │  Jointly encodes (query, sentence)
            │                        │  Re-scores top evidence sentences;
            │                        │  fixes negation blindness in stage 1
            └───────────┬────────────┘
                        │ re-ranked evidence
            ┌───────────▼────────────┐
            │  OUTPUT: Ranked        │  Restaurants ordered by score +
            │  restaurants +         │  top supporting review sentences
            │  explanations          │  shown as justification
            └────────────────────────┘
```

---

## Why each stage matters

### Stage 0 — Query rewriting
The bi-encoder (stage 1) is a **semantic similarity** model, not a relevance model. It
cannot handle:

- **Negation:** *"no waitline"* and *"long wait"* have near-identical embeddings because
  they discuss the same concept. Without rewriting, the top sentences are about the
  restaurant being busy — the opposite of what the user wants.
- **Idiomatic language:** *"cheat meal"* has no food-adjacent embedding; results are random.

The LLM rewriting step converts the query into explicit, positive attributes that reviewers
actually write (*"quick seating, walk-in welcome, no reservation"*), then the bi-encoder
retrieves against that.

### Stage 1 — Bi-encoder retrieval
`paraphrase-multilingual-mpnet-base-v2` encodes queries and review sentences into a 768-dim
embedding space. Cosine similarity at query time is fast because sentence embeddings are
**pre-computed and cached** (~740 MB for 240k RIRD sentences). This supports both English
and Thai without any language-specific fine-tuning.

**Key finding (A1/A2):** Dense retrieval significantly outperforms BM25 under all top-k
mean aggregations (Holm-corrected p ≤ 0.004). Dense top-10 mean at sentence level achieves
**nDCG@10 = 0.682** and **MAP = 0.618**, approaching the RIRD fine-tuned reference (MAP 0.626)
with no domain adaptation.

### Stage 2 — Score aggregation
Each restaurant's score is the **mean of the top-10 sentence scores** for that restaurant.
This was chosen over max (noisy, volume-sensitive) and mean-of-all (diluted by irrelevant
sentences). Volume control (A3) confirmed that sampling every restaurant to 400 reviews
reduces score/count correlation to near zero (Pearson −0.009 to +0.019).

**Key finding (A5):** Switching to sentence-level units helps top-k mean aggregations but
hurts mean aggregation (Dense|mean degrades by Δ = −0.055, Holm p = 0.000). This shows
that **sentence granularity + top-k mean is the right combination**.

### Stage 3 — Cross-encoder re-ranking
The cross-encoder jointly processes (query, sentence) pairs using a full transformer
attention mechanism. Unlike stage 1 (which encodes query and sentence independently), the
cross-encoder can detect **semantic mismatch and negation** because it sees both inputs
together.

Trade-off: cross-encoders are 50–100× slower than bi-encoders, so we only re-rank the
**top-k sentences already retrieved** in stage 1, not the full corpus.

---

## Datasets

| Dataset | Language | Size | Role |
|---|---|---|---|
| **RIRD** ([github.com/D3Mlab/rir_data](https://github.com/D3Mlab/rir_data)) | English | 50 restaurants, ~29k reviews, 100 queries, 5-annotator labels | Tracks A & B: restaurant ranking + explanation evaluation |
| **Wongnai** ([Kaggle](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction)) | Thai | ~1.2M reviews, star ratings only | Track C: Thai taste test set (sentence-level) |
| Thailand Foods | Thai/EN | ~320 dishes | Out of scope (future work) |
| M-ABSA Thai | Thai | Aspect-sentiment tuples | Out of scope (future work) |

> **Why three datasets?** RIRD provides the only English dataset with both restaurant IDs
> and human relevance labels, enabling end-to-end restaurant ranking evaluation. Wongnai
> provides real Thai restaurant reviews for the transfer test. Thailand Foods and M-ABSA
> were in the original proposal but are out of scope for the final evaluation.

---

## EDA summary

Notebooks in [`eda/`](eda/):

- **RIRD EDA:** 50 restaurants, 3–5 stars, 406–2281 reviews per restaurant (highly
  unequal — motivates volume control in A3). Average 5.5 sentences per review.
  100 queries, average 7.1 relevant restaurants per query.
- **Wongnai EDA:** ~99% Thai text. No restaurant IDs → cannot do restaurant-level ranking.
  Average 9.4 sentences per review after PyThaiNLP `crfcut` segmentation.
- **Language check:** RIRD is 100% English; Wongnai is ~99% Thai. The multilingual encoder
  handles both without translation.

---

## Evaluation results

### English — RIRD (Tracks A & B)

| Model | Unit | Aggregation | nDCG@10 | MAP |
|---|---|---|---|---|
| BM25 (baseline v1) | Review (concat) | — | 0.621 | — |
| Dense (proposed) | Sentence | Top-10 mean | **0.682** | **0.618** |
| Dense + rewrite + CE (B3) | Sentence | Top-10 mean + cross-encoder | *TBD after B3 run* | *TBD* |
| RIRD fine-tuned (external ref) | — | — | — | 0.626 |

**A5 (review vs. sentence):** No variant significantly improves by switching units after Holm
correction. Dense|top-10 mean has the best positive gap (Δ = +0.013, n.s.) — sentence units
only help with top-k mean aggregation.

**B1 (faithfulness):** Automatic deletion test was inconclusive (gap = 0, p = 1.0) because
the evidence pool per restaurant is too small (top-10 sentences) to distinguish shown vs.
random deletion. Human B2 support rating is the more meaningful faithfulness signal.

**B2 (human support):** Labelling in progress. Key failure modes identified:
- Negation blindness (see B3 fix)
- Vague/idiomatic queries (see B3 fix)

### Thai — Wongnai (Track C)

| Metric | Value |
|---|---|
| Pool size | 452 blind (query, sentence) pairs |
| Queries | 24 frozen Thai taste queries |
| Systems compared | BM25 vs. Dense (same encoder, sentence-level) |
| Labelling | In progress (C4) |
| nDCG@10, P@10 | TBD after C7 |

---

## Repository map

```
Food_Recommendation_Engine/
├── app/                          ← Demo web app (FastAPI + HTML/JS)
│   ├── main.py                   ← API: GET /, POST /api/query, GET /api/health
│   ├── retrieval.py              ← Retrieval engine (loads CSVs at startup)
│   ├── requirements.txt
│   └── static/index.html         ← Dark-mode SPA (English + Thai tabs)
│
├── baselines/                    ← All RIRD baselines and evaluation
│   ├── baseline_models_v2.ipynb  ← A1/A2 review-level (current)
│   ├── baseline_models_v2_sentence.ipynb  ← A1/A2 sentence-level
│   ├── a5_review_vs_sentence_test.ipynb   ← A5 paired significance test
│   ├── b1_faithfulness.ipynb     ← B1 automatic deletion faithfulness test
│   ├── b3_improved_retrieval.ipynb ← B3 query rewriting + cross-encoder
│   ├── b2_export_support_sheet.py ← generates human-labelling CSVs
│   ├── b2_score_agreement.py     ← computes Cohen's kappa + % supporting
│   ├── eval_utils.py             ← shared nDCG, MAP, Recall, P functions
│   └── cache/                    ← gitignored: embeddings, result CSVs, plots
│
├── thai/                         ← Thai taste test set (Track C)
│   ├── thai_queries.csv          ← 24 frozen Thai taste queries
│   ├── thai_retrieval_10k_colab.ipynb ← C3: 10k Wongnai retrieval (Colab/GPU)
│   ├── build_label_pool.py       ← builds blind label pool from topk CSV
│   ├── score_labels.py           ← C7: computes Thai nDCG@10, kappa
│   ├── annotation_guidelines.md  ← labelling instructions (Thai + English)
│   ├── labels/                   ← label sheets (gitignored key)
│   └── results/topk_n10000_s42.csv ← 10k retrieval output
│
├── eda/                          ← EDA notebooks for all datasets
├── datasets/                     ← gitignored raw data (download separately)
├── docs/
│   ├── DEPLOYMENT.md             ← AWS EC2 deployment guide
│   └── ton_5_paper.md            ← professor's earlier paper comments
│
├── Dockerfile                    ← Multi-stage; Python 3.11; port 8000
├── docker-compose.yml            ← Local dev with volume mounts
├── action_items.md               ← Live task tracker (owners, status, plan)
└── PROJECT.md                    ← Technical framing and pipeline detail
```

---

## UI tech stack

| Layer | Tech | Notes |
|---|---|---|
| Backend | **FastAPI** + **Uvicorn** (Python) | REST API at `http://localhost:8000` |
| Frontend | **Vanilla HTML/CSS/JS** (single file) | No build step, no framework |
| Styling | Custom dark-mode CSS | Glassmorphism, CSS variables, micro-animations |
| Fonts | Google Fonts — Inter + JetBrains Mono | Loaded via CDN |
| File | [`app/static/index.html`](app/static/index.html) | Served by FastAPI as a static file |

The frontend talks to the backend via three fetch calls:
- `GET /api/queries?lang=en|th` — loads autocomplete chips on page load
- `POST /api/query` — submits a query, receives ranked results + explanation sentences
- `GET /api/health` — liveness check

---

## Running the app

### Step 1 — Install dependencies

```bash
pip3 install fastapi uvicorn numpy pandas groq python-dotenv sentence-transformers scikit-learn
```

### Step 2 — Set your Groq API key

Create a `.env` file in the project root (already exists if you followed setup):

```
groq = your_groq_api_key_here
```

The server reads this automatically at startup. Without it, query rewriting is disabled
but pre-computed queries still work.

### Step 3 — Start the server

```bash
# From the project root (Food_Recommendation_Engine/)
python3 -m uvicorn app.main:app --port 8000
```

Startup takes **~30–60 seconds** — it loads:
- `evidence_sentence.csv` (~107 MB, 100 pre-computed queries)
- Sentence embeddings (~740 MB, for live inference)
- Bi-encoder + cross-encoder models (~85 MB each)
- Thai results CSV (24 pre-computed queries)

When you see `Application startup complete. Uvicorn running on http://127.0.0.1:8000`,
open **http://localhost:8000** in your browser.

### What happens for each query type

| Query type | Path | Latency |
|---|---|---|
| One of the 100 RIRD queries (exact/case-insensitive match) | Instant pre-computed lookup → no models touched | <1ms |
| New English query (anything else) | Stage 0: Groq rewrites → Stage 1: bi-encoder → Stage 2: cross-encoder → Stage 3: aggregate | ~3–8s |
| One of the 24 Thai queries | Instant pre-computed lookup | <1ms |
| New Thai query | Not supported (error shown) | — |

The UI shows:
- **⚡ Pre-computed** badge for instant lookups
- **🧠 Live B3** badge + *"Query rewritten: …"* banner for new queries
- Restaurant cards ranked by score, each with expandable explanation sentences

### With Docker (requires Docker 20.10+ for `docker compose`)

```bash
docker compose up --build
```

### On EC2

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the full step-by-step guide.

---

## B3 System Improvements (motivated by B2 labelling)

> This section is the reference for the **presentation improvement slide**.

### Problem 1 — Negation blindness
`paraphrase-multilingual-mpnet-base-v2` is a *semantic similarity* model, not a relevance model.
*"no waitline"* and *"long wait, make a reservation"* have near-identical embeddings (same topic: wait times).
The system was returning sentences about restaurants being *busy* for queries asking for *no wait*.

**Fix (B3a): LLM query rewriting — Groq / `qwen3.8-27b`**
A system prompt converts negations to positive reviewer-language equivalents before encoding:
- `"no waitline"` → `"quick seating walk-in no queue no reservation"`
- `"not too noisy"` → `"quiet calm peaceful relaxed atmosphere"`
- `"can I not have oily food"` → `"light fresh non-greasy clean healthy cooking"`

The rewritten query is what gets encoded by the bi-encoder. Results are cached in-memory (second call instant).

---

### Problem 2 — Idiomatic / vague queries return random results
*"Can I have a cheat meal"*, *"I feel chubby"*, *"Want a place to get over this depression"* — these phrases
have no food-adjacent embedding. The bi-encoder returns effectively random restaurants.

**Fix (B3a, same system):** the Groq prompt also expands idioms to concrete food attributes:
- `"cheat meal"` → `"indulgent rich comfort food burgers fries pizza calorie dense"`
- `"I feel chubby"` → `"healthy low calorie salad light grilled fresh vegetables"`

---

### Problem 3 — Short/generic sentences score too high (embedding length bias)
Very short sentences like `"Solid meal!"` (2 words) or `"Great place."` get artificially high cosine
similarity with short queries because the bi-encoder's normalisation amplifies short-text signals.
B2 labelling returned 0 (not supporting) for most of these.

**Fix (B3c): `MIN_SENTENCE_WORDS = 10` filter**
Applied at server startup. Any sentence under 10 words is removed from the live retrieval corpus.
`"Solid meal!"` → filtered. `"No consistency or timing in serving the meals."` (9 words) → filtered.
Only substantive sentences remain.

Additionally, `CE_DISPLAY_THRESH = 0.0` hides cross-encoder-scored sentences below the relevance
threshold from the UI explanation. Fallback: always show ≥2 sentences so cards are never empty.

---

### Problem 4 — Cross-encoder was too slow for live demo (3–8s per query)

**Root cause:** naïve implementation ran CE on all 50 restaurants × 30 candidates = 1,500 inference calls.

**Fix (B3b + B3d):**
| Setting | Before | After | Effect |
|---|---|---|---|
| `CE_TOP_RESTS` | 50 (all) | **20** | Only CE-rank top-20 bi-encoder restaurants |
| `CE_INPUT_K` | 30 | **15** | 15 CE candidates per restaurant |
| **Total CE calls** | 1,500 | **300** | **5× fewer inference calls** |
| `_live_cache` | none | in-memory | Repeated queries return instantly |

**Net latency:** ~8s → **~2–4s** for new live queries.
Pre-computed queries (100 RIRD + 24 Thai) still bypass all models entirely (<1ms).

---

### Why the cross-encoder (Stage 2) is better than bi-encoder alone (Stage 1)

| | Bi-encoder (Stage 1) | Cross-encoder (Stage 2) |
|---|---|---|
| Encodes | Query and sentence independently | Query + sentence jointly |
| Negation | Invisible — same-topic embeddings | Detectable — full attention sees mismatch |
| Speed | Fast (pre-computed embeddings) | Slow (no pre-computation possible) |
| Use in pipeline | Retrieval (all 240k sentences) | Re-ranking (top-300 candidates only) |

---

## Reproducing all results


Run notebooks in this order (each depends on the previous):

```
1. eda/                                 ← exploratory analysis (optional)

2. baselines/baseline_models_v2.ipynb           ← A1/A2 review-level
3. baselines/baseline_models_v2_sentence.ipynb  ← A1/A2 sentence-level → produces evidence_sentence.csv

4. python3 baselines/build_a5_notebook.py  && jupyter nbconvert --execute baselines/a5_review_vs_sentence_test.ipynb
5. python3 baselines/build_b1_notebook.py  && jupyter nbconvert --execute baselines/b1_faithfulness.ipynb
6. python3 baselines/b2_export_support_sheet.py   ← generate labelling sheets
   [label baselines/cache/b2_support_sheet.csv manually]
   python3 baselines/b2_score_agreement.py

7. python3 baselines/build_b3_notebook.py  && jupyter nbconvert --execute baselines/b3_improved_retrieval.ipynb
   [requires GROQ key in .env as 'groq']

Thai track (Track C):
8. thai/thai_retrieval_10k_colab.ipynb   ← run on Colab with GPU
9. python3 thai/build_label_pool.py
   [label thai/labels/label_sheet.csv manually]
   python3 thai/score_labels.py
```

> **Data:** RIRD and Wongnai datasets are not committed. Download them with
> `python3 datasets/download_datasets.py` (or follow the manual links in `PROJECT.md`).

---

## Key dependencies

| Package | Version | Purpose |
|---|---|---|
| `sentence-transformers` | ≥2.7 | Bi-encoder + cross-encoder |
| `rank-bm25` | any | BM25 baseline |
| `fastapi` + `uvicorn` | ≥0.111 | Demo web server |
| `groq` | any | LLM query rewriting API |
| `python-dotenv` | any | Load `.env` for API keys |
| `pythainlp` | any | Thai sentence segmentation |
| `numpy`, `pandas`, `scipy` | standard | Numerics and statistics |

Install: `pip3 install -r app/requirements.txt groq python-dotenv pythainlp rank-bm25`

---

## References

- RIRD dataset: <https://github.com/D3Mlab/rir_data>
- RIR paper (ECIR 2023): <https://arxiv.org/pdf/2308.00762>
- Korikov et al., *Multi-Aspect Reviewed-Item Retrieval via LLM Query Decomposition and Aspect Fusion* (2024): <https://arxiv.org/abs/2408.00878>
- Zhang et al., *Enhancing Recommendation Explanations through User-Centric Refinement*, Findings of EMNLP 2025: <https://aclanthology.org/2025.findings-emnlp.434/>
- Hasan et al., *Review-based Recommender Systems: A Survey*, ACM Computing Surveys: <https://doi.org/10.1145/3742421>
- Wongnai challenge: <https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction>
- PyThaiNLP: <https://github.com/PyThaiNLP/pythainlp>
