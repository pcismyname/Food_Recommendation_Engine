# Explainable Reviewed-Item Retrieval — Project Framing

Technical framing for Group Flower's project. Working title: *"Explainable reviewed-item retrieval,
with a Thai taste case study"*, revised after the proposal feedback
([professor_commmend.md](professor_commmend.md)). For tasks and owners see
[action_items.md](action_items.md); for setup see [README.md](README.md).

---

## 1. Problem

**Explainable Reviewed-Item Retrieval (RIR).** Given a free-text *taste preference* query, return a
ranked list of restaurants, each justified by the review sentences that support it.

- **Input:** one free-text query, e.g. `I am on a budget` (RIRD) or `อยากกินอะไรเปรี้ยวๆ` (Thai track)
- **Output:** ranked restaurants plus the supporting review sentences (the *explainable* part)

The proposal claimed a Thai restaurant recommender with fusion as the contribution. That claim went
beyond what we could evaluate (feedback F1), so the questions are now:

1. **Fair fusion (track A):** do BM25 and dense rankings hold up when both score the same unit, and
   are aggregations biased by how many reviews a restaurant has?
2. **Explanation evaluation (track B):** do the shown sentences genuinely support the ranking?
3. **Thai transfer (track C):** does the sentence retrieval step work on real Thai reviews with
   human taste labels?

## 2. Pipeline

```
query ──► encode query (BM25 tokens / dense embedding)
                         ┌─ split reviews into sentences (PyThaiNLP crfcut for Thai)
reviews (per restaurant) ┤─ tokenise for BM25 (newmm for Thai)
                         └─ encode with paraphrase-multilingual-mpnet-base-v2
                                  │
          ① score every unit (review or sentence) with BM25 and dense   ◄── A, C
                                  │
          ② aggregate unit scores ──► restaurant score                   ◄── A
             (max, mean, top-3/5/10 mean; same set for both models)
                                  │
          ③ rank restaurants + attach top sentences as the explanation  ◄── B
```

Stages ② and ③ only run on RIRD, because Wongnai has no restaurant ids.

## 3. Data

Datasets are reproduced (not committed) via `datasets/download_datasets.py`; see
[README.md](README.md) → **Dataset Setup**.

| Dataset | Language | Structure | Role now |
|---|---|---|---|
| **RIRD** (`datasets/rir_data-main`) | English | reviews + `business_id` + query→restaurant relevance labels | Tracks A and B: restaurant ranking and explanation evaluation |
| **Wongnai** (`datasets/wongnai`) | Thai | `review_body` + `star_rating` only; no restaurant id, query or label | Track C: sentence-level Thai test set with our own queries and human labels |
| Thailand Foods (`datasets/thailand_foods.csv`) | Thai/En | ~320 dishes | **Out of scope** (proposal only) |
| M-ABSA Thai (`datasets/m-absa/*/th`) | Thai | aspect-sentiment tuples | **Out of scope** (proposal only) |

**Language decision (from `eda/`):** RIRD is 100% English (Toronto / Yelp); Wongnai is ~99% Thai but
has no restaurant grouping, queries or labels. Therefore:

1. Restaurant ranking and explanations are **measured in English** on RIRD.
2. Thai is a **sentence-level case study**: 24 Thai taste queries
   ([thai/thai_queries.csv](thai/thai_queries.csv), frozen 2026-10-17) run over Wongnai sentences;
   pooled results are hand-labelled 0/1/2 by Thai speakers.
3. Thai restaurant-level ranking is future work. It needs Thai reviews with restaurant ids.
4. Never run Thai queries against RIRD; there is nothing Thai to match.

### Data at a glance (from EDA)
- **RIRD reviews:** 29,168 reviews, 50 restaurants (406 to 2,281 reviews each), 17,334 users, mean
  3.93★, median 94 words, 2008–2019. **RIRD PMD:** 100 queries × 50 restaurants = 5,000 judgments
  (+1 duplicate row to drop), ~27% positive, 5 annotators (κ ≈ 0.528).
- **Wongnai:** 40,000 train / 6,203 test, `star_rating` on a **0–4** scale (skewed to 2–3), median
  ~20 words per review, ~99% Thai (pure + mixed). The Thai notebook uses a 2,000-review subsample
  on CPU.

## 4. Repository layout

```
Food_Recommendation_Engine/
├── README.md                  # overview, status, setup, response to feedback
├── PROJECT.md                 # this file: technical framing
├── action_items.md            # tasks, owners, plan
├── professor_commmend.md      # proposal feedback
├── project_proposal.md        # submitted proposal (historical)
├── requirements.txt
├── datasets/                  # reproduced via download_datasets.py (not in git)
├── baselines/
│   ├── baseline_models_v2.ipynb   # current: review-level BM25 vs. dense, CIs, significance tests
│   ├── baseline_models.ipynb      # proposal-era v1 (BM25 on concatenated restaurant docs)
│   ├── build_baselines_notebook.py
│   ├── eval_utils.py              # metrics, bootstrap CIs, paired tests
│   └── cache/                     # result tables (embeddings are gitignored)
├── thai/
│   ├── thai_queries.csv           # frozen Thai taste queries (C1)
│   ├── thai_queries_ai_draft.csv  # original LLM draft (record)
│   ├── query_guidelines.md        # query rules, change log, sign-off
│   ├── thai_retrieval.ipynb       # Wongnai sentence-level BM25 vs. dense (C2)
│   └── build_thai_retrieval_notebook.py
├── eda/                       # RIRD and Thai dataset EDA
├── knowledge_base/            # background notes
└── docs/                      # paper notes, references, proposal LaTeX
```

## 5. Environment & setup

Portable path (any machine); see [README.md](README.md):

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python datasets/download_datasets.py
```

Local dev uses the **cpdsai** venv at `C:\Users\chids\OneDrive\Desktop\com_pro_DSAI\cpdsai`
(Jupyter kernel `cpdsai`), which has the full stack including `pythainlp`, `rank-bm25` and
`sentence-transformers`.

```powershell
& "C:\Users\chids\OneDrive\Desktop\com_pro_DSAI\cpdsai\Scripts\Activate.ps1"
jupyter lab   # select the cpdsai kernel
```

The sentence-level RIRD run is planned on a Colab GPU.

## 6. Milestones

- [x] RIRD and Thai datasets downloaded, EDA, reproducible setup
- [x] Proposal report, slides, video
- [x] v1 baselines (BM25 on concatenated documents vs. mean-pooled dense)
- [x] v2 review-level baselines: same unit, same aggregations, bootstrap CIs, Holm-corrected paired tests
- [x] A3 review-volume control (subsample to 400 reviews; score vs. review count): effect ≤ 0.011 nDCG@10
- [x] A4 save per-query rankings and evidence (review level)
- [ ] A1/A2 sentence-level rerun
- [x] C1 Thai taste queries (24, frozen 2026-10-17)
- [/] C2 Thai sentence pipeline (built on 2,000 reviews; segmentation spot-check pending)
- [ ] Progress report and slides (2026-10-19), presentation (2026-10-20)
- [ ] B1 faithfulness test, B2 human support rating
- [ ] C3–C7 Thai pooling, labelling, agreement, Thai nDCG@10 / P@10
- [ ] Final report, slides, video

## 7. Evaluation

**RIRD (tracks A, B).** Score on `PMD.csv` labels. Positives are sparse (~27%), so we use
rank-aware metrics: **nDCG@10**, **MAP**, P@k and Recall@k. Every model reports a bootstrap 95% CI;
pairs of models are compared with paired randomization tests and Holm correction. Three
query-independent floors (random, review count, popularity) show how much is just restaurant
popularity.

Current best (review level): dense with top-10 mean, nDCG@10 0.668 [0.625, 0.711], MAP 0.608.
Dense beats BM25 under every aggregation.

**Explanations (track B).**
- Faithfulness: rank drop when the shown top-k sentences are deleted, compared with deleting k
  random sentences.
- Support: two annotators rate the shown sentences 0/1/2. We report Cohen's kappa and the share
  rated as supporting.

**Thai (track C).** Results from BM25 and dense are pooled into a blind sheet and labelled 0/1/2 by
a Thai speaker. A second annotator labels a subset for agreement. Then nDCG@10 and P@10 compare BM25
and dense at sentence level.
