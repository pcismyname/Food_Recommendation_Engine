# Food Recommendation Engine — Project Framing

Technical framing for the **Thai Food Recommendation** system (Group: Flower).
For team roles, deliverables, and section ownership see [`README.md`](README.md).

---

## 1. Problem

**Explainable Reviewed-Item Retrieval (RIR).** Given a free-text *taste preference*
query, return a ranked list of restaurants, each justified by the specific review
sentences that support it.

- **Input:** one free-text query (Thai, e.g. `อยากกินอะไรเปรี้ยวๆ เผ็ดๆ ไม่แพง`)
- **Output:** ranked restaurants + the supporting review sentences (the *explainable* part)

The **research contribution is the fusion step** — how per-sentence relevance scores
are combined into a single restaurant score. Everything else is plumbing.

## 2. Pipeline

```
query ──► encode query
                         ┌─ split reviews into sentences
reviews (per restaurant) ┤─ segment (PyThaiNLP for Thai)
                         └─ encode sentences (multilingual / Thai model)
                                  │
                    retrieve top-k sentences per query (vector similarity)
                                  │
                    FUSE sentence scores ──► restaurant score   ◄── contribution
                                  │
                          rank restaurants + attach justifying sentences
```

## 3. Data & the language decision

Datasets are reproduced (not committed) via `datasets/download_datasets.py` —
see `README.md` → **Dataset Setup**.

| Dataset | Language | Structure | Role |
|---|---|---|---|
| **RIRD** (`datasets/rir_data-main`) | **English** | reviews + `business_id` + query→restaurant relevance labels | Method development + **evaluation** |
| **Wongnai** (`datasets/wongnai`) | **Thai** | `review_body` + `star_rating` only — *no restaurant id / query / label* | Thai review corpus for the live demo |
| **Thailand Foods** (`datasets/thailand_foods.csv`) | Thai/En | ~320 dishes: name, ingredients, course, region | Food-term gazetteer / normalization |
| **M-ABSA Thai** (`datasets/m-absa/*/th`) | Thai | aspect-sentiment tuples (coarse categories) | Thai aspect-sentiment signal |

**Key finding (see `eda/`):** RIRD is **100% English** (Toronto / Yelp) — 0% Thai in
both the 29k reviews and the 100 queries; Wongnai is ~99% Thai but has **no restaurant
grouping, queries, or relevance labels**. Therefore:

1. **RIRD (English)** is the *protocol + baseline* — the only source with restaurant
   grouping **and** relevance labels to score the fusion method.
2. **Wongnai (Thai)** powers the **Thai-input demo**, but at the **review/sentence
   level** (Thai query → relevant Thai review sentences), since it can't be grouped
   into restaurants or evaluated with labels.
3. **Thailand Foods** / **M-ABSA Thai** are supporting signal (term normalization,
   aspect sentiment), not retrieval corpora.
4. Never run Thai queries against RIRD — there is nothing Thai to match.

**Does "user types Thai → gets a recommendation" still hold?** Yes — that is the product
goal. The method is *built and measured* on RIRD (English, labelled); the *Thai
experience* is demonstrated on Wongnai at the review level. True Thai **restaurant-level**
recommendation needs either Wongnai metadata with restaurant ids, or translating RIRD's
queries to Thai for a restaurant-level demo.

### Data at a glance (from EDA)
- **RIRD reviews** — 29,168 reviews, 50 restaurants, 17,334 users, mean 3.93★, median
  94 words, 2008–2019. **RIRD PMD** — 100 queries × 50 restaurants = 5,000 judgments
  (+1 dup row to drop), ~27% positive, 5 annotators (κ ≈ 0.528).
- **Wongnai** — 40,000 train / 6,203 test, `star_rating` on a **0–4** scale (skewed to
  2–3), median ~20 words/review, ~99% Thai (pure + mixed).
- **Thailand Foods** — 325 rows (3 duplicate dish names), region mostly
  `Various`/`Unknown`; courses dominated by main dish / snack / dessert.
- **M-ABSA Thai** (food + restaurant) — categories are `food quality`, `food general`,
  `service general`, `ambience general` — **not** flavour axes (sour/spicy/cheap).

## 4. Repository layout

```
Food_Recommendation_Engine/
├── README.md                  # team proposal + setup + dataset setup
├── PROJECT.md                 # this file — technical framing
├── requirements.txt
├── .gitignore                 # datasets/ and venvs are not committed
├── datasets/                  # reproduced via download_datasets.py (not in git)
│   ├── download_datasets.py   # one-shot dataset setup
│   ├── rir_data-main/         # RIRD — English, labelled (PMD.csv + reviews/)
│   ├── wongnai/               # Thai reviews (train/test parquet)
│   ├── thailand_foods.csv     # Thai dish gazetteer
│   └── m-absa/                # M-ABSA Thai (per-domain aspect sentiment)
├── eda/
│   ├── eda_rir_data.ipynb     # English RIRD EDA
│   ├── eda_thai_datasets.ipynb# Thai datasets EDA (Wongnai + Foods + M-ABSA)
│   ├── *.py                   # notebook builders / scripts
│   └── figures/               # generated plots
└── docs/                      # paper notes, related work
```

## 5. Environment & setup

Portable path (any machine) — see `README.md`:

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python datasets/download_datasets.py              # rebuild datasets/ (Thailand Foods is manual)
```

Local dev also uses the **cpdsai** venv (Python 3.14) at
`C:\Users\chids\OneDrive\Desktop\com_pro_DSAI\cpdsai` — already has pandas 3.0.5, numpy
2.5.2, matplotlib 3.11.1, seaborn 0.13.2, `datasets`, `huggingface_hub`.

```powershell
& "C:\Users\chids\OneDrive\Desktop\com_pro_DSAI\cpdsai\Scripts\Activate.ps1"
jupyter lab eda\eda_rir_data.ipynb     # or eda_thai_datasets.ipynb (select cpdsai kernel)
```

Still needed for the Thai pipeline: `pythainlp`, `sentence-transformers` (multilingual),
a vector index (`faiss` / `scikit-learn` nearest-neighbour), `rank-bm25` (baseline).

## 6. Milestones

- [x] Download RIRD, confirm columns & licence, run EDA → **English, use as protocol**
- [x] Download Thai datasets (Wongnai, Thailand Foods, M-ABSA Thai) + EDA
- [x] Reproducible dataset setup (`datasets/download_datasets.py`)
- [ ] BM25 baseline on RIRD producing a retrieval number (MAP / nDCG)
- [ ] Sentence split + multilingual encoding + top-k retrieval
- [ ] Fusion method (the contribution) + ablation vs. baseline
- [ ] Port pipeline to the Thai (Wongnai) corpus — review-level demo
- [ ] Proposal report, slides, 15-min video

## 7. Evaluation

Score on RIRD's `PMD.csv` labels. Because positives are sparse (~27%), rank-aware
metrics — **MAP**, **nDCG@k**, **Recall@k** — matter more than accuracy. Compare the
fusion method against the BM25 baseline on the same queries.
