"""Generate baseline_models.ipynb — BM25 + Dense (no fusion) baselines on RIRD."""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE  = Path(__file__).resolve().parent
cells = []
md    = lambda t: cells.append(new_markdown_cell(t))
code  = lambda t: cells.append(new_code_cell(t))

# ── Title ──────────────────────────────────────────────────────────────────
md("""\
# Baseline Models — Thai Food Recommendation Engine
**Group:** Flower | **Authors:** Ton (st127004) · Sam (st127012)

This notebook establishes two baselines on the **RIRD (English, labelled)** dataset:

| # | Model | Description |
|---|---|---|
| 1 | **BM25** | Lexical keyword-matching baseline (no GPU, no training) |
| 2 | **Dense (no fusion)** | Mean-pooled multilingual sentence embeddings per restaurant |

Both models are evaluated with the **same harness** (`eval_utils.py`) so that any
future fusion variant can be added in one cell and compared in the same table.

**Evaluation metrics:** MAP · MRR · nDCG@5 · nDCG@10 · Recall@5 · Recall@10 · P@5 · P@10  
**Labels:** binary from `"If only Low or High"` (PMD.csv) + graded (sum of 5 annotator votes) for nDCG
""")

# ── Section 0 — Setup ──────────────────────────────────────────────────────
md("## 0. Setup & Imports")

code("""\
import sys, re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Point at eval_utils.py in this same baselines/ folder
HERE  = Path.cwd()
ROOT  = HERE if (HERE / "datasets").exists() else HERE.parent
DATA  = ROOT / "datasets"
sys.path.insert(0, str(ROOT / "baselines"))

from eval_utils import evaluate_model, compare_models

print("ROOT :", ROOT)
print("DATA :", DATA)
""")

# ── Section 1 — Data Loading ───────────────────────────────────────────────
md("""\
## 1. Data Loading

We use two files from RIRD:
- `reviews/50_restaurants_all_rates.csv` — 29 k reviews, 50 restaurants
- `PMD.csv` — 100 queries × 50 restaurants, 5 annotators each
""")

code("""\
# ── Reviews ──────────────────────────────────────────────────────────────
rev = pd.read_csv(DATA / "rir_data-main" / "reviews" / "50_restaurants_all_rates.csv")
rev["review_text"] = rev["review_text"].fillna("").astype(str)
print(f"Reviews: {rev.shape}  |  unique restaurants: {rev['business_id'].nunique()}")
rev.head(2)
""")

code("""\
# ── PMD (query-restaurant relevance labels) ───────────────────────────────
pmd_raw = pd.read_csv(DATA / "rir_data-main" / "PMD.csv")

# Drop the known duplicate row flagged in EDA
pmd = pmd_raw.drop_duplicates().copy()
print(f"PMD rows after dedup: {len(pmd)}  (was {len(pmd_raw)})")

ANNOTATORS = ["Annotator1","Annotator2","Annotator3","Annotator4","Annotator5"]
AGG_COL    = "If only Low or High"   # binary aggregated label

pmd.head(3)
""")

# ── Section 2 — Ground Truth ───────────────────────────────────────────────
md("""\
## 2. Build Ground Truth (qrels)

We build two qrel structures from `PMD.csv`:

| Name | Type | Used for |
|---|---|---|
| `binary_qrels` | `{query -> set of relevant business_ids}` | MAP, MRR, Recall, P |
| `graded_qrels` | `{query -> {business_id -> annotator_vote_sum (0–5)}}` | graded nDCG |
""")

code("""\
# Link PMD rows to business_ids via the reviews table
# PMD has restaurant names but not business_ids directly
name_to_bid = rev.drop_duplicates("business_id").set_index("name")["business_id"].to_dict()

# Some PMD rows use a 'restaurant' or 'name' column — inspect
print("PMD columns:", pmd.columns.tolist())
""")

code("""\
# Build qrels
# PMD must have a column that identifies the restaurant — find it
# Common column names in RIRD: 'restaurant_name', 'name', 'business_id'
name_col = next(
    (c for c in ["restaurant_name", "name", "business_id"] if c in pmd.columns),
    None
)
print(f"Restaurant identifier column: {name_col!r}")

binary_qrels: dict = {}
graded_qrels: dict = {}

for query, grp in pmd.groupby("query"):
    rel_set   = set()
    rel_grade = {}
    for _, row in grp.iterrows():
        restaurant = row[name_col]
        bid = name_to_bid.get(restaurant, restaurant)  # fall back to name if no mapping
        grade = int(sum(row[a] for a in ANNOTATORS if pd.notna(row[a])))
        if row[AGG_COL] == 1:
            rel_set.add(bid)
        rel_grade[bid] = grade
    binary_qrels[query] = rel_set
    graded_qrels[query] = rel_grade

n_queries    = len(binary_qrels)
avg_pos      = np.mean([len(v) for v in binary_qrels.values()])
print(f"Queries: {n_queries}  |  avg relevant restaurants/query: {avg_pos:.1f}")
""")

# ── Section 3 — BM25 Baseline ──────────────────────────────────────────────
md("""\
## 3. BM25 Baseline

**Strategy:** concatenate all reviews for each restaurant into one document,
tokenise with simple whitespace + lowercase, build a BM25 index, score every
restaurant for each query, rank by score.

BM25 is the standard lexical baseline from the ECIR 2023 RIR paper.
""")

code("""\
from rank_bm25 import BM25Okapi

def simple_tokenise(text: str):
    return re.sub(r"[^a-z0-9\\s]", " ", text.lower()).split()

# Build one document per restaurant (all reviews concatenated)
restaurant_docs = (
    rev.groupby("business_id")["review_text"]
       .apply(lambda texts: " ".join(texts))
       .reset_index()
)

corpus_ids   = restaurant_docs["business_id"].tolist()
corpus_texts = [simple_tokenise(t) for t in restaurant_docs["review_text"]]

bm25 = BM25Okapi(corpus_texts)
print(f"BM25 index built over {len(corpus_ids)} restaurants")
""")

code("""\
# Rank all restaurants for each query
bm25_rankings: dict = {}

for query in tqdm(binary_qrels.keys(), desc="BM25 queries"):
    q_tokens = simple_tokenise(query)
    scores   = bm25.get_scores(q_tokens)
    ranked   = [corpus_ids[i] for i in np.argsort(scores)[::-1]]
    bm25_rankings[query] = ranked

print("BM25 ranking complete")
""")

code("""\
# Evaluate BM25
bm25_results = evaluate_model(
    rankings     = bm25_rankings,
    binary_qrels = binary_qrels,
    graded_qrels = graded_qrels,
)

print("BM25 Results:")
for metric, val in bm25_results.items():
    print(f"  {metric:<12} {val:.4f}")
""")

# ── Section 4 — Dense (No Fusion) Baseline ────────────────────────────────
md("""\
## 4. Dense Baseline — Mean-Pooled Embeddings (No Fusion)

**Strategy:** encode every review sentence with a multilingual sentence transformer,
mean-pool all sentence embeddings per restaurant into one vector, then rank by
cosine similarity to the query embedding.

This isolates the contribution of the **fusion step** — everything else is identical
to the proposed model.

> **Note:** encoding 29 k reviews takes ~10–20 min on CPU. No caching is applied here.
""")

code("""\
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
embedder   = SentenceTransformer(MODEL_NAME)
print(f"Loaded: {MODEL_NAME}")
""")

code("""\
# Encode all reviews (one embedding per review)
all_texts = rev["review_text"].tolist()
all_bids  = rev["business_id"].tolist()

print(f"Encoding {len(all_texts):,} reviews...")
all_embeddings = embedder.encode(
    all_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"Embeddings shape: {all_embeddings.shape}")
""")

code("""\
# Mean-pool per restaurant
restaurant_embs = {}
for bid in set(all_bids):
    idxs = [i for i, b in enumerate(all_bids) if b == bid]
    restaurant_embs[bid] = all_embeddings[idxs].mean(axis=0)

bid_list    = list(restaurant_embs.keys())
rest_matrix = np.stack([restaurant_embs[b] for b in bid_list])  # (50, D)

print(f"Restaurant embedding matrix: {rest_matrix.shape}")
""")

code("""\
# Rank all restaurants for each query
dense_rankings: dict = {}
queries = list(binary_qrels.keys())

print(f"Encoding {len(queries)} queries...")
q_embeddings = embedder.encode(queries, show_progress_bar=True, convert_to_numpy=True)

for query, q_emb in tqdm(zip(queries, q_embeddings), total=len(queries), desc="Dense ranking"):
    sims   = cosine_similarity(q_emb.reshape(1, -1), rest_matrix)[0]
    ranked = [bid_list[i] for i in np.argsort(sims)[::-1]]
    dense_rankings[query] = ranked

print("Dense ranking complete")
""")

code("""\
# Evaluate Dense (no fusion)
dense_results = evaluate_model(
    rankings     = dense_rankings,
    binary_qrels = binary_qrels,
    graded_qrels = graded_qrels,
)

print("Dense (no fusion) Results:")
for metric, val in dense_results.items():
    print(f"  {metric:<12} {val:.4f}")
""")

# ── Section 5 — Cross-Model Comparison ────────────────────────────────────
md("""\
## 5. Cross-Model Comparison

Add any new model to `all_results` below to include it in the table automatically.
Every future fusion variant (max-pool, top-k mean, weighted sum, etc.) goes here.
""")

code("""\
# ── Aggregate all model results here ──────────────────────────────────────
# To add a new model: just append it to this dict before running this cell
all_results = {
    "BM25"            : bm25_results,
    "Dense (no fusion)": dense_results,
    # "Top-3 Mean Fusion": top3_results,   # ← add future variants here
}

comparison_df = compare_models(all_results)
print("\\n=== Cross-Model Comparison ===")
comparison_df
""")

code("""\
import matplotlib.pyplot as plt

metrics_to_plot = ["MAP", "nDCG@5", "nDCG@10", "Recall@5", "Recall@10"]
plot_df = comparison_df[metrics_to_plot]

ax = plot_df.plot(kind="bar", figsize=(11, 5), rot=0, width=0.6)
ax.set_title("Baseline Model Comparison — RIRD", fontsize=13)
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
""")

# ── Write notebook ─────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
out = HERE / "baseline_models.ipynb"
nbf.write(nb, str(out))
print(f"Written → {out}")
