"""Generate thai_retrieval_10k_colab.ipynb (task C3 input: top-10 BM25 + dense pool on 10k Wongnai reviews).

Same pipeline as build_thai_retrieval_notebook.py (the 2k C2 notebook, kept unchanged as a record),
scaled to N_REVIEWS = 10000 and made to run on Google Colab (GPU) as well as locally.
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = Path(__file__).resolve().parent
cells = []
md = lambda t: cells.append(new_markdown_cell(t))
code = lambda t: cells.append(new_code_cell(t))

md("""# Thai Taste Retrieval: 10k Wongnai reviews, top-10 export (task C3 input)
**Thai Food Recommendation Engine (Group: Flower)**

**Purpose.** Builds the candidate pool for the Thai taste test set, which answers professor comment
**F1** (our Thai claims were unmeasured). For each of the 24 frozen queries in `thai_queries.csv`
it takes the top 10 sentences from BM25 and the top 10 from the dense encoder and exports them to
one CSV, `topk_n10000_s42.csv`. The pooling script (C3) turns that CSV into the blind labelling sheet.

**Why 10k reviews.** In the 2k-review corpus (`thai_retrieval.ipynb`, C2), rare tastes may have
fewer than 10 relevant sentences in total, so P@10 would be capped by the corpus, not the systems.
Relevance labels are tied to the exact sentence set, so the corpus size is fixed **before**
labelling: 10,000 reviews, `SEED = 42`. Do not change these after C4 starts.

**Pipeline** (identical to the 2k notebook): PyThaiNLP `sent_tokenize(engine="crfcut")`,
whitespace normalisation, drop sentences shorter than 15 characters; BM25Okapi over `newmm` word
tokens (lower-cased, stop words kept); dense = `paraphrase-multilingual-mpnet-base-v2`, cosine on
L2-normalised embeddings.

**Colab steps**
1. *Runtime → Change runtime type → T4 GPU*.
2. If you did not clone the repo into Colab, upload `thai/thai_queries.csv` to `/content`
   (Files panel on the left).
3. *Runtime → Run all*. Wongnai is downloaded from Hugging Face; splitting takes several minutes.
4. The last cell downloads `topk_n10000_s42.csv`. Put it in `thai/results/` in the repo.

Locally the notebook also runs from the repo (reads `datasets/wongnai/train.parquet`), but the
machine is slow, so Colab is the intended runtime.
""")

md("## 0. Install packages (Colab)")
code('''%pip install -q pythainlp python-crfsuite rank-bm25 sentence-transformers datasets pyarrow''')

code('''from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

IN_COLAB = Path("/content").exists()
HERE = Path.cwd()
ROOT = HERE if (HERE / "datasets").exists() else HERE.parent
DATA = ROOT / "datasets" / "wongnai"
LOCAL_REPO = (ROOT / "thai" / "thai_queries.csv").exists() and (DATA / "train.parquet").exists()

BASE  = Path("/content") if IN_COLAB and not LOCAL_REPO else ROOT / "thai"
CACHE = BASE / "cache";   CACHE.mkdir(parents=True, exist_ok=True)
OUT   = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)

N_REVIEWS  = 10000
SEED       = 42
MIN_CHARS  = 15       # drop sentence fragments shorter than this
TOP_K      = 10       # per system, per query
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TAG = f"n{N_REVIEWS or 'all'}_s{SEED}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 256 if DEVICE == "cuda" else 64   # 256 fits a T4 (16 GB) comfortably for mpnet-base
pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 120)
print("colab:", IN_COLAB, "| local repo:", LOCAL_REPO, "| device:", DEVICE,
      torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
print("cache:", CACHE, "| out:", OUT, "| tag:", TAG)''')

md("""## 1. Load Wongnai and subsample
Locally: `datasets/wongnai/train.parquet`. On Colab: the same Hugging Face dataset that
`datasets/download_datasets.py` saves (`Highgroundbkk/wongnai_reviews`, train split, `to_pandas()`),
so row order matches and `review_id` = row index of the train split in both cases.""")
code('''if (DATA / "train.parquet").exists():
    train = pd.read_parquet(DATA / "train.parquet")
    print("loaded local parquet")
else:
    from datasets import load_dataset
    train = load_dataset("Highgroundbkk/wongnai_reviews", split="train").to_pandas()
    print("loaded from Hugging Face")
assert len(train) == 40000, f"expected 40,000 train reviews, got {len(train):,}"

wg = train.reset_index(names="review_id")
if N_REVIEWS:
    wg = wg.sample(N_REVIEWS, random_state=SEED).sort_values("review_id").reset_index(drop=True)
print(wg.shape)
wg.head(3)''')

md("""## 2. Sentence splitting
Same as the 2k notebook (`crfcut`, whitespace normalised, `MIN_CHARS` filter). Cached; this is
CPU-bound (about 1 s per 100 reviews), so expect a few minutes for 10k.""")
code('''from pythainlp.tokenize import sent_tokenize

sent_path = CACHE / f"sentences_{TAG}.parquet"
if sent_path.exists():
    sents = pd.read_parquet(sent_path)
else:
    rows = []
    for rid, text, star in tqdm(wg[["review_id", "review_body", "star_rating"]].itertuples(index=False),
                                total=len(wg)):
        for j, s in enumerate(sent_tokenize(text, engine="crfcut")):
            s = " ".join(s.split())
            if len(s) >= MIN_CHARS:
                rows.append((rid, j, s, star))
    sents = pd.DataFrame(rows, columns=["review_id", "sent_idx", "text", "star_rating"])
    sents.to_parquet(sent_path)

print(f"{len(sents):,} sentences from {sents.review_id.nunique():,} reviews "
      f"({len(sents) / sents.review_id.nunique():.1f} per review)")''')

md("""## 3. Queries (C1, frozen)
Looks for `thai_queries.csv` in `/content`, the working directory, `./thai`, and the repo layout.""")
code('''cands = [Path("/content/thai_queries.csv"), HERE / "thai_queries.csv", HERE / "thai" / "thai_queries.csv",
         ROOT / "thai" / "thai_queries.csv", Path("/content/Food_Recommendation_Engine/thai/thai_queries.csv")]
q_path = next((p for p in cands if p.exists()), None)
if q_path is None:
    raise FileNotFoundError("thai_queries.csv not found. Upload thai/thai_queries.csv from the repo to "
                            "/content (Files panel on the left), then re-run this cell.")
queries = pd.read_csv(q_path)
assert len(queries) == 24, f"expected 24 frozen queries, got {len(queries)}"
print("queries:", q_path, "|", queries.type.value_counts().to_dict())
queries''')

md("""## 4. BM25 on Thai word tokens
`newmm` dictionary segmentation; whitespace tokens dropped and Latin text lower-cased.
Thai stop words are **kept**, since negations like ไม่ are exactly what the taste queries depend on.""")
code('''from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

def th_tokenise(text):
    return [t.lower() for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]

bm25 = BM25Okapi([th_tokenise(s) for s in tqdm(sents.text, desc="tokenise")])
print(th_tokenise(queries.query_th.iloc[10]))''')

md("""## 5. Dense encoder
Same model as the English baseline and the 2k notebook. Embeddings are L2-normalised and cached.""")
code('''from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME, device=DEVICE)
emb_path = CACHE / f"emb_{MODEL_NAME.split('/')[-1]}_{TAG}_min{MIN_CHARS}.npy"
if emb_path.exists():
    S = np.load(emb_path)
else:
    S = model.encode(sents.text.tolist(), batch_size=BATCH, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(emb_path, S)
assert len(S) == len(sents), "embedding cache does not match sentences; delete the cache files"
Q = model.encode(queries.query_th.tolist(), normalize_embeddings=True, convert_to_numpy=True)
print("sentences:", S.shape, "| queries:", Q.shape)''')

md("""## 6. Top-10 per system and export
One row per (query, system, rank). BM25 can score fewer than 10 sentences above zero for a query;
the top 10 by score are still exported and the `score` column shows the zero-score hits.""")
code('''def top_bm25(qi, k=TOP_K):
    sc = bm25.get_scores(th_tokenise(queries.query_th.iloc[qi]))
    idx = np.argsort(-sc, kind="stable")[:k]
    return idx, sc[idx]

def top_dense(qi, k=TOP_K):
    sc = S @ Q[qi]
    idx = np.argsort(-sc, kind="stable")[:k]
    return idx, sc[idx]

COLS = ["qid", "query_th", "system", "rank", "sent_id", "review_id", "sent_idx", "text", "score",
        "star_rating", "n_reviews", "seed"]
rows = []
for qi, q in queries.iterrows():
    for system, fn in [("bm25", top_bm25), ("dense", top_dense)]:
        idx, sc = fn(qi)
        for r, (i, s) in enumerate(zip(idx, sc), 1):
            x = sents.iloc[i]
            rows.append((q.qid, q.query_th, system, r, f"{int(x.review_id)}_{int(x.sent_idx)}",
                         int(x.review_id), int(x.sent_idx), x.text, float(s), int(x.star_rating),
                         N_REVIEWS, SEED))
topk = pd.DataFrame(rows, columns=COLS)
assert len(topk) == len(queries) * 2 * TOP_K

out_csv = OUT / f"topk_{TAG}.csv"
topk.to_csv(out_csv, index=False, encoding="utf-8-sig")
print("wrote", out_csv, topk.shape)
topk.head(12)''')

md("## 7. Summary")
code('''n_rev = sents.review_id.nunique()
print(f"sentences: {len(sents):,} from {n_rev:,} reviews ({len(sents) / n_rev:.2f} per review)")

zero = topk[topk.system == "bm25"].groupby("qid").score.apply(lambda s: int((s <= 0).sum()))
print("\\nBM25 zero-score hits in top 10, per query (non-zero only):",
      zero[zero > 0].to_dict() or "none")

b = topk[topk.system == "bm25"].groupby("qid").sent_id.apply(set)
d = topk[topk.system == "dense"].groupby("qid").sent_id.apply(set)
ov = pd.DataFrame({"type": queries.set_index("qid").type,
                   "overlap@10": [len(b[q] & d[q]) for q in queries.qid],
                   "pool_size":  [len(b[q] | d[q]) for q in queries.qid]})
print(ov.groupby("type")[["overlap@10", "pool_size"]].mean().round(2))
print(f"\\nExpected pool to label: {ov.pool_size.sum()} (query, sentence) pairs over {len(queries)} queries; "
      f"{topk.sent_id.nunique()} unique sentences")
ov''')

md("## 8. Download the CSV (Colab only)\nThen put it in `thai/results/` in the repo.")
code('''try:
    from google.colab import files
    files.download(str(out_csv))
except ImportError:
    print("not on Colab; CSV is at", out_csv)''')

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
})
out = HERE / "thai_retrieval_10k_colab.ipynb"
nbf.write(nb, out)
print("wrote", out)
