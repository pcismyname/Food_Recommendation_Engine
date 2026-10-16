"""Generate thai_retrieval.ipynb (task C2: Thai sentence-level BM25 vs. dense on Wongnai)."""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = Path(__file__).resolve().parent
cells = []
md = lambda t: cells.append(new_markdown_cell(t))
code = lambda t: cells.append(new_code_cell(t))

md("""# Thai Taste Retrieval: Wongnai, sentence level (task C2)
**Thai Food Recommendation Engine (Group: Flower)**

Answers professor comment **F1**: our Thai claims were unmeasured. This notebook builds the
retrieval side of the Thai taste test set:

1. Split Wongnai reviews into sentences with PyThaiNLP (`crfcut`) and spot-check the segmentation.
2. Build a Thai BM25 index (`newmm` word tokens) and a dense index (same multilingual encoder as the
   English baseline, `paraphrase-multilingual-mpnet-base-v2`).
3. Run the Thai taste queries in `thai_queries.csv` (C1) through both and compare the top sentences.

For the progress submission this is **qualitative**: no relevance labels exist yet. Labels come
from C3–C5 (blind pooled sheet, hand labels, second annotator), then C7 computes nDCG@10 / P@10.

> Run with the **cpdsai** venv kernel. Needs `pythainlp` and `python-crfsuite`.
""")

code('''from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

HERE = Path.cwd()
ROOT = HERE if (HERE / "datasets").exists() else HERE.parent
DATA = ROOT / "datasets" / "wongnai"
THAI = ROOT / "thai"
CACHE = THAI / "cache"; CACHE.mkdir(exist_ok=True)

N_REVIEWS  = 2000     # subsample for CPU; set None for all 40k (~570k sentences, needs a GPU)
SEED       = 42
MIN_CHARS  = 15       # drop sentence fragments shorter than this
TOP_K      = 5
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TAG = f"n{N_REVIEWS or 'all'}_s{SEED}"

pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 120)
print("root:", ROOT, "| tag:", TAG)''')

md("## 1. Load Wongnai and subsample")
code('''wg = pd.read_parquet(DATA / "train.parquet").reset_index(names="review_id")
if N_REVIEWS:
    wg = wg.sample(N_REVIEWS, random_state=SEED).sort_values("review_id").reset_index(drop=True)
print(wg.shape)
wg.head(3)''')

md("""## 2. Sentence splitting
Thai has no sentence-final punctuation, so `crfcut` guesses boundaries mostly from spaces and
context. Results are cached because splitting takes about 1 s per 100 reviews.""")
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
      f"({len(sents) / sents.review_id.nunique():.1f} per review)")
print(sents.text.str.len().describe().round(1))''')

md("""### 2a. Spot-check (needs a Thai reader)
Print a few reviews as split. Look for: sentences cut mid-phrase, several ideas merged into one
"sentence", and food names split across boundaries. Write what you see in the notes cell below.""")
code('''for rid in sents.review_id.drop_duplicates().sample(5, random_state=1):
    print(f"--- review {rid} ---")
    for s in sents.loc[sents.review_id == rid, "text"]:
        print("  |", s)''')
md("""### 2b. Error patterns across all sentences
Counts the two failure patterns seen in 2a: cuts at abbreviation dots (กทม. / อ. / ม.ค.) and very
long "sentences" where several ideas were merged.""")
code(r"""t = sents.text
thai_abbr = t.str.contains(r"(?:^|[\s(])(?:[ก-ฮ]{1,3}\.)+$")
time_dot  = t.str.contains(r"\d\.\d\d\s*น\.$")
merged    = t.str.len() > 200
label     = t.str.contains(r"(?:\.\.\.|…)\s*$")
non_thai  = (t.str.count(r"[ก-๙]") / t.str.len()) < 0.2
for name, m in [("cut after a Thai abbreviation (excl. times like 18.00 น.)", thai_abbr & ~time_dot),
                ("ends with a '...' section label", label),
                ("> 200 chars (likely merged)", merged),
                ("mostly non-Thai (< 20% Thai characters)", non_thai)]:
    print(f"{name:58s} {m.sum():5,}  ({m.mean():.1%})")
per = sents.groupby("review_id").size()
print(f"\nsentences per review: median {per.median():.0f}; reviews left as one sentence: {(per == 1).mean():.1%}")""")
md("""**Spot-check notes** (drafted by Claude from 2a and 2b; confirmed by Ton 2026-10-17):

- **Reviews checked:** the 5 in 2a, plus pattern counts over all 19,294 sentences in 2b.
- **Short, informal reviews** (most of Wongnai) split cleanly at natural pauses, e.g.
  `ก๋วยเตี๋ยวเรือชามละ 20 บาท ปรุงมาอร่อยแล้วไม่ต้องปรุงเพิ่ม แต่ระวัง!` | `พริกเผ็ดมาก (ไม่กินเผ็ดก็บอกเค้าได้นะ ...)`.
- **Bad cuts (mid-phrase):** crfcut treats the dot in Thai abbreviations as a sentence end:
  `... ใกล้ ๆ กทม.` | `เกษตร วิทยาเขตกำแพงแสน`, `ร้านอาหาร K.U.` | `สโมสร แทน`, `ม.ค.` | `58 ไปจะมีเนื้อดี...`.
  About 0.7% of sentences end in such an abbreviation.
- **Merged sentences / shifted labels:** long blog-style reviews with sections like `รสชาติ ...`,
  `บรรยากาศ ...` get the label glued to the end of the previous section, e.g.
  `ถือว่าเป็นข้อดีข้อเดียวเลยแหละ รสชาตกาแฟ` | `ถ้าเทียบกับราคาคงแพงเกิน`, so the second sentence
  loses what it is about. About 1.6% of sentences are over 200 characters (several ideas merged).
- **Other:** about 0.7% of sentences are mostly English; Thai queries will rarely match them.
- **Verdict:** good enough for sentence retrieval. Taste statements (`เผ็ดมาก`, `ไม่หวานมาก`) are
  short and almost always survive intact. Errors affect a few percent of sentences, mostly in long
  blog-style reviews. State crfcut's published accuracy on review text (≈ 0.77 F1,
  scb-mt-en-th-2020) as a limitation; no fix needed before labelling.
""")

md("## 3. Queries (C1)")
code('''queries = pd.read_csv(THAI / "thai_queries.csv")
print(queries.type.value_counts().to_dict())
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
Same model as the English baseline (`baselines/baseline_models_v2.ipynb`), so the two tracks are
comparable. Embeddings are L2-normalised and cached.""")
code('''from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME)
emb_path = CACHE / f"emb_{MODEL_NAME.split('/')[-1]}_{TAG}_min{MIN_CHARS}.npy"
if emb_path.exists():
    S = np.load(emb_path)
else:
    S = model.encode(sents.text.tolist(), batch_size=64, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(emb_path, S)
Q = model.encode(queries.query_th.tolist(), normalize_embeddings=True, convert_to_numpy=True)
print("sentences:", S.shape, "| queries:", Q.shape)''')

md("## 6. Retrieve top sentences per query")
code('''def top_bm25(qi, k=TOP_K):
    sc = bm25.get_scores(th_tokenise(queries.query_th.iloc[qi]))
    idx = np.argsort(-sc)[:k]
    return idx, sc[idx]

def top_dense(qi, k=TOP_K):
    sc = S @ Q[qi]
    idx = np.argsort(-sc)[:k]
    return idx, sc[idx]

def show(qid, k=TOP_K):
    qi = queries.index[queries.qid == qid][0]
    q = queries.iloc[qi]
    print(f"=== {q.qid}: {q.query_th}  ({q.query_en}) ===")
    for name, fn in [("BM25", top_bm25), ("Dense", top_dense)]:
        idx, sc = fn(qi, k)
        print(f"\\n  [{name}]")
        for r, (i, s) in enumerate(zip(idx, sc), 1):
            print(f"   {r}. ({s:.3f}) {sents.text.iloc[i][:150]}")
    print()''')

md("""### 6a. Qualitative examples for the progress slides
One query of each type: single taste, combination, negation.""")
code('''for qid in ["T02", "T11", "T20"]:
    show(qid)''')
md("""**Observations** (drafted by Claude, confirmed by Ton 2026-10-17; eyeballed, not labels):

- **T02 อยากกินอะไรเผ็ดๆ แบบจัดๆ (really spicy):** BM25 fails. Its top 5 match the words
  อยากกินอะไร / จัด, not spiciness (`อยากกินปูเน้นๆเนื้อๆ เลยจัดเต็ม`). Dense ranks 1–2 are on target
  (`ถ้าชอบเผ็ดสั่งเพิ่มได้เลย`, `สำหรับคนชอบเผ็ดๆเลย`), then drifts, and its rank 5 is the opposite
  taste (`รสชาติจะออกอ่อนๆ`).
- **T11 ต้มยำเปรี้ยวๆ เผ็ดๆ (sour and spicy tom yum):** both are reasonable. BM25 finds exact
  sour-spicy statements (`รสชาติเปรี้ยว เผ็ด ลงตัวดี`) but also a negated one (`รสชาติต้มยำไม่เผ็ดมาก`).
  Dense finds sour-spicy sentences about *other* dishes (ลาบหมู, ยำหมูยอ, มาม่าต้มยำ).
- **T20 ของหวานที่ไม่หวานมาก (not too sweet):** both handle the negation well; most of the top 5
  from each say ไม่หวานมาก / หวานน้อย. One BM25 miss (`ทีเด็ดอยู่ที่ของหวาน`).
- **Pattern:** BM25 is fooled by filler words in casual queries; dense is fooled by topic (any taste
  word, any dish). Both look fine on this negation query, contrary to the English NevIR findings.
  The labels in C4 decide.
""")

md("""## 7. How different are the two systems?
Without labels we can still measure overlap: if BM25 and dense return almost the same sentences,
pooling them for annotation (C3) adds little; if they barely overlap, the pool is larger and the
comparison in C7 is more informative.""")
code('''POOL_K = 10
rows = []
for qi, q in queries.iterrows():
    b = set(top_bm25(qi, POOL_K)[0]); d = set(top_dense(qi, POOL_K)[0])
    rows.append((q.qid, q.type, len(b & d), len(b | d)))
ov = pd.DataFrame(rows, columns=["qid", "type", f"overlap@{POOL_K}", "pool_size"])
print(ov.groupby("type")[[f"overlap@{POOL_K}", "pool_size"]].mean().round(2))
print(f"\\nTotal pool to label at top-{POOL_K}: {ov.pool_size.sum()} pairs "
      f"({len(queries)} queries)")
ov''')

md("""## 8. Takeaways for the progress report

- Segmentation quality: good enough; abbreviation cuts and merged long sentences affect a few
  percent of sentences, mostly in blog-style reviews (from 2a/2b).
- Qualitative BM25 vs. dense: no clear winner. BM25 fails on filler-word queries (T02), dense drifts
  to related tastes or dishes (T11); both handle the T20 negation (from 6a).
- Pool size for annotation: 465 pairs at top-10 for 24 queries; BM25 and dense share fewer than one
  sentence per query on average (from 7). This sets the labelling budget for C4/C5.
- Limitation to state: Wongnai has no restaurant ids, so the Thai evaluation is **sentence level
  only**; restaurant-level Thai ranking is future work.
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (cpdsai)", "language": "python", "name": "cpdsai"},
    "language_info": {"name": "python"},
})
out = HERE / "thai_retrieval.ipynb"
nbf.write(nb, out)
print("wrote", out)
