"""Generate eda_rir_data.ipynb from structured cells, then it can be executed."""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = Path(__file__).resolve().parent
nb = new_notebook()
cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(text):
    cells.append(new_code_cell(text))


md("""# RIRD — Exploratory Data Analysis
**Thai Food Recommendation Engine (Group: Flower)**

Dataset: `datasets/rir_data-main/` — the Reviewed-Item Retrieval Dataset (RIRD, ECIR 2023).

This notebook explores the two data files and answers one framing question:
**should the engine's input query be Thai or English for this dataset?**

- `PMD.csv` — query × restaurant relevance annotations (ground truth)
- `reviews/50_restaurants_all_rates.csv` — Yelp reviews for the 50 restaurants

> Run with the **cpdsai** venv kernel.
""")

code("""import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# resolve repo root whether run from eda/ or repo root
HERE = Path.cwd()
ROOT = HERE if (HERE / "datasets").exists() else HERE.parent
DATA = ROOT / "datasets" / "rir_data-main"

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)
print("data dir:", DATA)""")

code('''# quick Unicode-based script detector (Thai vs Latin)
THAI = re.compile(r"[\\u0e00-\\u0e7f]")
LATIN = re.compile(r"[A-Za-z]")

def script_of(text):
    if not isinstance(text, str) or not text.strip():
        return "empty"
    t, l = bool(THAI.search(text)), bool(LATIN.search(text))
    if t and l: return "mixed"
    if t: return "thai"
    if l: return "latin"
    return "other"''')

md("## 1. Reviews — `50_restaurants_all_rates.csv`")
code('''rev = pd.read_csv(DATA / "reviews" / "50_restaurants_all_rates.csv")
rev = rev.drop(columns=[c for c in rev.columns if c.startswith("Unnamed")])
print("shape:", rev.shape)
rev.head(3)''')

code('''print(rev.dtypes, "\\n")
print("missing:\\n", rev.isna().sum(), "\\n")
print("restaurants:", rev["business_id"].nunique(),
      "| users:", rev["user_id"].nunique(),
      "| dup rows:", rev.duplicated().sum())''')

md("### 1a. Rating distribution\nRatings skew positive — the paper kept only restaurants with avg rating ≥ 3.")
code('''vc = rev["review_stars"].value_counts().sort_index()
print(vc, "\\nmean:", round(rev["review_stars"].mean(), 3))
vc.plot(kind="bar", color="#4C72B0", title="Review star distribution")
plt.xlabel("stars"); plt.ylabel("count"); plt.tight_layout(); plt.show()''')

md("### 1b. Reviews per restaurant")
code('''per = rev.groupby("name").size().sort_values(ascending=False)
print("min %d max %d median %d" % (per.min(), per.max(), per.median()))
print(per.head(5))
per.plot(kind="hist", bins=30, color="#55A868", title="Reviews per restaurant")
plt.xlabel("# reviews"); plt.tight_layout(); plt.show()''')

md("### 1c. Review length\nMedian ≈ 94 words with a long tail → reviews must be split into sentences before encoding.")
code('''txt = rev["review_text"].fillna("").astype(str)
rev["word_len"] = txt.str.split().map(len)
print(rev["word_len"].describe())
rev["word_len"].clip(upper=rev["word_len"].quantile(0.99)).plot(
    kind="hist", bins=50, color="#C44E52", title="Review length (words)")
plt.xlabel("words"); plt.tight_layout(); plt.show()''')

md("### 1d. Categories & dates")
code('''cats = (rev.drop_duplicates("business_id")["categories"]
        .dropna().str.split(",").explode().str.strip())
print("distinct tags:", cats.nunique())
print(cats.value_counts().head(10))
dates = pd.to_datetime(rev["date"], errors="coerce")
print("\\ndate range:", dates.min(), "->", dates.max())''')

md("""## 2. Language check — reviews
This is the decisive cell for the Thai-vs-English question.""")
code('''sample = txt.sample(min(5000, len(txt)), random_state=0)
print((sample.map(script_of).value_counts(normalize=True) * 100).round(2))
print("\\nexamples:")
for s in txt[txt.str.len() > 40].head(3):
    print(" -", s[:140].replace("\\n", " "))''')

md("## 3. PMD — query × restaurant relevance")
code('''pmd = pd.read_csv(DATA / "PMD.csv")
pmd.columns = [c.strip() for c in pmd.columns]
pmd = pmd.drop_duplicates()  # one exact dup row in the raw file
ann = [c for c in pmd.columns if c.lower().startswith("annotator")]
agg = [c for c in pmd.columns if "Low" in c or "High" in c][0]
print("shape:", pmd.shape, "| queries:", pmd["query"].nunique(),
      "| restaurants:", pmd["Restaurant name"].nunique())
pmd.head(3)''')

md("### 3a. Label balance — sparse positives (~27%)")
code('''print(pmd[agg].value_counts().sort_index())
print("positive rate: %.2f%%" % (pmd[agg].mean() * 100))''')

md("### 3b. Annotator behaviour & agreement")
code('''for c in ann:
    print("%s: %.2f%%" % (c, pmd[c].mean() * 100))
votesum = pmd[ann].sum(axis=1)
votesum.value_counts().sort_index().plot(
    kind="bar", color="#8172B3", title="# annotators voting 'relevant' (0-5)")
plt.xlabel("# positive votes"); plt.ylabel("rows"); plt.tight_layout(); plt.show()''')

md("### 3c. Language check — queries")
code('''q = pmd["query"].dropna().drop_duplicates()
print(q.map(script_of).value_counts())
print("\\nexamples:")
for x in q.head(8):
    print(" -", x)''')

md("""## 4. Conclusion — Thai or English?

| Text | Thai | English |
|------|------|---------|
| reviews | 0% | **100%** |
| queries | 0% | **100%** |

RIRD is entirely **English** (Toronto / Yelp). Decisions:

1. **Use RIRD in English** as the *evaluation protocol + baseline* — it is the only
   source here with ground-truth relevance labels to score the fusion method.
2. **Do not** run Thai queries against RIRD — there is nothing Thai to match.
3. **Thai input** belongs to the **Wongnai / Foods-in-Thailand** corpora; transfer
   the validated fusion method there with a multilingual encoder.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "cpdsai", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = HERE / "eda_rir_data.ipynb"
nbf.write(nb, out)
print("wrote", out)
