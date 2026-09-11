"""Generate eda_thai_datasets.ipynb (Wongnai + Thailand Foods + M-ABSA Thai)."""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = Path(__file__).resolve().parent
cells = []
md = lambda t: cells.append(new_markdown_cell(t))
code = lambda t: cells.append(new_code_cell(t))

md("""# Thai Datasets — EDA
**Thai Food Recommendation Engine (Group: Flower)**

The Thai side of the project. Companion to `eda_rir_data.ipynb` (the English RIRD baseline).

| Dataset | Path | Role |
|---|---|---|
| **Wongnai** | `datasets/wongnai/` | Thai review corpus (the real Thai input demo) |
| **Thailand Foods** | `datasets/thailand_foods.csv` | Dish gazetteer (ingredients, region, course) |
| **M-ABSA (Thai)** | `datasets/m-absa/*/th/` | Thai aspect-based sentiment signal |

**Product goal (unchanged):** a user types a **Thai** taste preference and gets ranked
recommendations with justifying review sentences. See the conclusion for how each
dataset supports (or limits) that.

> Run with the **cpdsai** venv kernel.
""")

code('''import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path.cwd()
ROOT = HERE if (HERE / "datasets").exists() else HERE.parent
DATA = ROOT / "datasets"

# Thai fonts are usually absent from matplotlib; keep Thai text out of plot labels.
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 40)

THAI = re.compile(r"[\\u0e00-\\u0e7f]"); LATIN = re.compile(r"[A-Za-z]")
def script_of(t):
    if not isinstance(t, str) or not t.strip(): return "empty"
    th, la = bool(THAI.search(t)), bool(LATIN.search(t))
    if th and la: return "mixed"
    if th: return "thai"
    if la: return "latin"
    return "other"
print("data dir:", DATA)''')

# ---- Wongnai ----
md("""## 1. Wongnai reviews
Source: HuggingFace `Highgroundbkk/wongnai_reviews`, exported to parquet.
Only two columns: `review_body` (Thai) and `star_rating`.""")
code('''wg_tr = pd.read_parquet(DATA / "wongnai" / "train.parquet")
wg_te = pd.read_parquet(DATA / "wongnai" / "test.parquet")
print("train:", wg_tr.shape, "| test:", wg_te.shape)
print("columns:", list(wg_tr.columns), "| nulls:", wg_tr.isna().sum().to_dict())
wg_tr.head(3)''')

md("### 1a. Rating distribution — note the scale is **0–4**, not 1–5")
code('''vc = wg_tr["star_rating"].value_counts().sort_index()
print(vc)
vc.plot(kind="bar", color="#4C72B0", title="Wongnai star_rating (train)")
plt.xlabel("star_rating (0-4)"); plt.ylabel("count"); plt.tight_layout(); plt.show()''')

md("### 1b. Review length\nMedian ~20 words — far shorter than RIRD's ~94. Sentence splitting still helps on the long tail.")
code('''wg_tr["words"] = wg_tr["review_body"].str.split().map(len)
print(wg_tr["words"].describe())
wg_tr["words"].clip(upper=wg_tr["words"].quantile(0.99)).plot(
    kind="hist", bins=50, color="#55A868", title="Wongnai review length (words)")
plt.xlabel("words"); plt.tight_layout(); plt.show()''')

md("### 1c. Language check")
code('''sc = wg_tr["review_body"].sample(5000, random_state=0).map(script_of)
print((sc.value_counts(normalize=True) * 100).round(2))
print("\\nexample (Thai):")
print(" ", wg_tr["review_body"].iloc[0][:160].replace(chr(10), " "))''')

md("""> **Structural limitation:** Wongnai has **no restaurant id, no query, no relevance label** —
> only review text + rating. It is a Thai *text* corpus, not a labelled
> reviewed-item-retrieval benchmark like RIRD.""")

# ---- Thailand Foods ----
md("""## 2. Thailand Foods (dish gazetteer)
A reference table of Thai dishes — not reviews. Useful for normalizing / expanding
food terms in queries and reviews.""")
code('''foods = pd.read_csv(DATA / "thailand_foods.csv")
print("shape:", foods.shape, "| cols:", list(foods.columns))
print("nulls:", foods.isna().sum().to_dict())
dups = foods.duplicated(subset=["en_name"]).sum()
print("duplicate en_name:", dups)
foods.head(3)''')

md("### 2a. Dedupe + category breakdown")
code('''foods_clean = foods.drop_duplicates(subset=["en_name"]).reset_index(drop=True)
print("after dedupe:", foods_clean.shape[0], "dishes")
print("\\ncourse:\\n", foods_clean["course"].value_counts().to_string())
print("\\nregion (note heavy Various/Unknown):\\n", foods_clean["region"].value_counts().to_string())
foods_clean["course"].value_counts().plot(
    kind="bar", color="#C44E52", title="Thailand Foods: course")
plt.ylabel("dishes"); plt.tight_layout(); plt.show()''')

md("### 2b. Ingredient vocabulary\nIngredients are `+`-separated; this gives a rough flavour/ingredient lexicon.")
code('''ing = (foods_clean["ingredients"].dropna().str.split("+").explode().str.strip().str.lower())
print("distinct ingredient tokens:", ing.nunique())
print("\\ntop 15:\\n", ing.value_counts().head(15).to_string())''')

# ---- M-ABSA ----
md("""## 3. M-ABSA (Thai) — aspect-based sentiment
Format per line: `sentence####[[target, aspect_category, polarity], ...]`.
Food-relevant domains: `restaurant` and `food`.""")
code('''def load_mabsa(domain):
    rows = []
    for split in ["train", "dev", "test"]:
        p = DATA / "m-absa" / domain / "th" / f"{split}.txt"
        for line in p.read_text(encoding="utf-8").splitlines():
            if "####" not in line: continue
            sent, tuples = line.split("####", 1)
            try:
                tup = eval(tuples)
            except Exception:
                tup = []
            rows.append({"domain": domain, "split": split, "sentence": sent, "tuples": tup})
    return pd.DataFrame(rows)

mabsa = pd.concat([load_mabsa("restaurant"), load_mabsa("food")], ignore_index=True)
print("rows:", len(mabsa))
print(mabsa.groupby(["domain", "split"]).size())''')

md("### 3a. Aspect categories & polarity\nThese are coarse ABSA labels — **not** flavour axes (sour/spicy/cheap).")
code('''exploded = mabsa.explode("tuples").dropna(subset=["tuples"])
exploded = exploded[exploded["tuples"].map(lambda x: isinstance(x, (list, tuple)) and len(x) == 3)]
cat = exploded["tuples"].map(lambda x: x[1])
pol = exploded["tuples"].map(lambda x: x[2])
print("aspect categories:\\n", cat.value_counts().to_string())
print("\\npolarity:\\n", pol.value_counts().to_string())
pol.value_counts().plot(kind="bar", color="#8172B3", title="M-ABSA Thai (food+restaurant) polarity")
plt.ylabel("aspect mentions"); plt.tight_layout(); plt.show()''')

# ---- Conclusion ----
md("""## 4. Conclusion — does "user types Thai → gets recommendation" still hold?

**Yes, that is still the product goal.** What the data dictates is *how* we get there:

| Need | Dataset | Status |
|---|---|---|
| Method development + **evaluation** (restaurant grouping + relevance labels) | **RIRD (English)** | ✅ has `business_id` + `PMD.csv` labels |
| **Thai** review text for the live demo | **Wongnai** | ✅ 46k Thai reviews — but **no restaurant id / query / label** |
| Food term normalization / expansion | **Thailand Foods** | ✅ ~320 dishes (after dedupe) |
| Thai aspect-sentiment signal | **M-ABSA Thai** | ✅ coarse ABSA, not flavour axes |

**Plan:**
1. Build & measure the fusion/ranking method on **RIRD** (English, labelled).
2. Demo in **Thai** on **Wongnai** — at the **review/sentence level** (Thai query →
   most relevant Thai review sentences), because Wongnai lacks restaurant grouping.
3. For true Thai *restaurant* recommendation, either obtain Wongnai metadata with
   restaurant ids, or translate RIRD queries to Thai for a restaurant-level demo.

So the Thai-input experience is real; only the *restaurant-level evaluation* stays on
RIRD until richer Thai metadata exists.
""")

nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "cpdsai", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = HERE / "eda_thai_datasets.ipynb"
nbf.write(nb, out)
print("wrote", out)
