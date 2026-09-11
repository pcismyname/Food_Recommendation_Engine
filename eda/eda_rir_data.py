"""
EDA for the RIRD dataset (datasets/rir_data-main).

Covers:
  - PMD.csv                         : query x restaurant relevance annotations
  - reviews/50_restaurants_all_rates.csv : Yelp reviews for the 50 restaurants

Also runs a lightweight language check on the review text and queries to
settle the "should the input be Thai or English?" question for the
Food Recommendation Engine.

Outputs:
  - printed summary to stdout
  - figures under eda/figures/
Run with the cpdsai venv python.
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "datasets" / "rir_data-main"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)


def h(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# Unicode ranges for quick script detection
THAI = re.compile(r"[฀-๿]")
LATIN = re.compile(r"[A-Za-z]")


def script_of(text):
    if not isinstance(text, str) or not text.strip():
        return "empty"
    thai = bool(THAI.search(text))
    latin = bool(LATIN.search(text))
    if thai and latin:
        return "mixed"
    if thai:
        return "thai"
    if latin:
        return "latin"
    return "other"


# --------------------------------------------------------------------------
# 1. Reviews
# --------------------------------------------------------------------------
h("1. REVIEWS  —  reviews/50_restaurants_all_rates.csv")
rev = pd.read_csv(DATA / "reviews" / "50_restaurants_all_rates.csv")

# drop junk index columns if present
junk = [c for c in rev.columns if c.startswith("Unnamed")]
if junk:
    print(f"Dropping index columns: {junk}")
    rev = rev.drop(columns=junk)

print(f"\nShape: {rev.shape[0]:,} rows x {rev.shape[1]} cols")
print("\nColumns / dtypes:")
print(rev.dtypes.to_string())

print("\nMissing values per column:")
print(rev.isna().sum().to_string())

print("\nUniques:")
print(f"  restaurants (business_id): {rev['business_id'].nunique()}")
print(f"  restaurants (name)       : {rev['name'].nunique()}")
print(f"  users (user_id)          : {rev['user_id'].nunique():,}")
print(f"  duplicate rows           : {rev.duplicated().sum():,}")

# rating distribution
h("1a. Review rating distribution")
vc = rev["review_stars"].value_counts().sort_index()
print(vc.to_string())
print(f"\nmean stars: {rev['review_stars'].mean():.3f}  median: {rev['review_stars'].median()}")

plt.figure(figsize=(6, 4))
vc.plot(kind="bar", color="#4C72B0")
plt.title("Review star rating distribution")
plt.xlabel("stars")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(FIG / "review_star_distribution.png", dpi=120)
plt.close()

# reviews per restaurant
h("1b. Reviews per restaurant")
per = rev.groupby("name").size().sort_values(ascending=False)
print(f"min {per.min()}, max {per.max()}, mean {per.mean():.0f}, median {per.median():.0f}")
print("\nTop 5:")
print(per.head(5).to_string())
print("\nBottom 5:")
print(per.tail(5).to_string())

plt.figure(figsize=(7, 4))
per.plot(kind="hist", bins=30, color="#55A868")
plt.title("Reviews per restaurant")
plt.xlabel("# reviews")
plt.tight_layout()
plt.savefig(FIG / "reviews_per_restaurant.png", dpi=120)
plt.close()

# review length
h("1c. Review text length (characters & words)")
txt = rev["review_text"].fillna("").astype(str)
rev["char_len"] = txt.str.len()
rev["word_len"] = txt.str.split().map(len)
print(rev[["char_len", "word_len"]].describe().to_string())

plt.figure(figsize=(7, 4))
rev["word_len"].clip(upper=rev["word_len"].quantile(0.99)).plot(
    kind="hist", bins=50, color="#C44E52"
)
plt.title("Review length (words, 99th pct clipped)")
plt.xlabel("words")
plt.tight_layout()
plt.savefig(FIG / "review_word_length.png", dpi=120)
plt.close()

# categories
h("1d. Restaurant categories")
cats = (
    rev.drop_duplicates("business_id")["categories"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)
print(f"distinct category tags: {cats.nunique()}")
print("\nTop 15 category tags (by # restaurants):")
print(cats.value_counts().head(15).to_string())

# dates
h("1e. Review dates")
dates = pd.to_datetime(rev["date"], errors="coerce")
print(f"parsed: {dates.notna().sum():,} / {len(dates):,}")
print(f"range: {dates.min()}  ->  {dates.max()}")

# --------------------------------------------------------------------------
# 2. LANGUAGE CHECK (the key question)
# --------------------------------------------------------------------------
h("2. LANGUAGE CHECK  —  review_text")
sample = txt.sample(min(5000, len(txt)), random_state=0)
rev_lang = sample.map(script_of).value_counts(normalize=True) * 100
print("Script distribution on 5k sampled reviews (%):")
print(rev_lang.round(2).to_string())
print("\nExample reviews:")
for s in txt[txt.str.len() > 40].head(3):
    print("  -", s[:160].replace("\n", " "))

# --------------------------------------------------------------------------
# 3. PMD — query/restaurant relevance
# --------------------------------------------------------------------------
h("3. PMD.csv  —  query x restaurant relevance annotations")
pmd = pd.read_csv(DATA / "PMD.csv")
pmd.columns = [c.strip() for c in pmd.columns]
print(f"\nShape: {pmd.shape[0]:,} rows x {pmd.shape[1]} cols")
print("Columns:", list(pmd.columns))

ann_cols = [c for c in pmd.columns if c.lower().startswith("annotator")]
agg_col = [c for c in pmd.columns if "Low" in c or "High" in c]
agg_col = agg_col[0] if agg_col else None

print(f"\nqueries     : {pmd['query'].nunique()}")
print(f"restaurants : {pmd['Restaurant name'].nunique()}")
print(f"query x restaurant rows: {len(pmd):,}")
print("\nMissing values:")
print(pmd.isna().sum().to_string())

h("3a. Relevance label balance")
if agg_col:
    bal = pmd[agg_col].value_counts(dropna=False).sort_index()
    print(f"Aggregated label '{agg_col}':")
    print(bal.to_string())
    pos = pmd[agg_col].mean()
    print(f"\npositive (relevant) rate: {pos:.3%}")

h("3b. Annotator behaviour & agreement")
print("Per-annotator positive rate:")
for c in ann_cols:
    print(f"  {c}: {pmd[c].mean():.3%}")

# simple pairwise agreement (% identical labels)
print("\nMean vote sum per row (0-5) describe:")
votesum = pmd[ann_cols].sum(axis=1)
print(votesum.describe().to_string())

plt.figure(figsize=(6, 4))
votesum.value_counts().sort_index().plot(kind="bar", color="#8172B3")
plt.title("Number of annotators voting 'relevant' (0-5)")
plt.xlabel("# positive votes")
plt.ylabel("rows")
plt.tight_layout()
plt.savefig(FIG / "pmd_vote_agreement.png", dpi=120)
plt.close()

h("3c. LANGUAGE CHECK  —  queries")
q_lang = pmd["query"].dropna().drop_duplicates().map(script_of).value_counts()
print("Script of distinct queries:")
print(q_lang.to_string())
print("\nExample queries:")
for q in pmd["query"].dropna().drop_duplicates().head(8):
    print("  -", q)

print("\n\nDone. Figures written to", FIG)
