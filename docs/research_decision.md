# Research Decision — Language, Cuisine, and Evaluation

Status: open for team discussion (Ton + Sam). Captures the core scope decision
surfaced during dataset EDA. See `PROJECT.md` for the technical framing and
`eda/` for the evidence.

---

## The decision in one sentence

Our project is pitched as **Thai food recommendation**, but our only *labelled*
dataset (RIRD) is **English and almost entirely non-Thai food** — so we must decide
what "Thai" means for us and which dataset carries the *evaluated* contribution.

## "Thai" means two different things

| Sense | Example | Dataset that provides it |
|---|---|---|
| Thai **language** (query + reviews in Thai) | `อยากกินอะไรเปรี้ยวๆ เผ็ดๆ ไม่แพง` | Wongnai |
| Thai **cuisine** (the food being recommended) | pad thai, som tam, khao soi | Wongnai; barely in RIRD |

A truly "Thai food recommendation" system wants **both**. Our datasets split them apart.

## How much Thai food is actually in each dataset?

**RIRD (English, labelled):** 50 Toronto restaurants — only **2 are Thai cuisine**
(Khao San Road, Pai Northern Thai Kitchen ≈ **4%**). The rest is general Toronto
dining:

```
8  Breakfast & Brunch   7  Japanese   5  Ramen   4  Noodles
4  Chinese   4  Canadian (New)   3  Asian Fusion   3  Sushi Bars ...
```

So **English-only on RIRD is not a Thai-food system** — it is a *general restaurant
recommendation* system that happens to include 2 Thai restaurants.

**Wongnai (Thai, unlabelled):** 46k reviews of Thai restaurants written by Thai users —
this is where the actual *Thai food* content lives. But it has **no restaurant id, no
queries, and no relevance labels**, so it cannot be evaluated quantitatively as-is.

**The tension:** the dataset with *labels* has no Thai food; the dataset with *Thai
food* has no labels.

## Options

### Option A — English-only (RIRD)
- **Becomes:** explainable *general* reviewed-item retrieval in Toronto. Drops the
  "Thai food" identity (both language and cuisine — only 4% is Thai).
- **Pros:** fully labelled end-to-end; directly comparable to the ECIR 2023 RIR paper;
  smallest scope/risk; the fusion contribution is unchanged (it is language-agnostic).
- **Cons:** we are no longer a "Thai food" project — this likely needs a retitle and a
  new pitch (*"explainable reviewed-item retrieval / better fusion"*).

### Option B — Thai (Wongnai) as the real system
- **Becomes:** a genuine Thai-food, Thai-language recommender.
- **Pros:** matches the project identity; real Thai cuisine + language.
- **Cons:** **no labels and no restaurant grouping** → no quantitative evaluation, no
  restaurant-level ranking, nothing to compare against. Would require us to hand-build a
  labelled Thai query set or obtain Wongnai metadata with restaurant ids.

### Option C — Hybrid (recommended)
- **Develop + evaluate** the fusion method on **RIRD** (English, labelled) — this is
  where the numbers and the paper comparison come from. The method is language-agnostic,
  so results transfer in principle.
- **Demo in Thai** on **Wongnai** at the **review/sentence level** (Thai query →
  relevant Thai review sentences), presented as the real-world application.
- Keep full Thai *restaurant-level* recommendation as **future work** (needs Wongnai
  restaurant ids or a small labelled Thai query set).
- Lets us keep the "Thai food" framing in the intro/demo while every *measured* claim
  rests on labelled data.

**Recommendation:** Option C. Anything we put a number on lives on RIRD; the Thai food
story is the application and demo. If the course requires the identity to be strictly
"Thai food," we must invest in labelling a small Thai query set on Wongnai (Option B
with added annotation effort) — budget for that explicitly.

---

## Evaluation protocol (how we test the recommender)

This is **query-based Information Retrieval**, not a classic user–item recommender, so
we evaluate it as a ranking problem — not with recsys hit-rate on held-out clicks.

**Ground truth:** `PMD.csv` gives, per query, a binary relevance label for each of the
50 restaurants (aggregated over 5 annotators). 100 queries total.

**Loop:** for each query → rank all 50 restaurants → score the ranking against the
labels → average over the 100 queries.

**Metrics** (rank-aware; positives are sparse ~27%):

| Metric | Purpose |
|---|---|
| **MAP** | headline overall ranking quality |
| **nDCG@k** (k = 5, 10) | position-weighted top-k quality |
| **Precision@k / Recall@k** | did relevant restaurants reach the top-k |
| **MRR** | rank of the first relevant restaurant |

Bonus: use the 5 annotator votes as **graded relevance (0–5)** for graded nDCG.

**Baselines (to prove the fusion helps):**
1. **BM25** — lexical baseline (one afternoon, no GPU).
2. **Dense, no fusion** — mean-pooled review embeddings per restaurant.
3. **Our fusion method** — should beat both; then **ablate** fusion variants
   (max / mean / top-k mean / learned). That ablation table is the contribution's evidence.

**Rigor:**
- **Significance:** only 100 queries → paired t-test / randomization test vs. each baseline.
- **Per-category:** break results down by the 5 RIRD query types (indirect, negation,
  contradictory, …).
- **Leakage:** if we fine-tune the fusion, split **by query** (not by review) into
  train/val/test — reviews are shared across queries.

**Explainability (justifying sentences):** no labels for this — use a proxy (top
sentences should come from relevant restaurants) plus a small qualitative human check.

**Why this reinforces the decision:** all of the above requires *labels*. RIRD has them;
Wongnai does not. That is the concrete reason the evaluated contribution belongs on RIRD
(Option A/C), and why a Thai-only evaluation (Option B) needs new annotation first.
