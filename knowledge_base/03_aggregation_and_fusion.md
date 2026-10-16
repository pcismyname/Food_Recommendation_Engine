# 03 · Aggregation and Fusion: From Review Scores to a Restaurant Score

> **Why this matters for our project**
> Fusion was the proposal's stated research contribution. The current scope (Track A) is a fair comparison instead: BM25 and dense on the same unit with the same aggregations, plus a review-volume control. This is exactly what the professor criticised (F2): different units for BM25 and dense, and max / top-k fusion biased by review volume.
> This file explains what "unit" and "aggregation" mean, why the bias happens, and how task A3 tests it.

---

## 1. The "unit" question

Before scoring anything, decide **what one scored piece of text is**. We call that the **unit**. There are three choices:

```
Restaurant  ─────────────────────────────────────────────  (1) restaurant document
 ├─ Review 1 ──────────────────────                        (2) review
 │   ├─ "The tom yum was fiery."                           (3) sentence
 │   └─ "Service was slow."
 ├─ Review 2 ──────────────────────
 │   └─ ...
 └─ ... (406 to 2,281 reviews per restaurant in RIRD)
```

| Unit | What gets scored | Pros | Cons |
|---|---|---|---|
| Restaurant document | All reviews glued together (or averaged) | One score, no aggregation needed | Huge, blurry text; no natural explanation |
| Review | Each review | Natural unit; matches RIRD's original setup | A review mixes several topics |
| Sentence | Each sentence | Usually one topic; is directly the explanation we show | Many more units; short sentences lack context |

### Why both models must use the same unit (professor's F2)

In v1, **BM25 scored one concatenated document per restaurant**, while **dense mean-pooled review embeddings into one vector per restaurant**. When dense won, we couldn't tell whether that was because dense is a better *model* or because its *unit and pooling* happened to suit the task better. Two things changed at once: a **confounded** comparison.

The fix is a controlled experiment: change one thing at a time. In v2 (`UNIT = 'review'`), **both models score the same reviews and go through the same aggregation functions**. Now a BM25 vs. dense gap is about the model. The sentence-level run (`UNIT = 'sentence'`) is coded but not yet run (task A1).

## 2. Early vs. late fusion

Two ways to go from many reviews to one restaurant:

- **Early fusion**: merge the *inputs* first, then score once.
  - v1 BM25: concatenate all review tokens into one document.
  - v1 dense: average the review embeddings into one restaurant vector.
- **Late fusion**: score every unit separately, then merge the *scores*.
  - v2: score each review, then take max / mean / top-k mean per restaurant.

```
EARLY:  reviews ──merge──► 1 text/vector ──score──► restaurant score
LATE:   reviews ──score each──► many scores ──aggregate──► restaurant score
```

Late fusion has a big advantage for us: we know **which** reviews/sentences scored highest, so we can show them as the explanation.

(Reminder from file 02: "fusion" here means combining unit scores into an item score. It is not hybrid BM25 + dense.)

## 3. Aggregation functions

Say restaurant $r$ has $n$ units with scores $s_1, \dots, s_n$ for a query. Sort them so $s_{(1)} \ge s_{(2)} \ge \dots$ (the bracketed index means "1st largest, 2nd largest, ...").

- **Max**: $\text{score}(r) = s_{(1)}$ — the single best unit.
- **Mean**: $\text{score}(r) = \frac{1}{n}\sum_{i=1}^{n} s_i$ — average over *all* units.
- **Top-k mean**: $\text{score}(r) = \frac{1}{k}\sum_{i=1}^{k} s_{(i)}$ — average of the $k$ best units. We use $k = 3, 5, 10$. Max is top-1; mean is top-$n$. Top-k sits between them.

### Worked example (toy numbers, not our results)

Query: "spicy food". Cosine scores of reviews:

| | Restaurant A: small spicy specialist | Restaurant B: big generalist | Restaurant C: popular, partly spicy |
|---|---|---|---|
| reviews | 6 | 2,000 | 1,000 |
| scores | 0.70, 0.68, 0.65, 0.60, 0.40, 0.35 | almost all ≈ 0.20; by chance the top 3 are 0.74, 0.72, 0.71 | 300 reviews ≈ 0.70, 700 reviews ≈ 0.20 |
| **max** | 0.70 | **0.74** | 0.70 |
| **top-3 mean** | 0.677 | **0.723** | 0.70 |
| **mean** | **0.563** | 0.20 | 0.35 |

Calculations: A's mean = 3.38 / 6 = 0.563. C's mean = (300 × 0.70 + 700 × 0.20) / 1000 = (210 + 140) / 1000 = 0.35.

What happened:

- **Max and top-k reward volume.** B is not a spicy place. But with 2,000 reviews, a few will mention chili or heat just by chance, and the max only needs *one*. B beats the genuine specialist A.
- **Mean dilutes.** C has 300 enthusiastic spicy reviews, yet its mean (0.35) loses to A because 700 reviews talk about other things. Mean punishes restaurants for having reviews about *anything else*, and long-tail chatter (parking, waiting time) drags everyone down.
- **Top-k mean** is a compromise: averaging several strong units is harder to win by one lucky review than max, but it still benefits from having more "lottery tickets".

### Why more reviews = more lucky high scores

Suppose any single review has a 0.1% chance of scoring above some high threshold just by accident. The chance that **at least one** of $n$ reviews does is $1 - 0.999^{\,n}$:

- smallest RIRD restaurant, $n = 406$: $1 - 0.999^{406} \approx 0.33$
- largest, $n = 2{,}281$: $1 - 0.999^{2281} \approx 0.90$

Same per-review luck, very different restaurant outcome. This is the **volume bias** the professor flagged.

Caution: a correlation with volume is not *automatically* bias. Popular restaurants may genuinely be relevant to more queries (that is why the notebook includes review-count and popularity floors). That's why we need a controlled test, not just a correlation.

## 4. What our review-level results say so far

From `action_items.md` §2 (review level, v2):

- **Top-k mean beats max and mean** for both BM25 and dense; **max is the weakest dense variant**.
- Best model: **Dense, top-10 mean**: nDCG@10 **0.668** [0.625, 0.711], MAP **0.608**.
- **Not yet done:** the sentence-level run and the review-volume control. So the volume half of F2 is still open.

## 5. The volume-control experiment (task A3)

**Idea:** remove the volume difference and see what changes.

1. **Subsample**: for every restaurant, randomly keep exactly **400 reviews**. The smallest RIRD restaurant has 406, so every restaurant can be cut to the same size.
2. **Re-score from the cache**: the review embeddings and BM25 scores already exist; we just keep the selected columns of the score matrix. No re-encoding needed (file 02 §8).
3. **Re-aggregate and re-evaluate** with every aggregation (max, mean, top-3/5/10).
4. **Measure the link between restaurant score and review count** for each aggregation, with Spearman correlation (below), on the full data and after subsampling (comparing to the *original* review count).
5. Ideally repeat with several random seeds, because one random subsample could be lucky.

**How to read it:** if max/top-k scores correlate strongly with review count on the full data, and that correlation (or the metric advantage) drops once everyone has 400 reviews, the advantage came partly from volume. If rankings and metrics barely change, volume was not driving them.

### Spearman correlation, explained

**Pearson** correlation measures how well two variables fit a straight line. **Spearman** correlation ($\rho$, "rho") is Pearson computed on **ranks** instead of raw values. So it asks only: "when one goes up, does the other tend to go up?" — not "is it linear?" That suits us: we care whether more reviews tends to mean a higher score, not about a linear relationship, and review counts are very skewed (406 to 2,281).

Without ties:

$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

where $n$ = number of items (restaurants), and $d_i$ = for restaurant $i$, its rank by review count minus its rank by score. $\rho = +1$: perfectly same order; $0$: no monotone relation; $-1$: reversed.

**Worked example** with 5 restaurants for one query:

| Restaurant | reviews | rank by reviews | max-score rank | $d$ | $d^2$ |
|---|---|---|---|---|---|
| R1 | 406 | 1 | 1 | 0 | 0 |
| R2 | 800 | 2 | 3 | −1 | 1 |
| R3 | 1,200 | 3 | 2 | 1 | 1 |
| R4 | 1,700 | 4 | 4 | 0 | 0 |
| R5 | 2,281 | 5 | 5 | 0 | 0 |

(ranks from smallest to largest in both columns)

$\sum d^2 = 2$, so $\rho = 1 - \frac{6 \times 2}{5 \times 24} = 1 - 0.1 = 0.9$. Here the restaurant's score almost perfectly follows its review count: a red flag. In practice you compute $\rho$ over the 50 restaurants for each query and summarise across the 100 queries (e.g., the mean), per aggregation (`scipy.stats.spearmanr` handles ties).

## 6. How Korikov et al. (2024) relates

Korikov et al., *Multi-Aspect Reviewed-Item Retrieval via LLM Query Decomposition and Aspect Fusion* ([arXiv:2408.00878](https://arxiv.org/abs/2408.00878)), study exactly our setup with **multi-aspect queries**, e.g. "cozy place **with** spicy food" = aspect *cozy* + aspect *spicy*.

The problem they highlight: aspects are **scattered across reviews**. One review praises the cozy décor, a different review mentions the spicy food; few reviews mention both.

```
Review 1: "Such a cozy little room."          cozy ✓  spicy ✗
Review 2: "The chili wings burn, in a good way." cozy ✗  spicy ✓
Query:    "cozy place with spicy food"          needs both
```

Standard **late fusion** (they define it as averaging an item's top-K review–query similarities, i.e. our top-k mean) scores each review against the *whole* query, so no single review matches well, and results depend heavily on how often each aspect appears in the reviews and how cleanly they are separated. Their fix, **aspect fusion**, uses an LLM to split the query into aspects, scores each aspect separately, then combines. They report that on imbalanced review corpora this improves MAP@10 from 0.36 to 0.52, and performs about the same as late fusion on balanced ones.

Why it matters to us:

- It confirms the professor's point: top-k late fusion is **sensitive to how evidence is distributed across reviews**, and review volume is one part of that distribution.
- At **sentence level** the scattering is even stronger (one sentence ≈ one aspect).
- LLM query decomposition is **out of scope** for us (`action_items.md` §7): we cite it and use it to explain our results and limitations, not reimplement it.

---

## Check yourself

1. Why was the v1 comparison between BM25 and dense "confounded"?
2. Is v1 dense (mean-pooled restaurant vector) early or late fusion? Is v2 top-5 mean?
3. Restaurant scores 0.9, 0.5, 0.4, 0.2. Compute max, mean and top-2 mean.
4. Why do max and top-k favour restaurants with many reviews?
5. Why subsample to 400 rather than, say, 1,000?
6. A Spearman $\rho$ of 0.8 between max score and review count: what does it suggest, and why isn't it proof of bias on its own?

<details>
<summary>Answers</summary>

1. Two things differed at once, the model *and* the unit/pooling (concatenated document vs. mean-pooled vector), so we couldn't attribute the gap to the model.
2. v1 dense is early fusion (merge vectors, then score once). v2 top-5 mean is late fusion (score each review, then aggregate scores).
3. Max = 0.9. Mean = 2.0 / 4 = 0.5. Top-2 mean = (0.9 + 0.5) / 2 = 0.7.
4. More reviews means more chances for some review to score high by accident; max needs only one, top-k only k.
5. The smallest RIRD restaurant has 406 reviews; 400 is the largest size every restaurant can be cut to. At 1,000, small restaurants couldn't be equalised.
6. Higher-volume restaurants tend to get higher scores, a warning sign. But popular restaurants might genuinely be relevant to more queries, so we need the subsampling experiment to separate volume from real relevance.

</details>

## Where this lives in our code

- `UNIT`, `TOP_KS`, `split_sentences`, `aggregate()` (max / mean / top-k), v1 concat-doc and pooled-vector references, review-count and popularity floors: [baselines/baseline_models_v2.ipynb](../baselines/baseline_models_v2.ipynb)
- Saved review-level results and per-query metrics (input for A3 and significance tests): [baselines/cache/results_review.csv](../baselines/cache/results_review.csv), [baselines/cache/per_query_review.csv](../baselines/cache/per_query_review.csv)
- Metrics used for every comparison: [baselines/eval_utils.py](../baselines/eval_utils.py)
- Task definitions A1–A4 and the F2 critique: [action_items.md](../action_items.md), [professor_commmend.md](../professor_commmend.md)
