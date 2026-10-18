# 04 — Evaluating a ranking system

> **Why this matters for our project**
> Every claim we make ("dense beats BM25", "top-10 mean is best") is only as good as the labels and the statistics behind it.
> Your Thai tasks C3–C7 build a small test collection from scratch, so you need to know how labels, metrics and significance tests fit together.

---

## 1. Relevance judgments and qrels

A **relevance judgment** is a human decision: "for query *q*, is item *d* relevant?" The item can be a restaurant (English RIRD) or a review sentence (our Thai set).

- **Binary labels**: 0 = not relevant, 1 = relevant.
- **Graded labels**: several levels. We will use **0 / 1 / 2**:
  - 0 = not relevant (does not talk about the asked taste, or says the opposite)
  - 1 = partly relevant (mentions the taste weakly, or only part of a combination query)
  - 2 = clearly relevant (directly says the dish/place has the asked taste)

The full set of judgments is called the **qrels** (query relevance file). In code it is usually a dictionary:

```python
qrels = {"อาหารรสเผ็ดจัด": {"sent_812": 2, "sent_77": 0, "sent_4031": 1}}
```

In RIRD, `PMD.csv` has 5 annotators per (query, restaurant). Our v2 notebook builds two qrels from it: `binary_qrels` from the column `If only Low or  High`, and `graded_qrels` where the grade is the **sum of the 5 annotators' votes (0–5)**. Graded qrels are used only for nDCG.

---

## 2. The metrics

Running example: one query. Graded labels are A=2, B=0, C=1, D=2, E=0, F=1 (all other items are 0). The system ranks:

| Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Item | A | B | C | D | E | G | H | F |
| Grade | 2 | 0 | 1 | 2 | 0 | 0 | 0 | 1 |

For binary metrics, call anything with grade ≥ 1 "relevant", so the relevant set is {A, C, D, F} (4 items).

**Precision@k (P@k)** = (number of relevant items in the top k) ÷ k. "How clean is the top of the list?"
P@5 = 3 relevant (A, C, D) ÷ 5 = **0.60**.

**Recall@k** = (relevant items in the top k) ÷ (total relevant items). "How much of what exists did we find?"
Recall@5 = 3 ÷ 4 = **0.75**.

**Reciprocal rank (RR)** = 1 ÷ (rank of the first relevant item). Here A is at rank 1, so RR = **1.0**. If the first hit were at rank 4, RR would be 0.25. **MRR** is the mean of RR over all queries.

**Average Precision (AP)**: walk down the list; every time you hit a relevant item, write down the precision at that point; then divide the sum by the total number of relevant items.

- rank 1 (A): 1/1 = 1.000
- rank 3 (C): 2/3 = 0.667
- rank 4 (D): 3/4 = 0.750
- rank 8 (F): 4/8 = 0.500

AP = (1.000 + 0.667 + 0.750 + 0.500) ÷ 4 = **0.729**. A relevant item that is never retrieved adds 0 but still counts in the "÷ 4", so missing items hurt. **MAP** = mean of AP over queries.

### DCG and nDCG@k (the main metric)

**DCG@k** (Discounted Cumulative Gain):

  DCG@k = Σ (from i = 1 to k) rel_i ÷ log₂(i + 1)

- *i* = rank position (1, 2, 3, …)
- rel_i = the grade of the item at rank *i*
- log₂(i + 1) = the **discount**: rank 1 is divided by 1, rank 3 by 2, rank 7 by 3. Being lower in the list is worth less, because users read from the top.

**IDCG@k** (Ideal DCG) = the DCG of the best possible ordering of the judged items (sort all grades high to low).
**nDCG@k** = DCG@k ÷ IDCG@k. It is between 0 and 1; 1 means "perfect order".

Worked example, k = 5:

| Rank i | Item | rel_i | log₂(i+1) | gain |
|---|---|---|---|---|
| 1 | A | 2 | 1.000 | 2.000 |
| 2 | B | 0 | 1.585 | 0.000 |
| 3 | C | 1 | 2.000 | 0.500 |
| 4 | D | 2 | 2.322 | 0.861 |
| 5 | E | 0 | 2.585 | 0.000 |
| | | | **DCG@5** | **3.361** |

Ideal order of grades: 2, 2, 1, 1, 0 → 2/1 + 2/1.585 + 1/2 + 1/2.322 + 0 = 2 + 1.262 + 0.5 + 0.431 = **4.193**.

nDCG@5 = 3.361 ÷ 4.193 = **0.802**.

**Why nDCG for graded labels?** P@k, Recall, MRR and AP only see "relevant or not". They give the same score whether rank 1 is a "2" or a "1". nDCG uses the grade itself and also rewards putting the 2s above the 1s. That is exactly what we want for 0/1/2 labels in C7. P@10 is still useful as a simple, easy-to-explain companion number.

Note: in [eval_utils.py](../baselines/eval_utils.py), IDCG is computed from all labelled items for the query, not only the retrieved ones (same as above: F counts in the ideal list even though it is at rank 8).

---

## 3. Pooling: where labels come from

We cannot label all ~95,000 Thai sentences (10,000-review corpus) for 24 queries (over 2 million pairs; even the 19,294-sentence pilot would be over 460,000). The standard trick, from the TREC evaluations, is **pooling**:

1. Run several systems.
2. Take the top-k results of each system for each query.
3. Merge them (remove duplicates) into one **pool**.
4. Humans judge only the pool. Anything outside the pool is **assumed not relevant**.

In the 2,000-review Thai pilot, pooling BM25 top-10 and dense top-10 gave 465 pairs for the 24 frozen queries (BM25 and dense share less than 1 sentence per query on average, so the pool is about 19 per query). The real labelling run uses the larger 10,000-review corpus (~95,000 sentences); `thai/build_label_pool.py` builds the pool from its top-10 lists and that is the labelling budget for C4.

**Scoring a pool (`thai/score_labels.py`).** Because only the pool is labelled, nDCG@10 is computed at **pool depth 10**: the ideal list (IDCG) is built from all labelled pool sentences of that query, so the score is "relative to the pool", not to every relevant sentence in the corpus. Since each system's whole top 10 is in the pool, BM25 and dense have no unjudged items; pool bias only bites for a future system. The script uses gains 0/1/2 for nDCG and reports P@10 two ways: lenient (grade ≥ 1 counts) and strict (only grade 2). With `--partial` it scores only the queries whose pool is fully labelled, so you can run a pilot before the whole sheet is done (treat those numbers as a sanity check, not a result).

**Pool bias**: a new system that was *not* in the pool (say, a future BGE-M3 run) may find relevant sentences nobody labelled. They count as 0, so the new system looks worse than it really is. Report this as a limitation, and if you add a system later, label its unjudged top-10 items before comparing.

### Blind, shuffled annotation (C3)

The sheet must **not** show which system found a sentence, and the rows should be in **random order**. Otherwise an annotator who expects dense to be better (or who just gets tired near the end) can bias the labels. Keep a hidden key column (sentence id → system) in a separate file.

---

## 4. Do two people agree? (C5, B2)

**Percent agreement** = share of items where both annotators gave the same label. Simple, but it ignores luck: if 90% of pairs are "0", two people guessing "0" all the time agree 90% of the time.

**Cohen's kappa (κ)** corrects for chance:

  κ = (p_o − p_e) ÷ (1 − p_e)

- p_o = observed agreement (the percent agreement, as a fraction)
- p_e = agreement expected by chance = Σ over labels of (share of annotator 1 using that label) × (share of annotator 2 using that label)

Worked example: Ton and a second annotator label 50 pairs.

| Ton ↓ / Other → | 0 | 1 | 2 | Row total |
|---|---|---|---|---|
| **0** | 14 | 3 | 1 | 18 |
| **1** | 2 | 12 | 4 | 18 |
| **2** | 0 | 3 | 11 | 14 |
| Column total | 16 | 18 | 16 | 50 |

- p_o = (14 + 12 + 11) ÷ 50 = **0.74**
- p_e = (18×16 + 18×18 + 14×16) ÷ 50² = (288 + 324 + 224) ÷ 2500 = **0.334**
- κ = (0.74 − 0.334) ÷ (1 − 0.334) = **0.61**

**Weighted kappa** is for ordinal labels like 0/1/2. Plain kappa treats "0 vs 1" and "0 vs 2" as equally wrong; weighted kappa gives partial credit for near misses. With linear weights (agreement weight = 1 − |a − b| ÷ 2, so a 1-step miss counts as 0.5 agreement and a 2-step miss as 0), the same table gives weighted p_o = 0.86, weighted p_e = 0.565, and **weighted κ ≈ 0.68**. Report both. In Python: `sklearn.metrics.cohen_kappa_score(y1, y2, weights="linear")`. `thai/score_labels.py` reports both (unweighted and linear-weighted) for the 60 pairs in `label_sheet_annotator2.csv`.

**Rough interpretation** (Landis & Koch, 1977; a convention, not a law):

| κ | Agreement |
|---|---|
| < 0 | worse than chance |
| 0.00–0.20 | slight |
| 0.21–0.40 | fair |
| 0.41–0.60 | moderate |
| 0.61–0.80 | substantial |
| 0.81–1.00 | almost perfect |

Low kappa usually means the **guidelines** (E1) are unclear, not that someone is careless. Look at the disagreements, add an edge-case rule, and relabel.

---

## 5. Is the difference real? Statistical testing

All tests start from **per-query scores**: one nDCG@10 per query per system. The mean hides how much queries vary; with 100 queries (RIRD) or 24 (Thai) the mean could shift a lot by luck of which queries we wrote.

### Bootstrap 95% confidence interval

1. You have 100 per-query scores.
2. Draw 100 queries **with replacement** (some appear twice, some not at all) and compute the mean.
3. Repeat 10,000 times (`N_RESAMPLES = 10_000` in the notebook).
4. The 2.5th and 97.5th percentiles of those 10,000 means are the 95% CI.

Our best English model: Dense top-10 mean, nDCG@10 = **0.668 [0.625, 0.711]**. Read it as "a plausible range for the true mean is about 0.63 to 0.71."

### Paired randomization (sign-flip) test

Question: "Is dense better than BM25, or could the gap be luck?" Because both systems run on the **same** queries, compare them **query by query** (paired):

1. For each query, d = score_dense − score_BM25.
2. If the systems were truly equal, the sign of each d would be a coin flip.
3. 10,000 times: flip each d's sign at random, take the mean.
4. p-value = share of shuffled means at least as far from 0 as the real mean (the code adds 1 to top and bottom so p is never exactly 0).

Small p (< 0.05) → the real gap is unusual under "no difference".

### Multiple comparisons and Holm

We compare many pairs (5 aggregations × BM25 vs dense, each model vs the floor, …). At p < 0.05, about 1 in 20 tests of *equal* systems would still look significant. **Holm correction**:

1. Sort the m p-values from smallest to largest.
2. Multiply the smallest by m, the next by m − 1, …, the last by 1.
3. Make the results non-decreasing (each adjusted p is at least the previous one) and cap at 1.

Example, m = 3, raw p = 0.01, 0.03, 0.04 → 0.03, 0.06, 0.04 → non-decreasing: **0.03, 0.06, 0.06**. Only the first stays significant. This is the `holm()` function in the v2 notebook and the `p (Holm)` column. `thai/score_labels.py` applies the same three tools (bootstrap CI, paired randomization, Holm) to the Thai labels, overall and by query type.

### How small a difference can 24 queries detect?

Fewer queries means wider CIs and less **power** (the chance a test detects a real difference). With only 24 Thai queries, a BM25 vs. dense gap is only likely to come out significant if it is roughly **0.11–0.17 nDCG@10** or larger. A smaller real gap will probably show as "not significant". So "not significant" on the Thai set means "we can't tell", not "they are equal". Say this in the report.

---

## 6. Floors: what does "good" mean?

A score of 0.6 means nothing until you know what a dumb system gets. The v2 notebook has three **query-independent floors** (they never read the query):

- **Random (expected)**: random order, averaged over many shuffles.
- **Review count**: restaurants with more reviews first, same order for every query.
- **Popularity (LOO)**: restaurants relevant to the most *other* queries first (leave-one-query-out, so a query never sees its own labels).

If a model barely beats "popular restaurants first", it is not really understanding queries. In v2, every model beats popularity on nDCG@10 and MAP; BM25 with mean aggregation is the exception that is not significantly better, and it loses to popularity on P@10 (0.377 vs. 0.41). For Thai, a useful floor is random sentences from the pool.

---

## 7. Common pitfalls

- **Tuning on the test set**: choosing k, the aggregation or the encoder by looking at the test nDCG, then reporting that same nDCG. The number is optimistic. Say clearly what was tuned, or hold out queries.
- **Reporting means without CIs**: with 24 Thai queries, a gap of 0.05 may be noise.
- **Unpaired tests on paired data**: always compare systems on the same queries.
- **Forgetting the pool**: unjudged ≠ irrelevant for a new system.
- **Changing labels after seeing system output**: label blind, freeze the labels, then score.
- **Cherry-picked examples**: qualitative examples are fine for slides, but pick them by a rule (e.g. one query of each type), not by "where we look good".
- **Pooling a statistic across queries**: when the question is about the order *within* each ranking, compute the statistic per query, then average. In A3 we first correlated restaurant score with review count over all queries pooled together and got only 0.05–0.12. Score levels differ from query to query, and pooling mixes those differences in and hides the pattern. Computed per query and averaged, the correlation is up to 0.36 (BM25 top-10 mean). Same idea as the metrics: nDCG is computed per query, then averaged. (Details in file 03 §5.)

---

## Check yourself

1. A system ranks a grade-1 item first and a grade-2 item second. Swapping them would raise which metric(s): P@2, MRR, nDCG@2?
2. Compute nDCG@3 for grades [0, 2, 1] if the ideal list is [2, 1, 0].
3. Why can percent agreement be 0.90 while kappa is close to 0?
4. Why do we use a *paired* test instead of comparing two bootstrap CIs?
5. What is pool bias, and who does it hurt?
6. Raw p-values 0.004, 0.02, 0.03, 0.30 (m = 4). Which survive Holm at 0.05?

<details>
<summary>Answers</summary>

1. Only nDCG@2. P@2 and MRR treat both as "relevant", so they do not change.
2. DCG = 0/1 + 2/1.585 + 1/2 = 1.762. IDCG = 2/1 + 1/1.585 + 0 = 2.631. nDCG@3 = 0.670.
3. When one label dominates, chance agreement p_e is also near 0.90, so p_o − p_e is near 0.
4. Per-query differences remove the variation between easy and hard queries; overlapping CIs can still hide a consistent, significant paired gap.
5. Items outside the pool are assumed irrelevant. It hurts a new system that was not pooled, because its unique relevant finds count as 0.
6. Adjusted: 0.016, 0.06, 0.06, 0.30. Only the first (0.004) survives.

</details>

---

## Where this lives in our code / tasks

- Metric functions (`precision_at_k`, `recall_at_k`, `mrr`, `average_precision`, `ndcg_at_k`): [baselines/eval_utils.py](../baselines/eval_utils.py)
- Bootstrap CI, paired randomization test, Holm correction and the three floors: `bootstrap_ci`, `paired_randomization_p`, `holm`, section 7 of [baselines/baseline_models_v2.ipynb](../baselines/baseline_models_v2.ipynb) (they are in the notebook, not in `eval_utils.py`)
- Pool size and BM25/dense overlap for the Thai set: section 7 of [thai/thai_retrieval.ipynb](../thai/thai_retrieval.ipynb)
- Blind pooled label sheet, hidden key and second-annotator sheet: [thai/build_label_pool.py](../thai/build_label_pool.py); scoring the Thai labels (pooled-depth nDCG@10, P@10 lenient/strict, bootstrap, paired randomization, Holm, kappa, `--partial`): [thai/score_labels.py](../thai/score_labels.py)
- Per-query vs. pooled correlation (A3): [baselines/cache/a3_volume_control_correlations_review.csv](../baselines/cache/a3_volume_control_correlations_review.csv)
- Tasks in [action_items.md](../action_items.md): **C3** (blind pooled sheet), **C4** (0/1/2 labels), **C5** (second annotator, kappa), **C7** (Thai nDCG@10 / P@10 with CIs), **B2** (support labels and kappa), **E1** (guidelines that make kappa high), **A2** (CIs and Holm on the English runs)
