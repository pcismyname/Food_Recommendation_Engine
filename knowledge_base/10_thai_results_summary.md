# 10 · Thai Results: Summary and Findings (for writing Chapters 5–8)

> **Who this is for:** Sam, writing Chapters 5–8 of the final report.
> **What it covers:** every Thai (Track C) number, what it means, what we may and may not claim,
> and exactly where it goes in the report. Chapters 1–4 are done
> ([report/progress/sections/](../report/progress/sections/)); Chapter 4 already describes how
> the Thai test set was built, so Chapters 5–8 only need the method summary and the results.

Written 2026-11-10 by Ton. All numbers come from files in the repo; nothing here is estimated by hand.

| Source | What it holds |
|---|---|
| [thai/results/thai_scores_summary.csv](../thai/results/thai_scores_summary.csv) | Means, 95% CIs, p-values (overall and by query type) |
| [thai/results/thai_scores_per_query.csv](../thai/results/thai_scores_per_query.csv) | nDCG@10 and P@10 per query and system |
| [thai/results/thai_agreement.csv](../thai/results/thai_agreement.csv) | Second-round agreement (kappa) |
| [thai/labels/label_sheet.csv](../thai/labels/label_sheet.csv) + `pool_key.csv` | The 452 labels and which system retrieved each sentence |
| [thai/score_labels.py](../thai/score_labels.py) | The scoring script (rerun with `python thai/score_labels.py`) |

---

## 1. TL;DR

1. **BM25 scores higher than dense on Thai, but the difference is not significant.** nDCG@10 0.654
   vs. 0.549 (difference −0.105, 95% CI [−0.298, +0.093], Holm p = 0.57).
2. **This is the opposite of English**, where dense beats BM25. It matches published Thai results
   (MIRACL: BM25 beats the older dense model mDPR).
3. **Combination queries are where BM25 clearly wins.** For strict precision (label = 2), BM25
   0.325 vs. dense 0.125, and this is the only CI that excludes zero.
4. **The two systems find different sentences** (only 1.2 of 10 overlap) and each wins on
   different queries (BM25 13, dense 9, tie 2). A hybrid of the two is the obvious next step.
5. **Labels are consistent** (κ = 0.837 on 60 pairs). Ton re-labelled the subset by hand in a
   second round, so this is a consistency check by the same annotator, not two different
   annotators.

---

## 2. Setup in one paragraph (for Chapter 5)

10,000 Wongnai reviews (seed 42) → 93,810 sentences (PyThaiNLP `crfcut`, sentences < 15 characters
dropped). BM25 over `newmm` tokens vs. the same dense encoder as English
(`paraphrase-multilingual-mpnet-base-v2`, cosine). 24 Thai taste queries frozen 2026-10-17 (11
single, 8 combination, 5 negation). Top 10 of each system pooled → 452 blind pairs, labelled 0/1/2
by Ton under guidelines frozen 2026-11-02. Metrics per query: **nDCG@10** with graded gains 0/1/2
(ideal DCG from all labelled pool sentences for that query), **P@10 lenient** (label ≥ 1) and
**P@10 strict** (label = 2). Over queries: mean with 95% percentile bootstrap CI (10,000
resamples, seed 42); paired sign-flip randomisation test, dense vs. BM25; Holm correction over the
three metrics. Details and figures are already in Chapter 4 (§ "Thai taste test set").

Note: T05 has no relevant sentence in its pool, so both systems score 0 on it. It is kept in the
average (removing it would raise both means slightly and does not change the conclusion).

---

## 3. Headline numbers

### 3.1 Overall (24 queries)

| Metric | BM25 [95% CI] | Dense [95% CI] | Dense − BM25 [95% CI] | p (randomisation) | Holm p |
|---|---|---|---|---|---|
| nDCG@10 | 0.654 [0.519, 0.773] | 0.549 [0.423, 0.674] | −0.105 [−0.298, +0.093] | 0.316 | 0.573 |
| P@10 lenient (≥ 1) | 0.667 [0.525, 0.796] | 0.558 [0.425, 0.688] | −0.108 [−0.292, +0.075] | 0.286 | 0.573 |
| P@10 strict (= 2) | 0.258 [0.150, 0.379] | 0.163 [0.088, 0.254] | −0.096 [−0.213, +0.029] | 0.157 | 0.471 |

### 3.2 By query type (descriptive only, no significance test)

| Type | n | nDCG@10 BM25 | nDCG@10 Dense | Strict P@10 BM25 | Strict P@10 Dense | Wins (dense / BM25 / tie) |
|---|---|---|---|---|---|---|
| Single | 11 | 0.577 | 0.494 | 0.255 | 0.227 | 4 / 6 / 1 |
| Combination | 8 | **0.809** | 0.588 | **0.325** | 0.125 | 1 / 6 / 1 |
| Negation | 5 | 0.574 | **0.607** | 0.160 | 0.080 | 4 / 1 / 0 |

Combination strict P@10 difference: −0.200, 95% CI [−0.350, −0.050] (excludes 0).
All other by-type CIs include 0.

### 3.3 Label mix in each system's top 10 (240 pairs per system)

| System | Label 2 | Label 1 | Label 0 |
|---|---|---|---|
| BM25 | 25.8% | 40.8% | 33.3% |
| Dense | 16.2% | 39.6% | 44.2% |

Whole pool (452 unique pairs): 88 × 2, 180 × 1, 184 × 0.

### 3.4 Agreement

60-pair subset: 90% exact agreement, 95% agreement on lenient binary (≥ 1), Cohen's κ = 0.837,
linearly weighted κ = 0.869.

Figure for Chapter 6: [report/progress/figures/ch6_thai_results.png](../report/progress/figures/ch6_thai_results.png)
(panel a: nDCG@10 with CIs by query type; panel b: dense − BM25 per query, coloured by type).
It is produced by [make_figures_ch1_4.py](../report/progress/make_figures_ch1_4.py).

---

## 4. Findings, with examples

### F1. BM25 ≥ dense on Thai; not significant

BM25 is ahead on every overall metric by about 0.1, but every CI includes zero. Wording: "no
evidence of a difference", **not** "the two systems are equal" and **not** "BM25 is better".

### F2. Very large per-query variance, so low power

The per-query differences have SD ≈ 0.51 nDCG@10. With 24 queries, the smallest difference we could
reliably detect (80% power, α = 0.05; ≈ 2.8 × SD / √24) is about **0.29 nDCG@10**. Our observed
difference (0.105) is far below that. Note: [09_related_work_novelty.md](09_related_work_novelty.md) §5
guessed 0.11–0.17 assuming SD 0.2–0.3; the real SD is higher, so use **≈ 0.29** in the report.

### F3. Combination queries favour BM25

BM25 wins 6 of 8 combination queries; dense wins only T13. BM25 rewards a sentence for matching
*several* query terms, so sentences that mention both tastes rise to the top. The dense model
encodes the whole query as one vector and tends to return sentences about one of the tastes.

Example, T11 ต้มยำเปรี้ยวๆ เผ็ดๆ (sour and spicy tom yum), nDCG@10 BM25 0.85 vs. dense 0.17:

| Rank | BM25 (label) | Dense (label) |
|---|---|---|
| 1 | น้ำต้มยำอร่อยๆ เปรี้ยวๆ ดี (1) | เผ็ดมากค่ะ -ข้าวผัดปู (0) |
| 2 | หวานนิดๆ เผ็ดๆเปรี้ยวๆถึงใจ. (1) | ตรมเ้วย ผัดซีอิ๊วหมู (0) |
| 3 | รสชาติดี เค็ม ๆ เปรี้ยว ๆ ครับเผ็ดนิดหน่อย (1) | แถมผัดมาจืดสนิท (0) |

This links to Korikov et al. (2024): multi-aspect queries are the hard case for single-vector
retrieval. Good sentence for Chapter 7.

### F4. Negation: dense wins 4 of 5, but one large failure hides it

Dense is better on T19, T20, T21, T23, yet the type means are close (0.607 vs. 0.574) because of
T22 อาหารไม่มัน ไม่เลี่ยน (not oily or greasy): BM25 0.89 vs. dense 0.09.

| Rank | BM25 (label) | Dense (label) |
|---|---|---|
| 1 | ไม่มันไม่เลี่ยน (2) | ไม่จืดแต่ก็ไม่ได้ถึงเครื่องแกงมากนัก (0) |
| 2 | มันไม่เยอะไม่เลี่ยน (2) | …ส่วนอาหารก็ธรรมดานะ ไม่สด (0) |
| 3 | ไม่มันและเลี่ยนเกินไป (2) | รสชาติไม่เผ็ดมาก (0) |

BM25 matched the exact phrase because ไม่ is kept as a token. Dense matched the *pattern* "ไม่ + some
taste" (not bland, not fresh, not spicy) instead of the negated property. This is the same weakness
NevIR describes for English bi-encoders, and the same reason the English demo uses a cross-encoder
for negation queries. With only 5 negation queries, report this as an observation, not a result.

### F5. Each system fails in a different way

- **BM25 fails on filler words.** T01 อยากกินอะไรเปรี้ยวๆ (I want something sour), BM25 0.12 vs.
  dense 0.76. BM25's top 3 all contain อยากกินอะไร ("I want to eat something") but say nothing
  about sour: และอยากกินอะไรง่ายๆ, อยากกินอะไรถูกๆแล้วไม่อยากออกจากบ้าน… Dense found
  ใครชอบทานเปรี้ยวๆ น่าจะชอบนะคะ (2). Same pattern on T02 spicy (0.00 vs. 0.68) and T06 cheap but
  tasty (0.19 vs. 0.99).
- **Dense fails on specific terms.** T18 บุฟเฟต์คุ้มๆ (good-value buffet), BM25 0.96 vs. dense 0.09.
  BM25 returns แบบบุฟเฟต์คุ้มกว่า (2); dense returns generic praise such as อร่อย ปังเย็น ให้เยอะ (0).

### F6. The systems are complementary

Only 28 sentences appear in both systems' top 10 across all 24 queries (1.2 per query). The shared sentences are mostly good:
26 of 28 labelled ≥ 1 and 13 labelled 2. Sentences that both systems agree on are much more
likely to be relevant than either list alone. This supports **hybrid retrieval (e.g. reciprocal
rank fusion) as future work**. Do not claim a hybrid result: a hybrid run was not evaluated, and the
pool is only complete for BM25 and dense.

### F7. Star ratings are not relevance

Spearman ρ between the review star rating and the sentence label = 0.10. 31% of sentences from
4-star reviews are labelled 0; 23% of sentences from 1-star reviews are labelled 2. Already in
Chapter 4; can be repeated in one line in Chapter 7 when explaining why we built human labels.

### F8. Strictly relevant sentences are rare

Strict P@10 is low for both (0.26 / 0.16): on average fewer than 3 of the top 10 clearly satisfy the
whole query. Many retrieved sentences are partial (one taste of a combination, or the right taste
for the wrong item). One query (T05, oily fried food) has no relevant sentence in its pool.

---

## 5. How the Thai result connects to English

| | English (RIRD) | Thai (Wongnai) |
|---|---|---|
| What is ranked | Restaurants | Sentences |
| Dense vs. BM25 | Dense better (sentence top-10 mean 0.682 vs. BM25 0.609) | BM25 ahead, not significant (0.549 vs. 0.654) |
| Negation | Needed a cross-encoder in the demo | Dense mostly better, one large BM25 win (T22) |

**Never compare the numbers across languages** (restaurant nDCG vs. sentence nDCG measure different
things). Compare BM25 vs. dense *within* each language and say the direction flips.

Possible reasons for the flip. These are hypotheses; we did not test them:
1. The encoder is a paraphrase-similarity model, not a retrieval model, and its Thai training data
   is much smaller than its English data.
2. Thai taste requests use short, explicit taste words (เปรี้ยว, เผ็ด, หวาน) that also appear
   literally in relevant sentences, which favours lexical matching.
3. Combination queries need several terms matched at once (F3).
4. The pattern matches MIRACL Thai, where BM25 (0.484) beats mDPR (0.358).
5. English restaurant scores average 10 sentences (top-10 mean), which smooths out single bad
   matches; Thai is judged sentence by sentence, so every bad match counts.

---

## 6. What we can and cannot claim

| Statement | OK? |
|---|---|
| "The same pipeline runs on real Thai reviews and returns relevant sentences for most queries (lenient P@10 0.67 BM25 / 0.56 dense)." | ✅ |
| "On Thai, BM25 scored higher than dense, but the difference was not significant." | ✅ |
| "Combination queries favour BM25 (strict P@10 0.33 vs. 0.13, CI excludes 0)." | ✅ (say it is a subgroup, 8 queries) |
| "The English finding (dense > BM25) does not transfer to Thai." | ✅ phrased as "we found no evidence that it transfers" |
| "BM25 is better than dense in Thai." | ❌ not significant |
| "The system ranks Thai restaurants well." | ❌ never tested; no restaurant IDs |
| "Two independent annotators agreed with κ = 0.84." | ❌ both rounds were labelled by Ton |
| "Labels were consistent when the annotator re-labelled a 60-pair subset by hand (κ = 0.84)." | ✅ |
| "Thai nDCG (0.55–0.65) is lower than English nDCG (0.68)." | ❌ different units, not comparable |
| "To our knowledge, this is the first human-labelled Thai taste retrieval test set." | ✅ with "to our knowledge" |

---

## 7. Where each piece goes in Chapters 5–8

Labels you can `\ref` from Chapters 1–4: `fig:intro-example`, `fig:intro-scope`, `fig:task`,
`fig:volume-sim`, `tab:rq-map`, `tab:scope`, `fig:related-map`, `tab:related-compare`,
`tab:datasets`, `fig:rird-eda`, `fig:wongnai-eda`, `tab:thai-queries`, `fig:thai-testset`.
New bib keys now in `references.tex`: `korikov2024multiaspect`, `zhang2025enhancing`,
`hasan2025reviewsurvey`, `jacovi2020faithfulness`, `deyoung2020eraser`, `li2021extra`,
`macdonald2008voting`, `balog2006formal`, `chen2024densex`, `zhang2023miracl`, `weller2024nevir`,
`phatthiyaphaibun2023pythainlp`, `reimers2020multilingual`, `cohen1960kappa`,
`smucker2007significance`, `jarvelin2002ndcg`, `lowphansirikul2020scb`.
`jarvelin2002ndcg` (nDCG) and `smucker2007significance` (randomisation tests) are not cited yet;
cite them in Chapter 5 § Evaluation design.

### Chapter 5 (Methodology)

- **§ Exploratory data analysis** now repeats Chapter 4. Suggest shortening it to 2–3 sentences
  that point to Figures `fig:rird-eda` and `fig:wongnai-eda`.
- **§ Evaluation design**, replace the last sentence about Thai with the paragraph in §2 above
  (metrics, CI, test, Holm). Point to Chapter 4 for the pool and labels.

### Chapter 6 (Results): add a section "Thai sentence retrieval" (ready to paste)

```latex
\section{Thai sentence retrieval}
\label{sec:thai-results}
\begin{table}[H]
\centering\small
\caption{Thai sentence retrieval on the 24 frozen taste queries (452 labelled pairs). Means over
queries with 95\% bootstrap confidence intervals; $p$ from a paired randomisation test with Holm
correction over the three metrics.}
\label{tab:thai-results}
\begin{tabular}{@{}lcccc@{}}
\toprule
Metric & BM25 & Dense & Dense $-$ BM25 & Holm $p$\\
\midrule
nDCG@10 & 0.654 [0.52, 0.77] & 0.549 [0.42, 0.67] & $-0.105$ [$-0.30$, $+0.09$] & 0.57\\
P@10 (label $\geq 1$) & 0.667 [0.53, 0.80] & 0.558 [0.43, 0.69] & $-0.108$ [$-0.29$, $+0.08$] & 0.57\\
P@10 (label $= 2$) & 0.258 [0.15, 0.38] & 0.163 [0.09, 0.25] & $-0.096$ [$-0.21$, $+0.03$] & 0.47\\
\bottomrule
\end{tabular}
\end{table}

BM25 scores higher than the dense model on all three metrics, but every confidence interval of
the difference includes zero, so there is no evidence of a difference between the two systems.
The per-query results vary widely (Figure~\ref{fig:thai-results}b): dense is better on 9
queries, BM25 on 13, and 2 are tied. With this variance, 24 queries can only detect a difference
of about 0.29 nDCG@10. By query type, BM25 is strongest on combination queries (nDCG@10 0.809 vs.\
0.588; strict P@10 0.325 vs.\ 0.125, difference CI $[-0.35, -0.05]$), while dense is better on
four of the five negation queries. The two systems retrieve largely different sentences (1.2 of
10 shared on average), and the shared sentences are mostly relevant (26 of 28 labelled at least
1). Labels were consistent when the annotator re-labelled 60 pairs by hand in a second round
($\kappa = 0.837$, weighted $0.869$).

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figures/ch6_thai_results.png}
\caption{Thai sentence retrieval: (a) nDCG@10 by query type with 95\% bootstrap confidence
intervals; (b) per-query difference between dense and BM25.}
\label{fig:thai-results}
\end{figure}
```

Also in Chapter 6: the current last section "Explanation and Thai evaluation" says the Thai pool
exists but gives no Thai results. Remove its Thai sentences (they are now in the new section and
in Chapter 4) and keep it as "Explanation evaluation" (B1 + B2).

### Chapter 7 (Discussion): suggested paragraph

> The Thai results do not reproduce the English advantage of dense retrieval. BM25 scored higher on
> all three metrics, although not significantly. The per-query analysis suggests why. BM25 fails when
> a query contains filler words that are common in reviews: for "I want something sour", its top
> sentences all begin with "I want to eat something" but do not mention sourness. The dense model
> fails when the query depends on a specific term or on several terms together: for "sour and spicy
> tom yum" it returned sentences about only one taste, and for "not oily or greasy" it returned other
> negated tastes ("not bland", "not spicy") instead of the negated property. This agrees with
> published Thai results, where BM25 outperforms older multilingual dense retrievers
> \citep{zhang2023miracl}, with the difficulty of negation for bi-encoders \citep{weller2024nevir},
> and with the difficulty of multi-aspect queries for single-vector retrieval
> \citep{korikov2024multiaspect}. Because the two systems retrieve largely different sentences and
> the sentences they share are mostly relevant, a hybrid of BM25 and dense retrieval is the most
> promising next step for Thai.

Add to **§ Limitations**: 24 queries (detectable difference ≈ 0.29); one annotator (the re-check
was by the same person); pool complete only for BM25 and dense; Thai segmentation
errors (crfcut F1 ≈ 0.77 on product reviews, `lowphansirikul2020scb`); sentence level only.

### Chapter 8 (Conclusions)

The current paragraph says "Future work should complete the human support and Thai annotation
studies". The **Thai annotation is complete**, so replace with something like:

> On 24 frozen Thai taste queries with 452 human-labelled sentences, the same pipeline returned
> relevant sentences for most queries, but BM25 scored higher than the dense model (nDCG@10 0.654
> vs. 0.549), a difference that was not significant. The English advantage of dense retrieval
> therefore did not transfer to Thai. Future work should evaluate a BM25–dense hybrid, a Thai or
> retrieval-trained encoder, and Thai restaurant-level ranking once a dataset with restaurant
> identifiers is available.

### Slides (F2)

One slide is enough: Table `tab:thai-results` + panel (b) of the figure + the T11 or T22 example.
