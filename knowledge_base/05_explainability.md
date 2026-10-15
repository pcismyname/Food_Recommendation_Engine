# 05 — Explanations: are the shown sentences really the reason?

> **Why this matters for our project**
> We show the top review sentences next to each recommended restaurant and call that an "explanation". The professor's comment F3 says we have no evidence these sentences are complete or faithful explanations.
> Tasks B1 and B2 exist to produce that evidence, and you will label data for B2.

---

## 1. What is an explanation in recommendation?

An **explanation** answers the user's question "why are you showing me this?". In a review-based recommender, the most natural answer is a piece of review text:

> Query: *quiet place for a date with good wine*
> Recommended: Restaurant X
> Because reviewers said: "Lovely candle-lit room, never too loud." · "The wine list is long and the staff know it well."

There are two families:

| | **Extractive** (what we do) | **Generated** |
|---|---|---|
| How | Select existing sentences from reviews | A model (often an LLM) writes new text |
| Pro | Every word was really written by a customer; no invented facts | Fluent, can summarise many reviews, can be personalised |
| Con | Can be off-topic, repetitive, or partial | Can **hallucinate** (state things no review says) |

Our scope ([action_items.md](../action_items.md), section 7) is extractive only. That removes the hallucination risk but **not** the question of whether the sentences explain the ranking.

---

## 2. Plausibility vs faithfulness

These two words are the core of this topic.

- **Plausible** = a human reading the explanation finds it convincing and on-topic.
- **Faithful** = the explanation reflects what *actually caused* the system's decision.

They can come apart. Example with top-10-mean aggregation:

> Restaurant X is ranked #1 for "spicy noodles". Its score is the mean of its 10 best-matching sentences. We display the top 3. They mention spicy noodles, so they look perfect (plausible).
> But X has 2,000 reviews. If we delete those 3 sentences, the next 3 sentences are almost as good, and X stays #1.

So the 3 sentences are *a* reason, but X would be ranked #1 without them. The real driver is "X has many reviews that mention noodles" (this is also the review-volume problem in F2). The display is plausible, but it is not a complete or faithful explanation.

The reverse can also happen: a sentence that is the strongest driver of a dense score may look odd to a human (for example, a sentence that mentions "noodles" in a negative context, which the encoder scores highly anyway).

Faithfulness is about the **model**. Plausibility is about the **human**. We need evidence for both:

- **B1** measures faithfulness automatically.
- **B2** measures support (a form of plausibility) with humans.

---

## 3. Faithfulness by deletion (task B1)

Idea: **if these sentences are the reason, removing them should hurt the restaurant's rank.** This is called an **erasure** or **deletion** test.

Procedure for one (query, restaurant) pair where the restaurant is in the system's top results:

1. Record the restaurant's original rank, r₀.
2. Delete the **k shown sentences** (the explanation) from that restaurant's reviews.
3. Recompute the restaurant's score with the same aggregation and re-rank all restaurants. New rank = r_expl.
4. Separately, delete **k random sentences** from the same restaurant, rescore, get r_rand. Repeat several times (e.g. 20) and average.
5. Compare the **rank drops**: Δ_expl = r_expl − r₀ and Δ_rand = r_rand − r₀.

Worked numbers (k = 3):

| Query | r₀ | r_expl | Δ_expl | mean r_rand | Δ_rand |
|---|---|---|---|---|---|
| Q1 | 1 | 6 | 5 | 1.1 | 0.1 |
| Q2 | 2 | 2 | 0 | 2.0 | 0.0 |
| Q3 | 1 | 3 | 2 | 1.0 | 0.0 |

Average Δ_expl = 2.33, average Δ_rand = 0.03. The shown sentences matter far more than random ones, which is evidence of faithfulness. Q2 is a warning: there, the explanation made no difference at all.

### Why the random control matters

Removing *any* sentences changes a score a little, especially with `mean` aggregation, where every sentence counts. Without a control, a rank drop of 2 could just mean "removing text hurts". The random deletion tells you how much drop you get **for free**. Only the *extra* drop of the explanation over random counts as evidence. Because both deletions are done on the same (query, restaurant) pairs, compare them with a paired test (see [04_evaluation.md](04_evaluation.md), section 5).

Expected pattern, worth checking:

- With **max** aggregation, deleting the top sentence always changes the score. Faithfulness looks high, almost by construction.
- With **mean** or **top-10 mean** and many reviews, there are many near-duplicates, so deletion may barely matter. This is a finding to report, not a bug.

You can also report a **score drop** (original score minus new score) instead of a rank drop. It is smoother, but less meaningful to a user.

---

## 4. Human support rating (task B2)

The deletion test cannot tell whether a sentence makes sense to a person. B2 asks humans directly.

- Export the shown sentences for about 15–30 queries × the top 3 restaurants.
- Each row: query, restaurant name, shown sentence(s). Do **not** show the system name or score.
- Two people (Sam and Ton) label each row separately:
  - **2 = supports**: the sentence clearly gives a reason this restaurant fits the query.
  - **1 = partly supports**: related, but weak, vague, or covers only part of the query (e.g. "spicy" but not "noodles").
  - **0 = does not support**: off-topic, or contradicts the query ("not spicy at all").
- Report the **% rated 2** (and % rated ≥ 1), plus **Cohen's kappa / weighted kappa** between the two of you.

The guidelines (E1) must settle edge cases before labelling, for example: is sarcasm a 0? What about a sentence that supports the query for a *different* dish at the same restaurant?

Support is not the same as relevance in C4. C4 asks whether a sentence matches the query. B2 asks whether it **justifies recommending this restaurant**.

---

## 5. User-centric criteria (Zhang et al., Findings of EMNLP 2025)

The professor pointed to **Zhang, Tian, Feng, Chen & Chen, "Enhancing Recommendation Explanations through User-Centric Refinement"**, Findings of EMNLP 2025 ([ACL Anthology](https://aclanthology.org/2025.findings-emnlp.434/), arXiv [2502.11721](https://arxiv.org/abs/2502.11721)). What we verified from the paper:

- It argues that explanations from existing explainable recommenders often fail three **user-centric** aspects:
  - **Factuality**: the content is correct and can be supported by the available information (reviews).
  - **Personalization**: the explanation highlights specific item features and the user's own characteristics or preferences.
  - **Sentiment coherence**: the explanation's sentiment (positive or negative) matches the system's predicted preference. For example, you should not "recommend" with a negative sentence.
- Their method, **RefineX**, refines explanations at inference time with an LLM-based multi-agent framework (a plan-then-refine pattern with hierarchical reflection). It is a *generation* method, so it is out of scope for us to reimplement.
- For evaluation they use automatic measures: an entailment ratio judged by an LLM for factuality, feature coverage and diversity for personalization, and a sentiment-match ratio for coherence. They also run a small expert human evaluation on a 5-point scale. See the paper for exact definitions and numbers.

**How we borrow it** (criteria only, not the method):

| Their aspect | Our extractive version | Measured by |
|---|---|---|
| Factuality | Automatic for us: the sentence is copied from a real review. But it must be about *this* restaurant | — (state it as a design property) |
| Personalization | Does the sentence address *this query's* specific need (dish, taste, price)? | B2 support label (2 vs 1) |
| Sentiment coherence | A positive recommendation should not be backed by "not spicy at all" or "too sweet" | Add a rule in E1: contradicting sentiment = 0 |

Sentiment coherence links directly to the Thai negation problem in [06_thai_nlp.md](06_thai_nlp.md): BM25 returns "ไม่เผ็ดจัด" (*not very spicy*) for the query "อาหารรสเผ็ดจัด" (*very spicy food*). As an explanation, that sentence fails sentiment coherence.

---

## 6. How this answers the professor (F3)

F3: *"A top-scoring sentence can be relevant without being a complete or faithful explanation. The group has not evaluated explanatory quality."*

Our answer has two legs:

1. **Faithful?** B1 deletion vs random deletion: do the shown sentences actually drive the ranking, and does that depend on the aggregation?
2. **Supportive to a human?** B2: what share of shown sentences do two annotators rate as supporting, and how well do they agree (kappa)?

Together they turn "we show sentences" into a measured claim. A negative result is also acceptable, as long as it is investigated (for example, "under top-10 mean, deleting the 3 shown sentences changes the rank no more than random deletion, so the display is plausible but not faithful").

---

## Check yourself

1. Give a one-line difference between plausibility and faithfulness.
2. Why is extractive explanation safer than generated, and what problem does it *not* solve?
3. In B1, deleting the 3 shown sentences moves a restaurant from rank 1 to rank 2, and random deletion also moves it to rank 2. What do you conclude?
4. Why would faithfulness look artificially high under `max` aggregation?
5. Name the three user-centric aspects in Zhang et al. (2025) and give one that maps to a labelling rule for us.
6. Which statistic do you report for agreement between Sam and Ton in B2, and why the weighted version?

<details>
<summary>Answers</summary>

1. Plausible = convincing to a human; faithful = actually the reason the model ranked it.
2. Every sentence is real customer text, so there are no hallucinated facts. It does not guarantee the sentence is relevant, complete, or the true cause of the ranking.
3. The explanation is no more important than random text, so there is no evidence of faithfulness for that pair.
4. The score *is* the top sentence's score. Removing it always lowers the score, so the drop comes from how max works, not from a good explanation.
5. Factuality, personalization, sentiment coherence. Example: sentiment coherence → a sentence that contradicts the query ("not spicy") gets 0.
6. Cohen's kappa, which corrects for chance agreement. Weighted kappa because 0/1/2 is ordinal: a 1-vs-2 disagreement is milder than 0-vs-2.

</details>

---

## Where this lives in our code / tasks

- Professor's comment: [professor_commmend.md](../professor_commmend.md) (F3 in [action_items.md](../action_items.md), section 1.1)
- Aggregations (max, mean, top-k mean) whose faithfulness B1 tests: `aggregate()` in [baselines/baseline_models_v2.ipynb](../baselines/baseline_models_v2.ipynb)
- Kappa and paired tests: [04_evaluation.md](04_evaluation.md) and [baselines/eval_utils.py](../baselines/eval_utils.py)
- Tasks: **B1** (deletion faithfulness, needs **A2** at sentence level and **A4** saved rankings), **B2** (human support labels + kappa; Sam and Ton label), **E1** (guidelines with the sentiment-coherence rule), **D1** (cite Zhang et al. 2025)
