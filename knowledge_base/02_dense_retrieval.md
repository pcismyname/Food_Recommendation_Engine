# 02 · Dense Retrieval (Embeddings)

> **Why this matters for our project**
> Our second baseline, and currently our best model, is a dense retriever: `paraphrase-multilingual-mpnet-base-v2` + cosine similarity.
> The same model is used for English (RIRD) and Thai (Wongnai), so you need to know what it does, why it can cross languages, and where it fails.

---

## 1. Embeddings: vectors of meaning

BM25 (file 01) asks "do the query and the review share **words**?" Dense retrieval asks "do they share **meaning**?"

An **embedding** is a list of numbers (a vector) that a neural network produces for a piece of text. Our model outputs **768 numbers** per text. The network is trained so that texts with similar meaning land close together in this 768-dimensional space:

```
            "warm, intimate atmosphere" ●  ● "cozy place"
                                          
                                                    ● "parking was easy"
   "mouth-numbing chili" ●  ● "very spicy"
```

Nobody decides what each of the 768 numbers means; the network learns the space from training data. All you need: **close = similar meaning**.

Called "dense" because almost all 768 numbers are non-zero, as opposed to a BM25 representation, which is "sparse": one slot per vocabulary word, almost all zero.

## 2. Transformers and BERT in one paragraph

A **transformer** is a neural network that reads a text as a sequence of sub-word tokens and, in each layer, lets every token "look at" every other token (this is called **attention**) to update its own representation. After a dozen layers, the vector for "spicy" in "not spicy at all" is different from the one in "very spicy", because it has absorbed its context. **BERT** (2018) is a transformer pre-trained on huge amounts of text by hiding words and learning to predict them. It produces one vector **per token**, not per sentence. Our model is from the same family: a multilingual transformer (XLM-RoBERTa architecture) trained to imitate the English sentence model `paraphrase-mpnet-base-v2` (see section 6).

## 3. Sentence-BERT and bi-encoders

To get **one vector per sentence**, Sentence-BERT (Reimers & Gurevych, 2019) does two things:

1. **Pooling**: average all the token vectors into one vector (mean pooling).
2. **Fine-tuning on sentence pairs** (paraphrases, question–answer pairs) so that the averaged vectors of similar sentences really are close.

This is a **bi-encoder**: the query and the document are encoded **separately**, by the same network, and compared afterwards.

```
query  ──► [encoder] ──► q (768 numbers) ─┐
                                          ├──► cosine(q, d) = score
review ──► [encoder] ──► d (768 numbers) ─┘
```

Key benefit: review vectors don't depend on the query, so we encode all 29,168 RIRD reviews **once**, save them, and then every new query costs one encoder pass plus a cheap matrix multiplication.

## 4. Cosine similarity and normalization

**Cosine similarity** measures the angle between two vectors, ignoring their length:

$$\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q}\cdot\mathbf{d}}{\|\mathbf{q}\|\,\|\mathbf{d}\|}$$

- $\mathbf{q}\cdot\mathbf{d}$ (dot product) = multiply matching entries and add them up.
- $\|\mathbf{q}\|$ = the length of $\mathbf{q}$ = $\sqrt{\text{sum of squared entries}}$.
- Result is between −1 (opposite) and +1 (same direction); 0 means unrelated (perpendicular).

### Worked 2-D example

Pretend embeddings have only 2 numbers. Query $\mathbf{q} = (3, 4)$, so $\|\mathbf{q}\| = \sqrt{9+16} = 5$.

| Review | vector | dot with q | length | cosine |
|---|---|---|---|---|
| A | (4, 3) | 12 + 12 = 24 | 5 | 24 / (5·5) = **0.96** |
| B | (−4, 3) | −12 + 12 = 0 | 5 | **0.00** |
| C | (6, 8) | 18 + 32 = 50 | 10 | 50 / (5·10) = **1.00** |

C points exactly the same way as q, just longer, so cosine says "identical direction". Length is ignored.

**Normalization trick.** If we first divide every vector by its length (**L2 normalization**), all vectors have length 1, and cosine becomes a plain dot product. Normalized q = (0.6, 0.8), normalized A = (0.8, 0.6), dot = 0.48 + 0.48 = 0.96. Same answer.

That is what our code does: `l2_normalise` on review and query embeddings, then `S_dense = Q @ E.T` gives a (100 queries × 29,168 reviews) matrix of cosine scores in one multiplication. The Thai notebook does the same with `normalize_embeddings=True` and `S @ Q[qi]`.

## 5. Bi-encoder vs. cross-encoder (rerankers)

A **cross-encoder** feeds the query and the document **together** into one transformer and outputs a single relevance score:

```
[query + review] ──► [encoder] ──► score
```

Because every query token can attend to every review token, it can model things like "the query says *not* spicy and the review says *very* spicy". It is usually more accurate.

The cost: nothing can be precomputed. For RIRD, 100 queries × 29,168 reviews ≈ 2.9 million transformer passes, versus 29,168 + 100 passes for the bi-encoder.

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Encodes query and doc | separately | together |
| Precompute docs? | yes | no |
| Speed at query time | fast (one matrix multiply) | slow (one pass per pair) |
| Accuracy | good | usually better |
| Typical use | first-stage retrieval | **reranking** the top 50–100 from stage 1 |

The common pattern is "retrieve with a bi-encoder or BM25, rerank the top-N with a cross-encoder". We don't use a reranker yet; it would be a natural extension.

## 6. Multilingual models: one encoder for Thai and English

`paraphrase-multilingual-mpnet-base-v2` was made by **multilingual knowledge distillation** (Reimers & Gurevych, 2020): an English "teacher" model already produced good sentence embeddings; a multilingual "student" was trained so that an English sentence *and its translation* (for example Thai) both map to the teacher's vector for the English sentence. The result: `อาหารรสเผ็ดจัด` and "very spicy food" land near each other.

Consequences for us:

- The **same code and model** run on RIRD (English) and Wongnai (Thai). Only tokenization for BM25 changes.
- A Thai query could in principle match an English review. But RIRD has no Thai reviews and we deliberately never run Thai queries on RIRD (see `PROJECT.md`); the Thai evaluation (C1–C7) is Thai query → Thai sentence.
- Quality is usually somewhat lower in languages with less training data. This is one reason our Thai claims must be measured (task C7), not assumed.

## 7. Maximum sequence length and truncation

A transformer has a limit on how many **tokens** (sub-word pieces, not words) it reads. Anything beyond the limit is **cut off and ignored**. This model's default limit is 128 tokens. v1 used that default and so silently ignored the end of long reviews.

In v2 we set `MAX_SEQ_LEN = 256`. That is why the cache file is named `..._review_len256.npy`. The notebook also prints the share of reviews still longer than 256 tokens, so we know how much text is lost:

| Setting | Units longer than the limit |
|---|---|
| v1: reviews, 128 tokens | 50.6% |
| v2: reviews, 256 tokens | 19.9% |
| v2: sentences, 256 tokens | 0% |

So doubling the limit fixed most of the problem, but about one review in five is still cut. Sentence units remove truncation completely. (The Thai notebook keeps the default, but it encodes short sentences, so truncation matters much less there.)

Why not just set 10,000? Attention cost grows roughly with the square of the length, and the model was trained on short texts, so very long inputs get slow and the single averaged vector gets "blurry": one vector cannot represent a review that covers food, service, parking and price equally well. This is another argument for **sentence-level** units (file 03).

## 8. Precomputing and caching embeddings

Encoding ~29k reviews takes minutes on a GPU and much longer on a CPU. So we encode once and save to a `.npy` file:

- RIRD: `baselines/cache/emb_paraphrase-multilingual-mpnet-base-v2_review_len256.npy` (raw, normalized after loading)
- Thai: `thai/cache/emb_paraphrase-multilingual-mpnet-base-v2_n2000_s42_min15.npy` (already normalized)

The file name records the **model, unit, max length** (and for Thai the sample size, seed and minimum sentence length). If you change any of these, you must get a new file, or you'd silently be using stale vectors. The RIRD notebook also re-encodes if the number of rows doesn't match the current units. Caching is also what makes task A3 (the volume control) cheap: subsampling reviews only needs picking rows from the existing matrix, no re-encoding.

## 9. When dense wins, when BM25 wins

**Dense tends to win** on paraphrase and vocabulary mismatch: "cozy" vs "intimate atmosphere", "cheap" vs "good value". On RIRD, at the same review unit and with the same aggregation, **dense beats BM25 under every aggregation**, significant after Holm correction (nDCG@10 gap +0.03 to +0.09). Our best review-level model is **Dense, top-10 mean: nDCG@10 0.668 [0.625, 0.711], MAP 0.608** (from `action_items.md` §2).

**Switching to sentence units widens the gap.** At sentence level, dense top-10 mean rises to **0.682** (MAP 0.618), while BM25 top-10 mean falls from 0.634 to 0.609. Dense gains because a sentence is usually one topic, so its single vector is sharp and nothing is truncated (§7). BM25 loses because a short sentence has few words to match. Dense beats BM25 significantly under max and top-k at sentence level, but not under plain mean (+0.039, p 0.072). Details in file 03 §4.

**BM25 tends to win** when the exact string matters:

- **Exact names**: a dish, brand or restaurant name ("khao soi", a specific restaurant's name) that the encoder may map near generic food words.
- **Rare terms and numbers**: a rare ingredient, "24 hours", a specific price.
- **Out-of-domain jargon** the encoder rarely saw in training.

**Neither is reliable on negation.** Bi-encoders are known to place "not spicy" fairly close to "spicy", because the sentences share most of their content. Our Thai negation queries (T19–T23) are there to test this.

**Dense matches topic, not polarity.** Looking at the evidence the best dense model (top-10 mean, review level) shows for its top restaurants:

- *"I am on a budget"*: rank 1 is Fishman Lobster Clubhouse (not relevant in RIRD), and its top evidence is "It's not bad but kinda pricey." The review is about **price**, so the model is being faithful to what it computed, but it says the *opposite* of what the user wants.
- *"Non-spicy Chinese meals"*: rank 1 Ding Tai Fung is relevant, but its second evidence review says "...Wonton was spicy ... red chili oil": the "non-" was ignored. Rank 2 is Pai Northern Thai Kitchen, which is Thai, not Chinese.

The embedding knows the text is about price or spiciness; it does not reliably know *which way*. A cross-encoder reranker (§5) or a negation-aware step would be the natural fix. These examples also show that whole reviews (one evidence review here is 94 words) are too coarse to serve as explanations.

## 10. Hybrid retrieval in one paragraph

Since the two make different mistakes, many systems **combine** them: run both, then merge the result lists, for example with a weighted sum of normalized scores or with **reciprocal rank fusion** (each document gets $\sum 1/(k + \text{rank})$ over the systems, with $k$ a constant, often 60). Hybrid retrieval is a common, strong baseline in the literature, but it is **not** in our current plan. In our project, "fusion" means aggregating *review or sentence scores into a restaurant score* (file 03), compared under the same unit for both models in Track A. It never means combining BM25 with dense, and learned or weighted fusion is out of scope. Don't confuse the two meanings of "fusion".

---

## Check yourself

1. What does it mean that our embeddings are "dense", and how big are they?
2. Compute the cosine between q = (1, 0) and d = (1, 1).
3. Why can we precompute review embeddings with a bi-encoder but not with a cross-encoder?
4. Why does the same model work on Thai and English text?
5. What happens to the end of a 400-token review with `MAX_SEQ_LEN = 256`?
6. Give one query where you'd expect BM25 to beat dense, and why.

<details>
<summary>Answers</summary>

1. Each text becomes a vector of 768 numbers, almost all non-zero. (BM25's word-count vectors are sparse: mostly zeros.)
2. Dot = 1. Lengths 1 and √2. Cosine = 1/√2 ≈ 0.707 (45° apart).
3. A bi-encoder encodes the review without seeing the query, so its vector can be computed once. A cross-encoder needs the query and review together, so it must run again for every new query.
4. It was trained by multilingual distillation so that a sentence and its translation map to nearly the same vector. Thai was among its training languages.
5. Tokens beyond 256 are cut off and never seen by the model; whatever the review says there has no effect on its embedding.
6. A query with an exact dish or place name, or a rare term, e.g. a specific dish name. BM25 matches the exact token; dense may blur it into generic "food" meaning.

</details>

## Where this lives in our code

- Dense encoding, `MAX_SEQ_LEN = 256`, cache, `l2_normalise`, `S_dense = Q @ E.T`, v1 pooled-vector reference: [baselines/baseline_models_v2.ipynb](../baselines/baseline_models_v2.ipynb)
- Sentence-level run (same code, `UNIT = 'sentence'`): [baselines/baseline_models_v2_sentence.ipynb](../baselines/baseline_models_v2_sentence.ipynb) (its result CSVs are gitignored, local only)
- Saved results table: [baselines/cache/results_review.csv](../baselines/cache/results_review.csv)
- Evidence shown per restaurant (source of the topic-vs-polarity examples in §9): `baselines/cache/evidence_review.csv` (gitignored, local only)
- Thai sentence encoding, `top_dense` vs `top_bm25`, overlap of their top-10 lists: [thai/thai_retrieval.ipynb](../thai/thai_retrieval.ipynb)
- Metrics: [baselines/eval_utils.py](../baselines/eval_utils.py)
