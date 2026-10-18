# 01 · Retrieval Basics (and BM25)

> **Why this matters for our project**
> Our system is a *search engine over reviews*: a user types a preference, we score reviews, and we rank restaurants.
> BM25 is one of our two baselines, so you need to know exactly what it rewards, what it misses, and why.

---

## 1. Retrieval vs. recommendation

People often say "recommendation" for anything that suggests items. There are two quite different families:

**Collaborative filtering (CF).** You have a big table of *users × items* (who rated or visited what). The idea is "people who liked what you liked also liked X". The classic method is **matrix factorization**: squeeze the user × item table into two small tables of hidden "taste factors" — one row of numbers per user and one per item — so that a user's predicted rating for an item is the dot product of their two rows. CF never reads text. It needs a history for each user, and it cannot answer a one-off request like "somewhere cozy with spicy food".

**Content / retrieval-based.** The user states what they want *right now* in words (the **query**), and we look for items whose **content** matches. No user history needed.

We do the second because:

- RIRD gives us queries plus human relevance labels, not a per-user rating history to predict.
- The product goal is "type what you feel like eating", which is a query.
- We want **explanations**: review sentences that justify the ranking. CF has no text to show.

## 2. Core vocabulary

| Term | Meaning | In our project |
|---|---|---|
| **Query** | What the user types | "cozy place with spicy food", or a Thai query like `อาหารรสเผ็ดจัด` |
| **Document** | One thing we score against the query | One review (v2), or later one sentence |
| **Corpus** | All documents | 29,168 RIRD reviews; for Thai, 10,000 sampled Wongnai reviews split into about 95,000 sentences (the test set; an earlier pilot used 2,000 reviews) |
| **Relevance** | Does the item satisfy the query? Judged by humans | `PMD.csv`: 5 annotators per (query, restaurant) |
| **Ranking** | Items sorted best-first by score | 50 restaurants per query, evaluated by nDCG@10, MAP |

## 3. Reviewed-item retrieval (RIR)

In normal web search the thing you score *is* the thing you return (a web page). In **reviewed-item retrieval** they are different:

```
   query ──► score each REVIEW ──► combine into a RESTAURANT score ──► rank restaurants
                                                                     └► show best review sentences
```

An item (restaurant) is not described by one text but by **many reviews** — in RIRD between 406 and 2,281 per restaurant. Each review talks about only a few things (one mentions the chili, another the décor). So we need a step that turns many review scores into one restaurant score. That step, **aggregation / fusion**, is covered in file 03.

## 4. Tokenization

Computers can't match "meaning" directly in BM25; they match **tokens** (usually words).

- **English** (our code, `simple_tokenise`): lowercase, replace anything that isn't a letter or digit with a space, split on spaces.
  `"Super SPICY noodles!!"` → `["super", "spicy", "noodles"]`
- **Thai** has no spaces between words, so we need a word segmenter. We use PyThaiNLP `word_tokenize(..., engine="newmm")`, a dictionary-based segmenter:
  `อาหารไม่เผ็ดเหมาะกับเด็ก` → `["อาหาร", "ไม่", "เผ็ด", "เหมาะกับ", "เด็ก"]` ("food", "not", "spicy", "suitable for", "kids").

If segmentation is wrong, BM25 is wrong: a dish name split in a strange place will never match the query. That is why task C2 includes a human spot-check.

## 5. The inverted index

Scanning every review for every query word is slow. Search engines build an **inverted index**: for each word, the list of documents containing it (plus counts).

```
spicy   → D1 (×2), D2 (×1)
cozy    → D3 (×1)
noodles → D1 (×1)
```

At query time you only touch the lists for the query words. (`rank_bm25` is a simple library and actually loops over all documents, which is fine for our 29k reviews; real engines like Elasticsearch use an inverted index.)

## 6. TF, IDF and TF-IDF

Two intuitions:

1. **Term frequency (TF):** if a review says "spicy" three times, it is probably more about spice than one that says it once.
2. **Inverse document frequency (IDF):** a word that appears in almost every review ("food", "the", "good") tells you little. A rarer word ("tom yum", "mala") is informative.

Classic TF-IDF score of a word $t$ in document $D$:

$$\text{tfidf}(t, D) = \text{tf}(t, D) \times \log\frac{N}{n(t)}$$

where $\text{tf}(t,D)$ = how many times word $t$ occurs in document $D$, $N$ = number of documents in the corpus, and $n(t)$ = number of documents that contain $t$. A document's score for a query is the sum over the query words.

Problem: TF grows without limit (saying "spicy" 20 times is not 20× more relevant), and long reviews contain more words, so they win just by being long. BM25 fixes both.

## 7. BM25

BM25 ("Best Matching 25") scores document $D$ for query $Q$ as:

$$\text{score}(D,Q) = \sum_{q \in Q} \text{IDF}(q)\cdot \frac{f(q,D)\,(k_1+1)}{f(q,D) + k_1\left(1 - b + b\,\frac{|D|}{\text{avgdl}}\right)}$$

In words:

- $q \in Q$: loop over each word $q$ of the query and add up the contributions.
- $f(q,D)$: how many times $q$ occurs in document $D$ (the TF).
- $|D|$: length of $D$ in tokens; $\text{avgdl}$: average document length in the corpus.
- $k_1$: controls **term saturation**. `rank_bm25` default is $k_1 = 1.5$.
- $b$: controls **length normalization**, between 0 and 1. Default $b = 0.75$.

**IDF as implemented in `rank_bm25.BM25Okapi`** (we checked the installed source):

$$\text{IDF}(q) = \ln\frac{N - n(q) + 0.5}{n(q) + 0.5}$$

with $N$ = number of documents and $n(q)$ = number containing $q$. If a word appears in **more than half** the documents this becomes negative, so the library replaces any negative IDF with a small floor: $\varepsilon \times$ (average IDF over the vocabulary), with $\varepsilon = 0.25$.

### Saturation ($k_1$), intuitively

Take an average-length document (so the length part equals 1). The TF part becomes $\frac{f\,(k_1+1)}{f + k_1}$. With $k_1 = 1.5$:

| times word appears $f$ | 1 | 2 | 3 | 5 | 10 | 100 |
|---|---|---|---|---|---|---|
| TF part | 1.00 | 1.43 | 1.67 | 1.92 | 2.17 | 2.46 |

It can never exceed $k_1 + 1 = 2.5$. The first mention matters most; spamming a word stops helping. Smaller $k_1$ = saturates faster.

### Length normalization ($b$), intuitively

The term $1 - b + b\frac{|D|}{\text{avgdl}}$ is 1 for an average-length document, bigger for long documents (their TF part shrinks), smaller for short ones. $b = 0$ turns this off; $b = 1$ fully normalizes. A word appearing once in a 10-word review is stronger evidence than once in a 500-word review.

### Tiny worked example (numbers produced by `rank_bm25`)

Three "reviews" (already tokenized):

```
D1: spicy spicy noodles               (length 3)
D2: spicy soup with rice and egg      (length 6)
D3: cozy cafe                         (length 2)
N = 3, avgdl = 11/3 ≈ 3.67
```

**IDF.**
- `cozy` is in 1 of 3 docs: $\ln\frac{3-1+0.5}{1+0.5} = \ln(2.5/1.5) \approx 0.511$. Same for every word that occurs in one document.
- `spicy` is in 2 of 3 docs: $\ln(1.5/2.5) \approx -0.511$, negative, so it gets the floor. Average IDF over the 9 vocabulary words is $(8 \times 0.511 - 0.511)/9 \approx 0.397$, floor $= 0.25 \times 0.397 \approx 0.099$.

**Query "spicy":**
- D1: $f=2$, length factor $0.25 + 0.75 \times 3/3.67 \approx 0.864$. TF part $= \frac{2 \times 2.5}{2 + 1.5 \times 0.864} = 5/3.296 \approx 1.517$. Score $\approx 0.099 \times 1.517 \approx \mathbf{0.151}$.
- D2: $f=1$, longer doc, length factor $\approx 1.477$. TF part $= 2.5 / (1 + 2.216) \approx 0.777$. Score $\approx \mathbf{0.077}$.
- D3: no "spicy" → **0**.

D1 wins: more mentions *and* shorter.

**Query "cozy spicy":** D3 scores $0.511 \times 1.257 \approx \mathbf{0.642}$ for `cozy` alone, beating D1 (0.151). The rare word dominates the common one. That is IDF doing its job.

> Side note for our v1 baseline: v1 concatenated all reviews of a restaurant into one giant document, giving only $N = 50$ documents. Any word that appears in more than 25 of those 50 mega-documents gets the same tiny floor IDF, so common food words probably stop discriminating at all. This is another reason v2 scores individual reviews.

## 8. BM25's weaknesses

BM25 only counts exact token overlap. That causes:

1. **Vocabulary mismatch.** Query "cozy" will not match a review saying "warm, intimate atmosphere" — zero shared tokens, score 0.
2. **Synonyms and paraphrase.** "cheap" vs "inexpensive" vs "good value"; Thai `ถูก` vs `ไม่แพง` ("cheap" vs "not expensive").
3. **Negation.** Query "not spicy" — the word `not` either isn't in the corpus or is very common, and `spicy` matches. In our example, the query `["not", "spicy"]` gives exactly the same scores as `["spicy"]` (0.151, 0.077, 0), because `not` does not appear. A review saying "this is NOT spicy at all" and one saying "extremely spicy" both get credit for `spicy`. BM25 has no idea what `not` does to the meaning.
4. **Word order is ignored** ("bag of words"): "spicy not sweet" and "sweet not spicy" score the same.

Dense retrieval (file 02) addresses 1–2 and sometimes 3.

## 9. Stop words, and why we keep ไม่

**Stop words** are very common function words ("the", "a", "is") that many pipelines delete before indexing to save space. BM25's IDF already down-weights them, so deleting them gains little.

For us, deleting them would be harmful. PyThaiNLP's standard Thai stop-word list (1,030 words) **contains `ไม่` ("not") and `ไม่ค่อย` ("not very")**. If we removed stop words, the query `ขนมที่ไม่หวานเกินไป` ("desserts that are not too sweet") would lose its negation and become a query for *sweet* desserts — the exact opposite. Our Thai queries include a whole negation group (T19–T23). So **our tokenizers do not remove stop words**, in either language. (BM25 still can't *understand* negation, but at least the token is there for a later model or analysis to use.)

Pilot observation: for T20 `ของหวานที่ไม่หวานมาก` ("desserts that are not too sweet"), keeping `ไม่` let BM25 match sentences saying `ไม่หวานมาก` ("not very sweet") literally, so its top results looked right. That keeping stop words actually helps BM25 on negation queries is still a hypothesis; the hand labels (file 06, §8) will test it.

---

## Check yourself

1. Why can't collaborative filtering answer "somewhere cozy with spicy food" for a brand-new user?
2. In reviewed-item retrieval, what is the difference between the thing we *score* and the thing we *return*?
3. A review says "spicy" 10 times; another says it once. With $k_1 = 1.5$ and equal lengths, is the first one 10× better?
4. What does $b = 0$ do?
5. Why did `spicy` get a tiny IDF in the worked example, and what does `rank_bm25` do with negative IDFs?
6. Why would removing Thai stop words break query T20?

<details>
<summary>Answers</summary>

1. CF needs the user's past ratings/visits and works from item IDs, not text. A one-off natural-language request with no history gives it nothing to work with.
2. We score reviews (or sentences) but return restaurants. Many review scores must be aggregated into one restaurant score.
3. No. The TF part is 2.17 vs 1.00, only about 2.2×, and it can never exceed $k_1 + 1 = 2.5$ (saturation).
4. Turns off length normalization: long and short documents are treated the same.
5. It appears in 2 of 3 documents (more than half), so $\ln\frac{N-n+0.5}{n+0.5}$ is negative. `rank_bm25` replaces negative IDFs with $0.25 \times$ average IDF.
6. `ไม่` is on the PyThaiNLP stop-word list. Removing it turns "not too sweet" into "too sweet", flipping the meaning.

</details>

## Where this lives in our code

- BM25 at review level, `simple_tokenise`, and the v1 concatenated-document reference: [baselines/baseline_models_v2.ipynb](../baselines/baseline_models_v2.ipynb) (BM25 cells)
- Thai tokenizer `th_tokenise` (PyThaiNLP `newmm`) and Thai BM25 over sentences: [thai/thai_retrieval.ipynb](../thai/thai_retrieval.ipynb)
- Thai queries including the negation group: [thai/thai_queries.csv](../thai/thai_queries.csv)
- Evaluation metrics (MAP, nDCG@k, Recall@k): [baselines/eval_utils.py](../baselines/eval_utils.py)
