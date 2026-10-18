# 06 — Thai text processing for retrieval

> **Why this matters for our project**
> Before BM25 or a dense model can rank Thai review sentences, we must cut reviews into sentences and sentences into words. Thai makes both steps hard.
> Errors here flow straight into the C3 pool you will label, and they explain some failures we already see (BM25 returning "ไม่เผ็ดจัด" for "อาหารรสเผ็ดจัด").

---

## 1. Why Thai is hard for a computer

English gives us two free signals: **spaces between words** and **a full stop at the end of a sentence**. Thai has neither.

| Signal | English | Thai |
|---|---|---|
| Word boundary | space | **none**: `อาหารรสเผ็ดจัด` = อาหาร · รส · เผ็ด · จัด |
| Sentence end | `.` `!` `?` | **usually none**: a **space** marks the end of a phrase *or* a sentence |
| Full stop `.` | sentence end | mostly used in **abbreviations**: กทม. (Bangkok), อ. (อำเภอ district, or อาจารย์ teacher), ม.ค. (January) |

So a Thai space is ambiguous (phrase break or sentence break?), and a `.` is a trap (abbreviation, not an end).

Real example from our spot-check in [thai/thai_retrieval.ipynb](../thai/thai_retrieval.ipynb) (section 2a):

```
| คราวนี้เลยตั้งใจไปชิม ระยะทางจาก กทม.
| เกษตร วิทยาเขตกำแพงแสน
...
| แล้วไปร้านอาหาร K.U.
| สโมสร แทน อยู่ไม่ไกล
```

The splitter treated the `.` in `กทม.` and `K.U.` as sentence ends. "From Bangkok to Kasetsart, Kamphaeng Saen campus" became two broken pieces. The same happened with `อ.`: "the steak at อ. (district) Kamphaeng Saen" was cut right after `อ.`.

---

## 2. Word segmentation (word tokenisation)

### Dictionary-based maximal matching: `newmm`

PyThaiNLP's default word tokeniser is **`newmm`** ("new multi-cut"). According to the PyThaiNLP docs, it is *dictionary-based maximal matching, constrained by Thai Character Cluster (TCC) boundaries*.

- **Dictionary-based**: it has a list of known Thai words.
- **Maximal matching**: among all ways to cut the string into dictionary words, prefer the one with the **fewest words** (in other words, the longest pieces).
- **TCC constraint**: some characters can never be separated (a consonant and its vowel or tone mark), so cuts are only allowed at cluster boundaries. This avoids nonsense cuts and makes it faster.

```python
from pythainlp.tokenize import word_tokenize
word_tokenize("อาหารรสเผ็ดจัด", engine="newmm")
# ['อาหาร', 'รส', 'เผ็ด', 'จัด']
word_tokenize("ไม่ค่อยเผ็ด", engine="newmm")
# ['ไม่ค่อย', 'เผ็ด']        <- "ไม่ค่อย" is one dictionary entry
```

(Outputs above were run with PyThaiNLP 5.4.0 in the project's `cpdsai` environment.)

### Ambiguity

The same letters can be cut in different valid ways. The classic example is **ตากลม**:

- ตา · กลม = "round eyes"
- ตาก · ลม = "to be exposed to the wind"

Only context decides. A dictionary method picks one cut by its rule (`newmm` here keeps `ตากลม` as one token). Words that are not in the dictionary (new slang, shop names, misspellings) get chopped into odd pieces:

```python
word_tokenize("อร่อยมากกกก แซ่บจ้าาา", engine="newmm")
# ['อร่อย', 'มา', 'กก', 'กก', ' ', 'แซ่บ', 'จ้า', 'าา']
```

"มากกกก" (very-y-y) became `มา` + `กก` + `กก`. The word "มาก" is gone, and BM25 now sees a token `มา` ("come")!

Other engines exist (`attacut` and `deepcut`, which are neural; `longest`, which is plain longest matching). They are more robust to unknown words but slower. For this project, `newmm` is fine. Just know where it breaks.

---

## 3. Sentence segmentation: `crfcut`

```python
from pythainlp.tokenize import sent_tokenize
sent_tokenize("เลยตั้งใจออกไปหาอะไรกินกันต่างจังหวัดใกล้ ๆ กทม. พวกเราเลยนึกถึงสเต็กที่ อ. กำแพงแสน เคยอยากชิมแต่ยังไม่มีโอกาส",
              engine="crfcut")
# ['เลยตั้งใจออกไปหาอะไรกินกันต่างจังหวัดใกล้ ๆ กทม.',
#  ' พวกเราเลยนึกถึงสเต็กที่ อ.',
#  ' กำแพงแสน ',
#  'เคยอยากชิมแต่ยังไม่มีโอกาส']
```

**What is crfcut?** A **Conditional Random Field (CRF)**, a classic sequence-labelling model. It looks at every space and decides "sentence boundary or not" from features of the words around it (e.g. is the next word a typical sentence starter such as พวกเรา, เลย, แต่?). The PyThaiNLP docs say its default model is **trained on the TED dataset** (Thai transcripts of TED talks). They report about 82% "space-correct" accuracy on TED and 87% on the ORCHID corpus (the docs give 95% for the best published system on ORCHID).

Consequences for us:

- TED talks are spoken, edited language. Wongnai reviews are informal, full of emojis, English menu names and abbreviations, so expect **lower accuracy** on our data.
- A `.` after an abbreviation makes a cut very likely (`กทม.`, `อ.`, `K.U.`).
- Some "sentences" merge several ideas because the reviewer used no spaces.

We drop fragments under 15 characters (`MIN_CHARS = 15`). On the 2,000-review pilot this gave 19,294 sentences (9.6 per review); the 10,000-review test corpus gives about 95,000 (see §7).

**C2 spot-check result (done).** In the hand-checked sample of split sentences:

| Problem | Share of sentences |
|---|---|
| Cut right after an abbreviation (`กทม.`, `อ.`, `K.U.`) | 0.7% |
| Several sentences merged into one piece longer than 200 characters | 1.6% |
| Mostly non-Thai (English menu names, emojis, numbers) | 0.7% |

**Verdict: good enough.** Bad cuts are rare, and a sentence that is a bit too long or too short still carries its taste words, so it can still be retrieved and labelled. We keep `crfcut` as it is and do not add an abbreviation fix. We report the remaining weakness as a limitation: published results put `crfcut` at only about **0.77 F1** for sentence boundaries, so it is not a solved step, just an acceptable one for our purpose. (If we ever need a fix, the simple one is to protect common abbreviations, e.g. replace `กทม.` with `กทม` before splitting.)

---

## 4. Negation: the biggest problem for BM25

Thai negation is a separate word placed **before** the thing it negates:

| Pattern | Meaning |
|---|---|
| ไม่ + adj | not ___ (ไม่เผ็ด = not spicy) |
| ไม่ค่อย + adj | not very ___ (ไม่ค่อยเผ็ด = not very spicy) |
| ไม่ + adj + เกินไป | not too ___ (ไม่หวานเกินไป = not too sweet, usually a **compliment**) |
| ไม่ + adj + เลย | not ___ at all |

**BM25 is a bag of words.** It counts which query tokens appear in a sentence (weighted by rarity) and ignores order. Our real output for an earlier draft of T02, `อาหารรสเผ็ดจัด` ("very spicy food"; the frozen wording is `อยากกินอะไรเผ็ดๆ แบบจัดๆ`):

```
[BM25]
 3. (10.215) รสไม่จัด มาเครื่องดื่มบ้าง          <- "flavour not strong"
 5. ( 9.839) รสแซบเบาๆไม่เผ็ดจัด ...              <- "mildly zesty, NOT very spicy"
```

Sentence 5 contains `รส`, `เผ็ด` and `จัด`, all query words, so it scores high. The extra `ไม่` is simply not in the query, so it costs almost nothing. BM25 cannot know that `ไม่` flips the meaning.

The opposite case works *by accident*: when the query itself contains the negation, BM25 rewards sentences with the same phrase. Note that the notebook deliberately **keeps stop words** so that `ไม่` is not thrown away.

**Pilot observation (2,000 reviews).** For T20 `ของหวานที่ไม่หวานมาก` ("desserts that are not too sweet"; an earlier draft worded it `ขนมที่ไม่หวานเกินไป`), **both** systems returned sentences that really say ไม่หวานมาก (not very sweet). BM25 got there by literal matching of `ไม่` + `หวาน` + `มาก`; dense got there by meaning. So the negation group is not automatically a BM25 loss. Whether this holds across all of T19–T23 is something the labels will show, not something we can claim yet.

Dense encoders read the whole sentence, so they *can* model negation, but multilingual models are known to be weak at it. That is why C1 includes negation queries (T19–T23) and why these sentences need careful labels in C4.

**Guideline rule for negation queries** ([thai/annotation_guidelines.md](../thai/annotation_guidelines.md), R1 and the degree table in 3.2): read every `ไม่` before deciding.

- For a query that *wants* the taste, a negated taste is contradicted: `ไม่เผ็ด` (not spicy) is a **0** for a spicy query.
- For a query that wants *not / less* of the taste (T20–T22), it flips: for "not too sweet", `ไม่หวานมาก` (not very sweet) or `หวานกำลังดี` (just sweet enough) = **2**, while `หวานเกินไป` (too sweet) = **0**.
- T19 (no spice, for kids) is strictest: `ไม่เผ็ด` = 2, but `ไม่เผ็ดมาก` (not very spicy) is only 1.
- A `ไม่` that does not apply to the food does not count: in *"ลืมบอกทางร้านว่าไม่เผ็ด … จะทำรสแซ่บจัดจ้านมากกก"* ("forgot to tell the shop not spicy … they make it really fiery") the food was very spicy.

---

## 5. Colloquial spelling in reviews

Review writers type the way they speak:

- **Spelling variants**: แซ่บ / แซบ / แซ่บบ (zesty-spicy). These are different tokens to BM25, so a query with `แซ่บ` misses `แซบ`.
- **Repeated letters for emphasis**: อร่อยมากกกก, จ้าาา, ดีมากๆๆ. As shown above, these break `newmm`.
- **ๆ (mai yamok)**: the repetition mark, meaning "say the previous word again": เปรี้ยวๆ = เปรี้ยวเปรี้ยว ("sourish"). Tokenisers emit it as its own token `ๆ`. Our query T01 `อยากกินอะไรเปรี้ยวๆ` contains it.
- Mixed Thai/English: "Shibuya Honey Toast", "medium-rare".

Partial help: `pythainlp.util.normalize` removes duplicated **vowels and tone marks**, e.g. `normalize("จ้าาา")` returns `จ้า`. It does **not** collapse repeated consonants (`มากกกก` stays as it is), so a small regex such as `re.sub(r"(.)\1{2,}", r"\1", text)` is a useful extra step. A tiny variant map (แซบ→แซ่บ) also helps BM25. Dense models are naturally more tolerant of these variants.

---

## 6. Models: Thai-specific vs multilingual

| Model | What it is | Use for us |
|---|---|---|
| **WangchanBERTa** (`airesearch/wangchanberta-base-att-spm-uncased`) | RoBERTa-base pretrained only on Thai (about 78 GB of text, reported in the paper). Beats mBERT/XLM-R on Thai classification and tagging | A strong Thai **encoder**, but it is a masked language model, not a sentence-embedding model. It needs fine-tuning (e.g. on sentence pairs) before cosine similarity is meaningful. Fine-tuning is out of scope, so we don't use it |
| **paraphrase-multilingual-mpnet-base-v2** (sentence-transformers) | Multilingual sentence encoder (50+ languages including Thai), 768-dim vectors | **What we use**, same as the English baseline, so the two tracks are comparable |
| **multilingual-e5** (`intfloat/multilingual-e5-base` / `-large`) | Multilingual retrieval embeddings built on XLM-RoBERTa; inputs are prefixed with `"query: "` and `"passage: "` | Stronger retrieval alternative |
| **BGE-M3** (`BAAI/bge-m3`) | Multilingual (100+ languages), long inputs (up to 8,192 tokens), gives dense, sparse and multi-vector outputs | Stronger alternative; its sparse output is a learned BM25-like signal |

Benchmarks disagree on whether e5 or BGE-M3 is better for Thai (e.g. the SEA-BED benchmark puts multilingual-e5-large slightly ahead). If time allows, adding one of them is a clean extra row in C7. Remember **pool bias** ([04_evaluation.md](04_evaluation.md)): a model added after labelling needs its unjudged top-10 labelled.

---

## 7. The Wongnai dataset and its limits

Facts (Hugging Face card `Wongnai/wongnai_reviews` and our local files in `datasets/wongnai/`):

- **Train: 40,000 reviews; test: 6,204** (the test split has no ratings; it uses -1).
- Only **two columns**: `review_body` (text) and `star_rating`. The card describes 1–5 stars; our parquet stores them as **0–4** (0 = 1 star). Train distribution: 0: 415, 1: 1,845, 2: 12,171, 3: 18,770, 4: 6,799, so mostly positive.
- **No restaurant id, name or location.**

What this limits:

1. **No restaurant-level ranking.** We cannot group reviews by restaurant, so we cannot aggregate sentences into restaurant scores the way we do on RIRD. Thai evaluation is **sentence level only**, and restaurant-level Thai ranking is out of scope (future work).
2. **Stars are not taste labels.** A 4-star review says the person was happy, not that the food was sour or spicy. That is why we need human 0/1/2 labels (C4).
3. **Subsample: 10,000 reviews (decided).** The first notebook was a CPU pilot on 2,000 reviews (19,294 sentences). The test set uses **10,000 sampled reviews (about 95,000 sentences)**, run on Colab. Why the bigger sample: rare tastes and negations (e.g. ไม่หวานมาก, "not very sweet") need enough relevant sentences in the corpus, or the top 10 is mostly filler. Why fix it *now*: labels are tied to the corpus. If the corpus changed after labelling, each system's top 10 would change and new unjudged sentences would appear, so the size is fixed before anyone labels. Results on all 40k may still differ (more near-duplicates, larger pool).

---

## 8. From retrieval to labels

Wongnai has no relevance labels, so we make them by hand. The workflow:

```
thai_retrieval_10k_colab.ipynb ──► results/topk_n10000_s42.csv ──► build_label_pool.py
  (10k reviews, seed 42;              (BM25 + dense top 10 per         (blind label sheet)
   BM25 + dense)                       query, 480 rows)                       │
                                                                              ▼
score_labels.py ◄── hand labels 0/1/2 (annotation_guidelines.md) ◄── label_sheet.csv
```

1. **Retrieve.** The Colab notebook splits the 10,000 reviews into sentences, runs BM25 and the dense encoder for every query and writes each system's top 10 to `thai/results/topk_n10000_s42.csv` (480 rows = 24 queries × 2 systems × 10).
2. **Pool.** `build_label_pool.py` takes the union of both top-10 lists per query, removes duplicates and **shuffles** the order, so the sheet does not show which system found a sentence or at what rank (a *blind* pool). That is about **19 sentences per query**. It writes `label_sheet.csv` (to label), `pool_key.csv` (the hidden key; annotators don't open it) and `label_sheet_annotator2.csv`, a seeded sample of **60 pairs** spread across queries for the second annotator.
3. **Label.** Each (query, sentence) gets 0 / 1 / 2 by [thai/annotation_guidelines.md](../thai/annotation_guidelines.md): taste must be *described*, not guessed from a dish name (`ต้มยำ` alone is not "sour"); a combination needs both parts for a 2, one part = 1; a contradiction = 0; hedges lower the label by one.
4. **Score.** `score_labels.py` computes **nDCG@10** (gains 0/1/2) and **P@10** (lenient: label ≥ 1; strict: label = 2) per system, with bootstrap 95% CIs, a paired randomization test dense vs. BM25 (Holm-corrected), a descriptive breakdown by query type (single / combo / negation), and agreement with annotator 2 (Cohen's kappa, unweighted and linear-weighted). `--partial` scores only fully labelled queries, for a pilot.

Why "about 19" and not 20: the two systems rarely find the same sentence. In the 2,000-review pilot (pool of 465 pairs), BM25 and dense shared **fewer than 1 sentence per query** in their top 10 on average.

**What the pilot top 10s looked like** (no labels yet, just reading):

- **BM25 is fooled by filler words.** For T02 `อยากกินอะไรเผ็ดๆ แบบจัดๆ` ("I want something really spicy") it matches sentences that share `อยากกินอะไร` ("want to eat something") or `จัด`, not the spice.
- **Dense drifts by topic.** For T11 it returns sour-spicy sentences about *other* dishes; for T02 its rank 5 says `รสชาติจะออกอ่อนๆ` ("the taste is rather mild"), on topic but the wrong degree.
- **Both handle the T20 negation** (§4).

These are impressions; the labels decide. Keep the scale in mind too: with only 24 queries, the test can detect nDCG@10 differences of roughly **0.11–0.17** at best, so a smaller gap would show up as "not significant" even if it is real.

---

## Check yourself

1. Give two reasons why `sent_tokenize` cut `ระยะทางจาก กทม.` from the rest of its sentence.
2. What does "maximal matching" prefer, and why does `newmm` turn `มากกกก` into strange tokens?
3. Explain in one sentence why BM25 ranks `รสแซบเบาๆไม่เผ็ดจัด` high for `อาหารรสเผ็ดจัด`.
4. Why does the notebook keep stop words for BM25?
5. Can we use WangchanBERTa as a drop-in replacement for the mpnet sentence encoder? Why or why not?
6. Name two things Wongnai lacks that RIRD has, and what each prevents.

<details>
<summary>Answers</summary>

1. The `.` in the abbreviation looks like a sentence end, and there is a space right after it; crfcut was trained on TED transcripts, not reviews.
2. The segmentation with the fewest (longest) dictionary words. `มากกกก` is not in the dictionary, so it is cut into whatever dictionary pieces fit (`มา`, `กก`, `กก`).
3. It contains every query token (รส, เผ็ด, จัด), and BM25 ignores word order, so it never notices that `ไม่` reverses the meaning.
4. `ไม่` and other function words are usually in stop-word lists. Removing them would erase negation.
5. Not directly. It is a masked language model; without sentence-level fine-tuning its pooled vectors are not good for cosine-similarity retrieval.
6. Restaurant ids (prevent restaurant-level aggregation and ranking) and query relevance labels (prevent any measured evaluation until we label C4).

</details>

---

## Where this lives in our code / tasks

- 2,000-review CPU pilot (sentence splitting, BM25 on `newmm` tokens, dense encoding, overlap/pool size): [thai/thai_retrieval.ipynb](../thai/thai_retrieval.ipynb) (generated by [thai/build_thai_retrieval_notebook.py](../thai/build_thai_retrieval_notebook.py))
- 10,000-review test run on Colab, exports `thai/results/topk_n10000_s42.csv`: [thai/thai_retrieval_10k_colab.ipynb](../thai/thai_retrieval_10k_colab.ipynb) (generated by [thai/build_thai_retrieval_colab_notebook.py](../thai/build_thai_retrieval_colab_notebook.py))
- Blind label pool: [thai/build_label_pool.py](../thai/build_label_pool.py) → `thai/labels/` (how to label: [thai/labels/README.md](../thai/labels/README.md))
- Labelling rules, 0/1/2 scale, per-query rules and real examples: [thai/annotation_guidelines.md](../thai/annotation_guidelines.md)
- Scoring and agreement: [thai/score_labels.py](../thai/score_labels.py)
- Queries, including the negation type: [thai/thai_queries.csv](../thai/thai_queries.csv)
- English baseline using the same encoder: [baselines/baseline_models_v2.ipynb](../baselines/baseline_models_v2.ipynb)
- Status: the Colab run has not been executed yet and labelling has not started.
- Tasks in [action_items.md](../action_items.md): **C1** (queries with negations), **C2** (segmentation + spot-check, done: good enough), **C3** (blind pool from the 10k run), **C4** (labels by the guidelines: negated taste = 0 unless the query asks for less), **C5** (second annotator, 60 pairs), **C7** (BM25 vs dense with `score_labels.py`), **E1** (annotation guidelines with Thai edge cases), **F1** (why Thai is sentence-level only)
