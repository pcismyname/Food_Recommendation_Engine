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

We drop fragments under 15 characters (`MIN_CHARS = 15`) and get 19,294 sentences from 2,000 reviews (9.6 per review). The C2 spot-check is where you record how often cuts are bad. A simple fix to try: protect common abbreviations (replace `กทม.` with `กทม` before splitting).

---

## 4. Negation: the biggest problem for BM25

Thai negation is a separate word placed **before** the thing it negates:

| Pattern | Meaning |
|---|---|
| ไม่ + adj | not ___ (ไม่เผ็ด = not spicy) |
| ไม่ค่อย + adj | not very ___ (ไม่ค่อยเผ็ด = not very spicy) |
| ไม่ + adj + เกินไป | not too ___ (ไม่หวานเกินไป = not too sweet, usually a **compliment**) |
| ไม่ + adj + เลย | not ___ at all |

**BM25 is a bag of words.** It counts which query tokens appear in a sentence (weighted by rarity) and ignores order. Our real output for T02 `อาหารรสเผ็ดจัด` ("very spicy food"):

```
[BM25]
 3. (10.215) รสไม่จัด มาเครื่องดื่มบ้าง          <- "flavour not strong"
 5. ( 9.839) รสแซบเบาๆไม่เผ็ดจัด ...              <- "mildly zesty, NOT very spicy"
```

Sentence 5 contains `รส`, `เผ็ด` and `จัด`, all query words, so it scores high. The extra `ไม่` is simply not in the query, so it costs almost nothing. BM25 cannot know that `ไม่` flips the meaning.

The opposite case works *by accident*: for T20 `ขนมที่ไม่หวานเกินไป` ("desserts not too sweet"), the query itself contains `ไม่` and `เกินไป`, so BM25 rewards sentences with the same phrase. Note that the notebook deliberately **keeps stop words** so that `ไม่` is not thrown away.

Dense encoders read the whole sentence, so they *can* model negation, but multilingual models are known to be weak at it. That is why C1 includes negation queries (T19–T23) and why these sentences need careful labels in C4 (a "not spicy" sentence is a **0** for a spicy query).

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
| **WangchanBERTa** (`airesearch/wangchanberta-base-att-spm-uncased`) | RoBERTa-base pretrained only on Thai (about 78 GB of text, reported in the paper). Beats mBERT/XLM-R on Thai classification and tagging | A strong Thai **encoder**, but it is a masked language model, not a sentence-embedding model. It needs fine-tuning (e.g. on sentence pairs) before cosine similarity is meaningful |
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

1. **No restaurant-level ranking.** We cannot group reviews by restaurant, so we cannot aggregate sentences into restaurant scores the way we do on RIRD. Thai evaluation is **sentence level only**, and restaurant-level Thai ranking is future work.
2. **Stars are not taste labels.** A 4-star review says the person was happy, not that the food was sour or spicy. That is why we need human 0/1/2 labels (C4).
3. **Subsample.** The notebook uses 2,000 reviews (19,294 sentences) for CPU. Results on all 40k may differ (more near-duplicates, larger pool).

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

- Sentence splitting, BM25 on `newmm` tokens, dense encoding, overlap/pool size: [thai/thai_retrieval.ipynb](../thai/thai_retrieval.ipynb) (generated by [thai/build_thai_retrieval_notebook.py](../thai/build_thai_retrieval_notebook.py))
- Queries, including the negation type: [thai/thai_queries.csv](../thai/thai_queries.csv)
- English baseline using the same encoder: [baselines/baseline_models_v2.ipynb](../baselines/baseline_models_v2.ipynb)
- Tasks in [action_items.md](../action_items.md): **C1** (queries with negations), **C2** (segmentation + spot-check), **C3** (pool), **C4** (labels: negated taste = 0), **C7** (BM25 vs dense), **E1** (Thai edge-case examples), **F1** (why Thai is sentence-level only)
