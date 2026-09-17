# Project Proposal: Thai Food Recommendation Engine

**Group:** Flower | **Members:** Ton (st127004) · Sam (st127012)
**Section ownership:** Datasets (§4), Methodology (§5), Baseline & Proposed Approach (§6) — Sam

---

## Datasets

### 1. RIRD — Reviewed-Item Retrieval Dataset (English, Labelled)
- **Source:** [D3Mlab/rir_data](https://github.com/D3Mlab/rir_data) · ECIR 2023 paper: [arxiv 2308.00762](https://arxiv.org/pdf/2308.00762)
- **Files:** `reviews/50_restaurants_all_rates.csv`, `PMD.csv`
- **Size:**
  - Reviews: **29,168 rows** × 9 columns (~23 MB)
  - PMD (query-restaurant relevance labels): **5,000 rows** (100 queries × 50 restaurants)
- **Features:**

| Column | File | Description |
|---|---|---|
| `business_id` | reviews | Unique restaurant identifier |
| `user_id` | reviews | Reviewer identifier |
| `review_stars` | reviews | Star rating (1–5) |
| `review_text` | reviews | Full review text (English) |
| `name` | reviews | Restaurant name |
| `categories` | reviews | Cuisine type (e.g. Ramen, Japanese) |
| `date` | reviews | Review date (2008–2019) |
| `query` | PMD | Free-text query expressing a food preference |
| `Annotator1–5` | PMD | Binary relevance label per annotator (0 or 1) |
| `If only Low or High` | PMD | Aggregated binary relevance label |

- **Target Variable:** Binary relevance label per (query, restaurant) pair — used to evaluate the ranking quality of the recommendation system.
- **Relevant Characteristics:** 50 Toronto restaurants; ~27% of (query, restaurant) pairs are labelled positive; 5 annotators with inter-annotator agreement κ ≈ 0.528; 100 unique queries covering indirect, negation, and contradictory preference types; 2 out of 50 restaurants are Thai cuisine (~4% Thai food content); reviews span 2008–2019; mean review length ~94 words.

---

### 2. Wongnai Reviews (Thai, Unlabelled)
- **Source:** [HuggingFace – Highgroundbkk/wongnai_reviews](https://huggingface.co/datasets/Highgroundbkk/wongnai_reviews) · [Wongnai Kaggle Challenge](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction)
- **Files:** `wongnai/train.parquet` (40,000 reviews), `wongnai/test.parquet` (6,203 reviews)
- **Size:** ~46,000 rows total (~30 MB combined)
- **Features:**

| Column | Description |
|---|---|
| `review_body` | Full review text in Thai (~99% Thai, some English mixed) |
| `star_rating` | Customer satisfaction on a **0–4** scale |

- **Target Variable (for demo purposes):** `star_rating` — ordinal satisfaction score; used for filtering and ranking review sentences in the Thai-language demo component.
- **Relevant Characteristics:** ~99% Thai-language content; no restaurant IDs, no query labels, and no restaurant grouping — making quantitative end-to-end evaluation impossible without additional annotation; star rating skewed toward 2–3; median review length ~20 words; useful for Thai-query-to-review-sentence retrieval in the live demo.

---

### 3. Thailand Foods — Dish Gazetteer
- **Source:** [Kaggle – ponthakornsodchun/foods-in-thailand](https://www.kaggle.com/datasets/ponthakornsodchun/foods-in-thailand)
- **File:** `datasets/thailand_foods.csv`
- **Size:** **325 rows** × 6 columns (~28 KB)
- **Features:**

| Column | Description |
|---|---|
| `en_name` | English dish name (e.g. Pad Thai, Som Tam) |
| `th_name` | Thai script dish name |
| `ingredients` | Key ingredients (comma-separated) |
| `course` | Meal course: main dish, snack, dessert, etc. |
| `province` | Province of origin |
| `region` | Thai region: Central, Northeast, North, South, Various |

- **Target Variable:** N/A — used as a food-term normalisation gazetteer (not a model target).
- **Relevant Characteristics:** 325 Thai dishes with 3 duplicate dish names; region is dominated by `Various`/`Unknown`; courses are mainly main dish, snack, and dessert; provides structured ingredient vocabulary for mapping free-text Thai queries to food concepts used in retrieval.

---

### 4. M-ABSA Thai — Multilingual Aspect-Based Sentiment
- **Source:** [HuggingFace – Multilingual-NLP/M-ABSA](https://huggingface.co/datasets/Multilingual-NLP/M-ABSA)
- **Files:** `datasets/m-absa/food/`, `datasets/m-absa/restaurant/`
- **Size:** Thousands of annotated aspect-sentiment tuples across 7 domains (food, restaurant, hotel, laptop, phone, sight, coursera)
- **Features:**

| Column | Description |
|---|---|
| `text` | Review sentence in Thai |
| `aspect_category` | Coarse aspect label: `food quality`, `food general`, `service general`, `ambience general` |
| `sentiment` | Positive / Negative / Neutral |

- **Target Variable:** N/A — used as supporting signal for Thai aspect-sentiment detection, not as a model target.
- **Relevant Characteristics:** Thai-language aspect-sentiment annotations at the sentence level; aspect categories are coarse and do not directly map to flavour axes (sour, spicy, cheap) used in user queries; provides a fine-grained sentiment layer to complement the star-rating signal from Wongnai.

---

## Methodology

### Exploratory Data Analysis (EDA)

- **RIRD:** Plot review count per restaurant (bar chart), star rating distribution (histogram), review length distribution (boxplot), and word clouds of review text. Inspect the PMD relevance label distribution across query types and annotators. Check inter-annotator agreement and flag queries with low consensus.
- **Wongnai:** Plot star rating distribution (histogram), review length distribution, and language detection breakdown (Thai vs. mixed vs. English). Identify the most common words and topic clusters using frequency analysis.
- **Thailand Foods:** Plot dish count by region (bar chart) and by course (pie chart). Identify duplicate dish names and inspect ingredient vocabulary coverage.
- **M-ABSA Thai:** Plot aspect category distribution and sentiment polarity breakdown (positive / negative / neutral) within the food and restaurant domains.

### Data Preprocessing

**Missing-value handling**
- RIRD reviews: `categories` may be missing for some restaurants — fill with `"Unknown"`.
- PMD.csv: one duplicate row was identified in EDA and will be dropped before evaluation.
- Wongnai: `review_body` is never null (it is the primary column); no imputation needed.
- Thailand Foods: `province` and `region` contain `"Various"` and `"Unknown"` placeholder values — these are kept as-is since they are valid categories, not true missing values.
- M-ABSA: subset to Thai-language entries only; drop rows with missing `text` or `aspect_category`.

**Outlier analysis**
- RIRD reviews: flag and inspect reviews shorter than 10 words (likely spam or empty entries); keep them unless they are entirely whitespace.
- PMD: the ~27% positive rate is expected given the sparse nature of query-restaurant relevance — this is not an outlier, but it informs metric selection (rank-aware metrics favoured over accuracy).
- Wongnai star ratings: the 0–4 scale is non-standard; no transformation is applied since the scale is used only as a filter, not as a regression target.

**Data distribution and skewness**
- Review lengths in both RIRD and Wongnai are right-skewed (most reviews are short, a few are very long). For sentence-level retrieval, each review is split into individual sentences, making the unit of analysis the sentence rather than the full review — which reduces the impact of length skewness.
- Star ratings in Wongnai are concentrated at 2–3; this will be noted in the demo but does not affect the retrieval pipeline directly.

**Feature preparation / selection**
- **RIRD:** Split each `review_text` into sentences using NLTK sentence tokenizer. Encode each sentence using a pre-trained multilingual sentence transformer (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`). Store per-sentence embeddings grouped by `business_id`.
- **Wongnai:** Split `review_body` into Thai sentences using PyThaiNLP sentence segmenter. Encode each sentence with the same multilingual model to allow cross-lingual query-to-sentence retrieval.
- **Query encoding:** Each query (RIRD PMD or user-typed Thai query) is encoded with the same model to produce a single query embedding.
- **Thailand Foods:** Tokenise `en_name`, `th_name`, and `ingredients` into keyword lists to use as a gazetteer for normalising food-term mentions in queries (e.g. mapping `เผ็ด` → "spicy").
- No classical feature scaling is applied — transformer embeddings are already normalised to unit vectors before cosine similarity is computed.

---

## Baseline and Proposed Approach

### Baseline Model

The **baseline model is BM25** — a classical lexical keyword-matching retrieval method. For each query, BM25 scores all restaurant reviews using term frequency–inverse document frequency weighting, and ranks restaurants by their aggregate BM25 score across all their reviews.

BM25 is chosen as the baseline because:
- It is the standard retrieval baseline used in the original RIRD/RIR paper (ECIR 2023), making results directly comparable.
- It requires no GPU, no training, and can be run in a single afternoon.
- It performs surprisingly well on short focused queries but fails on indirect or negation-style queries (e.g. "I want to lose weight" — BM25 cannot infer this means "low-calorie food").
- Once a BM25 MAP/nDCG number exists, §6 and §7 of the proposal can be written regardless of model progress.

A **second baseline** is also included: **Dense retrieval without fusion** — mean-pooling all sentence embeddings for each restaurant into a single restaurant embedding, then ranking by cosine similarity to the query embedding. This isolates the contribution of the fusion step.

---

### Proposed Model / Approach

The proposed approach is **sentence-level dense retrieval with a learned fusion step**:

1. **Split** each restaurant's reviews into individual sentences.
2. **Encode** all sentences and the query using a multilingual sentence transformer.
3. **Retrieve** the top-k most relevant sentences per query using cosine similarity.
4. **Fuse** the per-sentence relevance scores into a single restaurant-level score (this is the research contribution).
5. **Rank** restaurants by their fused score and attach the top supporting sentences as justification (the explainability component).

**The fusion step** is where the system differs from existing solutions. Several fusion variants will be compared:

| Fusion Variant | Description |
|---|---|
| Max pooling | Restaurant score = highest sentence score |
| Mean pooling | Restaurant score = average of all sentence scores |
| Top-k mean | Average of the top-k sentence scores (k = 3, 5, 10) |
| Weighted sum | Sentence scores weighted by sentence position or length |

The variant that best improves MAP/nDCG on the RIRD evaluation set becomes the proposed method.

---

### How the Proposed System Improves on Existing Solutions

| Dimension | Existing Solutions (Kaggle / HuggingFace) | Proposed Approach |
|---|---|---|
| **Query type** | Most recommenders use collaborative filtering or tag-based search; none handle free-text preference queries | Handles indirect, subjective, and negation-style queries (e.g. "I'm on a diet" → low-calorie restaurants) |
| **Explainability** | Black-box rankings with no justification | Each recommendation includes the specific review sentences that justify its ranking |
| **Language** | Existing Thai restaurant apps are keyword-search only; no dense retrieval | Thai-language query → Thai review sentence retrieval via multilingual model |
| **Evaluation** | Most Kaggle food notebooks report accuracy on star rating prediction — not retrieval quality | Evaluated with MAP, nDCG@5, nDCG@10, Recall@k against human-annotated relevance labels from RIRD |
| **Fusion** | Dense retrieval baselines use mean-pooled document embeddings (single vector per restaurant) | Sentence-level retrieval with fusion: captures which specific parts of a review are relevant, not just the overall restaurant embedding |

**Unique features and user benefits:**
- **Explainability by design:** Users do not just get a ranked list — they see exactly which review sentences triggered the recommendation (e.g. "This place is perfect for spicy food lovers on a budget").
- **Indirect query handling:** The system understands intent, not just keywords. A query like "อยากกินอะไรเปรี้ยวๆ เผ็ดๆ ไม่แพง" (want something sour, spicy, cheap) is matched to review sentences that describe those sensory qualities without requiring the exact words.
- **Multilingual by design:** The same model handles both the English evaluation (RIRD) and the Thai demo (Wongnai) without separate pipelines or translation steps.
- **Ablation transparency:** The contribution table comparing BM25 vs. dense-no-fusion vs. each fusion variant makes it clear exactly what drives the improvement — a level of rigour not present in most Kaggle restaurant recommendation notebooks.
