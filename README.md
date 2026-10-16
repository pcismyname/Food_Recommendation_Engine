# Thai Food Recommendation — CP Proposal (Group: Flower)

## Setup

Requires Python 3.10+.

```bash
git clone <repo-url>
cd Food_Recommendation_Engine

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Then open any notebook under `eda/` to explore the data.

## Dataset Setup

Datasets are **not committed to git** (see `.gitignore`) — they are reproduced from
their sources. After installing requirements, run:

```bash
python datasets/download_datasets.py
```

This rebuilds `datasets/` with everything that can be fetched without credentials:

| Dataset | Source | Path | Auto? |
|---|---|---|---|
| RIRD (English, labelled — evaluation protocol) | GitHub `D3Mlab/rir_data` | `datasets/rir_data-main/` | ✅ |
| M-ABSA (Thai aspect sentiment) | HF `Multilingual-NLP/M-ABSA` | `datasets/m-absa/` | ✅ |
| Wongnai (Thai reviews) | HF `Highgroundbkk/wongnai_reviews` | `datasets/wongnai/` | ✅ |
| Thailand Foods (dish gazetteer) | Kaggle `ponthakornsodchun/foods-in-thailand` | `datasets/thailand_foods.csv` | ⚠️ manual |

**Thailand Foods (manual):** Kaggle requires an account. Either download the CSV from
the [dataset page](https://www.kaggle.com/datasets/ponthakornsodchun/foods-in-thailand)
and save it as `datasets/thailand_foods.csv`, or with a Kaggle API token
(`~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d ponthakornsodchun/foods-in-thailand -p datasets --unzip
```

See `PROJECT.md` for what each dataset is for and how they fit the pipeline.

Members: Ton (st127004, Chidsanuphong Pengchai) · Sam (st127012, Samichi Rungta)

## Response to proposal feedback

Status as of 2026-10-17. The professor's feedback is in [professor_commmend.md](professor_commmend.md).
Tasks, owners and deadlines are in [action_items.md](action_items.md).

### Q1 (F1): Application claims don't match the evaluation

> RIRD tests broad English restaurant preferences, not Thai taste. Wongnai can't support
> restaurant-level evaluation.

**Answer:** Agreed. We are fixing it from both sides.

- **Narrowing the claims (D2).** We only claim measured results where we measured them. Working
  title: *"Explainable reviewed-item retrieval, with a Thai taste case study"*.
- **Measuring Thai (C).** A small Thai taste test set on real Wongnai review sentences:
  - 24 queries covering single tastes, combinations and negations
    ([thai/thai_queries.csv](thai/thai_queries.csv)). They were drafted by an LLM, revised against
    written rules ([thai/query_guidelines.md](thai/query_guidelines.md)), then reviewed and frozen by
    a native Thai speaker before any results were seen.
  - The sentence-level BM25 and dense pipeline is built and run on a 2,000-review subsample
    ([thai/thai_retrieval.ipynb](thai/thai_retrieval.ipynb)).
- **Limit.** Thai is evaluated at sentence level only. Wongnai has no restaurant ids, so restaurant
  ranking in Thai is future work.
- **Status:** queries frozen and pipeline built. Labelling, agreement and Thai nDCG@10 come in
  Phase 2.

### Q2 (F2): BM25 and dense use different units; fusion is biased by review volume

**Answer:** The unit half is fixed; the volume half is next.

- **Fixed:** in [baselines/baseline_models_v2.ipynb](baselines/baseline_models_v2.ipynb), BM25 and
  dense both score individual reviews and use the same aggregations (max, mean, top-3/5/10 mean).
  Results come with bootstrap 95% CIs and paired significance tests with Holm correction.
- **Results:**
  - Dense beats BM25 under every aggregation, all significant (+0.03 to +0.09 nDCG@10).
  - The best model is dense with top-10 mean: nDCG@10 0.668 [0.625, 0.711] and MAP 0.608. That is
    close to the RIRD paper's fine-tuned 0.626.
  - Top-k mean beats both max and mean.
- **Open:** A3, the volume control. Every restaurant is subsampled to 400 reviews, and we report how
  restaurant score correlates with review count for each aggregation. A sentence-level rerun
  (A1/A2) follows.

### Q3 (F3): No evidence for explainability

> A top-scoring sentence can be relevant without being a faithful or complete explanation.

**Answer:** We agree it hasn't been measured yet. Two tests are planned:

- **B1, faithfulness (automatic):** delete the top-k sentences shown as the explanation, re-score,
  and compare the rank drop with deleting k random sentences. If the shown sentences really drive
  the ranking, the drop should be clearly larger.
- **B2, support (human):** two people separately rate the shown sentences for 15–30 queries × top 3
  restaurants as 0/1/2 support. We report kappa and the share rated as supporting.
- **Scope:** we only select existing review sentences; we don't generate explanations.
- **Status:** Phase 2. It depends on the sentence-level rerun.

### Q4 (F4): Check more literature

**Answer:** All three papers are being added to related work (D1):

- **Korikov et al. 2024**, [arXiv:2408.00878](https://arxiv.org/abs/2408.00878): multi-aspect
  retrieval and late fusion. It supports our F2 analysis; we cite it but don't reimplement LLM query
  decomposition.
- **Zhang et al., Findings of EMNLP 2025**,
  [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.434/): criteria for judging
  explanations, which frame B1/B2.
- **Hasan et al., ACM Computing Surveys**, [doi:10.1145/3742421](https://doi.org/10.1145/3742421):
  positioning among review-based recommenders.

### Final question

> Do retrieved review sentences genuinely support the system's restaurant rankings, and can the
> same approach work on real Thai restaurant reviews with independently annotated taste preferences?

**Answer so far:**

- **First half (support):** not answered yet. B1 and B2 are designed to answer it; results come
  after the progress presentation.
- **Second half (Thai):** partly answered. The same method runs on real Thai reviews, and an
  independently labelled Thai test set is underway. The final answer will be at sentence level,
  not restaurant level, and we say so.

## What we are building

A search system where the **query is a taste preference in Thai** and the
**documents are restaurant reviews**.

- **Input:** one free-text Thai query, e.g. `อยากกินอะไรเปรี้ยวๆ เผ็ดๆ ไม่แพง`
- **Output:** ranked list of restaurants, each with the review sentences that
  justify its position (the "explainable" part)

Pipeline: split reviews into sentences → segment with PyThaiNLP → encode with a
Thai/multilingual model → retrieve top-k sentences for the query → **fuse**
sentence scores up to a restaurant score → rank.

The fusion step is the research contribution. Everything else is plumbing.

## Section ownership

| § | Section | Owner |
|---|---|---|
| 1 | Introduction | Ton |
| 2 | Problem Statement | Ton |
| 3 | Related Works | Ton |
| 4 | Datasets | Sam |
| 5 | Methodology | Sam |
| 6 | Baseline and Proposed Approach | Sam |
| 7 | Preliminary Results | **unassigned — decide** |

**Both:** find 5 papers each (10 total) for §3 Related Works. Ton writes the
section, but Sam's 5 should come from the modelling side (retrieval, fusion,
Thai NLP models) so §5 and §6 have something to cite too. Drop them in a shared
doc with a one-line note on why each is relevant — not just the link.

## Data ownership

| Dataset | Owner |
|---|---|
| RIRD (professor's recommendation) — D3Mlab/rir_data | Ton |
| Wongnai corpus | Sam |
| Third dataset (TBC — Kaggle Foods in Thailand / M-ABSA) | Sam |

## First unblocking task each

Do these before anything else. If either fails, the plan changes.

- **Ton:** download RIRD, open the CSVs, confirm the licence and column
  structure. Note that it is **Toronto / Yelp / English**, not Thai — decide and
  write down how we use it (protocol template vs. direct data).
- **Sam:** get a BM25 baseline producing a number. One afternoon, no fine-tuning,
  no GPU. Once a number exists, §6 and §7 can be written even if everything
  else slips.

## Deliverables

1. Proposal report — AIT thesis template, https://languages.ait.ac.th/thesis-format/
   (start in the template, do not reformat later)
2. Proposal slides
3. 15-minute presentation video on YouTube — submit the link

## Video split (~7 min each, one handoff)

- **Ton:** §1–2 framing (~2 min), §3 related works (~2 min), plus demo (~3 min)
- **Sam:** §4 datasets (~2 min), §5 models (~3 min), §6 baseline (~3 min)

## Open items

- §7 Preliminary Results has no owner
- Sam writes §4 Datasets but Ton owns the professor's dataset — Ton must supply
  the RIRD description and licence paragraph to Sam
- Google Places licensing / reproducibility still unconfirmed
- Confirm whether M-ABSA's Thai aspect list actually overlaps our flavour axes
  before citing it

## References

- RIRD dataset: https://github.com/D3Mlab/rir_data
- RIR paper (ECIR 2023): https://arxiv.org/pdf/2308.00762
- Wongnai challenge: https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction
- Foods in Thailand: https://www.kaggle.com/datasets/ponthakornsodchun/foods-in-thailand
- M-ABSA: https://huggingface.co/datasets/Multilingual-NLP/M-ABSA
