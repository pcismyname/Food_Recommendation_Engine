# Explainable Reviewed-Item Retrieval, with a Thai Taste Case Study (Group: Flower)

Members: Ton (st127004, Chidsanuphong Pengchai) · Sam (st127012, Samichi Rungta)

Working title, revised after the proposal feedback. The project was proposed as a *Thai Food
Recommendation Engine*; the problem and pipeline are the same, but the claims are now limited to
what we can measure (see [Response to proposal feedback](#response-to-proposal-feedback)).

## What we are building

Given a free-text taste preference query, rank restaurants and show the review sentences that
justify each restaurant's position (the "explainable" part).

```
query → ① retrieve review sentences → ② fuse into a restaurant score → ③ show top sentences as the explanation
```

We compare BM25 and a dense multilingual encoder (`paraphrase-multilingual-mpnet-base-v2`) at
stage ①, and several aggregations (max, mean, top-k mean) at stage ②.

| Track | Question | Data | Stages | Owner |
|---|---|---|---|---|
| **A. Fair fusion** | Do the rankings hold up when BM25 and dense score the same unit, and are they biased by review volume? | RIRD (English, labelled) | ① ② | Sam |
| **B. Explanation evaluation** | Do the shown sentences really support the ranking (deletion test, human rating)? | RIRD | ③ | Sam |
| **C. Thai taste test set** | Does sentence retrieval carry over to real Thai reviews with human taste labels? | Wongnai (Thai) | ① only | Ton |

- RIRD is the only dataset with restaurant ids **and** relevance labels, so restaurant ranking and
  explanations are measured in English.
- Wongnai has no restaurant ids, so the Thai track is a sentence-level transfer test. Thai
  restaurant ranking is future work.
- C reuses A's method (same encoder, same BM25 vs. dense comparison, same sentence unit), not A's
  results.

## Status and plan

Task list, owners, dependencies and the plan are in **[action_items.md](action_items.md)**.

- **Next deadline:** Monday 2026-10-19, 11:59 PM, updated report (PDF) and slides (PDF), including a
  feedback summary page. Progress presentation Tuesday 2026-10-20 (10 min + 5 min Q&A).
- **Done:** review-level baselines with CIs and significance tests
  ([baselines/baseline_models_v2.ipynb](baselines/baseline_models_v2.ipynb)); Thai queries frozen
  and Thai sentence pipeline built ([thai/](thai/)).
- **Done:** A3 review-volume control was executed with 400 reviews per restaurant and `SEED = 42`.
  Dense top-10 mean remained the best controlled run: nDCG@10 0.672 and MAP 0.611, versus 0.668
  and 0.608 before control. Correlations with original uncapped review counts were reduced to
  near zero after control (Pearson range -0.009 to 0.019; Spearman range -0.021 to 0.001).
- **Done:** A4 ranking/evidence export produced 50,000 review-level restaurant-ranking rows and
  500,000 evidence rows across 10 BM25/dense aggregation runs. The files are
  `rankings_review.csv` and `evidence_review.csv` in the notebook cache.
- **Next:** run the sentence-level A1/A2 experiment and export `rankings_sentence.csv` and
  `evidence_sentence.csv`. The feedback page and related work (D3, D1) can proceed in parallel.

### Repository map

| Path | What it is |
|---|---|
| [action_items.md](action_items.md) | Current tasks, owners, plan |
| [professor_commmend.md](professor_commmend.md) | Professor's feedback on the proposal |
| [PROJECT.md](PROJECT.md) | Technical framing: pipeline, data, evaluation |
| [baselines/](baselines/) | RIRD baselines. `baseline_models_v2.ipynb` is current; `baseline_models.ipynb` is the proposal-era v1 |
| [thai/](thai/) | Thai taste queries, query rules, Wongnai sentence retrieval notebook |
| [eda/](eda/) | EDA for RIRD and the Thai datasets |
| [knowledge_base/](knowledge_base/) | Background notes on retrieval, fusion, evaluation, explainability, Thai NLP |
| [docs/](docs/) | Paper notes, references, proposal LaTeX sections |
| [project_proposal.md](project_proposal.md), `CP_Proposal (1).pdf` | The submitted proposal (historical record) |

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

Then open any notebook under `eda/`, `baselines/` or `thai/`. The Thai notebook also needs
`pythainlp` and `python-crfsuite` (both in `requirements.txt`).

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

Only **RIRD** and **Wongnai** are used in the current scope. M-ABSA and Thailand Foods were part of
the proposal and are now out of scope (see [action_items.md](action_items.md) §7); the downloader
still fetches them so the proposal EDA can be reproduced.

**Thailand Foods (manual, optional):** Kaggle requires an account. Either download the CSV from
the [dataset page](https://www.kaggle.com/datasets/ponthakornsodchun/foods-in-thailand)
and save it as `datasets/thailand_foods.csv`, or with a Kaggle API token
(`~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d ponthakornsodchun/foods-in-thailand -p datasets --unzip
```

See `PROJECT.md` for what each dataset is for and how they fit the pipeline.

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

**Answer:** Both halves are now tested at review level; the volume effect turned out to be small.

- **Fixed:** in [baselines/baseline_models_v2.ipynb](baselines/baseline_models_v2.ipynb), BM25 and
  dense both score individual reviews and use the same aggregations (max, mean, top-3/5/10 mean).
  Results come with bootstrap 95% CIs and paired significance tests with Holm correction.
- **Results:**
  - Dense beats BM25 under every aggregation, all significant (+0.03 to +0.09 nDCG@10).
  - The best model is dense with top-10 mean: nDCG@10 0.668 [0.625, 0.711] and MAP 0.608. That is
    close to the RIRD paper's fine-tuned 0.626.
  - Top-k mean beats both max and mean.
- **Completed:** A3 sampled every restaurant to exactly 400 source reviews with a fixed seed,
  reused cached embeddings, evaluated all aggregations for BM25 and dense, and saved metrics plus
  score/count correlations. The controlled results show that the review-volume relationship is
  substantially reduced when the original uncapped count is used as the diagnostic count.
  A sentence-level rerun (A1/A2) remains.

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

## References

- RIRD dataset: https://github.com/D3Mlab/rir_data
- RIR paper (ECIR 2023): https://arxiv.org/pdf/2308.00762
- Korikov et al., Multi-Aspect Reviewed-Item Retrieval via LLM Query Decomposition and Aspect Fusion: https://arxiv.org/abs/2408.00878
- Zhang et al., Enhancing Recommendation Explanations through User-Centric Refinement (Findings of EMNLP 2025): https://aclanthology.org/2025.findings-emnlp.434/
- Hasan et al., Review-based Recommender Systems: A Survey (ACM Computing Surveys): https://doi.org/10.1145/3742421
- Wongnai challenge: https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction
- PyThaiNLP: https://github.com/PyThaiNLP/pythainlp
