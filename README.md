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
