# Knowledge Base — Group Flower

Background notes for team members new to retrieval and recommendation. Read these before working on
the tasks in [action_items.md](../action_items.md).

## The project in one picture

```
 query ("อาหารรสเผ็ดจัด")
      │
      ▼
 ┌──────────────┐   score every review / sentence     ┌───────────────┐
 │ BM25 (words) │ ──────────────────────────────────► │ sentence/     │
 │ Dense (meaning)                                    │ review scores │
 └──────────────┘                                     └──────┬────────┘
                                                             │ aggregate (max / mean / top-k mean)
                                                             ▼
                                                    restaurant ranking
                                                             │
                                                             ▼
                                         top sentences shown as the explanation
```

- **Evaluate the ranking** against human labels: RIRD (English, restaurant level) and our own Thai labels (sentence level).
- **Evaluate the explanation:** is it faithful, and does it support the ranking?

## Reading order

| # | Note | What you learn | Time | Needed for |
|---|---|---|---|---|
| 1 | [01_retrieval_basics.md](01_retrieval_basics.md) | Query, document, relevance, TF-IDF, BM25 | 30 min | Everything |
| 2 | [02_dense_retrieval.md](02_dense_retrieval.md) | Embeddings, Sentence-BERT, cosine, bi- vs. cross-encoder | 30 min | C2, C7 |
| 3 | [06_thai_nlp.md](06_thai_nlp.md) | Thai segmentation, PyThaiNLP, negation | 25 min | C1, C2, C4 |
| 4 | [04_evaluation.md](04_evaluation.md) | P@k, MAP, nDCG, pooling, kappa, significance tests | 45 min | C3–C7, B2 |
| 5 | [03_aggregation_and_fusion.md](03_aggregation_and_fusion.md) | Retrieval unit, max/mean/top-k, volume bias | 25 min | A1–A3, D3 |
| 6 | [05_explainability.md](05_explainability.md) | Faithfulness vs. plausibility, deletion test | 25 min | B1, B2, E1 |
| — | [07_reading_list.md](07_reading_list.md) | Papers to read, in priority order, with links | — | D1 and all tasks |
| — | [glossary.md](glossary.md) | One-line definitions of every term | — | Lookup |

**Ton (Thai track):** before Monday, read 1, 2 and 3, plus the "Must read before Monday" papers in
the reading list. Read 4 before starting C3 and C4.

## Learn by running the code

These notes link to the real code. The fastest way to make the concepts stick:

1. Open [thai/thai_retrieval.ipynb](../thai/thai_retrieval.ipynb) and run one query through BM25 and dense. Compare the results with what [01](01_retrieval_basics.md) and [02](02_dense_retrieval.md) predict.
2. Open [baselines/eval_utils.py](../baselines/eval_utils.py) and find each metric from [04](04_evaluation.md). The bootstrap CIs and significance tests are in the v2 notebook.
3. In [baselines/baseline_models_v2.ipynb](../baselines/baseline_models_v2.ipynb), compare max with top-10 mean, using [03](03_aggregation_and_fusion.md) as a guide.
