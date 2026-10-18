# Glossary

Short definitions of the terms used in this project. Each entry points to the note that explains it
properly.

| Term | Meaning | More |
|---|---|---|
| **Query** | What the user types, e.g. "อาหารรสเผ็ดจัด" or "cozy place with good coffee". | [01](01_retrieval_basics.md) |
| **Document** | Whatever the system scores against the query. In our project this is a review or a sentence. | [01](01_retrieval_basics.md) |
| **Corpus** | The whole collection of documents, e.g. all RIRD reviews or the Wongnai sentences. | [01](01_retrieval_basics.md) |
| **Item** | The thing being recommended. Here, a restaurant. | [01](01_retrieval_basics.md) |
| **Reviewed-item retrieval (RIR)** | Ranking items (restaurants) for a query when each item is described only by its reviews. | [01](01_retrieval_basics.md), [03](03_aggregation_and_fusion.md) |
| **Relevance** | Whether a document or item satisfies the query. Judged by people, either binary (0/1) or graded (0/1/2). | [04](04_evaluation.md) |
| **Qrels** | "Query relevance judgments": the table of human labels. For RIRD this is `PMD.csv`. | [04](04_evaluation.md) |
| **Token / tokenization** | Splitting text into units (words or sub-words). Thai needs a segmenter because it has no spaces between words. | [01](01_retrieval_basics.md), [06](06_thai_nlp.md) |
| **Inverted index** | A map from each word to the documents that contain it, which makes keyword search fast. | [01](01_retrieval_basics.md) |
| **TF / IDF** | Term frequency (how often a word appears in a document) and inverse document frequency (how rare the word is across the corpus). | [01](01_retrieval_basics.md) |
| **BM25** | A classic keyword-matching score built from TF and IDF, with saturation and document-length normalization. This is our *sparse* baseline. | [01](01_retrieval_basics.md) |
| **Sparse vs. dense** | Sparse: a vector with one dimension per word, mostly zeros (BM25). Dense: a short vector of meaning, e.g. 768 numbers (embeddings). | [02](02_dense_retrieval.md) |
| **Embedding** | A vector a neural model produces from a text, positioned so that texts with similar meaning end up close together. | [02](02_dense_retrieval.md) |
| **Bi-encoder** | Encodes the query and each document separately and compares the vectors. Fast, and the document vectors can be cached. Our dense model works this way. | [02](02_dense_retrieval.md) |
| **Cross-encoder** | Reads the query and a document together and outputs a score. More accurate but slow, so it is used for reranking. | [02](02_dense_retrieval.md) |
| **Cosine similarity** | The angle-based similarity of two vectors. On normalized vectors it equals the dot product. | [02](02_dense_retrieval.md) |
| **Unit** | The level that is scored: restaurant, review or sentence. BM25 and dense must use the same unit for a fair comparison (professor comment F2). | [03](03_aggregation_and_fusion.md) |
| **Aggregation / fusion** | Combining many review or sentence scores into one restaurant score (max, mean, top-k mean). | [03](03_aggregation_and_fusion.md) |
| **Volume bias** | Max and top-k aggregation favour restaurants with many reviews simply because they have more chances to score high. | [03](03_aggregation_and_fusion.md) |
| **P@k** | Precision at k: the fraction of the top k results that are relevant. | [04](04_evaluation.md) |
| **MAP** | Mean average precision. Rewards putting relevant items early, averaged over queries. | [04](04_evaluation.md) |
| **nDCG@k** | Normalized discounted cumulative gain. Uses graded labels and discounts results further down the list. Our main metric. | [04](04_evaluation.md) |
| **Pooling** | Labelling only the union of the top-k results from several systems instead of the whole corpus. This is what C3 does. | [04](04_evaluation.md) |
| **Cohen's kappa (κ)** | Agreement between two annotators, corrected for the agreement you would expect by chance. | [04](04_evaluation.md) |
| **Linear-weighted kappa** | Cohen's kappa for ordered labels (0/1/2) where a disagreement counts by its distance, so 1-vs-2 is half as bad as 0-vs-2. `score_labels.py` reports it next to plain kappa. | [04](04_evaluation.md) |
| **Volume control / capping** | A3: keep the same number of reviews (400) for every restaurant and re-run, to see how much of the ranking comes from review count. | [03](03_aggregation_and_fusion.md) |
| **Pooled vs. per-query correlation** | Pooled: one correlation over all (query, restaurant) pairs, which mixes queries with different score scales. Per-query: one correlation per query, then averaged. We report per-query. | [03](03_aggregation_and_fusion.md) |
| **Comprehensiveness** | ERASER's faithfulness measure: how much the model's output drops when the explanation is removed. B1 is a ranking version of it. | [05](05_explainability.md) |
| **Topic vs. polarity** | Topic is what a text is about (price); polarity is whether it is for or against (cheap vs. pricey). Dense retrieval matches topic, not polarity. | [05](05_explainability.md) |
| **Bootstrap CI** | A confidence interval made by resampling queries with replacement. | [04](04_evaluation.md) |
| **Randomization test** | A significance test that randomly swaps the two systems' per-query scores to see whether the observed gap could be chance. | [04](04_evaluation.md) |
| **Holm correction** | An adjustment to p-values when many comparisons are tested at once. | [04](04_evaluation.md) |
| **Floor / baseline** | A trivial system (random, review count, popularity) that a real model must beat. | [04](04_evaluation.md) |
| **Extractive explanation** | An explanation built by selecting existing text (review sentences) rather than generating new text. | [05](05_explainability.md) |
| **Faithfulness** | Whether the explanation is really *why* the model ranked the item that way. | [05](05_explainability.md) |
| **Plausibility** | Whether the explanation *looks* convincing to a human. It can be plausible without being faithful. | [05](05_explainability.md) |
| **Deletion test** | Remove the shown sentences, re-score, and check how far the item drops in the ranking compared with removing random sentences. This is B1. | [05](05_explainability.md) |
| **newmm** | PyThaiNLP's default dictionary-based Thai word segmenter. | [06](06_thai_nlp.md) |
| **crfcut** | PyThaiNLP's CRF-based Thai sentence segmenter. | [06](06_thai_nlp.md) |
| **RIRD** | The English reviewed-item retrieval benchmark: 100 queries, 50 restaurants and human labels. | [07](07_reading_list.md) |
| **Wongnai** | A Thai restaurant review dataset of 40k training reviews with stars from 0 to 4 and no restaurant ids. | [06](06_thai_nlp.md) |
