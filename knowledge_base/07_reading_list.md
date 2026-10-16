# 07 — Reading List for Ton

Prepared 2026-10-16. Every entry below was checked against the publisher, the ACL Anthology, arXiv, or the Crossref DOI record: the title, first author, venue, year and link. Anything that could not be checked was left out.

Task IDs refer to [action_items.md](../action_items.md): **A3** volume control, **B1** faithfulness test, **B2** human support rating, **C1–C7** Thai taste test set, **D1** related work, **E1** annotation guidelines. **F1–F3** are the professor's concerns: F1 claims vs. evaluation, F2 unit/volume bias in fusion, F3 explanation quality.

Difficulty: **Easy** means readable with no IR background. **Medium** needs one or two definitions from the glossary. **Hard** has dense math or assumes the field.

---

## How to read a paper in 30 minutes

The method comes from S. Keshav, "How to Read a Paper," *ACM SIGCOMM Computer Communication Review* 37(3):83–84, 2007. [doi:10.1145/1273445.1273458](https://doi.org/10.1145/1273445.1273458). It is a two-page paper and takes about 10 minutes to read. Keshav reads a paper in up to three passes and stops as soon as he has what he needs:

1. **Pass 1, about 5–10 min: the bird's-eye view.** Read the title, abstract and introduction, then the section headings only, then the conclusion. Glance at the reference list. Then answer the "five Cs":
   - *Category*: what kind of paper is it?
   - *Context*: what work does it build on?
   - *Correctness*: do the assumptions look valid?
   - *Contributions*: what is new?
   - *Clarity*: is it well written?
2. **Pass 2, up to about 1 hour: the content.** Read the whole paper but skip proofs. Look closely at figures, tables and axis labels. Mark references you might want to read.
3. **Pass 3, several hours: virtual re-implementation.** Rebuild the paper in your head and challenge every assumption. You will rarely need this. In this project, only the RIRD paper deserves a pass 3, because we reproduce its setup.

**A 30-minute version for this project:**
- Do pass 1 (10 min).
- Then read only the sections listed under "Read:" for that entry (15 min).
- Then write 3 lines in the team notes (5 min):
  1. What problem the paper solves.
  2. What it means for our task ID.
  3. One sentence we could cite in the report, with its page number.

When a paper reports numbers, always check two things: what unit was scored (review, sentence or item), and what metric was used.

---

## Tier 1 — Must read before Monday (2026-10-19)

You will need these four for the progress report, the feedback page (D3), related work (D1), and your Monday tasks C1 and C2. Total time is about 3.5 hours.

### 1.1 The RIRD paper (our benchmark and anchor)
M. M. Abdollah Pour, P. Farinneya, A. Toroghi, A. Korikov, A. Pesaranghader, T. Sajed, M. Bharadwaj, B. Mavrin, S. Sanner. "Self-Supervised Contrastive BERT Fine-tuning for Fusion-Based Reviewed-Item Retrieval." *Advances in Information Retrieval (ECIR 2023)*, LNCS 13980, pp. 3–17, Springer, 2023. [doi:10.1007/978-3-031-28244-7_1](https://doi.org/10.1007/978-3-031-28244-7_1) · free preprint [arXiv:2308.00762](https://arxiv.org/abs/2308.00762)

- **Why for us:**
  - It defines our task (score reviews, then fuse them into an item score), our English dataset, and the query categories you will copy for the Thai queries (**C1**).
  - Its comparison of early and late fusion is the background to the professor's fusion-bias concern (**F2**, **A3**).
  - Its MAP 0.626 (fine-tuned) is the reference point for our off-the-shelf dense top-10 model (MAP 0.608). We don't fine-tune (out of scope), so this is not a like-for-like comparison.
- **Read:**
  - Abstract and introduction.
  - The task definition and the early vs. late fusion section.
  - The dataset section: how queries were written, the five categories, and annotator agreement.
  - The main results table.
  - Skip the contrastive-learning details on the first read.
  - About 75 min; this is the one paper that deserves a pass 3 later.
- **Difficulty:** Medium

### 1.2 Multi-aspect reviewed-item retrieval (professor's paper #1)
A. Korikov, G. Saad, E. Baron, M. Khan, M. Shah, S. Sanner. "Multi-Aspect Reviewed-Item Retrieval via LLM Query Decomposition and Aspect Fusion." SIGIR 2024 Workshop on Information Retrieval's Role in RAG Systems (IR-RAG), 2024. [arXiv:2408.00878](https://arxiv.org/abs/2408.00878)

- **Why for us:**
  - This is the professor's direct pointer for **F2**: top-K late fusion breaks down when a query's aspects are unevenly spread across reviews.
  - It must appear in related work (**D1**).
  - It also explains why your multi-aspect Thai queries, such as "sour *and* spicy *and* cheap" (**C1**), are hard for single-vector retrieval.
- **Read:**
  - Abstract and introduction.
  - The problem setup, especially the "aspect imbalance" argument and the figure illustrating it.
  - The results on balanced vs. imbalanced data.
  - Skim the LLM prompting parts. LLM query decomposition is out of scope for us (cite it, don't reimplement it).
  - About 45 min.
- **Difficulty:** Medium

### 1.3 PyThaiNLP (your toolkit for C2)
W. Phatthiyaphaibun, K. Chaovavanich, C. Polpanumas, A. Suriyawongkul, L. Lowphansirikul, P. Chormai, P. Limkonchotiwat, T. Suntorntip, C. Udomcharoenchaikit. "PyThaiNLP: Thai Natural Language Processing in Python." *Proc. 3rd Workshop for NLP Open Source Software (NLP-OSS 2023)*, pp. 25–36, ACL, 2023. [aclanthology.org/2023.nlposs-1.4](https://aclanthology.org/2023.nlposs-1.4/) · [doi:10.18653/v1/2023.nlposs-1.4](https://doi.org/10.18653/v1/2023.nlposs-1.4)

- **Why for us:**
  - **C2** splits Wongnai reviews into sentences and words with PyThaiNLP. You need to know which tokenizer engines exist, and why Thai, which has no spaces between words, needs them at all.
  - This is also the citation for the Thai preprocessing in the methods section.
- **Read:**
  - Introduction.
  - The sections on word tokenization and sentence tokenization (the engines and their trade-offs).
  - Skim the rest.
  - About 30 min.
- **Difficulty:** Easy

### 1.4 IR evaluation basics (IIR Chapter 8)
C. D. Manning, P. Raghavan, H. Schütze. *Introduction to Information Retrieval*, Chapter 8 "Evaluation in information retrieval." Cambridge University Press, 2008. Free online: [PDF](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf) · [HTML](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)

- **Why for us:**
  - On Monday you will present and defend nDCG@10, MAP, P@10 and confidence intervals.
  - **C7** computes Thai nDCG@10 and P@10.
  - This chapter is the clearest free explanation of these metrics and of relevance judgments and pooling.
- **Read:**
  - 8.1–8.5: test collections, unranked and ranked evaluation, MAP, R-precision, NDCG, and assessing relevance (which covers kappa and pooling).
  - About 60 min.
- **Difficulty:** Easy

---

## Tier 2 — Must read for Phase 2 (Ton's tasks)

Read these after the presentation, before starting E1, C3–C7 and the B2 labelling.

### Professor's papers #2 and #3

#### 2.1 User-centric refinement of recommendation explanations
J. Zhang, Z. Tian, X. Feng, X. Chen, C. Chen. "Enhancing Recommendation Explanations through User-Centric Refinement." *Findings of the Association for Computational Linguistics: EMNLP 2025*, pp. 8177–8191, Suzhou, China, 2025. [aclanthology.org/2025.findings-emnlp.434](https://aclanthology.org/2025.findings-emnlp.434/) · [doi:10.18653/v1/2025.findings-emnlp.434](https://doi.org/10.18653/v1/2025.findings-emnlp.434)

- **Why for us:**
  - The professor raised **F3** (explanation quality is not evaluated). This paper gives explicit criteria for judging an explanation: factuality, personalization and sentiment coherence.
  - Borrow these criteria when writing the 0/1/2 support scale in **E1**, and when you label explanations in **B2**.
  - We do not generate or refine explanations; we only take the criteria.
- **Read:**
  - Introduction, especially the definition of the three user-centric criteria.
  - The evaluation section: how they measure each criterion.
  - Skip the multi-agent LLM framework.
  - About 40 min.
- **Difficulty:** Medium

#### 2.2 Review-based recommender systems survey
E. Hasan, M. Rahman, C. Ding, J. X. Huang, S. Raza. "Review-based Recommender Systems: A Survey of Approaches, Challenges and Future Perspectives." *ACM Computing Surveys* 58(1):1–41, 2025. [doi:10.1145/3742421](https://doi.org/10.1145/3742421) · free preprint [arXiv:2405.05562](https://arxiv.org/abs/2405.05562)

- **Why for us:**
  - Use it to position the project in related work (**D1**, D2): where "retrieving items by their reviews from a text query" sits among review-based recommenders.
  - Use its datasets and evaluation discussion for the limitations section (**F1**).
- **Read:**
  - Introduction and taxonomy figure.
  - The sections on explainability and on datasets and evaluation metrics.
  - The challenges and future directions section.
  - Skim the model catalogue.
  - About 60 min.
- **Difficulty:** Medium

### Annotation and agreement (E1, C4, C5, C7, B2)

#### 2.3 Inter-coder agreement (the main reference)
R. Artstein, M. Poesio. "Inter-Coder Agreement for Computational Linguistics." *Computational Linguistics* 34(4):555–596, 2008. [aclanthology.org/J08-4004](https://aclanthology.org/J08-4004/) · [doi:10.1162/coli.07-034-R2](https://doi.org/10.1162/coli.07-034-R2)

- **Why for us:**
  - **C5** and **C7** ask you to report agreement between two Thai labellers (or between you and yourself a week later); B2 does the same for explanations.
  - This survey explains:
    - why raw percent agreement is not enough;
    - Cohen's κ vs. Krippendorff's α;
    - weighted κ for ordered 0/1/2 labels;
    - how to interpret the values.
- **Read:**
  - Section 1 (intro).
  - Section 2: the coefficients, focusing on percent agreement, κ, and weighted coefficients.
  - Section 4, on using agreement measures in practice, including interpretation.
  - Skip the derivations.
  - About 90 min.
- **Difficulty:** Medium–Hard

#### 2.4 Cohen's kappa (the original)
J. Cohen. "A Coefficient of Agreement for Nominal Scales." *Educational and Psychological Measurement* 20(1):37–46, 1960. [doi:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104)

- **Why for us:** this is the standard citation whenever you write "κ = …" in **C7** or **B2**. Cite it; you don't need to study it.
- **Read:**
  - The definition of κ and the worked example.
  - About 20 min.
  - Optional: Landis & Koch 1977 (Tier 4) for the commonly quoted interpretation bands.
- **Difficulty:** Easy–Medium

### Graded evaluation, pooling and judging (C3, C7)

#### 2.5 nDCG (the original)
K. Järvelin, J. Kekäläinen. "Cumulated Gain-Based Evaluation of IR Techniques." *ACM Transactions on Information Systems* 20(4):422–446, 2002. [doi:10.1145/582415.582418](https://doi.org/10.1145/582415.582418)

- **Why for us:** your Thai labels are graded (0/1/2), and **C7** reports nDCG@10. This is the paper to cite for nDCG, and it explains why graded relevance and position discounting matter.
- **Read:**
  - Introduction.
  - The section defining CG, DCG and nDCG.
  - One results figure.
  - About 40 min.
- **Difficulty:** Medium

#### 2.6 Relevance judgments vary between assessors
E. M. Voorhees. "Variations in Relevance Judgments and the Measurement of Retrieval Effectiveness." *Information Processing & Management* 36(5):697–716, 2000. [doi:10.1016/S0306-4573(00)00010-8](https://doi.org/10.1016/S0306-4573(00)00010-8)

- **Why for us:**
  - **C5** and **C7**: two assessors will disagree. This paper shows that system *rankings* (is BM25 better than dense?) are usually stable even when the absolute scores change.
  - That is exactly the argument you need if your κ is only moderate.
- **Read:**
  - Introduction.
  - The overlap and agreement analysis.
  - The conclusions.
  - About 45 min.
- **Difficulty:** Medium

#### 2.7 Pooling bias
C. Buckley, D. Dimmick, I. Soboroff, E. Voorhees. "Bias and the Limits of Pooling for Large Collections." *Information Retrieval* 10(6):491–508, 2007. [doi:10.1007/s10791-007-9032-x](https://doi.org/10.1007/s10791-007-9032-x)

- **Why for us:**
  - **C3** pools the top sentences from BM25 and dense, and only pooled sentences get judged.
  - This paper explains how pooling can favour the systems that built the pool, or one style of system, e.g. lexical systems that match the query words. Use it to write the C3/C7 limitations honestly: unjudged sentences, and only two contributing systems.
- **Read:**
  - Introduction.
  - The description of the bias (title-word matching).
  - The conclusions.
  - About 40 min.
- **Difficulty:** Medium

### Statistical testing (C7, and reading Sam's A-results)

#### 2.8 Which significance test to use in IR
M. D. Smucker, J. Allan, B. Carterette. "A Comparison of Statistical Significance Tests for Information Retrieval Evaluation." *Proc. 16th ACM CIKM*, pp. 623–632, 2007. [doi:10.1145/1321440.1321528](https://doi.org/10.1145/1321440.1321528)

- **Why for us:**
  - The v2 baselines already use paired randomization tests. This paper is the standard justification for that choice over the sign test and the Wilcoxon test.
  - Use the same test for Thai BM25 vs. dense in **C7**. Note that with only 24 frozen Thai queries, statistical power will be low.
- **Read:**
  - Introduction.
  - The section describing the randomization (permutation) test.
  - The conclusions/recommendations.
  - About 45 min.
- **Difficulty:** Medium

### Faithfulness of explanations (B1, B2, F3)

#### 2.9 What "faithful" means
A. Jacovi, Y. Goldberg. "Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?" *Proc. 58th ACL*, pp. 4198–4205, 2020. [aclanthology.org/2020.acl-main.386](https://aclanthology.org/2020.acl-main.386/) · [doi:10.18653/v1/2020.acl-main.386](https://doi.org/10.18653/v1/2020.acl-main.386)

- **Why for us:**
  - The professor wrote that a top sentence "can be relevant without being a complete or faithful explanation" (**F3**).
  - This short paper separates *plausibility* (a human finds it convincing, which is what B2 measures) from *faithfulness* (it reflects why the model ranked the item, which is what B1 measures). The distinction should appear word-for-word in the report.
- **Read:**
  - The whole paper. It is short and non-mathematical.
  - About 40 min.
- **Difficulty:** Easy–Medium

#### 2.10 ERASER: comprehensiveness and sufficiency
J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher, B. C. Wallace. "ERASER: A Benchmark to Evaluate Rationalized NLP Models." *Proc. 58th ACL*, pp. 4443–4458, 2020. [aclanthology.org/2020.acl-main.408](https://aclanthology.org/2020.acl-main.408/) · [doi:10.18653/v1/2020.acl-main.408](https://doi.org/10.18653/v1/2020.acl-main.408)

- **Why for us:**
  - **B1** (delete the shown sentences, re-score, compare with deleting random sentences) is ERASER's *comprehensiveness* metric applied to restaurants.
  - Its *sufficiency* metric (score with *only* the shown sentences) is a cheap second test.
  - Cite it as the source of the B1 design.
- **Read:**
  - Introduction.
  - The section defining comprehensiveness and sufficiency (the faithfulness metrics).
  - Skip the dataset descriptions.
  - About 40 min.
- **Difficulty:** Medium

---

## Tier 3 — Foundations (read when you need them)

#### 3.1 BM25
S. Robertson, H. Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval* 3(4):333–389, 2009. [doi:10.1561/1500000019](https://doi.org/10.1561/1500000019)

- **Why for us:**
  - This is the lexical baseline on both tracks.
  - For **C2/C7**, know what k1 and b do. The b parameter normalizes for document length, which matters because Thai sentences are short and the token counts depend on the segmenter.
- **Read:**
  - The sections deriving BM25 and explaining term-frequency saturation and length normalization (k1, b).
  - Skip BM25F and the parameter-optimization math.
  - About 60 min.
- **Difficulty:** Hard

#### 3.2 Sentence-BERT
N. Reimers, I. Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proc. EMNLP-IJCNLP 2019*, pp. 3980–3990. [doi:10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410) · [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)

- **Why for us:**
  - This is the bi-encoder architecture behind our dense scorer.
  - It explains why we can embed every sentence once, cache it (C2), and compare by cosine similarity.
- **Read:**
  - Introduction.
  - The model section (siamese network, pooling).
  - About 40 min.
- **Difficulty:** Medium

#### 3.3 Multilingual sentence embeddings by knowledge distillation
N. Reimers, I. Gurevych. "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation." *Proc. EMNLP 2020*, pp. 4512–4525. [aclanthology.org/2020.emnlp-main.365](https://aclanthology.org/2020.emnlp-main.365/) · [doi:10.18653/v1/2020.emnlp-main.365](https://doi.org/10.18653/v1/2020.emnlp-main.365)

- **Why for us:** this is how the `paraphrase-multilingual-*` models we use for Thai were trained. A student model learns to map a translation to the same vector as the English sentence.
- **Read:**
  - Introduction.
  - The method section and its figure.
  - About 30 min.
- **Difficulty:** Medium

#### 3.4 Dense Passage Retrieval (DPR)
V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, W. Yih. "Dense Passage Retrieval for Open-Domain Question Answering." *Proc. EMNLP 2020*, pp. 6769–6781. [doi:10.18653/v1/2020.emnlp-main.550](https://doi.org/10.18653/v1/2020.emnlp-main.550) · [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)

- **Why for us:**
  - This is the classic paper showing that dense bi-encoders beat BM25 for passage retrieval.
  - It is useful context for why dense wins on RIRD, and for the "BM25 vs. dense" framing in **C7**.
- **Read:**
  - Introduction.
  - The model section.
  - The main results table.
  - About 40 min.
- **Difficulty:** Medium

#### 3.5 Explainable recommendation survey
Y. Zhang, X. Chen. "Explainable Recommendation: A Survey and New Perspectives." *Foundations and Trends in Information Retrieval* 14(1):1–101, 2020. [doi:10.1561/1500000066](https://doi.org/10.1561/1500000066) · [arXiv:1804.11192](https://arxiv.org/abs/1804.11192)

- **Why for us:**
  - It catalogues "review sentence as explanation", which is exactly our explanation style.
  - Its evaluation discussion supports **B2** and **F3**.
- **Read:**
  - Chapter 1 (introduction and taxonomy).
  - The sections on textual/sentence explanations and on evaluating explanations.
  - About 60 min.
  - Skip the model catalogue.
- **Difficulty:** Medium

#### 3.6 Introduction to Information Retrieval (free online book)
C. D. Manning, P. Raghavan, H. Schütze. *Introduction to Information Retrieval*. Cambridge University Press, 2008. [nlp.stanford.edu/IR-book](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)

- **Why for us:** the textbook background for everything above. Chapter 8 is already in Tier 1.
- **Read:**
  - **Ch. 1 "Boolean retrieval"** ([PDF](https://nlp.stanford.edu/IR-book/pdf/01bool.pdf)): basic vocabulary such as index, postings and tokenization. About 45 min.
  - **Ch. 6 "Scoring, term weighting & the vector space model"** ([PDF](https://nlp.stanford.edu/IR-book/pdf/06vect.pdf)): tf-idf and cosine similarity, which you need before BM25. About 75 min.
  - **Ch. 11 "Probabilistic information retrieval"** ([PDF](https://nlp.stanford.edu/IR-book/pdf/11prob.pdf)): where BM25 comes from; read 11.4.3 on Okapi BM25. About 60 min.
- **Difficulty:** Easy (Ch. 1), Medium (Ch. 6, Ch. 11)

#### 3.7 WangchanBERTa
L. Lowphansirikul, C. Polpanumas, N. Jantrakulchai, S. Nutanong. "WangchanBERTa: Pretraining Transformer-based Thai Language Models." arXiv preprint, 2021. [arXiv:2101.09635](https://arxiv.org/abs/2101.09635)

- **Why for us:**
  - This is the main Thai-only encoder. It needs fine-tuning, which is out of scope, so it is background only: the citation for the claim that Thai NLP has matured.
  - It also discusses Thai tokenization choices.
- **Read:**
  - Introduction.
  - The pretraining data and tokenization sections.
  - The downstream results tables.
  - About 45 min.
- **Difficulty:** Medium

#### 3.8 Thai sentence segmentation
C. Saetia, E. Chuangsuwanich, T. Chalothorn, P. Vateekul. "Semi-supervised Thai Sentence Segmentation Using Local and Distant Word Representations." *Engineering Journal* 25(6), 2021. [doi:10.4186/ej.2021.25.6.15](https://doi.org/10.4186/ej.2021.25.6.15) · preprint [arXiv:1908.01294](https://arxiv.org/abs/1908.01294)

- **Why for us:**
  - **C2**: Thai has no full stop, so "sentence" boundaries are a modelling decision. Cite this when you report the segmentation spot-check.
  - Note that the preprint and the journal version report different error-reduction numbers. Quote the journal version.
- **Read:**
  - Introduction, which explains why Thai sentence boundaries are ambiguous.
  - The datasets and results sections (ORCHID, UGWC).
  - About 30 min.
- **Difficulty:** Medium

---

## Tier 4 — Optional / if curious

- **MTEB.** N. Muennighoff, N. Tazi, L. Magne, N. Reimers. "MTEB: Massive Text Embedding Benchmark." *Proc. EACL 2023*, pp. 2014–2037. [doi:10.18653/v1/2023.eacl-main.148](https://doi.org/10.18653/v1/2023.eacl-main.148)
  - Why for us: shows how to choose an embedding model when someone asks "why this encoder?" (C2).
  - Read: intro and the task overview. About 30 min.
  - Difficulty: Easy.
- **Multilingual E5.** L. Wang, N. Yang, X. Huang, et al. "Multilingual E5 Text Embeddings: A Technical Report." arXiv, 2024. [arXiv:2402.05672](https://arxiv.org/abs/2402.05672)
  - Why for us: a stronger multilingual encoder we could try on Thai as an extra dense run in C7.
  - Read: the whole report (6 pages). About 20 min.
  - Difficulty: Easy.
- **BGE-M3.** J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, Z. Liu. "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." *Findings of ACL 2024*, pp. 2318–2335. [doi:10.18653/v1/2024.findings-acl.137](https://doi.org/10.18653/v1/2024.findings-acl.137)
  - Why for us: one model that gives both dense and sparse (lexical-like) scores, and it supports Thai. Relevant to the BM25-vs-dense question in C7.
  - Read: intro and the method overview. About 30 min.
  - Difficulty: Medium.
- **Cross-encoder reranking.** R. Nogueira, K. Cho. "Passage Re-ranking with BERT." arXiv, 2019. [arXiv:1901.04085](https://arxiv.org/abs/1901.04085)
  - Why for us: shows the step beyond bi-encoders, reranking the top results with a cross-encoder. Useful for "future work" in D2.
  - Read: the whole paper (short). About 20 min.
  - Difficulty: Easy.
- **Thai sentence-embedding benchmark.** P. Tasawong, P. Limkonchotiwat, W. Ponwitayarat, et al., S. Nutanong. "Advancing Thai Sentence Embedding: Benchmark Development." *Forum for Linguistic Studies* 7(12), 2025. [doi:10.30564/fls.v7i12.12023](https://doi.org/10.30564/fls.v7i12.12023)
  - Why for us: evidence on which multilingual or Thai encoders work for Thai retrieval and similarity tasks. Use it to justify the encoder choice in C2/C7.
  - Read: intro and the results table. About 30 min.
  - Difficulty: Medium.
- **Interpreting kappa.** J. R. Landis, G. G. Koch. "The Measurement of Observer Agreement for Categorical Data." *Biometrics* 33(1):159–174, 1977. [doi:10.2307/2529310](https://doi.org/10.2307/2529310)
  - Why for us: the source of the widely quoted bands ("0.41–0.60 moderate", etc.) you may use in C7/B2. Artstein & Poesio (2.3) warn against over-trusting these bands.
  - Read: the interpretation table only. About 10 min.
  - Difficulty: Easy.
- **Scoring with unjudged items.** T. Sakai. "Alternatives to Bpref." *Proc. SIGIR 2007*, pp. 71–78. [doi:10.1145/1277741.1277756](https://doi.org/10.1145/1277741.1277756)
  - Why for us: if many retrieved Thai sentences are unjudged in C7, this shows the "condensed list" trick: drop unjudged items, then compute nDCG.
  - Read: intro and the condensed-list section. About 40 min.
  - Difficulty: Hard.
- **Incomplete judgments.** C. Buckley, E. M. Voorhees. "Retrieval Evaluation with Incomplete Information." *Proc. SIGIR 2004*, pp. 25–32. [doi:10.1145/1008992.1009000](https://doi.org/10.1145/1008992.1009000)
  - Why for us: background to Sakai, on what happens to metrics when judgments are incomplete (C3/C7).
  - Read: intro and conclusions. About 30 min.
  - Difficulty: Medium.

---

## Status in `docs/references.bib`

| Entry | In `references.bib`? | Action |
|---|---|---|
| RIRD, Abdollah Pour et al. 2023 | Yes (`rird2023`) | — |
| Sentence-BERT 2019 | Yes (`reimers2019sbert`) | — |
| DPR 2020 | Yes (`karpukhin2020dpr`) | — |
| BM25, Robertson & Zaragoza 2009 | Yes (`robertson2009bm25`) | — |
| Zhang & Chen 2020 | Yes (`zhang2020explainable`) | — |
| WangchanBERTa 2021 | Yes (`lowphansirikul2021wangchanberta`) | — |
| Korikov et al. 2024 | Yes (`korikov2024multiaspect`) | — |
| Zhang et al. 2025 | Yes (`zhang2025enhancing`) | — |
| Hasan et al. 2025 | Yes (`hasan2025reviewsurvey`) | — |
| PyThaiNLP 2023 | **No** | Add (C2 methods) |
| IIR book 2008 | **No** | Add if we cite metric definitions |
| Multilingual distillation 2020 | **No** | Add: this is our Thai encoder |
| Cohen 1960, Artstein & Poesio 2008 | **No** | Add in Phase 2 (C7, B2) |
| Järvelin & Kekäläinen 2002 | **No** | Add (nDCG) |
| Voorhees 2000, Buckley et al. 2007 | **No** | Add in Phase 2 (C3/C7 limitations) |
| Smucker et al. 2007 | **No** | Add (significance tests) |
| Jacovi & Goldberg 2020, DeYoung et al. 2020 | **No** | Add in Phase 2 (B1, F3) |
| Saetia et al. 2021 | **No** | Add if we discuss segmentation |
| Tier 4 entries | No | Only if cited |

The existing entries in `references.bib` were re-checked against Crossref/arXiv, and their DOIs resolve to the right papers.

---

## Ready-to-paste BibTeX (verified)

```bibtex
% ---------- Professor's three papers ----------
@inproceedings{korikov2024multiaspect,
  author    = {Korikov, Anton and Saad, George and Baron, Ethan and Khan, Mustafa and Shah, Manav and Sanner, Scott},
  title     = {Multi-Aspect Reviewed-Item Retrieval via {LLM} Query Decomposition and Aspect Fusion},
  booktitle = {SIGIR 2024 Workshop on Information Retrieval's Role in RAG Systems (IR-RAG)},
  address   = {Washington, D.C., USA},
  year      = {2024},
  eprint    = {2408.00878},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2408.00878}
}

@inproceedings{zhang2025enhancing,
  author    = {Zhang, Jingsen and Tian, Zihang and Feng, Xueyang and Chen, Xu and Chen, Chong},
  title     = {Enhancing Recommendation Explanations through User-Centric Refinement},
  booktitle = {Findings of the Association for Computational Linguistics: {EMNLP} 2025},
  pages     = {8177--8191},
  address   = {Suzhou, China},
  publisher = {Association for Computational Linguistics},
  year      = {2025},
  doi       = {10.18653/v1/2025.findings-emnlp.434},
  url       = {https://aclanthology.org/2025.findings-emnlp.434/}
}

@article{hasan2025reviewsurvey,
  author  = {Hasan, Emrul and Rahman, Mizanur and Ding, Chen and Huang, Jimmy Xiangji and Raza, Shaina},
  title   = {Review-based Recommender Systems: A Survey of Approaches, Challenges and Future Perspectives},
  journal = {ACM Computing Surveys},
  volume  = {58},
  number  = {1},
  pages   = {1--41},
  year    = {2025},
  doi     = {10.1145/3742421},
  note    = {arXiv:2405.05562}
}

% ---------- New must-reads (Tier 1) ----------
@inproceedings{phatthiyaphaibun2023pythainlp,
  author    = {Phatthiyaphaibun, Wannaphong and Chaovavanich, Korakot and Polpanumas, Charin and Suriyawongkul, Arthit and Lowphansirikul, Lalita and Chormai, Pattarawat and Limkonchotiwat, Peerat and Suntorntip, Thanathip and Udomcharoenchaikit, Can},
  title     = {{PyThaiNLP}: {T}hai Natural Language Processing in Python},
  booktitle = {Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023)},
  pages     = {25--36},
  address   = {Singapore},
  publisher = {Association for Computational Linguistics},
  year      = {2023},
  doi       = {10.18653/v1/2023.nlposs-1.4}
}

@book{manning2008iir,
  author    = {Manning, Christopher D. and Raghavan, Prabhakar and Sch{\"u}tze, Hinrich},
  title     = {Introduction to Information Retrieval},
  publisher = {Cambridge University Press},
  year      = {2008},
  url       = {https://nlp.stanford.edu/IR-book/}
}

% ---------- New must-reads (Tier 2, Phase 2) ----------
@article{cohen1960kappa,
  author  = {Cohen, Jacob},
  title   = {A Coefficient of Agreement for Nominal Scales},
  journal = {Educational and Psychological Measurement},
  volume  = {20},
  number  = {1},
  pages   = {37--46},
  year    = {1960},
  doi     = {10.1177/001316446002000104}
}

@article{artstein2008agreement,
  author  = {Artstein, Ron and Poesio, Massimo},
  title   = {Inter-Coder Agreement for Computational Linguistics},
  journal = {Computational Linguistics},
  volume  = {34},
  number  = {4},
  pages   = {555--596},
  year    = {2008},
  doi     = {10.1162/coli.07-034-R2}
}

@article{jarvelin2002ndcg,
  author  = {J{\"a}rvelin, Kalervo and Kek{\"a}l{\"a}inen, Jaana},
  title   = {Cumulated Gain-Based Evaluation of {IR} Techniques},
  journal = {ACM Transactions on Information Systems},
  volume  = {20},
  number  = {4},
  pages   = {422--446},
  year    = {2002},
  doi     = {10.1145/582415.582418}
}

@article{voorhees2000variations,
  author  = {Voorhees, Ellen M.},
  title   = {Variations in Relevance Judgments and the Measurement of Retrieval Effectiveness},
  journal = {Information Processing \& Management},
  volume  = {36},
  number  = {5},
  pages   = {697--716},
  year    = {2000},
  doi     = {10.1016/S0306-4573(00)00010-8}
}

@article{buckley2007pooling,
  author  = {Buckley, Chris and Dimmick, Darrin and Soboroff, Ian and Voorhees, Ellen},
  title   = {Bias and the Limits of Pooling for Large Collections},
  journal = {Information Retrieval},
  volume  = {10},
  number  = {6},
  pages   = {491--508},
  year    = {2007},
  doi     = {10.1007/s10791-007-9032-x}
}

@inproceedings{smucker2007significance,
  author    = {Smucker, Mark D. and Allan, James and Carterette, Ben},
  title     = {A Comparison of Statistical Significance Tests for Information Retrieval Evaluation},
  booktitle = {Proceedings of the 16th ACM Conference on Information and Knowledge Management (CIKM)},
  pages     = {623--632},
  year      = {2007},
  doi       = {10.1145/1321440.1321528}
}

@inproceedings{jacovi2020faithfulness,
  author    = {Jacovi, Alon and Goldberg, Yoav},
  title     = {Towards Faithfully Interpretable {NLP} Systems: How Should We Define and Evaluate Faithfulness?},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages     = {4198--4205},
  year      = {2020},
  doi       = {10.18653/v1/2020.acl-main.386}
}

@inproceedings{deyoung2020eraser,
  author    = {DeYoung, Jay and Jain, Sarthak and Rajani, Nazneen Fatema and Lehman, Eric and Xiong, Caiming and Socher, Richard and Wallace, Byron C.},
  title     = {{ERASER}: A Benchmark to Evaluate Rationalized {NLP} Models},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages     = {4443--4458},
  year      = {2020},
  doi       = {10.18653/v1/2020.acl-main.408}
}

% ---------- Foundations we already rely on but have not cited ----------
@inproceedings{reimers2020multilingual,
  author    = {Reimers, Nils and Gurevych, Iryna},
  title     = {Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {4512--4525},
  year      = {2020},
  doi       = {10.18653/v1/2020.emnlp-main.365}
}

@article{saetia2021thaisent,
  author  = {Saetia, Chanatip and Chuangsuwanich, Ekapol and Chalothorn, Tawunrat and Vateekul, Peerapon},
  title   = {Semi-supervised {T}hai Sentence Segmentation Using Local and Distant Word Representations},
  journal = {Engineering Journal},
  volume  = {25},
  number  = {6},
  year    = {2021},
  doi     = {10.4186/ej.2021.25.6.15}
}
```
