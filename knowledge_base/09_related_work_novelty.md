# 09 · Related Work and Novelty Check

> **Why this matters for our project**
> Before we claim anything as new, we need to know who has done it already. This note summarises a
> literature search (2026-10-17, seven parallel searches) across every part of the project: what is
> already known (cite it), what looks new (claim it carefully), and what reviewers will expect us to
> position against.

Every paper below was found on a real page (arXiv, ACL Anthology, ACM, Springer, institutional
repository). Most were checked at abstract level only; items marked ⚠️ need checking before citing.
Papers already in [07_reading_list.md](07_reading_list.md) (RIRD, Korikov 2024, Zhang 2025, Hasan,
SBERT, DPR, BM25, ERASER, Jacovi & Goldberg, WangchanBERTa, PyThaiNLP…) are not repeated.

---

## 1. Verdict at a glance

| Our piece | Already done? | What we can claim | Must cite |
|---|---|---|---|
| **A.** Dense beats BM25 on RIRD under max / top-k / mean | ✅ **Yes**, by the RIRD paper itself (Table 4: off-the-shelf CoCondenser beats BM25 at K = 1, 10, all) | A **replication** with a multilingual encoder | Abdollah Pour et al. 2023 (Table 4) |
| **A.** Aggregation choice (max / sum / mean / top-k) as a variable | ✅ Yes, expert finding and blog distillation, 2006–2009 | Not new; describe ours as CombMAX / CombSUM-style voting | Macdonald & Ounis 2006/2008; Balog et al. 2006; Zhang & Balog 2017 |
| **A.** Bias toward items with many documents | ✅ Known in general (candidate length normalisation, prolific bloggers) | **New for RIR**: no paper tests a review-volume control on RIRD | Glasgow TREC 2006; Macdonald & Ounis CIKM 2008; Elsas et al. 2008 |
| **A.** Sentence unit vs review unit on RIR | ❌ Not found (every RIR paper scores whole reviews) | **New for RIR**. We now have the result (A1/A2): sentences help dense top-k (top-10 mean 0.682 vs 0.668 review level) but hurt BM25 and dense mean; paired review-vs-sentence test still to do | Dense X Retrieval 2024; Luan et al. 2021; Birch 2019 |
| **B.** Review sentences ranked as explanations | ✅ Yes (EXTRA, graph extractor, Rank-Don't-Generate) | Not new as a framing | Li et al. 2021 (EXTRA) |
| **B.** Deletion faithfulness test + human support rating with kappa, on the same sentences, for a reviewed-item ranker | ❌ Not found together | **New as a combined protocol**; recent preprints are close, so don't say "first" | ERASER; CountER 2021; AIS 2023; TRACE 2026 |
| **C.** Thai taste-preference retrieval test set | ❌ Not found | **New**: "to our knowledge, the first small human-labelled test set for this domain" | MIRACL; NitiBench; Wongnai aspect-sentiment paper |
| **C.** BM25 vs dense on Thai review sentences | ❌ Not found (Thai IR benchmarks are Wikipedia / QA / legal) | **New** setting | MIRACL; Mr. TyDi |
| **C.** Thai negation in retrieval (ไม่หวาน) | ❌ Not found in any non-English retrieval work | Open gap; report as an analysis slice | NevIR; ExcluIR |
| Taste queries over review sentences (any language) | ❌ Not found | Closest are subjective search (OpineDB) and Japanese food-texture mining | OpineDB; Kato et al. 2012 |
| Thai restaurant reviews **with restaurant ids** | ❌ No public dataset | Confirms our sentence-level-only design | — |

**One-sentence framing for the report:**

> We replicate the RIRD finding that off-the-shelf dense retrieval beats BM25 with a multilingual
> encoder, extend it to sentence units with a review-volume control for the known count bias of late
> fusion, test whether the shown sentences are faithful and supportive, and run the retrieval step on
> a new human-labelled Thai taste test set.

---

## 2. Track A — fusion, units and volume bias

| Paper | Year · Venue | Link | Why it matters |
|---|---|---|---|
| Abdollah Pour et al., *Self-Supervised Contrastive BERT Fine-tuning for Fusion-based RIR* | 2023 · ECIR | [arXiv:2308.00762](https://arxiv.org/abs/2308.00762) | **Already compares BM25 vs off-the-shelf dense at K = 1 / 10 / all.** Our main A result replicates their Table 4 |
| Macdonald & Ounis, *Voting for Candidates* | 2006 · CIKM | [eprints.gla.ac.uk/3547](https://eprints.gla.ac.uk/3547) | Document scores → person scores with CombMAX / CombSUM / CombMNZ. Count-based voting rewards prolific candidates |
| Macdonald & Ounis, *Voting Techniques for Expert Search* | 2008 · KAIS | [doi:10.1007/s10115-007-0105-3](https://dx.doi.org/10.1007/s10115-007-0105-3) | Journal version, 12 voting techniques |
| Lioma, Macdonald et al., *University of Glasgow at TREC 2006* | 2006 · TREC | [PDF](https://trec.nist.gov/pubs/trec15/papers/uglasgow.tera.ent.final.pdf) | **Candidate length normalisation** "to prevent candidates with too much expertise evidence from gaining an unfair advantage". This is our volume bias |
| Macdonald & Ounis, *Key Blog Distillation: Ranking Aggregates* | 2008 · CIKM | [eprints.gla.ac.uk/44372](https://eprints.gla.ac.uk/44372) | A blog is a bag of posts, like a restaurant is a bag of reviews. ⚠️ DOI differs between listings; check ACM |
| Balog, Azzopardi & de Rijke, *Formal Models for Expert Finding* | 2006 · SIGIR | [ir.webis.de](https://ir.webis.de/anthology/2006.sigirconf_conference-2006.7) | Model 1 (one profile per candidate, like our v1 concatenated doc) vs Model 2 (score documents, then aggregate, like v2). Model 2 wins |
| Zhang & Balog, *Design Patterns for Fusion-Based Object Retrieval* | 2017 · ECIR | [arXiv:1708.08715](https://arxiv.org/abs/1708.08715) | Formalises early vs late fusion across tasks |
| Elsas et al., *Retrieval and Feedback Models for Blog Feed Search* | 2008 · SIGIR | [ir.webis.de](https://ir.webis.de/anthology/2008.sigirconf_conference-2008.46/) | Large-document vs small-document models |
| Ganesan & Zhai, *Opinion-Based Entity Ranking* | 2012 · Information Retrieval | [doi:10.1007/s10791-011-9174-8](https://doi.org/10.1007/s10791-011-9174-8) | Early review-based entity ranking with preference queries |
| Singhal, Buckley & Mitra, *Pivoted Document Length Normalization* | 1996 · SIGIR | [ir.webis.de](https://ir.webis.de/anthology/1996.sigirconf_conference-96.4) | Classic length-bias correction; background for the volume control |
| Fox & Shaw, *Combination of Multiple Searches* | 1994 · TREC-2 | [trec.nist.gov](https://trec.nist.gov/pubs/trec2/papers/txt/23.txt) | Defines CombMAX / CombSUM / CombMNZ |
| Dai & Callan, *Deeper Text Understanding…* (FirstP / MaxP / SumP) | 2019 · SIGIR | [arXiv:1905.09217](https://arxiv.org/abs/1905.09217) | Passage → document aggregation with neural scores |
| Yang, Zhang & Lin, *Simple Applications of BERT for Ad Hoc Document Retrieval* (Birch) | 2019 · arXiv | [arXiv:1903.10972](https://arxiv.org/abs/1903.10972) | Sentence scores aggregated into a document score |
| Li et al., *PARADE* | 2024 · TOIS | [arXiv:2008.09093](https://arxiv.org/abs/2008.09093) | Learned aggregation beats max on broad topics |
| Luan et al., *Sparse, Dense, and Attentional Representations* | 2021 · TACL | [ACL](https://aclanthology.org/2021.tacl-1.20) | Dense vectors lose precision on long texts, so dense should score short units |
| Chen et al., *Dense X Retrieval: What Retrieval Granularity Should We Use?* | 2024 · EMNLP | [arXiv:2312.06648](https://arxiv.org/abs/2312.06648) | Unit size matters a lot for dense; finer units help off-the-shelf retrievers |

**Implications for A:**
- Cite RIRD Table 4 next to our numbers and call our result a replication.
- Describe the aggregations as voting / CombMAX- and CombSUM-style, not as our invention.
- RIRD only kept restaurants with ≥ 400 reviews (406–2,281 in our data). Report this range when
  presenting A3, because the volume effect may be limited by it.
- The scorer × aggregator interaction (max is weakest *for dense*) and sentence units on RIR are the
  parts no one has reported.

---

## 3. Track B — explanation evaluation

| Paper | Year · Venue | Link | Why it matters |
|---|---|---|---|
| Li, Zhang & Chen, *EXTRA: Explanation Ranking Datasets* | 2021 · SIGIR | [arXiv:2102.10315](https://arxiv.org/abs/2102.10315) | Established framing: explanation = ranking existing review sentences. No faithfulness test or human rating |
| Tan et al., *Counterfactual Explainable Recommendation (CountER)* | 2021 · CIKM | [arXiv:2108.10539](https://arxiv.org/abs/2108.10539) | Removal/counterfactual faithfulness in recsys; the established relative of our deletion test |
| Rashkin et al., *Measuring Attribution in NLG Models (AIS)* | 2023 · Computational Linguistics | [ACL](https://aclanthology.org/2023.cl-4.2/) | Human protocol for "is this supported by the source"; model for our 0/1/2 support guidelines (E1) |
| Gao et al., *Enabling LLMs to Generate Text with Citations (ALCE)* | 2023 · EMNLP | [arXiv:2305.14627](https://arxiv.org/abs/2305.14627) | Automatic NLI check of citation support, validated against humans |
| Zhao et al., *TRACE: Tourism Recommendation with Accountable Citation Evidence* | 2026 · arXiv | [arXiv:2605.07677](https://arxiv.org/abs/2605.07677) | **Closest domain**: Yelp places, cites review spans, NLI grounding score vs humans. Finds non-LLM retrievers ground better than LLMs. No deletion test or kappa |
| Kabongo et al., *Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation* | 2026 · ⚠️ ICTIR '26 per one listing | [alphaxiv 2604.03724](https://www.alphaxiv.org/abs/2604.03724) | Ranks review statements as explanations; ranking metrics only |
| Kabongo & Guigue, *On the Factual Consistency of Text-based Explainable Recommendation Models* | 2025 · arXiv | [arXiv:2512.24366](https://arxiv.org/abs/2512.24366) | Generated explanations: high BERTScore but only 4–33% statement precision. This supports extractive explanations |
| Wang, Cai & Wang, *Graph-based Extractive Explainer for Recommendations* | 2022 · WWW | [arXiv:2202.09730](https://arxiv.org/abs/2202.09730) | Selects review sentences as explanations |
| Xie, McAuley & Majumder, *On Faithfulness and Coherence of Language Explanations for Recommendation* | 2022 · arXiv | [arXiv:2209.05409](https://arxiv.org/abs/2209.05409) | Generated review "explanations" don't explain the predicted ratings |
| Balog & Radlinski, *Measuring Recommendation Explanation Quality* | 2020 · SIGIR | [ir.webis.de](https://ir.webis.de/anthology/2020.sigirconf_conference-2020.41/) | Dimensions of explanation quality for user evaluation |
| Hernandez-Bocanegra & Ziegler, *Assessing the Helpfulness of Review Content for Explaining Recommendations* | 2020 · EARS workshop | [arXiv:2010.06328](https://arxiv.org/abs/2010.06328) | User study on review content as explanation (hotels; restaurants named as future work) |
| Singh & Anand, *EXS: Explainable Search* | 2019 · WSDM | [arXiv:1809.03857](https://arxiv.org/abs/1809.03857) | Perturbation-based explanation of neural rankers |
| Sani et al., *FIRE: Faithful Interpretable Recommendation Explanations* | 2025 · arXiv | [arXiv:2508.05225](https://arxiv.org/abs/2508.05225) | Separates faithfulness to the model from plausibility |

**Implications for B (protocol to copy):**
- **B1:** ERASER comprehensiveness recast for ranking. Measure the rank and score drop after deleting
  the top-k shown sentences vs. k random sentences, average the random condition over several seeds,
  and report more than one k.
- **B2:** AIS-style guidelines. Does the sentence support the query attribute, without outside
  knowledge? Map to 0/1/2 and report Cohen's kappa. Optionally add an ALCE-style NLI score as an
  automatic proxy.
- Say explicitly: B1 measures **faithfulness to the ranker**, B2 measures **plausibility/support**
  (Jacovi & Goldberg; FIRE).
- Don't claim "first faithfulness study of review explanations". Claim the combined protocol on a
  reviewed-item ranker.

---

## 4. Track C — Thai

| Item | Year · Venue | Link | Why it matters |
|---|---|---|---|
| Zhang et al., *MIRACL* | 2023 · TACL | [ACL](https://aclanthology.org/2023.tacl-1.63) | Thai split, native-speaker judgments to top-10 (same depth as ours). Thai nDCG@10: BM25 0.484, mDPR 0.358, mContriever 0.517, hybrid 0.599; newer mE5-large is around 0.80 |
| Zhang et al., *Mr. TyDi* | 2021 · MRL | [ACL](https://aclanthology.org/2021.mrl-1.12.pdf) | Zero-shot mDPR loses badly to BM25 on Thai; a hybrid still helps |
| Lowphansirikul et al., *scb-mt-en-th-2020* | 2020 · arXiv | [arXiv:2007.03541](https://arxiv.org/abs/2007.03541) | Reports **crfcut sentence-boundary F1 ≈ 0.77 on product reviews**; cite in the segmentation limitation |
| PyThaiNLP paper | 2023 · arXiv | [arXiv:2312.04649](https://arxiv.org/abs/2312.04649) | newmm ≈ 71% on BEST-2010 vs 95.6% state of the art; BM25 quality depends on segmentation |
| TranSentCut | 2022 · Songklanakarin J. Sci. Tech. | [sjst.psu.ac.th](https://sjst.psu.ac.th/article.php?art=2861) | Stronger Thai sentence segmenter than crfcut out of domain |
| Wongnai corpus (search queries + food dictionary) | 2019 · GitHub | [wongnai/wongnai-corpus](https://github.com/wongnai/wongnai-corpus) | **Real Thai food search queries** (500K words, 10K labelled) and a food dictionary; can back up the realism of our 24 queries |
| Phetruchaeng et al., *Multi-Label Sentiment Analysis … Thai Restaurant Reviews* | 2025 · PMR 3(5) | [tci-thaijo](https://so09.tci-thaijo.org/index.php/PMR/article/view/6469) | Nearest Wongnai work: aspect sentiment (food, price, service, ambience). Classification only |
| SEACrowd | 2024 · EMNLP | [ACL](https://aclanthology.org/2024.emnlp-main.296/) | Has a Wongnai dataloader; cite for the dataset and its licence (LGPL-3.0) |
| SEA-BED | 2026 · ACL | [ACL](https://aclanthology.org/2026.acl-long.397/) | SEA embedding benchmark: no single model is best |
| NitiBench (WangchanX-Legal) | 2025 · arXiv | [arXiv:2502.10868](https://arxiv.org/abs/2502.10868) | Closest hand-built Thai retrieval test set, in the legal domain |
| Weller et al., *NevIR: Negation in Neural IR* | 2024 · EACL | [arXiv:2305.07614](https://arxiv.org/abs/2305.07614) | Most retrievers are near random on negation; only cross-encoders do well |
| van den Elsen et al., *Reproducing NevIR* | 2025 · SIGIR | [arXiv:2502.13506](https://arxiv.org/abs/2502.13506) | Confirms NevIR |
| Zhang et al., *ExcluIR* | 2025 · AAAI | [arXiv:2404.17288](https://arxiv.org/abs/2404.17288) | "X but not Y" queries are hard for retrievers |
| Petcu et al., *A Comprehensive Taxonomy of Negation* | 2025 · Findings of EMNLP | [arXiv:2507.22337](https://arxiv.org/abs/2507.22337) | Use to name negation types in our 5 negation queries |
| Malaviya et al., *QUEST* | 2023 · ACL | [ACL](https://aclanthology.org/2023.acl-long.784/) | Conjunction + negation queries are hardest; relevant to combo queries |
| Li et al., *Subjective Databases (OpineDB)* | 2019 · PVLDB | [arXiv:1902.09661](https://arxiv.org/abs/1902.09661) | Subjective restaurant/hotel queries over reviews; nearest precedent for taste queries |
| Kato et al., *Onomatopoeia for Foods from Food Reviews … Restaurant Search* | 2012 · WWW Companion | [PDF](https://archives.iw3c2.org/www2012/proceedings/companion/p719.pdf) | Japanese taste/texture words from reviews for restaurant search; nearest non-English precedent |
| Bjerva et al., *SubjQA* | 2020 · EMNLP | [arXiv:2004.14283](https://arxiv.org/abs/2004.14283) | Subjective QA over reviews, with restaurants |
| Min et al., *A Survey on Food Computing* | 2019 · ACM CSUR | [arXiv:1808.07202](https://arxiv.org/abs/1808.07202) | Domain framing |

**Implications for C:**
- **Don't expect "dense beats BM25 in Thai".** Published Thai results favour BM25 over older
  zero-shot dense models, and a hybrid usually wins. Our queries are more paraphrastic (which favours
  dense), but our encoder is a similarity model, not a retriever. Expect no clear winner and large
  per-query variance. Phrase the result as "both produce usable sentence rankings, nDCG@10 X vs. Y".
- **Report negation queries as their own slice** (NevIR style). For Thai, keeping ไม่ as a token
  might let BM25 match the negated phrase literally. That is our hypothesis to test, not a published
  finding.
- **Name the segmenters** (crfcut, newmm) and their published accuracy as a limitation.
- Our query freeze before any results matches the advice in Webber et al. 2008.

---

## 5. Small test collections and labelling (C3–C7, B2)

| Paper | Year · Venue | Link | Why it matters |
|---|---|---|---|
| Webber, Moffat & Zobel, *Statistical Power in Retrieval Experimentation* | 2008 · CIKM | [unimelb](https://people.eng.unimelb.edu.au/ammoffat/abstracts/wmz08cikm.html) | Don't add queries until significant; many shallow topics beat few deep ones |
| Sakai, *Topic Set Size Design* | 2016 · IR Journal | [Springer](https://link.springer.com/article/10.1007/s10791-015-9273-z) | How many queries you need for a given power |
| Sakai, *Statistical Significance, Power, and Sample Sizes* | 2016 · SIGIR | [ir.webis.de](https://ir.webis.de/anthology/2016.sigirconf_conference-2016.3/) | Many IR experiments are underpowered; reporting checklist |
| Carterette, Allan & Sitaraman, *Minimal Test Collections* | 2006 · SIGIR | [ir.webis.de](https://ir.webis.de/anthology/2006.sigirconf_conference-2006.36) | Precedent for a small judging budget |
| Zobel, *How Reliable are the Results of Large-Scale IR Experiments?* | 1998 · SIGIR | [unimelb](https://people.eng.unimelb.edu.au/ammoffat/sigir98/abstracts/zobel.html) | Rankings survive incomplete pools |
| Voorhees, Soboroff & Lin, *Can Old TREC Collections Reliably Evaluate Modern Neural Retrieval Models?* | 2022 · arXiv | [arXiv:2201.11086](https://arxiv.org/abs/2201.11086) | Pool reuse with neural runs |
| Thomas et al., *LLMs Can Accurately Predict Searcher Preferences* | 2024 · SIGIR | [arXiv:2309.10621](https://arxiv.org/abs/2309.10621) | LLM labels can match humans, but are prompt-sensitive (relevant to optional C6) |
| Faggioli et al., *Perspectives on LLMs for Relevance Judgment* | 2023 · ICTIR | [arXiv:2304.09161](https://arxiv.org/abs/2304.09161) | Human–LLM collaboration in judging; don't replace humans |
| Upadhyay et al., *UMBRELA* | 2024 · arXiv | [arXiv:2406.06519](https://arxiv.org/abs/2406.06519) | Open LLM relevance-judging prompt |

**What to state for 24 Thai queries and two systems:**
- Paired randomization or bootstrap test on per-query nDCG@10. Report the mean difference, a 95% CI,
  and a win/tie/loss count.
- **Power limit:** with n = 24, only differences of roughly **0.11–0.17 nDCG@10** can be detected
  reliably. This is a back-of-envelope estimate (≈ 0.57 × the SD of the per-query differences,
  assumed 0.2–0.3); recompute it from our own data. A non-significant result means "no evidence of
  a difference", not "equal".
- **Pooling BM25 + dense top-10 is fine for those two systems** (complete judgments at depth 10, the
  same as MIRACL). The collection is **not validated for reuse**: any new run (hybrid, reranker,
  another encoder) needs its unjudged top-10 labelled. A cheap improvement is to add an RRF hybrid
  as a third pooled run.
- If an LLM labels (C6), treat it as a third annotator and report its kappa against both humans.

---

## 6. LLM-era work reviewers will ask about

| Paper | Year · Venue | Link | How to position |
|---|---|---|---|
| TRACE (see §3) | 2026 · arXiv | [arXiv:2605.07677](https://arxiv.org/abs/2605.07677) | Closest benchmark; we add labelled single-turn item ranking + a deletion test |
| Kemper et al., *RA-Rec* (Sanner lab, cites RIRD) | 2024 · SIGIR demo | [arXiv:2406.00033](https://arxiv.org/abs/2406.00033) | LLM conversational upper bound; out of scope by design |
| Hou et al., *BLaIR / Amazon-C4* | 2024 · arXiv | [arXiv:2403.03952](https://arxiv.org/abs/2403.03952) | Review-aware encoder. Expect "why not BLaIR?"; it is English-only, and we need one encoder for both languages |
| *KALM4Rec* | 2025 · WWW Companion | [arXiv:2405.19612](https://arxiv.org/abs/2405.19612) | Yelp reviews + LLM reranking; out of scope |
| Xie et al., *Factual and Informative Review Generation* | 2023 · AAAI | [arXiv:2209.12613](https://arxiv.org/abs/2209.12613) | Retrieval reduces hallucinated explanations; supports extractive choice |
| Mysore et al., *Mint: LLM-Augmented Narrative-Driven Recommendation* | 2023 · RecSys | [arXiv:2306.02250](https://arxiv.org/abs/2306.02250) | Narrative queries over items represented by reviews |
| Kim et al., *Pearl* | 2024 · Findings of ACL | [arXiv:2403.04460](https://arxiv.org/abs/2403.04460) | Review-grounded conversational dataset |
| Eberhard et al., *LLMs as Narrative-Driven Recommenders* | 2025 · WWW | [arXiv:2410.13604](https://arxiv.org/abs/2410.13604) | Zero-shot LLMs on free-text requests |

Papers citing RIRD are almost all from Scott Sanner's group (Korikov 2024, RA-Rec, TravelDest,
Recipe-MPR). None is a controlled BM25 vs. dense comparison or tests explanation faithfulness.

**Why the project is not obsolete:** TRACE finds non-LLM retrievers ground answers better than LLMs,
and Kabongo 2025 finds generated explanations mostly unsupported by reviews. Both make extractive,
retrieval-based evidence a principled baseline. Justify the off-the-shelf scope with cost,
reproducibility, and the low-resource Thai setting.

---

## 7. Suggested follow-ups (not yet in action_items.md)

Status checked 2026-10-18 against `\cite` keys in [report/progress/main.tex](../report/progress/main.tex).

| Suggestion | Track | Cost | Status |
|---|---|---|---|
| Put RIRD Table 4 next to our A numbers; call it a replication | A, D2 | Low | ✅ Done: report Ch. 3 calls our dense > BM25 result a replication of RIRD (table number not cited) |
| Report the 406–2,281 review range with A3 | A | Low | ✅ Done in the progress report |
| Add these must-cites to related work (D1): Macdonald & Ounis, Balog 2006, Dense X, EXTRA, CountER, AIS, TRACE, MIRACL, NevIR | D1 | Low | ✅ Mostly: Macdonald & Ounis 2008, Balog 2006, Dense X (Chen 2024), EXTRA, ERASER, Jacovi & Goldberg, MIRACL, NevIR are cited. ⏳ CountER, AIS, TRACE still missing |
| Report Thai negation queries as a separate slice | C7 | Low | ✅ Planned: `thai/score_labels.py` breaks results down by query type (single / combo / negation) |
| State the power limit and "not validated for reuse" for the Thai collection | C7 | Low | ✅ Power limit (~0.11–0.17 nDCG@10) stated in the report; ⏳ "not validated for reuse" not yet |
| Add an RRF hybrid as a third pooled Thai run | C3 | Medium (adds ~5–7 sentences per query to label) | ⏳ Not done |
| Cite the Wongnai search-query corpus as a realism check on our queries | C1, D2 | Low | ⏳ Not done |
