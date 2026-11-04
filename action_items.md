# Action Items — Group Flower

Last updated 2026-11-03.

**Progress presentation:** ✅ Completed 2026-10-20.
**Next deadline: Final submission — December 1** — report (PDF, AIT thesis template), slides (PDF), and code.
Presentation: 10 minutes plus 5 minutes Q&A. Focus on bullets 5–8 (methodology through conclusions);
bullets 1–4 were covered in the proposal/progress presentations.

---

## 1. What the professor wants

### 1.1 Feedback on the proposal ([professor_commmend.md](professor_commmend.md))

**Strengths (keep these):** strong problem formulation as reviewed-item retrieval; RIRD gives
independent human relevance labels; EDA and two baselines with clear numbers; honest split between
measured English results and a qualitative Thai demo; coherent report and slides.

**Concerns we must answer:**

| # | Concern | What it means for us | Tasks |
|---|---|---|---|
| F1 | Application claims don't match the evaluation | RIRD tests broad restaurant preferences in English, not Thai taste. Wongnai has no restaurant ids, so it can't support restaurant-level evaluation. Our "Thai food" claims are unmeasured. | C1–C7, D2 |
| F2 | BM25 and dense use different units; fusion biased by review volume | v1 BM25 scored one concatenated document per restaurant, dense scored mean-pooled reviews. Max / top-k fusion favours restaurants with more reviews (406 to 2,281 reviews per restaurant in RIRD). | A1–A4 ✅ |
| F3 | No evidence for explainability | A top-scoring sentence can be relevant without being a complete or faithful explanation. Explanation quality has not been evaluated. | B1, B2 |
| F4 | Check more literature | Read and cite the three papers below. | D1 ✅ |

Papers the professor pointed to:

- Korikov et al., *Multi-Aspect Reviewed-Item Retrieval via LLM Query Decomposition and Aspect Fusion* — [arXiv:2408.00878](https://arxiv.org/abs/2408.00878). Shows top-K late fusion is sensitive to how aspects are spread across reviews; directly relevant to F2.
- Zhang et al., *Enhancing Recommendation Explanations through User-Centric Refinement*, Findings of EMNLP 2025 — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.434/). Criteria for judging explanations; relevant to F3.
- Hasan et al., *Review-based Recommender Systems: A Survey of Approaches, Challenges and Future Perspectives*, ACM Computing Surveys — [doi:10.1145/3742421](https://doi.org/10.1145/3742421). Background and positioning.

**The question the final project must answer:**

> Do retrieved review sentences genuinely support the system's restaurant rankings, and can the
> same approach work on real Thai restaurant reviews with independently annotated taste preferences?

- First half → B1 (faithfulness) and B2 (human support rating).
- Second half → C1–C7 (Thai taste test set). Expect a partial answer: sentence level on Thai, not restaurant level.

### 1.2 Requirements for the final submission (December 1)

**Presentation (10 min) must cover (focus on bullets 5–8):**

1. ~~Project overview — problem, objectives, datasets, methodology (brief).~~ (covered in proposal)
2. ~~Responses to previous feedback.~~ (covered in progress presentation)
3. ~~Progress and preliminary results.~~ (covered in progress presentation)
4. ~~Next steps.~~ (covered in progress presentation)
5. **Methodology** — full pipeline: EDA, preprocessing, BM25 vs. dense, aggregation, explanation extraction.
6. **Model evaluation results** — nDCG@10, MAP with CIs; B1 faithfulness; B2 human support kappa; Thai nDCG@10 / P@10.
7. **Discussions** — comparison with RIRD fine-tuned baseline (MAP 0.626); performance / interpretability / complexity trade-offs vs. competing systems.
8. **Conclusions** — summary of contributions; limitations; future work.

**Final report must:**

- Use the **AIT thesis template** from the AIT Language Center.
- Include a detailed explanation of how the system **outperforms others on Kaggle/HuggingFace**, highlighting unique features, advantages, and user benefits.
- Describe the **baseline model** (BM25 review-level, concatenated document) vs. the **proposed model** (dense sentence-level, top-10 mean aggregation).
- Properly cite all figures and datasets obtained from the internet.
- Submit both **report PDF** and **slides PDF**.

---

## 2. Progress so far

### Track A — Fair fusion (English / RIRD) ✅ all done

- **A1/A2 complete** at review level and sentence level. BM25 and dense now score the same unit with the
  same aggregation variants (max, mean, top-3/5/10 mean), with bootstrap 95% CIs and Holm-corrected
  paired tests. [baseline_models_v2.ipynb](baselines/baseline_models_v2.ipynb) (review),
  [baseline_models_v2_sentence.ipynb](baselines/baseline_models_v2_sentence.ipynb) (sentence).
  - **Best result:** Dense top-10 mean — nDCG@10 **0.682**, MAP **0.618** (sentence level).
    Review level: 0.668 / 0.608. RIRD fine-tuned reference: MAP 0.626.
  - Dense beats BM25 under all aggregations at review level (all Holm-significant). At sentence level,
    dense beats BM25 for max and top-k means (Holm p ≤ 0.004); mean gap not significant (p = 0.072).
  - BM25 gets worse with sentences (top-10 mean 0.609 vs 0.634 at review level); dense mean also drops
    (0.587 vs 0.642). Sentence units help dense top-k only.
  - Sentence result CSVs are gitignored; exist only in local `baselines/cache/` copies.
- **A3 complete:** volume control — every restaurant sampled to 400 reviews (SEED=42). Score/count
  correlation drops to near zero (Pearson −0.009 to +0.019) after control; effect on nDCG@10 ≤ 0.011.
- **A4 complete:** 50,000 ranking rows + 500,000 evidence rows exported (query, model, aggregation,
  rank/score, source review row ID and text).
- **A5 complete:** paired review-vs-sentence significance test is in
  `cache/a5_review_vs_sentence_test.csv` with its forest plot.

### Track C — Thai (Wongnai)

- **C1/C1b complete:** 24 Thai taste queries frozen 2026-10-17 ([thai/thai_queries.csv](thai/thai_queries.csv)).
- **C2 complete:** PyThaiNLP sentence pipeline on 2,000-review subsample; segmentation spot-check done;
  qualitative T02/T11/T20 examples documented.
- **C3 complete (2026-10-19):** Ton ran 10k Colab notebook on T4 GPU
  ([thai/thai_retrieval_10k_colab.ipynb](thai/thai_retrieval_10k_colab.ipynb)).
  - 10,000 Wongnai reviews (seed 42) → 93,810 sentences (9.4 per review).
  - BM25 and dense (`paraphrase-multilingual-mpnet-base-v2`) top-10 for all 24 frozen queries.
  - Output: [thai/results/topk_n10000_s42.csv](thai/results/topk_n10000_s42.csv) (480 rows).
  - Pool built with [thai/build_label_pool.py](thai/build_label_pool.py): **452 blind pairs**, mean 18.8
    per query (range 13–20). Two systems barely overlap: mean 1.2 shared sentences in top 10. No BM25
    zero-score hits in any top 10.
  - Files committed: [thai/labels/label_sheet.csv](thai/labels/label_sheet.csv) (452 pairs), [thai/labels/label_sheet_annotator2.csv](thai/labels/label_sheet_annotator2.csv)
    (60 pairs for second annotator / C5), [thai/labels/pool_key.csv](thai/labels/pool_key.csv) (hidden
    key; annotators must not open it).
- **C4 complete (2026-11-02):** all 452 pairs labelled by Ton under the frozen guidelines.
  Final: 184 × 0, 180 × 1, 88 × 2; one query has no relevant sentence in its pool.
- **C7 metrics (2026-11-02):** BM25 beats dense on Thai, but not significantly (24 queries):

  | Metric | BM25 | Dense | Diff [95% CI] | Holm p |
  |---|---|---|---|---|
  | nDCG@10 | 0.654 | 0.549 | −0.105 [−0.298, +0.093] | 0.57 |
  | P@10 (≥1) | 0.667 | 0.558 | −0.108 [−0.292, +0.075] | 0.57 |
  | P@10 (=2) | 0.258 | 0.163 | −0.096 [−0.212, +0.029] | 0.47 |

  By type: combo queries favour BM25 most (strict P@10 0.325 vs. 0.125, CI excludes 0); negation
  queries are a tie (nDCG@10 0.574 vs. 0.607). Opposite of English, where dense wins.
- **C5 complete (2026-11-04):** second-round labels on the 60-pair subset were LLM-drafted under the frozen guidelines (no access to round-1 labels; draft kept in [thai/labels/label_sheet_annotator2_llm_draft.csv](thai/labels/label_sheet_annotator2_llm_draft.csv)) and reviewed/corrected by Ton. Agreement with round 1: κ 0.837 (weighted 0.869, 90% exact, n=60) → [thai/results/thai_agreement.csv](thai/results/thai_agreement.csv). Report as LLM-drafted, annotator-reviewed — not an independent second annotator.

### Writing / Report

- **D1 complete:** three professor-assigned papers added to related work.
- **D3 complete:** feedback summary page in [report/progress/main.pdf](report/progress/main.pdf).
- **D4 complete:** progress report PDF and slides PDF submitted (2026-10-19).
- **E1 complete (2026-11-02):** [thai/annotation_guidelines.md](thai/annotation_guidelines.md) signed off by Ton and frozen.

### Demo app (2026-11-03)

- **Thai tab fixed (PR #1, merged):** every Thai query used to return HTTP 500 (the response schema only
  allowed restaurant results). Thai results now come back as sentences, BM25 and dense interleaved by
  rank, each with Ton's C4 relevance label. [tests/test_thai_app.py](tests/test_thai_app.py) adds 8
  tests (all passing).
- **English live inference + Groq rewriting verified locally:** the gitignored sentence embeddings are
  rebuilt with [app/build_sentence_embeddings.py](app/build_sentence_embeddings.py). New English
  queries now run Groq rewrite → bi-encoder → selective CE in ~3–5s on a GTX 1650. The 100 suggested
  queries still use instant pre-computed results (no Groq).

---

## 3. Tasks

Status: `[ ]` not started, `[/]` in progress, `[x]` done.
**Thai?** marks tasks that need someone who reads Thai.
**Priority:** High / Medium / Low for the final submission.

### Owners

| Area | Owner |
|---|---|
| A. Fair fusion study (English) | **Sam** |
| B. Explanation evaluation (English) | **Sam** |
| C. Thai taste test set | **Ton** |
| D. Writing | **Both** |
| E. Annotation guidelines | **Ton** (finalize), **Sam** (B2 sheet prep) |
| F. Final deliverables | **Both** |

---

### A. Fair fusion study on RIRD (English) — answers F2 · Owner: Sam

| ID | Task | Priority | Thai? | Depends on |
|---|---|---|---|---|
| A1 | `[x]` Shared retrieval unit for BM25 and dense — review and sentence level | — | No | — |
| A2 | `[x]` Same aggregation variants (max, mean, top-k) with CIs and Holm tests — review and sentence level | — | No | A1 |
| A3 | `[x]` Volume control: every restaurant sampled to 400 reviews; score/count correlation near zero after control | — | No | A1 |
| A4 | `[x]` Per-query ranking and evidence export (50k ranking rows + 500k evidence rows) | — | No | A2 |
| A5 | `[x]` Paired review-vs-sentence test. Results in `cache/a5_review_vs_sentence_test.csv`, forest plot in `cache/a5_forest_plot.png`. **Key finding:** no variant shows a Holm-significant improvement from switching to sentences. Dense\|mean is significantly *worse* at sentence level (Holm p=0.0, Δ=−0.055). Only Dense\|top-10 mean shows a positive but non-significant gap (Δ=+0.013, Holm p=1.0). | Medium | No | A2 |

The sentence-level evidence output from A4 is the direct input for B1 and B2.

---

### B. Explanation evaluation on RIRD (English) — answers F3 · Owner: Sam

| ID | Task | Priority | Thai? | Depends on |
|---|---|---|---|---|
| B1 | `[x]` **Faithfulness (automatic):** [b1_faithfulness.ipynb](baselines/b1_faithfulness.ipynb) executed. **Result: mean faithfulness gap = 0.000, p=1.0 (n.s.).** Shown and random deletions hurt rank equally (both mean rank drop = 48). nDCG@10 drops from 0.682 to 0.379 under either deletion — deletion is very destructive but not differentially so. Interpretation: the pooled evidence store is small (top-10 per restaurant) so random and shown deletions always hit the same sentences. See `cache/b1_faithfulness_summary.csv` and `cache/b1_faithfulness_plot.png`. | **High** | No | A4 ✅ |
| B2 | `[/]` **Support (human):** [b2_export_support_sheet.py](baselines/b2_export_support_sheet.py) run. Sheets generated: `cache/b2_support_sheet.csv` (full, Sam labels), `cache/b2_support_sheet_ann2.csv` (40-pair subset, Ton labels), `cache/b2_support_key.csv` (hidden key). **Status 2026-11-03:** Ton's 40-pair `b2_support_sheet_ann2.csv` is done (19 × 1, 12 × 0, 9 × 2). **Next: Sam fills `label` column (0/1/2) in `b2_support_sheet.csv` (0 / 375 labelled). Then run [b2_score_agreement.py](baselines/b2_score_agreement.py).** | **High** | No | A4 ✅, E1 |
| B3 | `[/]` **Improved retrieval** — notebook executed and integrated into [app/retrieval.py](app/retrieval.py). The live system uses selective CE because applying CE to every query reduced nDCG@10 from 0.682 to 0.658 and MAP from 0.618 to 0.586. **2026-11-03 (Ton):** live pipeline verified end-to-end locally — sentence embeddings rebuilt with [app/build_sentence_embeddings.py](app/build_sentence_embeddings.py) (240,660 × 768, GTX 1650), Groq rewrite + selective CE confirmed on new queries ("somewhere quiet without a long wait" → CE path, 5.4s; "cheat meal after the gym" → bi-encoder only, 3.1s). Remaining: B3f, and note in limitations that negation is still imperfect (e.g. a "no reservations… hassle to get seated" sentence ranked #3 for the no-wait query). | **High** | No | B2, A4 ✅ |
| B3f | `[ ]` **Align app with B3 notebook:** (1) app's `GROQ_SYSTEM_PROMPT` is missing the notebook's *"healthy option"* example, so live rewrites can differ from what B3 measured; (2) the 100 pre-computed queries show Dense results and never use the B3 rewrites in `cache/b3_rewritten_queries.json` — decide whether the demo should show them. CE scores also display as raw logits (e.g. −8.78) next to 0–1 cosine scores. | Medium | No | B3 |
| B3a | `[x]` **LLM query rewriting (Groq / qwen3.8-27b):** B2 labelling found two failure modes: (1) *negation blindness* — "no waitline" retrieves sentences about busy restaurants because the bi-encoder sees same-topic embeddings regardless of negation; (2) *idiomatic query failure* — "cheat meal", "I feel chubby" land in undefined embedding regions. Fix: call Groq API to rewrite query into explicit positive reviewer-language keywords before encoding. "No waitline" → "quick seating walk-in no queue no reservation". Results cached in-memory (second call instant). **Reason for PPT:** addresses a fundamental bi-encoder limitation; shows LLM integration for IR. | **High** | No | B2 |
| B3b | `[x]` **Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2):** bi-encoder encodes query and sentence *independently* (no joint attention), so negation mismatch is invisible. Cross-encoder reads (rewritten_query, sentence) together with full transformer attention — detects semantic mismatch. Only top-20 bi-encoder restaurants are sent to CE (pre-filter), and only top-15 bi-encoder sentences per restaurant (300 CE calls total vs. 1,500 in naive approach = **5× faster**). **Reason for PPT:** higher precision evidence; cross-encoder is the standard IR "re-ranker" pattern. | **High** | No | B3a |
| B3c | `[x]` **Evidence quality filtering:** B2 also showed very short sentences ("Solid meal!", "No consistency.") scoring high due to embedding length bias. Fix: `MIN_SENTENCE_WORDS=10` filter applied at load time removes sub-10-word sentences from the live retrieval corpus. `CE_DISPLAY_THRESH=0.0` hides CE-scored sentences below the threshold (fallback: always show ≥2). **Reason for PPT:** improves faithfulness of shown evidence; directly motivated by B2 labelling observations. | **High** | No | B3b |
| B3d | `[x]` **Demo speed optimisation:** cross-encoder over all restaurants was 3–8s per query. Fix: (1) `CE_TOP_RESTS=20` — only CE-rank top-20 bi-encoder restaurants; (2) `CE_INPUT_K=15` — 15 candidates/restaurant; (3) in-memory `_live_cache` — repeated queries return instantly. Pre-computed queries (100 RIRD + 24 Thai) still bypass all models (<1ms). **Net result:** live query latency reduced from ~8s to ~2–4s. | **High** | No | B3b |
| B3e | `[x]` **B3 notebook results + selective CE fix:** b3_improved_retrieval.ipynb executed. **Key finding:** rewrite+CE improved 44/100 queries but *degraded* 53/100. Root cause: `ms-marco-MiniLM-L-6-v2` is trained on web search (MS MARCO), not restaurant review retrieval — it confidently re-ranks wrong sentences for food/location queries. Top degraded: `"An ice cream shop with bubble tea"` (−0.74), `"I am vegan but my friends want good steaks"` (−0.49). Top improved: `"not Thai but Asian"` (+0.42), `"not sushi but Asian"` (+0.40), `"no waitline"` queries. **Fix applied to [app/retrieval.py](app/retrieval.py):** CE is now *selective* — only activates when the query contains explicit negation/exclusion patterns (`no`, `not`, `without`, `don't`, `never`, `except`, `avoid`, `non-`, `instead of`, etc.) via `_NEGATION_RE`. For plain positive queries, bi-encoder alone is used. **Reason for PPT:** honest reporting of when and why the CE helps vs. hurts — this is the nuanced finding that makes the work credible. | **High** | No | B3d |

---

### C. Thai taste test set on Wongnai — answers F1 · Owner: Ton

| ID | Task | Priority | Thai? | Depends on |
|---|---|---|---|---|
| C1 | `[x]` 24 frozen Thai taste queries in [thai/thai_queries.csv](thai/thai_queries.csv) | — | Yes | — |
| C1b | `[x]` Ton review and sign-off 2026-10-17; queries frozen; notebook rerun | — | Yes | C1 |
| C2 | `[x]` Sentence pipeline on 2,000-review subsample; segmentation spot-check done | — | Code: No. Spot-check: Yes | — |
| C3 | `[x]` 10k Colab run → `topk_n10000_s42.csv` → blind label pool built. 452 pairs in [thai/labels/label_sheet.csv](thai/labels/label_sheet.csv). 60-pair annotator-2 subset in [thai/labels/label_sheet_annotator2.csv](thai/labels/label_sheet_annotator2.csv). Key in `pool_key.csv` (do not open). | — | No | C1b, C2 |
| C4 | `[x]` **Done 2026-11-02.** All 452 pairs in [thai/labels/label_sheet.csv](thai/labels/label_sheet.csv) labelled 0/1/2 by Ton under the frozen guidelines. Final counts: 184 × 0, 180 × 1, 88 × 2. | — | Yes | C3, E1 |
| C5 | `[x]` **Done 2026-11-04.** 60-pair subset in [thai/labels/label_sheet_annotator2.csv](thai/labels/label_sheet_annotator2.csv): LLM-drafted under the frozen guidelines (blind to round-1 labels; draft with confidence/reasons in [label_sheet_annotator2_llm_draft.csv](thai/labels/label_sheet_annotator2_llm_draft.csv)), reviewed and corrected by Ton. κ 0.837 (weighted 0.869, 90% exact, n=60). **Method note for report:** describe as LLM-drafted, annotator-reviewed second round, not an independent second annotator. | — | Yes | C4 ✅ |
| C6 | `[-]` Not used. | — | — | — |
| C7 | `[x]` **Thai metrics done 2026-11-02** with [thai/score_labels.py](thai/score_labels.py) → [thai/results/thai_scores_summary.csv](thai/results/thai_scores_summary.csv), [thai/results/thai_scores_per_query.csv](thai/results/thai_scores_per_query.csv). nDCG@10 BM25 0.654 vs. dense 0.549 (diff −0.105, CI [−0.298, +0.093], Holm p 0.57, not significant). Agreement κ 0.837 (weighted 0.869, 90% exact, n=60) (C5). | **High** | No | C4 ✅, C5 |

---

### D. Writing · Owner: Both

| ID | Task | Priority | Thai? | Depends on |
|---|---|---|---|---|
| D1 | `[x]` Related work — three professor-assigned papers added to progress report | — | No | — |
| D3 | `[x]` Feedback summary page (progress report) | — | No | — |
| D4 | `[x]` Progress report PDF + progress slides PDF submitted | — | No | — |
| D2 | `[ ]` **Rewrite title, claims, and limitations** to match actual scope. Working title: *"Explainable reviewed-item retrieval, with a Thai taste case study."* Write once B2 and C7 numbers are in hand. | **High** | No | B2, C7 |

---

### E. Annotation guidelines · Owner: Ton (finalize), Sam (B2 sheet)

| ID | Task | Priority | Thai? | Depends on |
|---|---|---|---|---|
| E1 | `[x]` **Signed off by Ton 2026-11-02.** **Finalize annotation guidelines:** Ton rechecks [thai/annotation_guidelines.md](thai/annotation_guidelines.md) using its own checklist (real Wongnai examples, per-query rules, B2 appendix). Must be signed off before C4 or B2 labelling starts. | **High — do first** | Thai examples: **Yes** | — |

---

### F. Final deliverables · Owner: Both

| ID | Task | Priority | Thai? | Depends on |
|---|---|---|---|---|
| F1 | `[ ]` **Final report** — AIT thesis template. Sections: Introduction, Problem, Related Work, Datasets (≥3), Methodology (EDA → pipeline → evaluation), Results (A2 + B1 + B2 + C7), Discussion (vs. RIRD fine-tuned MAP 0.626 + Kaggle/HF systems), Conclusions. Cite all figures and datasets. | **High** | No | B1, B2, C7, D2 |
| F2 | `[ ]` **Final presentation slides** — 10 min covering bullets 5–8. Lead with evaluation results; frame dense top-10 mean as proposed vs. BM25 review-level as baseline. Include key results table and one qualitative Thai example. | **High** | No | F1 draft |
| F3 | `[ ]` **Code / repository cleanup** — all notebooks fully executed with outputs saved, dead code removed, `requirements.txt` verified, top-level README section on how to reproduce every result table. | **High** | No | B1, B2, C7 |
| F4 | `[ ]` **Baseline comparison write-up** (section inside F1): (a) v1 baseline — BM25 concatenated docs vs. dense mean-pooled reviews; (b) proposed — sentence-level dense, top-10 mean, volume-controlled; (c) external reference — RIRD fine-tuned MAP 0.626. Argue outperformance in terms of performance, interpretability, and complexity. | **High** | No | A2, B1, B2 |
| F5 | `[ ]` **Video** (if required by submission portal) — screen-capture walkthrough of notebook demo and key results. | Medium | No | F2, F3 |

---

### G. Demo app & AWS deployment · Owner: Sam

FastAPI backend + single-page HTML/JS frontend, Dockerized for EC2. English tab: live pre-computed lookup for 100 RIRD queries + optional live inference for new queries. Thai tab: pre-computed lookup for 24 frozen Wongnai queries.

| ID | Task | Priority | Depends on |
|---|---|---|---|
| G1 | `[x]` **Backend API** — [app/main.py](app/main.py) (FastAPI) + [app/retrieval.py](app/retrieval.py). Endpoints are implemented; validate health, query validation, precomputed lookup, live routing, and response schemas. *2026-11-03: Thai 500 fixed (PR #1); health, precomputed EN/TH, and live EN routing (CE and non-CE paths) checked by hand.* *2026-11-04: validation and schema covered by [tests/test_english_app.py](tests/test_english_app.py); blank queries now rejected; live cache fixed to honour `n_results` and capped at 256 queries.* | **High** | A4 ✅, C3 ✅ |
| G2 | `[/]` **Frontend UI** — [app/static/index.html](app/static/index.html). English and Thai result cards, explanations, and query chips are implemented; browser smoke-test remains. *2026-11-03: Thai cards now show human labels and system ranks (PR #1).* *2026-11-04: fixed status bar never reappearing after the first search (hid the spinner, errors and the "Query rewritten" notice), score bars breaking on negative CE scores, Thai tab showing the "Live B3" badge, and "[object Object]" validation errors. Browser smoke-test still remains.* | **High** | G1 |
| G3 | `[ ]` **Local Docker test** — `docker compose up --build`, verify both tabs work, check `/api/health`. Fix any import/path issues. | **High** | G1, G2 |
| G4 | `[ ]` **EC2 deployment** — once professor provides instance ID/IP: SSH in, install Docker, `rsync` or `git clone` project, `docker build -t food-recommender . && docker run ...`. Open port 8000 in security group. Follow [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). | **High** | G3, professor provides EC2 access |
| G5 | `[ ]` **Live inference (optional)** — uncomment the embeddings `COPY` line in [Dockerfile](Dockerfile) to enable arbitrary English query inference (requires t2.large, ~750 MB extra). Only do this if EC2 instance has ≥ 8 GB RAM. The `.npy` is gitignored: build it with `python -m app.build_sentence_embeddings` (~30 min on a GTX 1650; much slower on CPU, so build locally and `rsync` it). Groq key must be in `.env` (model `qwen/qwen3.8-27b`; `llama-3.1-8b-instant` is not available on our account). | Low | G4 |

---

### H. Engineering validation and additional metrics · Owner: Sam

These are the remaining code-facing checks beyond the existing ranking notebook results.

| ID | Task | Priority | Depends on |
|---|---|---|---|
| H1 | `[x]` **API and retrieval smoke tests** — *Thai: [tests/test_thai_app.py](tests/test_thai_app.py) (8 tests). English: [tests/test_english_app.py](tests/test_english_app.py) (17 tests, incl. live-path caching with stubbed models). 25/25 passing 2026-11-04.* Test health, query listing, precomputed English/Thai queries, unknown English fallback, negation routing, empty/oversized requests, and response schema. | High | G1 |
| H2 | `[/]` **Live-pipeline regression checks** — *2026-11-03: corpus count (240,660) and embedding alignment confirmed — the build script uses the app's own splitter, and the app raises on a size mismatch at startup.* Still to do: turn these into asserted tests, and check that all 50 restaurants are represented, and explanations contain at most 5 sentences. | High | B3, G1 |
| H3 | `[ ]` **B3 subgroup analysis** — report nDCG@10/MAP separately for explicit negation/exclusion queries and plain positive queries, plus query rewrite change rate and per-query deltas. This verifies that selective CE is justified rather than relying only on the overall average. | High | B3 |
| H4 | `[ ]` **Operational metrics** — measure cold-start time, precomputed latency, live-query p50/p95 latency, CE-vs-bi-encoder latency, and cache-hit latency; record peak memory during startup. | Medium | G1, G3 |
| H5 | `[ ]` **Robustness checks** — verify behavior with no Groq key, Groq timeout/error, missing optional embeddings, malformed CSV rows, and repeated/case-variant queries. | Medium | H1, H2 |
| H6 | `[/]` **Automated test suite** — started: `tests/` with pytest (`python -m pytest -q tests`). Add focused tests for helpers, routing, corpus alignment, API validation, and result contracts; run them before final packaging. | Medium | H1, H2 |

### Metrics to report

- **Ranking:** nDCG@5/@10, MAP, MRR, Recall@5/@10, P@5/@10, bootstrap 95% CIs, and paired per-query significance tests.
- **B3 diagnosis:** rewrite rate, selective-CE subgroup scores, improved/degraded query counts, and delta distributions.
- **Explanations:** percentage of evidence labelled supporting, mean label, support precision at 1/5, and inter-annotator Cohen's kappa or Krippendorff's alpha.
- **Thai track:** nDCG@10/P@10 (and MAP only if the annotation design supports it), per-query variance, and annotator agreement.
- **System quality:** cold start, p50/p95 latency, cache-hit latency, peak memory, and graceful-degradation behavior.

## 4. Plan (now → December 1)

### Priority order (dependency-aware)

```
Start immediately (unblocked or nearly so):
  1. E1   — ✅ done 2026-11-02
  2. B2   — Sam/Ton: complete human support labels and agreement
  3. H1/H2 — Sam: validate API and live corpus alignment

After E1:
  4. C4   — ✅ done 2026-11-02
  5. C5   — ✅ done 2026-11-04 (κ 0.837, LLM-drafted + Ton-reviewed)

After C4 + C5:
  6. C7   — ✅ done (nDCG@10 / P@10 + kappa)

After B2, C7, and H1–H3:
  7. D2  — Both: rewrite title / claims / limitations
  8. F4  — Both: baseline comparison write-up
  9. F1  — Both: final report (AIT thesis template)
  10. F2 — Both: final slides
  11. F3 — Sam: repository / notebook cleanup
  12. F5 — Both: video (if required)
```

### Owner summary

| Priority | Sam | Ton |
|---|---|---|
| **Do now** | Fill `cache/b2_support_sheet.csv` with 0/1/2 labels (open in Excel/Numbers, fill `label` col) | Label `cache/b2_support_sheet_ann2.csv` (B2); find second Thai annotator (C5) |
| **After labelling** | Run `python3 baselines/b2_score_agreement.py` for kappa + % supporting | C5: second Thai annotator (or self-relabel ≥ 1 week later), then rerun `thai/score_labels.py` for kappa |
| **Report / slides** | F4 baseline write-up (use A5 + B1 results now available), F1/F2 | F1/F2 together |

---

## 5. What can run in parallel

- **Sam:** B1 is fully unblocked — start now. B2 sheet can be prepared in parallel while waiting for E1.
- **Ton:** E1 can be done independently; C4 labelling starts immediately after E1. C5 annotator should be found this week.
- The RIRD track (A, B) and Thai track (C) are independent until the final report.

Dependency chains:

```
A4 (evidence export) ──► B1
A4 (evidence export) ──► B2 ◄── E1

C3 (done ✅) ──► C4 ──► C5 ──┐
E1 ──────────────────────────┤
C3 ──► C6 (optional) ────────┴──► C7

B1, B2, C7 ──► D2 ──► F1 (report) ──► F2 (slides) ──► F3 (repo) ──► F5 (video)
                └──► F4 (baseline comparison, inside F1)
```

---

## 6. Key numbers for the final presentation

| Model | Unit | Aggregation | nDCG@10 | MAP |
|---|---|---|---|---|
| Dense **(proposed)** | Sentence | Top-10 mean | **0.682** | **0.618** |
| Dense | Review | Top-10 mean | 0.668 | 0.608 |
| BM25 (baseline) | Sentence | Top-10 mean | 0.609 | — |
| BM25 (baseline) | Review | Top-10 mean | 0.634 | — |
| RIRD fine-tuned (external reference) | — | — | — | 0.626 |

Our off-the-shelf dense already approaches the fine-tuned external reference (MAP 0.618 vs 0.626),
with no domain adaptation. B1 and B2 add the interpretability argument.

**Thai (Wongnai, sentence level, 24 queries, C7):** BM25 nDCG@10 0.654 vs. dense 0.549; P@10 0.667
vs. 0.558. Not significant (Holm p 0.57). Kappa pending (C5).

---

## 7. If time runs short, cut in this order

1. Drop C6 (LLM labelling) — pool is 452 pairs; hand-label all of them.
2. Shrink B2 to 15 queries (minimum for meaningful kappa).
3. Drop C5 (second annotator) and use self-relabelling fallback.
4. Drop A5 (paired review-vs-sentence test) — informative but not blocking any deliverable.

**Do not cut B1 or C4–C7.** Without B1 there is no faithfulness answer; without C7 there is no
Thai result — both are required to answer the professor's central question.

---

## 8. Out of scope

- Thai restaurant-level ranking (Wongnai has no restaurant ids) — future work.
- Fine-tuning or contrastive training of the encoder.
- LLM query decomposition / aspect fusion — cite it, don't reimplement it.
- Generated explanations; we only select existing review sentences.
- The flavour-feature layer, the Thailand Foods gazetteer and M-ABSA.
- Learned or weighted fusion.
- A polished demo app; a notebook cell is enough.
- Machine-translating RIRD into Thai.

---

## 9. Open questions

- **Final submission portal:** confirm whether a video is required and where to submit code.
- **Second Thai annotator (C5):** Ton needs to identify this person before or during C4 labelling.
- **AIT thesis template:** confirm which version/link to use for the final report (LaTeX vs Word).
- **Earlier proposal comments:** [docs/ton_5_paper.md](docs/ton_5_paper.md) lists other critiques
  (dish table too small, circular flavour vectors, Wongnai stars, M-ABSA alignment, Google Places
  licensing). If any are still unaddressed, cover them in the final report's limitations section.
- **Per-category breakdown:** `PMD.csv` has no query-category column; left out unless time permits.
