# Action Items — Group Flower

Last updated 2026-10-17.

**Next deadline: Monday 2026-10-19, 11:59 PM** — updated report (PDF) and slides (PDF).
Progress presentation on Tuesday 2026-10-20: 10 minutes plus 5 minutes Q&A.

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
| F2 | BM25 and dense use different units; fusion biased by review volume | v1 BM25 scored one concatenated document per restaurant, dense scored mean-pooled reviews. Max / top-k fusion favours restaurants with more reviews (406 to 2,281 reviews per restaurant in RIRD). | A1–A4 |
| F3 | No evidence for explainability | A top-scoring sentence can be relevant without being a complete or faithful explanation. Explanation quality has not been evaluated. | B1, B2 |
| F4 | Check more literature | Read and cite the three papers below. | D1 |

Papers the professor pointed to:

- Korikov et al., *Multi-Aspect Reviewed-Item Retrieval via LLM Query Decomposition and Aspect Fusion* — [arXiv:2408.00878](https://arxiv.org/abs/2408.00878). Shows top-K late fusion is sensitive to how aspects are spread across reviews; directly relevant to F2.
- Zhang et al., *Enhancing Recommendation Explanations through User-Centric Refinement*, Findings of EMNLP 2025 — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.434/). Criteria for judging explanations; relevant to F3.
- Hasan et al., *Review-based Recommender Systems: A Survey of Approaches, Challenges and Future Perspectives*, ACM Computing Surveys — [doi:10.1145/3742421](https://doi.org/10.1145/3742421). Background and positioning.

**The question the final project must answer:**

> Do retrieved review sentences genuinely support the system's restaurant rankings, and can the
> same approach work on real Thai restaurant reviews with independently annotated taste preferences?

- First half → B1 (faithfulness) and B2 (human support rating).
- Second half → C1–C7 (Thai taste test set). Expect a partial answer: sentence level on Thai, not restaurant level.

### 1.2 Requirements for the progress submission (due 2026-10-19)

**Presentation (10 min) must cover:**

1. Project overview — problem, objectives, datasets, methodology (brief).
2. Responses to previous feedback — how each comment above was addressed.
3. Progress and preliminary results — experiments done, findings, challenges.
4. Next steps — remaining tasks and plan.

**Updated report must:**

- Include a **dedicated feedback summary page** addressing each comment: action taken, revision made, or reason for not implementing it.
- Update methodology, experimental results and remaining work.

**Grading notes:** worth 5%. Show meaningful progress, not a repeat of the proposal. Preliminary
results must be backed by experiments and analysis. Negative or inconclusive findings are fine if
properly investigated and explained.

---

## 2. Progress so far

[baselines/baseline_models_v2.ipynb](baselines/baseline_models_v2.ipynb) has been run at **review level**:

- BM25 and dense now score the **same unit** (review) with the **same aggregation** (max, mean, top-3/5/10 mean). This addresses the unit half of F2.
- Added bootstrap 95% CIs, paired randomization tests with Holm correction, and three query-independent floors (random, review count, popularity).
- **Dense beats BM25 under every aggregation**, all significant after Holm correction (nDCG@10 gap +0.03 to +0.09).
- **Best model:** Dense, top-10 mean — nDCG@10 0.668 [0.625, 0.711], MAP 0.608. The RIRD paper's fine-tuned best is MAP 0.626, so off-the-shelf dense is already close.
- **Top-k mean beats max and mean** for both models; max is the weakest dense variant.
- All models beat the popularity floor except BM25 mean (not significant).
- **Not yet done:** the sentence-level run and the review-volume control, so the volume half of F2 is still open.
- **A3 implementation prepared:** [baselines/baseline_models_v2.ipynb](baselines/baseline_models_v2.ipynb)
  now tracks source review rows, samples exactly 400 reviews per restaurant with `SEED = 42`, reuses
  the existing score matrices/cached embeddings, and saves A3 metrics and score/count correlations.
  The experiment is not marked complete until it is run and the outputs are inspected.
- **A4 implementation prepared:** the same notebook now retains the shared-unit BM25/dense runs and
  exports reusable restaurant rankings plus top evidence units for each query, model and aggregation.
  The export is not marked complete until the notebook has been run and the files have been checked.

---

## 3. Tasks

Status: `[ ]` not started, `[/]` in progress, `[x]` done.
**Thai?** marks tasks that need someone who reads Thai.
**Phase:** 1 = due for the progress submission (Monday 2026-10-19), 2 = after the presentation.
**Priority:** High / Medium / Low within that phase.

### Owners

| Area | Owner | Focus until Monday |
|---|---|---|
| A. Fair fusion study (English) | **Sam** | **Main focus.** A3 first, then the sentence-level rerun |
| B. Explanation evaluation (English) | **Sam** | Not before Monday; starts in Phase 2 |
| C. Thai taste test set | **Ton** | C1 and a first Thai example; continues in Phase 2 |
| D. Writing, E. Guidelines | Not assigned yet | D1, D3, D4 are due Monday |

### A. Fair fusion study on RIRD (English) — answers F2 · Owner: Sam

| ID | Task | Phase · Priority | Thai? | Depends on |
|---|---|---|---|---|
| A3 | `[/]` **Code prepared; execution pending.** Subsample every restaurant to 400 reviews with a fixed seed, reuse cached embeddings, rerun every aggregation, and report score/count correlations plus controlled metrics | **1 · High** (do first) | No | Can run now on the review-level cache |
| A1 | `[/]` Shared retrieval unit for BM25 and dense. Done at review level in v2; sentence splitting is coded but not yet run (`UNIT = 'sentence'`) | **1 · High** | No | — |
| A2 | `[/]` Same fusion variants on both models, with CIs and significance tests. Done at review level; rerun at sentence level | **1 · High** | No | A1 |
| A4 | `[/]` **Code prepared; execution pending.** Save per-query restaurant rankings and the top-10 selected source units, including source row IDs and text, so A3, B1 and B2 can reuse them | 1 · Medium | No | — |

If the sentence-level run is not finished by Monday, present it as a next step; A3 on review level is the must-have.

### B. Explanation evaluation on RIRD (English) — answers F3 · Owner: Sam

| ID | Task | Phase · Priority | Thai? | Depends on |
|---|---|---|---|---|
| B1 | `[ ]` Faithfulness (automatic): delete the top-k shown sentences, re-score, compare the rank drop against deleting k random sentences | 2 · High | No | A2 at sentence level |
| B2 | `[ ]` Support (human): export shown sentences for ~15–30 queries × top 3 restaurants; two people label 0/1/2 separately; report kappa and % rated as supporting. Sam prepares the sheet; Sam and Ton both label | 2 · High | No | A2 at sentence level, E1 |

### C. Thai taste test set on Wongnai — answers F1 · Owner: Ton

| ID | Task | Phase · Priority | Thai? | Depends on |
|---|---|---|---|---|
| C1 | `[x]` Write 15–30 Thai taste queries (sour, spicy, sweet, rich, cheap, combinations, a few negations). 24 queries in [thai/thai_queries.csv](thai/thai_queries.csv): drafted by an LLM, then revised by an LLM against the rules in [thai/query_guidelines.md](thai/query_guidelines.md). The original draft is kept in [thai/thai_queries_ai_draft.csv](thai/thai_queries_ai_draft.csv) | **1 · High** | **Yes** | — |
| C1b | `[x]` Ton reviews and rewords the revised queries in natural Thai using the checklist in [thai/query_guidelines.md](thai/query_guidelines.md), **without opening the notebook outputs**, then signs off and freezes the file. After that, rerun [thai/thai_retrieval.ipynb](thai/thai_retrieval.ipynb). **Signed off by Ton 2026-10-17 with no changes; queries frozen; notebook rerun on the frozen queries** | **1 · High** | **Yes** | C1 |
| C2 | `[x]` Split Wongnai reviews into sentences with PyThaiNLP, build a Thai BM25 index, encode and cache sentences. Spot-check the segmentation. For Monday, one qualitative Thai query → top sentences example is enough. Pipeline built and run on a 2,000-review subsample (19,294 sentences, [thai/thai_retrieval.ipynb](thai/thai_retrieval.ipynb)). Spot-check done (§2a–2b): good enough for retrieval; ~0.7% of sentences cut at abbreviation dots, ~1.6% merged long sentences. Qualitative T02/T11/T20 notes and takeaways written (§6a, §8). Notes drafted by Claude, confirmed by Ton 2026-10-17 | 1 · Medium | Code: no. Spot-check: **yes** | — |
| C3 | `[ ]` For each query, pool the top sentences from BM25 and dense into one shuffled sheet that doesn't show which system found each sentence | 2 · High | No | C1b, C2 |
| C4 | `[ ]` Hand-label the sampled pairs 0/1/2 | 2 · High | **Yes** | C3, E1 |
| C5 | `[ ]` Second opinion: a second Thai speaker labels 50–80 of the same pairs. Fallback: re-label 50 of your own pairs at least a week later. Start looking for a second Thai speaker now | 2 · Medium | **Yes** | C4 |
| C6 | `[ ]` *Optional:* two LLMs from different providers label the full pool. Skip if the pool is small enough to label by hand | 2 · Low | No (but checking their output: **yes**) | C3, E1 |
| C7 | `[ ]` Agreement figures, then Thai nDCG@10 / P@10 for BM25 vs. dense | 2 · High | No | C4, C5 (C6 if used) |

### D. Writing · Owner: not assigned yet

| ID | Task | Phase · Priority | Thai? | Depends on |
|---|---|---|---|---|
| D3 | `[ ]` **Feedback summary page** for the progress report (one row per comment in 1.1: action taken / revision / reason not done) | **1 · High** | No | — |
| D4 | `[ ]` Progress report PDF and slides PDF, following the four sections in 1.2 | **1 · High** | No | D3, plus whatever of A is finished |
| D1 | `[ ]` Related work: add the three papers in 1.1 | **1 · High** | No | — |
| D2 | `[ ]` Rewrite title, claims and limitations. Working title: "Explainable reviewed-item retrieval, with a Thai taste case study" | Draft: 1 · Medium. Final: 2 · High | No | Final after A3, B2, C7 |
| D5 | `[ ]` Final report, slides and video | 2 · High | No | everything above |

### E. Shared setup · Owner: not assigned yet

| ID | Task | Phase · Priority | Thai? | Depends on |
|---|---|---|---|---|
| E1 | `[ ]` Annotation guidelines in English: 0/1/2 scale, worked examples, decided edge-case rules, for both the Thai sheet and the explanation sheet | 2 · High (blocks all labelling; finish early in Phase 2) | Thai examples: **yes** | — |

---

## 4. Plan

### Phase 1 — before the progress submission (2026-10-15 to 10-19)

Aim: show the fixed baselines, an honest answer to each comment, and a concrete plan.

| Order | Sam | Ton | Not assigned yet |
|---|---|---|---|
| 1 | Run and inspect A3 volume control (review-level cache) | ~~C1b review and freeze Thai queries~~ (done) | D3 feedback summary page |
| 2 | Implement/run A4 ranking export, then run A1/A2 sentence-level rerun (Colab GPU) | ~~C2 one qualitative Thai example~~ (done) | D1 related work |
| 3 | Summarize A3/A1/A2 findings and draft D2 claims | | D4 report and slides |

If time is short, present the sentence-level run and the Thai work as next steps; the review-level
results in section 2 plus A3 are enough to show progress.

### Phase 2 — after the presentation, to the final deadline

| Priority | Sam | Ton | Not assigned yet |
|---|---|---|---|
| High | B1, B2 (sheet + labelling with Ton) | C3, C4, C7; B2 labelling | E1 (early, blocks labelling), D2 final, D5 |
| Medium | Finish any Phase 1 leftovers in A | C5 | |
| Low | | C6 (optional) | |

---

## 5. What can run in parallel

Can start **now**, independently of each other:

- **Sam:** run/inspect A3, implement A4 ranking export, then run A1/A2 sentence-level rerun
- **Ton:** C1, C1b, C2 done; next is E1 Thai examples and finding a second annotator (C5)
- **Unassigned:** D1, D3 (writing, English)

Dependency chains:

```
A1 ──► A2 ──┬──► A3
            ├──► B1
            └──► B2 ◄── E1

C1 ──► C1b ──┐
              ├──► C3 ──┬──► C4 ──► C5 ──┐
C2 ───────────┘         │                ├──► C7
                        └──► C6 ─────────┘   (C6 optional)
          E1 ──► C4, C6

D3 + current results ──► D4 (progress, Oct 19)
A3, B1, B2, C7 ──► D2 (final) ──► D5
```

- The RIRD track (A, B) and the Thai track (C) are independent until the report.
- **A2 at sentence level is the main bottleneck** for the explanation work (B1, B2).
- **E1 blocks all labelling** (B2, C4, C6).
- **Thai tasks (C1b, C2 spot-check, C4, C5, E1 examples) all fall on Ton**, so Ton should not also
  carry the English code tasks.

---

## 6. If time runs short, cut in this order

1. Drop C6 (LLM labelling) and hand-label a pool of ≤ 300 pairs.
2. Shrink C1 to 15 queries.
3. Shrink B2 to 15 queries.
4. Drop the subsampling rerun in A3 but keep the score vs. review-count correlation.

Do not cut A2 at sentence level or B1; without them there is no answer to the professor's question.

## 7. Out of scope

- Thai restaurant-level ranking (Wongnai has no restaurant ids) — future work.
- Fine-tuning or contrastive training of the encoder.
- LLM query decomposition / aspect fusion — cite it, don't reimplement it.
- Generated explanations; we only select existing review sentences.
- The flavour-feature layer, the Thailand Foods gazetteer and M-ABSA.
- Learned or weighted fusion.
- A polished demo app; a notebook cell is enough.
- Machine-translating RIRD into Thai.

## 8. Open questions

- **Final deadline** is not confirmed yet; Phase 2 should be scheduled once it is.
- **Earlier proposal comments:** [docs/ton_5_paper.md](docs/ton_5_paper.md) lists other critiques (dish table too small, circular flavour vectors, Wongnai stars, M-ABSA alignment, Google Places licensing). If those came from the proposal presentation, the feedback page (D3) must answer them too.
- **Per-category breakdown:** `PMD.csv` has no query-category column, so it would need hand-tagging of the 100 queries. Left out unless there is time.
