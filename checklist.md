# ✅ Project Completion Checklist — Explainable Food Recommendation Engine
**Group Flower · AIT DSAI 2026 · Deadline: December 1**

> Use this checklist to verify the project is complete, reproducible, and submission-ready.
> Status: `[ ]` = not done · `[/]` = in progress · `[x]` = done

---

## 1. 📊 Research Tracks — Results & Analysis

### Track A — Fair Fusion Study (English / RIRD)
- [x] **A1/A2** — BM25 vs. Dense at review level (`baseline_models_v2.ipynb`) with bootstrap CIs and Holm-corrected paired tests
- [x] **A1/A2** — BM25 vs. Dense at sentence level (`baseline_models_v2_sentence.ipynb`)
- [x] **A3** — Volume control: every restaurant sampled to 400 reviews; Pearson correlation near zero
- [x] **A4** — Per-query ranking and evidence exported (50k ranking rows + 500k evidence rows)
- [x] **A5** — Paired review-vs-sentence significance test (`a5_review_vs_sentence_test.ipynb`), forest plot saved
- [x] `baselines/cache/sentence/evidence_sentence.csv` exists (the main pre-computed result file)
- [x] `baselines/cache/sentence/emb_paraphrase-multilingual-mpnet-base-v2_sentence_len256.npy` exists (~740 MB, for live inference)

### Track B — Explanation Evaluation (English / RIRD)
- [x] **B1** — Faithfulness automatic deletion test (`b1_faithfulness.ipynb`) executed; result: gap = 0, p = 1.0
- [x] **B2** — Human support labelling: Sam's `b2_support_sheet.csv` — all 375 rows labelled ✅
- [x] **B2** — `python3 baselines/b2_score_agreement.py` run: κ = 0.544 (moderate), 60.5% supporting, 22.4% clearly supporting ✅
- [x] **B2** — Ton's 40-pair `b2_support_sheet_ann2.csv` labelled; agreement: 70.0% exact, κ = 0.544 ✅
- [x] **B3** — Improved retrieval notebook (`b3_improved_retrieval.ipynb`) executed; selective CE logic integrated into `app/retrieval.py`
- [ ] **B3f** — Align app's `GROQ_SYSTEM_PROMPT` with the notebook's (the app is missing the *"healthy option"* example)
- [x] **B3a–e** — Groq rewriting, cross-encoder re-ranking, `MIN_SENTENCE_WORDS=10` filter, speed optimisation (5× fewer CE calls), selective CE all implemented

### Track C — Thai Taste Test (Wongnai)
- [x] **C1** — 24 frozen Thai taste queries in `thai/thai_queries.csv`
- [x] **C2** — PyThaiNLP sentence pipeline on 2,000-review subsample
- [x] **C3** — 10k Colab notebook run (`thai_retrieval_10k_colab.ipynb`); output: `thai/results/topk_n10000_s42.csv`
- [x] **C4** — All 452 label pairs labelled by Ton (184×0, 180×1, 88×2) in `thai/labels/label_sheet.csv`
- [x] **C5** — 60-pair consistency check: κ = 0.837 (weighted 0.869, 90% exact); `thai/results/thai_agreement.csv`
- [x] **C7** — Thai nDCG@10 (BM25 0.654 vs. Dense 0.549) and P@10 scores in `thai/results/thai_scores_summary.csv`

---

## 2. 🖥️ Demo App — Backend & Frontend

### Backend (FastAPI)
- [x] `app/main.py` — All 4 endpoints implemented: `GET /`, `GET /api/queries`, `POST /api/query`, `GET /api/health`
- [x] `app/retrieval.py` — Pre-computed English lookup (100 RIRD queries), pre-computed Thai lookup (24 queries), live B3 English pipeline
- [x] Thai HTTP 500 bug fixed (PR #1 merged); Thai tab returns `ThaiSentenceResult` schema
- [x] Groq key present in `.env` as `groq = "..."` ← **key is set**
- [ ] **Manually test the app locally** — run `python3 -m uvicorn app.main:app --port 8000`, confirm startup log says:
  - `Loaded 50 restaurant names`
  - `Loaded 100 English pre-computed queries`
  - `Loaded 24 Thai pre-computed queries`
  - `Live inference: yes (bi-encoder + cross-encoder)`
  - `Query rewriting: yes (Groq)`
- [ ] **Test a pre-computed English query** — open `http://localhost:8000`, select any chip → ⚡ badge, results appear
- [ ] **Test a new live English query** — type *"quiet place with no wait"* → 🧠 Live B3 badge, rewritten query shown, ~2–4s latency
- [ ] **Test a Thai query** — switch to Thai tab, select any chip → Thai sentence results appear with relevance labels

### Frontend (HTML/JS)
- [x] `app/static/index.html` — Dark-mode SPA with English + Thai tabs, query chips, restaurant cards, explanation sentences
- [ ] **Browser smoke-test** — verify score bars don't break on negative CE scores, status bar reappears after search, "Query rewritten: …" banner shows
- [ ] **Check UI for negative CE scores** — confirm raw logits (e.g., −8.78) display correctly vs. cosine 0–1 scores (or are hidden below `CE_DISPLAY_THRESH`)

---

## 3. 🧪 Tests & Engineering Validation

- [x] `tests/test_english_app.py` — 17 tests (health, pre-computed, live-path caching, negation routing, schema, blank/oversized queries)
- [x] `tests/test_thai_app.py` — 8 tests (Thai queries, schema, error handling)
- [x] 25/25 tests passing (as of 2026-11-04)
- [ ] **Re-run the test suite** (install pytest in venv first):
  ```bash
  cd Food_Recommendation_Engine
  source venv/bin/activate    # or: pip install pytest
  python -m pytest -q tests/
  ```
  → All 25 tests must pass before submission
- [ ] **H2** — Confirm all 50 restaurants appear in sentence metadata, explanations contain ≤ 5 sentences
- [ ] **H3** — B3 subgroup analysis: nDCG@10/MAP for negation queries vs. plain positive queries separately (justifies selective CE)
- [ ] **H5** — Robustness: test with no Groq key, Groq timeout, missing `.npy` embeddings, malformed CSV rows
- [ ] **H4** — Measure and record: cold-start time, pre-computed latency (<1ms), live query p50/p95, peak memory

---

## 4. 🐳 Docker & Deployment

### Local Docker
- [ ] **G3** — `docker compose up --build` completes without errors
- [ ] Verify both English and Thai tabs work in the Docker container
- [ ] `GET /api/health` returns `{"status": "ok", "engine_ready": true}` in Docker

### EC2 Deployment (G4)
- [x] EC2 deployment complete ✅ (confirmed by user)
- [ ] SSH access confirmed
- [ ] Docker installed on EC2 instance
- [ ] Project files `rsync`'d or `git clone`'d to EC2
- [ ] `.env` file with Groq key copied to EC2 (do NOT commit this to git)
- [ ] `docker build -t food-recommender .` completes on EC2
- [ ] `docker run -p 8000:8000 food-recommender` starts successfully
- [ ] Port 8000 opened in EC2 security group inbound rules
- [ ] App accessible at `http://<EC2-public-IP>:8000`
- [ ] `/api/health` returns `ok` from outside EC2
- [ ] Both English and Thai tabs tested on live EC2 URL

### Live Inference on EC2 (G5 — Optional)
- [ ] EC2 instance has ≥ 8 GB RAM (t2.large or larger)
- [ ] Sentence embeddings `.npy` file (~750 MB) built locally and `rsync`'d to EC2
- [ ] `COPY` line in `Dockerfile` for the `.npy` file uncommented
- [ ] Live English queries (not pre-computed) work on EC2

---

## 5. 📝 Final Report (F1) — AIT Thesis Template

- [ ] Report is in the **AIT thesis template** format (LaTeX `report/final/main.tex` → compile to PDF)
- [ ] **Section 01 — Introduction**: problem motivation, overview, contributions stated clearly
- [ ] **Section 02 — Problem Statement**: formal RIR definition, research questions (Track A, B, C)
- [ ] **Section 03 — Related Work**: 3 professor-assigned papers cited (Korikov 2024, Zhang EMNLP 2025, Hasan ACM Survey)
- [ ] **Section 04 — Datasets**: RIRD + Wongnai described; Thailand Foods & M-ABSA noted as out-of-scope
- [ ] **Section 05 — Methodology**: full pipeline documented (EDA → preprocessing → BM25 vs. Dense → aggregation → B3 improvements)
- [ ] **Section 06 — Evaluation**: all results tables populated:
  - [ ] A2: nDCG@10/MAP review-level and sentence-level with 95% CIs
  - [ ] A3: volume control correlation table
  - [ ] A5: forest plot included, review vs. sentence significance results
  - [ ] B1: faithfulness gap = 0, deletion test interpretation
  - [ ] B2: % supporting, Cohen's kappa (Sam vs. Ton annotation)  ← **run `b2_score_agreement.py` first**
  - [ ] B3: nDCG@10 with/without CE, improved/degraded query counts (44 improved, 53 degraded)
  - [ ] C7: Thai BM25 0.654 vs. Dense 0.549, P@10, Holm p=0.57 (n.s.)
  - [ ] C5: annotator consistency κ = 0.837
- [ ] **Section 07 — Discussion**: comparison vs. RIRD fine-tuned (MAP 0.626); why dense outperforms BM25 in English; why BM25 leads on Thai; negation blindness analysis; limitations
- [ ] **Section 08 — Conclusion**: contributions summary, limitations, future work (Thai restaurant-level ranking, fine-tuning, etc.)
- [ ] **F4** — Baseline comparison write-up: v1 (BM25 concat docs) vs. proposed (Dense sentence top-10 mean) vs. RIRD fine-tuned
- [ ] **D2** — Title, abstract, and claims match actual scope: *"Explainable reviewed-item retrieval, with a Thai taste case study"*
- [ ] All figures properly cited; all datasets cited with original sources
- [ ] Report compiles to PDF with no errors (run `pdflatex main.tex` or use Overleaf)
- [ ] Final report PDF saved and ready for submission

---

## 6. 📊 Presentation Slides (F2)

- [ ] Slides cover the required 4 bullets (5–8 from the brief):
  - [ ] **5 — Methodology**: pipeline diagram, EDA highlights, BM25 vs. dense, B3 improvements
  - [ ] **6 — Evaluation Results**: key results table (Dense top-10 mean: nDCG@10 0.682, MAP 0.618), B1/B2/C7 numbers
  - [ ] **7 — Discussion**: vs. RIRD fine-tuned (MAP 0.626), why CE helps/hurts, Thai BM25 vs. dense contrast
  - [ ] **8 — Conclusions**: contributions, limitations, future work
- [ ] At least one qualitative Thai example (a Thai query and its top sentences)
- [ ] Presentation stays within 10-minute time limit (rehearsed)
- [ ] Slides PDF saved and ready for submission

---

## 7. 🗂️ Repository Cleanup (F3)

- [ ] All notebooks fully executed with **saved outputs** (not empty cells):
  - [ ] `baselines/baseline_models_v2.ipynb`
  - [ ] `baselines/baseline_models_v2_sentence.ipynb`
  - [ ] `baselines/a5_review_vs_sentence_test.ipynb`
  - [ ] `baselines/b1_faithfulness.ipynb`
  - [ ] `baselines/b3_improved_retrieval.ipynb`
  - [ ] `thai/thai_retrieval_10k_colab.ipynb` (run on Colab; output committed)
  - [ ] `eda/eda_rir_data.ipynb`
  - [ ] `eda/eda_thai_datasets.ipynb`
- [ ] Dead/scratch code removed from all notebooks
- [ ] `requirements.txt` and `app/requirements.txt` verified — `pip install -r requirements.txt` succeeds cleanly
- [ ] Top-level `README.md` updated: how to reproduce every result table from scratch (clear numbered steps)
- [ ] `.gitignore` confirmed: `.env`, `venv/`, `.npy` embeddings, raw datasets are **not** in git
- [ ] No API keys committed anywhere (Groq key only in `.env`)
- [ ] `git log` is clean — no large binary files accidentally committed

---

## 8. 🎥 Video (F5 — if required)

- [ ] Confirm with professor / submission portal whether a video is required
- [ ] If yes: screen-capture walkthrough covering:
  - [ ] Running the app locally (or EC2 URL)
  - [ ] Demo of a pre-computed English query
  - [ ] Demo of a new live English query with Groq rewrite
  - [ ] Demo of a Thai query
  - [ ] Brief tour of a key notebook result (e.g., nDCG@10 results table)
- [ ] Video exported as MP4, compressed to a reasonable size

---

## 9. 🚦 Final Submission Checklist

| Item | Status | Notes |
|---|---|---|
| Final report PDF | [ ] | AIT thesis template, compiled |
| Slides PDF | [ ] | 10 min, bullets 5–8 |
| Code / repository | [ ] | All notebooks executed, clean |
| Video (if required) | [ ] | Confirm requirement |
| EC2 / deployment URL | [ ] | Shared with professor |

> [!IMPORTANT]
> **Submission deadline: December 1.** Confirm the exact submission portal and whether a video is required.

---

## 10. 🔥 Outstanding High-Priority Items (Do These First)

> These are the items most likely to be incomplete based on the current project state:

1. ~~**B2 → Run `python3 baselines/b2_score_agreement.py`**~~ ✅ Done — κ = 0.544, 60.5% supporting
2. ~~**G4 → EC2 deployment**~~ ✅ Done
3. **H3 → B3 subgroup analysis** — Report nDCG@10/MAP for negation vs. plain positive queries separately.
4. **G2 → Browser smoke-test** — Visual check of UI edge cases (negative CE scores, status bar, Thai labels).
5. **F1 → Compile final report** — All report sections complete; compile PDF (`pdflatex` or Overleaf).
6. **F2 → Finalize slides** — See PPT notes below; B2 numbers need updating in the explanation slide.
7. **F3 → Repository cleanup** — Execute all notebooks with saved outputs; verify `requirements.txt`.
