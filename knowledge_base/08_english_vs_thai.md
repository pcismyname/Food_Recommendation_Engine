# 08 · English vs. Thai: What We Test Where

> **Why this matters for our project**
> The professor's final question has two halves, and we answer them on two different datasets.
> This file puts the English (RIRD) and Thai (Wongnai) tracks side by side so it is clear what each
> one tests, what each one can claim, and how they connect.

Numbers are as of 2026-10-17.

---

## 1. The two halves of the professor's question

> Do retrieved review sentences genuinely support the system's restaurant rankings, and can the
> same approach work on real Thai restaurant reviews with independently annotated taste preferences?

| Half | Plain words | Answered on | Tasks |
|---|---|---|---|
| **1. Support** | When we show a sentence as the reason for a restaurant, is it a real reason? | 🇬🇧 English (RIRD) | A (ranking), B (explanations) |
| **2. Thai** | Does the same search method work on Thai reviews, judged by Thai speakers? | 🇹🇭 Thai (Wongnai) | C |

---

## 2. Side by side

| | 🇬🇧 English (RIRD) | 🇹🇭 Thai (Wongnai) |
|---|---|---|
| Reviews | 29,168 reviews of 50 Toronto restaurants (Yelp) | 40,000 reviews; 2,000-review subsample → 19,294 sentences |
| Restaurant ids? | ✅ Yes | ❌ No |
| Queries | 100, written by the RIRD authors (e.g. *"I am on a budget"*) | 24, ours, frozen 2026-10-17 (e.g. *อยากกินอะไรเปรี้ยวๆ*) |
| Query types | Broad restaurant needs (budget, friends, late night…) | Taste: 11 single, 8 combo, 5 negation |
| Labels already exist? | ✅ Yes, 5 annotators per query × restaurant | ❌ No, we create them (C4, C5) |
| What a label says | "Restaurant R fits query Q" (yes/no; vote count 0–5) | "Sentence S fits query Q" (0/1/2) |
| Who labels | RIRD's annotators (independent of us) | Ton, plus a second Thai speaker on a subset |
| What the system scores | Reviews now; sentences next (A1) | Sentences |
| What gets evaluated | **Restaurant ranking** | **Sentence ranking** |
| Models | BM25 vs. dense (`paraphrase-multilingual-mpnet-base-v2`) | Same: BM25 vs. the same dense model |
| Aggregation to restaurant | max, mean, top-3/5/10 mean | None (no restaurants) |
| Metrics | nDCG@10, MAP, P@10, with bootstrap CIs and paired tests | nDCG@10, P@10 on the labelled pool |
| Explanations evaluated? | ✅ B1 deletion test, B2 human 0/1/2 rating | ❌ Not planned |
| Owner | Sam | Ton |

---

## 3. Which pipeline step is tested where

```
query → ① find sentences → ② rank restaurants → ③ show sentences as the reason
```

| Step | 🇬🇧 English | 🇹🇭 Thai | Why |
|---|---|---|---|
| ① Find matching sentences/reviews | ✅ A | ✅ C | Both datasets have review text |
| ② Rank restaurants | ✅ A | ❌ | Wongnai has no restaurant ids |
| ③ Sentences really explain the ranking? | ✅ B | ❌ | No ranking in Thai to explain |

**Analogy:** English tests the whole car. Thai tests only the engine, because there is no Thai car
to put it in.

---

## 4. Real examples

### 🇬🇧 English: query *"I am on a budget"* (dense, top-10 mean, review level)

| Rank | Restaurant | Sentence shown as the reason | RIRD label |
|---|---|---|---|
| 1 | Fishman Lobster Clubhouse | *"It's not bad but kinda pricey."* | ❌ 0/5 votes |
| 2 | Dragon Boat Fusion | *"Good dim sum, pretty standard prices/service."* | ✅ 5/5 |
| 3 | Buk Chang Dong Soon Tofu | *"Student friendly budget…"* | ✅ 5/5 |

Rank 1 matches the *topic* (price) but not the *meaning*. "Pricey" argues against the ranking. This
is the kind of error B1 and B2 measure.

### 🇹🇭 Thai: query T02 *อยากกินอะไรเผ็ดๆ แบบจัดๆ* ("I want something really spicy")

| Rank | BM25 (matches words) | Dense (matches meaning) |
|---|---|---|
| 1 | อยากกินปูเน้นๆเนื้อๆ เลยจัดเต็ม | ถ้าชอบเผ็ดสั่งเพิ่มได้เลย |
| 2 | อยากกินอะไรเบาๆและชิมชานมเย็น | สำหรับคนชอบเผ็ดๆเลย |

BM25 matched the words อยากกินอะไร and จัด but not "spicy"; dense found spicy sentences. These are
eyeballed examples only. The real judgement comes from the 0/1/2 labels (C4, C5).

---

## 5. What we can and can't conclude

| Statement | Allowed? | Why |
|---|---|---|
| "The full method (find, rank, explain) is measured in English." | ✅ | RIRD has restaurant labels; B evaluates explanations |
| "The sentence-finding step is measured in Thai with independent human labels." | ✅ | C4/C5 labels |
| "Dense beats BM25 in English." | ✅ | Already shown, significant under every aggregation |
| "Dense beats BM25 in Thai too, so the pattern transfers." | ✅ only after C7 | Needs the Thai labels first |
| "Thai scores are higher/lower than English scores." | ❌ | Restaurant-level vs. sentence-level nDCG measure different things |
| "The system ranks Thai restaurants well." | ❌ | Never tested; no restaurant ids |
| "Thai explanations are faithful." | ❌ | B is English only |

**Rule of thumb:** compare BM25 vs. dense *within* a language, never one language's numbers against
the other's.

---

## 6. How the two tracks connect

| Link | Status |
|---|---|
| Same models (BM25 + the same dense encoder) | ✅ Done |
| Same comparison (BM25 vs. dense) | ✅ Done |
| Same scoring unit (sentence) | ⏳ Needs A1, the English sentence-level rerun. Until then English scores reviews and Thai scores sentences, so the bridge is weaker |
| Shared annotation scale (0/1/2) and guidelines | ⏳ E1 covers both the Thai sheet and the English explanation sheet |

After A1, the report can say: *"The identical sentence-retrieval step is evaluated in both
languages. In English we also show it leads to good restaurant rankings; in Thai we can test the
step itself."*

---

## 7. Status

| | 🇬🇧 English | 🇹🇭 Thai |
|---|---|---|
| Done | Review-level BM25 vs. dense with CIs and significance tests | 24 queries frozen; sentence pipeline built; pool of 465 pairs ready |
| Next (Phase 1) | A3 volume control; A1/A2 sentence-level rerun | C2 segmentation spot-check; one example for the slides |
| Phase 2 | B1 deletion test; B2 human support rating | C3 blind sheet, C4/C5 labels, C7 Thai nDCG@10 |
| Final answer will be | "X% of shown sentences support the ranking; deleting them drops rank by Y vs. Z for random" | "On 24 labelled Thai queries, dense = X vs. BM25 = Y nDCG@10 at sentence level" |
