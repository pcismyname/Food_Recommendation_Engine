# Thai Taste Queries — Writing Rules and Change Log (task C1)

Last updated 2026-10-17. **Frozen 2026-10-17 (signed off by Ton). Do not edit.**

## Status

| File | What it is |
|---|---|
| [thai_queries.csv](thai_queries.csv) | Written by Ton (native Thai speaker) using the rules below. **Frozen 2026-10-17** |

> **Correction (2026-11-10, Ton):** an earlier version of this file said the queries were drafted
> and revised by an LLM and signed off by Ton without changes. That description was wrong and the
> sign-off was made by mistake: the queries in `thai_queries.csv` were written by Ton. The
> unused LLM draft file has been removed.

The queries were written **without looking at any retrieval output**, frozen, and only then used
to build the labelling pool (C3).

**Wording for the report:** "A native Thai speaker in the team wrote the queries against written
rules and froze them before any retrieval results were seen. Relevance labels are fully human."

## Rules

1. **Casual register.** Write the way someone types into Wongnai search or asks a friend
   (เปรี้ยวๆ, แซ่บๆ, กิน not ทาน). Avoid menu-style or formal written Thai.
2. **A want, not a review.** Each query asks for something. A sentence like "this food was bland"
   is a complaint, not a search.
3. **Clear enough to label.** A labeller must be able to decide 0/1/2 by reading one review
   sentence. Drop vague queries such as รสชาติกลมกล่อม on its own.
4. **Taste first, few dish names.** At most about a third of the queries name a specific dish
   (ต้มยำ, ก๋วยเตี๋ยว). Otherwise BM25 can win just by matching the dish word, and the test measures
   dish lookup instead of taste. Broad words (ของหวาน, น้ำซุป, เครื่องดื่ม) are fine.
5. **Blind to the data.** Don't copy wording from reviews and don't look at retrieval results
   while writing or editing.
6. **No undesirable tastes as wants.** Don't ask for เลี่ยน (cloying); people only mention it to
   avoid it, so it belongs in negation queries.
7. **Short.** About 2–8 words.
8. **Avoid idioms that look like negation.** Write ถูกๆ / ราคาถูก instead of ไม่แพง, so that
   `type = negation` only marks real negations.

## Metadata conventions

- **`taste_tags`**: only from this list: `sour, spicy, sweet, salty, rich, oily, bold, mild, herbal, cheap`.
  Tags list every taste the query *mentions*, including the negated one (T19 "ไม่เผ็ด" → `spicy`).
  `negation` is no longer a tag; the `type` column carries it.
- **`type`**:
  - `negation` if the query negates or limits a taste (ไม่, หวานน้อย).
  - Otherwise `combo` if it has two or more tags.
  - Otherwise `single`.
- `rich` = thick, concentrated, creamy (เข้มข้น). `oily` = greasy, fried (มัน, เลี่ยน). `bold` = strongly
  seasoned overall (รสจัด).

Balance: 11 single, 8 combo, 5 negation. Five queries name a specific dish (T11, T14,
T15, T21, T23).

## Checklist for Ton

- [x] Each query sounds like something a real person would type. Reword freely
- [x] Each query is clear enough to label 0/1/2 from one sentence
- [x] The tags and type match the Thai
- [x] Didn't open the notebook outputs while reviewing
- [x] Sign-off: **Ton, 2026-10-17**. File frozen
