# Thai Taste Queries — Writing Rules and Change Log (task C1)

Last updated 2026-10-17.

## Status

| File | What it is |
|---|---|
| [thai_queries_ai_draft.csv](thai_queries_ai_draft.csv) | Original LLM draft, kept unchanged as a record |
| [thai_queries.csv](thai_queries.csv) | Revised by an LLM (Claude) using the rules below. **Not final**: Ton must review it before it is frozen |

The revision was made **without looking at any retrieval output** (the notebook results were not
read). Ton should also review it without opening the outputs of `thai_retrieval.ipynb`.

Once Ton has signed off, freeze the file (no edits after that), rerun `thai_retrieval.ipynb`, and only
then build the labelling pool (C3).

**Wording for the report:** "Queries were drafted with an LLM from a list of taste categories,
revised against written rules, then checked and corrected by a native Thai speaker before any
retrieval results were seen. Relevance labels are fully human."

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

Balance after revision: 11 single, 8 combo, 5 negation. Five queries name a specific dish (T11, T14,
T15, T21, T23).

## Change log (AI draft → revised)

| qid | AI draft | Revised | Reason |
|---|---|---|---|
| T01 | อยากกินอะไรเปรี้ยวๆ | (unchanged) | Already casual |
| T02 | อาหารรสเผ็ดจัด | อยากกินอะไรเผ็ดๆ แบบจัดๆ | Rule 1 |
| T03 | ขนมหวานที่หวานกำลังดี | ของหวานหวานกำลังดี | Rule 1, shorter |
| T04 | ซุปน้ำข้นรสเข้มข้น | น้ำซุปข้นๆ เข้มข้น | Rule 1 |
| T05 | เมนูมันๆ เลี่ยนๆ แบบของทอด | อยากกินของทอดมันๆ | Rule 6; tag `rich` → `oily` |
| T06 | ร้านราคาถูกแต่อร่อย | ร้านถูกๆ แต่อร่อย | Rule 1 |
| T07 | อาหารรสจัดจ้าน | อยากกินอะไรรสจัดๆ | Rule 1 |
| T08 | รสชาติกลมกล่อม | อาหารรสเค็มนำ | Rule 3 (too vague); fills the missing single `salty` query |
| T09 | น้ำซุปหอมกลิ่นสมุนไพร | น้ำซุปหอมสมุนไพร | Shorter |
| T10 | อาหารรสอ่อน ทานง่าย | อาหารรสอ่อนๆ กินง่าย | Rule 1 |
| T11 | ต้มยำเปรี้ยวเผ็ด | ต้มยำเปรี้ยวๆ เผ็ดๆ | Rule 1 (dish kept, within quota) |
| T12 | ส้มตำแซ่บๆ เปรี้ยวเผ็ดเค็ม | อะไรแซ่บๆ เปรี้ยวเผ็ดเค็ม | Rule 4: dish removed, taste-only |
| T13 | ของหวานไม่แพง | ของหวานถูกๆ | Rule 8 |
| T14 | แกงกะทิรสเข้มข้นหวานมัน | แกงกะทิเข้มข้น หวานมัน | Rule 1 (dish kept) |
| T15 | ยำรสจัดเปรี้ยวหวาน | ยำรสจัด เปรี้ยวหวาน | Tags fixed: `spicy` → `bold` to match รสจัด; English fixed |
| T16 | อาหารเผ็ดแต่ราคาไม่แพง | ร้านอาหารเผ็ดๆ ราคาถูก | Rules 1, 8 |
| T17 | เครื่องดื่มหวานน้อย สดชื่น | น้ำผลไม้เปรี้ยวหวาน สดชื่น | Tag `fresh` not in the list; "less sweet" duplicated T20 |
| T18 | บุฟเฟต์คุ้มราคา | บุฟเฟต์คุ้มๆ | Type `combo` → `single` (one tag) |
| T19 | อาหารไม่เผ็ดเหมาะกับเด็ก | อาหารไม่เผ็ด เด็กกินได้ | Rule 1 |
| T20 | ขนมที่ไม่หวานเกินไป | ของหวานที่ไม่หวานมาก | Rule 1 |
| T21 | ก๋วยเตี๋ยวน้ำซุปไม่เค็ม | (unchanged) | Already clear (dish kept) |
| T22 | อาหารไม่มันไม่เลี่ยน | อาหารไม่มัน ไม่เลี่ยน | Spacing; tag `rich` → `oily` |
| T23 | กาแฟเข้มข้นไม่เปรี้ยว | กาแฟเข้มๆ ไม่เปรี้ยว | Rule 1 (dish kept) |
| T24 | อาหารรสชาติจืด ไม่ค่อยมีรสชาติ | อยากกินอะไรหวานๆ เค็มๆ | Rule 2 (complaint, not a want); `mild` already covered by T10 |

## Checklist for Ton

- [ ] Each query sounds like something a real person would type. Reword freely
- [ ] Each query is clear enough to label 0/1/2 from one sentence
- [ ] The tags and type match the Thai
- [ ] Didn't open the notebook outputs while reviewing
- [ ] Sign-off: name + date → file frozen
