# Annotation Guidelines: Thai Taste Relevance (E1)

Written 2026-10-18 for C4 (Ton), C5 (second annotator) and C6 (LLM labellers, if used).
Appendix A covers the English explanation-support sheet (B2).

## 1. Purpose

The labels are the human ground truth for the Thai case study. Each row pairs one of the 24
frozen queries in [thai_queries.csv](thai_queries.csv) with one Wongnai review sentence taken from
the BM25 and dense top-10 pool. Your 0/1/2 labels are used to compute Thai **nDCG@10 and P@10 for
BM25 vs. dense** (C7) and agreement (kappa, C5). Together they answer the professor's comment F1:
does the approach work on real Thai reviews with independently annotated taste preferences? The
sheet is **blind**: it does not show which system found a sentence, and rows are shuffled within
each query. **Do not open `pool_key.csv`, the outputs of `thai_retrieval.ipynb` or any results
file while labelling.** If a star rating is shown, ignore it, because stars are not taste labels.

## 2. The scale

Ask one question: *if I typed this query, does this sentence tell me the food, drink or place has
what I asked for?*

| Label | Meaning |
|---|---|
| **2** | The sentence clearly says the item has the queried taste or property, at the degree asked. It satisfies the **whole** query: all parts of a combination, the right kind of item, and the negation if there is one. |
| **1** | Partial. The taste is weak or hedged, only some parts of a combination are met, the right taste is described for the wrong kind of item, or a specific dish the query names appears with no taste information. |
| **0** | Not about the query, **or contradicts it** (e.g. "not spicy" for a spicy query), or the query word appears only in an unrelated sense (ถูกปาก, มันฝรั่ง) or only inside a dish name. |

You are labelling **relevance to the query, not sentiment**. A complaint can be a 2 (see R3).

## 3. Decided rules

### 3.1 Three checks per row

1. **Item.** Is the sentence about the kind of item the query asks for? (Column "Item" in 3.3.)
2. **Taste.** For each taste part of the query, is it *met*, *partial* or *contradicted*?
   Use the degree table in 3.2.
3. **Combine:**
   - all parts met and the item is right → **2**
   - any part contradicted → **0**, even if other parts are met
   - otherwise, if something relevant is met or partial → **1**
   - right taste fully met but the wrong or unstated item → **1**
   - partial taste *and* the wrong item → **0**

### 3.2 Degree table: how a taste mention counts

Take X = the taste word (เผ็ด, เค็ม, หวาน …). Find the column for what the query wants.

| What the sentence says about X | Query wants X (T01, T04, T05, T09, T11–T18, T24) | Query wants strong X (T02, T07, T08, strong part of T23) | Query wants not / less X (T20–T22, not-sour part of T23) | T19 (no spice, for kids) |
|---|---|---|---|---|
| Strong: Xมาก, Xจัด, Xไป, Xเกิน, Xมากกก, Xดดดด, Xน้ำตาไหล | met | met | contradicted | contradicted |
| Plain: X, Xๆ, รสX, ออกX | met | partial | contradicted | contradicted |
| Just right: Xกำลังดี, Xพอดี, Xลงตัว | met | partial | met | contradicted |
| Slight: Xนิดๆ, Xหน่อยๆ, อมX | met | partial | met | partial |
| Moderate: ไม่Xมาก, ไม่ค่อยX, ไม่Xเกินไป | partial | contradicted | met | partial |
| Absent: ไม่X, ไม่Xเลย, (for taste) จืด | contradicted | contradicted | met | met |

T03 and T10 have their own degree rules in 3.3.

### 3.3 Per-query rules

"Any" means any food or drink counts. **อาหาร** = savoury food or meals, so desserts and drinks
count as the wrong item (1 at most).

| qid | Query | Item | 2 when | Typical 1 | Typical 0 |
|---|---|---|---|---|---|
| T01 | อยากกินอะไรเปรี้ยวๆ | any | sour met | ไม่เปรี้ยวมาก | ไม่เปรี้ยว; sour only in a name (ส้มตำ, แกงส้ม) |
| T02 | อยากกินอะไรเผ็ดๆ แบบจัดๆ | any | strong spicy (เผ็ดมาก, เผ็ดจัด, เผ็ดจี๊ด, แซ่บจัดจ้าน) | plain/medium spicy, เผ็ดกลางๆ, แซ่บ alone | ไม่เผ็ดมาก, ไม่เผ็ดจัด |
| T03 | ของหวานหวานกำลังดี | dessert | dessert sweetness กำลังดี/พอดี/ลงตัว or ไม่หวานเกินไป | dessert plainly sweet (หวานอร่อย); ไม่หวานมาก/หวานน้อย; just-right sweetness in a drink | dessert too sweet or very sweet; not sweet/bland |
| T04 | น้ำซุปข้นๆ เข้มข้น | broth: soup, แกง, noodle broth | broth ข้น **or** เข้มข้น (one is enough) | rich sauce/dip/drink; rich flavour, item not stated; น้ำข้น only as a menu style | ไม่เข้มข้น, จืด, ใส watery |
| T05 | อยากกินของทอดมันๆ | fried food | fried item described มัน/มันๆ/อมน้ำมัน/เยิ้ม (complaints count) | crispy fried food with no oiliness; oily but not fried | ทอดมาไม่มัน; ทอดมัน (fish cake) or มันฝรั่ง as names only |
| T06 | ร้านถูกๆ แต่อร่อย | any | cheap **and** tasty | cheap only; ราคากลางๆ/สมเหตุสมผล + tasty | tasty only; expensive; ถูกปาก/ถูกใจ |
| T07 | อยากกินอะไรรสจัดๆ | any | รสจัด, จัดจ้าน, ถึงเครื่อง, รสเข้มข้น, แซ่บมาก, or any one taste at strong degree | แซ่บ alone; one plain taste (เผ็ด) | รสอ่อน, ไม่จัด, จืด, กลมกล่อม alone |
| T08 | อาหารรสเค็มนำ | อาหาร | salt leads: เค็มนำ, ออกเค็ม, หนักเค็ม, เค็มมาก/เค็มไป (complaints count) | เค็มนิดๆ; salt equal to other tastes (หวานเค็ม); เค็มพอดี | ไม่เค็ม; หวานนำ with no salt |
| T09 | น้ำซุปหอมสมุนไพร | broth | broth with herb aroma or herbs: สมุนไพร, ตะไคร้, ข่า, ใบมะกรูด, ขิง, เครื่องต้มยำ, สมุนไพรจีน | herbs on a non-broth (fried fish, herb butter); herbal drink น้ำสมุนไพร; broth หอม with no herb named | herb in a dish name only; สีสมุนไพร (herbal food colouring) |
| T10 | อาหารรสอ่อนๆ กินง่าย | อาหาร | รสอ่อน, รสไม่จัด, รสกลางๆ, รสเบาๆ, กินง่าย about taste | only ไม่เผ็ด; จืด/จืดไป (bland goes too far); mild dessert or drink | รสจัด, เผ็ด, เค็มจัด |
| T11 | ต้มยำเปรี้ยวๆ เผ็ดๆ | ต้มยำ (incl. tom yum noodles, รสต้มยำ) | tom yum sour **and** spicy (แซ่บ = spicy) | tom yum with one part; tom yum named, no taste; another soup sour+spicy | tom yum ไม่เปรี้ยว or ไม่เผ็ด |
| T12 | อะไรแซ่บๆ เปรี้ยวเผ็ดเค็ม | any | at least 2 of {sour, spicy/แซ่บ, salty}, or ครบรส | one of them, incl. แซ่บ alone | any of the three contradicted |
| T13 | ของหวานถูกๆ | dessert or dessert shop | dessert + cheap | dessert, no price; cheap non-dessert | expensive dessert |
| T14 | แกงกะทิเข้มข้น หวานมัน | coconut curry: แกงเขียวหวาน, แกงเผ็ด, พะแนง, มัสมั่น, ฉู่ฉี่, แกงคั่ว, ข้าวซอย broth | coconut curry + at least 2 of {เข้มข้น, หวาน, มัน} | curry with 1 part; curry named, no taste; non-coconut curry or ต้มข่า, or item not stated, with 2+ parts | กะทิจืด/ไม่เข้มข้น; coconut desserts (ไอศกรีมกะทิ) |
| T15 | ยำรสจัด เปรี้ยวหวาน | ยำ (any ยำ…, น้ำยำ) | yam + at least 2 of {bold/แซ่บ, sour, sweet} | yam with 1 part; yam named, no taste; ส้มตำ/ลาบ/พล่า with 2+ parts | yam ไม่เปรี้ยว/จืด |
| T16 | ร้านอาหารเผ็ดๆ ราคาถูก | any | spicy + cheap | one of them | expensive; ไม่เผ็ด |
| T17 | น้ำผลไม้เปรี้ยวหวาน สดชื่น | fruit drink: juice, fruit smoothie (ปั่น), lime/lemon drink, fruit soda | fruit drink + at least 2 of {sour, sweet, สดชื่น} | fruit drink with 1 part; fruit drink named, no taste; non-fruit drink with 2+ parts | fruit drink ไม่เปรี้ยว/จืด |
| T18 | บุฟเฟต์คุ้มๆ | buffet (บุฟเฟ่/บุฟเฟ่ต์, หัวละ…) | buffet + value: คุ้ม, คุ้มค่า, ราคาไม่แพง/เบาๆ, แค่/เพียง X บาท | buffet with variety or food only; buffet named, no value; value but not a buffet | ไม่คุ้ม; buffet expensive |
| T19 | อาหารไม่เผ็ด เด็กกินได้ | อาหาร | no spice: ไม่เผ็ด, ไม่เผ็ดเลย, ไม่ใส่พริก; kids part optional | ไม่เผ็ดมาก/ไม่เผ็ดเกินไป, เผ็ดนิดๆ; kid-friendly with no spice info | any spicy, incl. เผ็ดกำลังดี |
| T20 | ของหวานที่ไม่หวานมาก | dessert | dessert ไม่หวาน, ไม่หวานมาก, ไม่หวานเกิน, หวานน้อย, หวานนิดๆ, หวานกำลังดี | dessert plainly sweet (no degree); same phrases about a drink or unstated item | dessert หวานมาก/หวานไป/หวานเกิน |
| T21 | ก๋วยเตี๋ยวน้ำซุปไม่เค็ม | noodle-soup broth | noodle broth not/less salty (ไม่เค็ม, ไม่เค็มมาก, เค็มพอดี, รสอ่อน) | other soup not salty; noodle broth กลมกล่อม with no salt info; noodles named, no taste; not salty, item not stated | noodle broth เค็ม/เค็มไป |
| T22 | อาหารไม่มัน ไม่เลี่ยน | อาหาร | ไม่มัน, ไม่เลี่ยน, ไม่เยิ้ม, ไม่อมน้ำมัน (one is enough) | dessert/drink ไม่เลี่ยน; a side dish that "cuts the greasiness" (ช่วยลดความเลี่ยน) | มัน/เลี่ยน/อมน้ำมัน |
| T23 | กาแฟเข้มๆ ไม่เปรี้ยว | coffee | coffee strong **and** not sour | coffee strong only, or not sour only; another drink strong and not sour | coffee sour; coffee ไม่ค่อยเข้ม/อ่อน |
| T24 | อยากกินอะไรหวานๆ เค็มๆ | any | sweet + salty (หวานเค็ม, หวานๆเค็มๆ, หวานนำ เค็มนิดๆ) | one of them | ไม่หวาน or ไม่เค็ม |

### 3.4 Edge cases

- **R1 Negation.** Read every ไม่ before deciding. ไม่เผ็ด in the text means 0 for a spicy query
  and can mean 2 for T19. A ไม่ in the sentence that does not apply to the food does not count. In
  *"ลืมบอกทางร้านว่าไม่เผ็ด … จะทำรสแซ่บจัดจ้านมากกก"* the food was very spicy (T19 = 0, T02 = 2).
- **R2 Combinations.** All parts met = 2. Some parts met or partial, none contradicted = 1. Any
  part contradicted = 0. Three-taste queries (T12, T14, T15, T17) need 2 of the 3 tastes for a 2.
  Near-synonym pairs count as one part (ข้น/เข้มข้น in T04, ไม่มัน/ไม่เลี่ยน in T22).
- **R3 Complaint vs. praise.** Label whether the property is present at the degree asked, not
  whether the reviewer liked it. *"เค็มไปหน่อย"* (a bit too salty) is 2 for T08 (salt-forward) but
  0 for T21 (not salty). *"อมน้ำมัน"* (soaked in oil) is 2 for T05.
- **R4 Taste word only inside a name.** The taste must be *described*, not inferred from what
  the dish is. แกงเขียวหวาน, ตับหวาน, ทอดมัน, เนื้อเค็ม, ต้มแซ่บ, ส้มตำ, ร้าน Sweet cake, "Spicy
  salmon": no taste evidence from the name alone. A dish name with no taste words is 1 only when
  the query names that dish (T11 ต้มยำ, T14, T15, T17, T18, T21, T23). Otherwise it is 0.
  *"ยำวุ้นเส้น ต้มยำกุ้ง ฯลฯ"* = 1 for T11 and 0 for T01.
- **R5 Unrelated senses.** ถูก is cheap only as price. ถูกปาก/ถูกใจ (to one's taste), เลือกไม่ถูก
  (can't choose) and passive ถูก are 0. The pronoun มัน ("it"), มันฝรั่ง/มันบด (potato) and
  คุ้มค่าการรอคอย (worth the wait) do not count either.
- **R6 Price (T06, T13, T16, T18).** Counts as cheap: ถูก, ราคาถูก, ไม่แพง, ราคาเบาๆ, ย่อมเยา,
  สบายกระเป๋า, คุ้ม/คุ้มค่า/คุ้มราคา, แค่/เพียง X บาท. Partial: ราคากลางๆ, ราคาโอเค, สมเหตุสมผล.
  A bare price with no judgement word (*"ราคา 85 บาท"*) gives no price evidence. Contradicted: แพง,
  ราคาแรง, ไม่ถูก, ไม่คุ้ม. In T06 the price is the core, so tasty without price = 0.
- **R7 Item type.** Desserts and drinks are the wrong item for อาหาร queries (T08, T10, T19, T22),
  but count for "อะไร" queries (T01, T02, T07, T12, T24). Drinks are not desserts (T03, T13, T20).
  Sauces, dips and dressings count as part of their dish.
- **R8 Item not stated.** For a fragment such as *"รสชาติเข้มข้นมากๆ"*, judge the taste, but if the
  query needs a specific item, treat it as the wrong item (1 at most).
- **R9 Hedges, conditionals, advice.** Lower by one grade for น่าจะ, เหมือนจะ, อาจจะ, ไม่แน่ใจ, or
  ถ้าชอบเผ็ด…แนะนำ. Advice, ordering instructions or general preferences that do not describe a real
  item (*"เอาเผ็ดแค่ไหนอย่าลืมสั่งก่อน"*, *"ไม่ชอบกาแฟเปรี้ยวๆ"*) = 0. If they imply a description
  (*"สั่งเผ็ดน้อย เพราะมันเผ็ดมากจริงๆ"*), judge the implied description.
- **R10 Sarcasm.** Label the intended meaning when it is obvious to a Thai reader. For
  *"เผ็ดนิดเดียวเองค่ะ น้ำตาไหลพรากเลย 555"* (illustrative: "only a tiny bit spicy, tears
  streaming 555"), T02 = 2. If you can't tell, take it literally and write `sarcasm?` in `note`.
- **R11 Segmentation errors.** Judge only the text shown; don't guess the previous sentence. A cut
  fragment follows R8. In merged text, judge the best-matching clause. A contradiction in a clause
  about the *same* item still gives 0. A clause about a different item does not lower the label.
- **R12 Spelling, elongation, emoji, code-mixing.** เผ็ดดดด/มากกกก = strong. 555 (laughter), emoji
  and broken `????` are ignored unless they clearly change the meaning. Common variants have the
  same meaning: รสชาด = รสชาติ, แซบ/แซ๊บ = แซ่บ, กระทิ = กะทิ, เจ้มจ้น = เข้มข้น, บุฟเฟ่ = บุฟเฟต์.
  English taste words count like Thai (*"spicy มาก"* = เผ็ดมาก). English in menu or shop names
  follows R4.
- **R13 Unreadable or not a review.** Ads, opening hours, headers, or text you can't parse = 0,
  with `unreadable` in `note`.

## 4. Worked examples

All sentences below are real Wongnai sentences from `thai/cache/sentences_n2000_s42.parquet`
(`review_id:sent_idx`). Glosses are literal where needed.

**Label 2**

| Query | Sentence (Wongnai) | Gloss | Why |
|---|---|---|---|
| T02 | ใครสั่งแกงป่า กับขนมจีนระวังเรื่องความเผ็ดให้ดี เพราะเผ็ดมากกกก แต่โดยรวมแล้วรสชาติอร่อยมากครับ (1900:4) | Whoever orders the jungle curry with khanom jeen, be careful about the heat, because it's sooo spicy; overall it's very tasty | Strong spicy (elongated มากกกก). A warning, but the degree is what counts (R3) |
| T11 | น้ำต้มยำเปรี้ยวแซ่บ หอมกลิ่นพริกขี้หนู (31167:2) | The tom yum broth is sour and zesty, smelling of bird's-eye chilli | Right item; sour met; แซ่บ + chilli = spicy |
| T20 | รสชาติไม่หวานเกิน และเนื้อเค้กนุ่มมาก (32027:9) | Not overly sweet, and the cake is very soft | Dessert; ไม่หวานเกิน = met (also 2 for T03) |
| T06 | สรุปว่าถูกและอร่อยคะ เอาไว้ทานตอนรีบๆ (3177:6) | In short, cheap and tasty; good for when you're in a hurry | Both parts met |
| T22 | เผ็ดนิดหน่อย แต่ไม่มันและไม่เลี่ยน ให้ผ่านเลยสำหรับผัดไท ปอเปี๊ยะทอดเราเฉยๆ … (12119:5) | A bit spicy but not oily and not greasy, a pass for the pad thai; the fried spring rolls were so-so for us … | Merged sentence: the pad thai clause meets the query (R11) |

**Label 1**

| Query | Sentence (Wongnai) | Gloss | Why |
|---|---|---|---|
| T03 | ชาเขียวที่นี่เข้มข้นมากและหวานกำลังดี (1483:35) | The green tea here is very strong and just sweet enough | Right degree, but a drink, not a dessert (R7) |
| T11 | รสชาติต้มยำไม่เผ็ดมาก (2494:4) | The tom yum flavour is not very spicy | Right item; spicy only moderate = partial; sour not mentioned |
| T08 | หวานนำ เค็มนิดๆ (28662:22) | Sweet-forward, slightly salty | Salt present but not leading (also 2 for T24) |
| T18 | จัดกันไปเอาให้คุ้มกับราคาหัวละ 690 บาทถ้วน (18926:5) | We went all out to get our money's worth from the 690 baht per head | Buffet implied, but it doesn't say the buffet is good value |
| T14 | สั่งมาลองเพราะไม่เคยทาน รสชาติ กะทิข้นและมันมาก (1413:30) | Ordered it to try since I'd never had it; the coconut milk is thick and very rich | Two parts met, but no curry is named (R8) |

**Label 0**

| Query | Sentence (Wongnai) | Gloss | Why |
|---|---|---|---|
| T02 | คือไม่เผ็ดจัดจ้านเลย (29574:35) | It's not fiery at all | Contradicts (R1). BM25 matches เผ็ด/จัด |
| T19 | ลืมบอกทางร้านว่าไม่เผ็ด ร้านบ้านจิบ จะทำรสแซ่บจัดจ้านมากกก แต่อร่อยสุดๆ เช่นกัน (4156:18) | Forgot to tell the shop "not spicy"; Baan Jib makes it really fiery, but it's also super tasty | The ไม่เผ็ด is about the request; the food was very spicy |
| T06 | กลมกล่อมถูกปากคนไทยแน่นอนค่ะ บริการดี (15102:1) | Well-balanced, sure to suit Thai taste; good service | ถูกปาก ≠ cheap (R5); tasty alone = 0 for T06 |
| T05 | สายลมพัดเอื่อยๆ เมนูที่สั่งวันนี้ ฉู่ฉี่ปลาสายยู ทอดมันปลากราย ยำถั่วพลู น้ำพริกกุ้งสด (7870:3) | A gentle breeze; today we ordered chu chee sai-yu fish, featherback-fish cakes (tod man), wing-bean salad, fresh-shrimp chilli dip | ทอดมัน is a dish name; no oiliness described (R4) |
| T23 | กาแฟไม่ค่อยเข้มมาก (18918:8) | The coffee isn't very strong | Contradicts the "strong" part |

## 5. Procedure

1. Read sections 2–3 once, then the worked examples. Label the first ~20 rows as a warm-up and
   recheck them at the end.
2. Label **one query at a time**. Rows are grouped by query, and filtering to one `qid` is fine.
   Read the query and its row in 3.3 before starting its block.
3. Spend about **10–15 seconds per pair**. Go with your first reading under these rules, without
   overthinking. Write doubts, sarcasm or unreadable text in `note`.
4. Fill only `label` (0/1/2) and `note`. **Don't reorder, sort, delete or edit rows**, `pair_id`
   or `text`. Every row gets a label. Save as CSV UTF-8.
5. **Take a break every ~100 pairs.** The pool is about 465 pairs, roughly 2 hours.
6. Record the date of each session: write `session YYYY-MM-DD` in the `note` of the first row
   you label that day.
7. **Second annotator (C5)** labels `label_sheet_annotator2.csv` alone, after reading these
   guidelines, without seeing or discussing the first annotator's labels. If there is no second
   Thai speaker, Ton relabels the same subset **at least a week later** without looking at the
   first labels.
8. After kappa is computed, disagreements may be discussed and a rule added here, but the
   **first-pass labels are kept** and reported. Don't change labels after seeing system scores.

## 6. Recheck checklist for Ton

- [ ] The glosses in section 4 and in R1–R13 are accurate (especially 18926:5, 1413:30, 7870:3).
- [ ] The degree table (3.2) matches how Thai speakers read ไม่ค่อยX, ไม่Xมาก, Xนิดๆ and Xกำลังดี.
- [ ] **แซ่บ alone = 1** for T02, T07 and T12 feels right. Change it to 2 for T12 if แซ่บ by
      itself already means "sour-spicy-salty" to you.
- [ ] **จืด = 1 for T10** (bland goes too far) and **อมน้ำมัน = 2 for T05** (complaints count) feel
      natural.
- [ ] The item lists are complete: T14 curries, T15 (ส้มตำ/ลาบ/พล่า as near-items only), T17
      fruit drinks (ชามะนาว in or out?), T09 herbs.
- [ ] **T03 vs. T20:** ไม่หวานมาก/หวานน้อย is 1 for T03 but 2 for T20.
- [ ] R4 (dish names give no taste evidence) does not feel unfair to obvious cases such as ต้มยำ
      for T01.
- [ ] Every query has a clear rule. Note any row that needed a judgement call these rules don't
      cover.

## Appendix A. The same scale for the explanation-support sheet (B2)

In B2 each row is an English query from RIRD, a restaurant the system ranked in its top 3, and
one review sentence the system *shows* as the explanation. Label whether **the sentence supports
why this restaurant matches the query**. Judge the sentence, not whether the restaurant or the
ranking is right. **2** = the sentence clearly gives a reason the restaurant fits the whole query.
**1** = it is related or covers only part of the query (one part of a combination, or vague
praise). **0** = it is irrelevant to the query or contradicts it. The rules on combinations, negation,
hedges and complaints (R1–R3, R9) apply unchanged. Examples are illustrative (invented), for the
query *"quiet place with good vegetarian options"*:

- **2**: "The back room was calm enough to talk, and half the menu is vegetarian."
- **1**: "They have a couple of veggie dishes." (one part, weakly)
- **0**: "The burgers are huge." (irrelevant) / "So loud we couldn't hear each other." (contradicts)
