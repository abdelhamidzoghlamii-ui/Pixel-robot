# M2-Q1 blind grades (Local AI) — fixed BEFORE key.json is opened

Run 20261017T213604Z. Graded from sheet_m2.pdf (sha256 3445…9585) and sheet_ab.pdf (sha256 bfb0…99ed).
Key not seen. Scored on 50 unique scenes. ~10 test_photos are byte-identical copies of bench_photos;
duplicates were skipped. Examples: stereo_a = blocked_2, stereo_b = blocked_3, table_top = blocked_1,
table_stereo_a = blocked_4, test empty_1..4 = bench empty_1..4, chiara_far_front = person_facing_300cm,
abdel_far_left = person_sideon_300cm.

## A. M2 shown lines (what Gemma would see)
Each line is graded as one of:
- R: right.
- V: vague but true.
- B: YOLO label wrong; the relation itself would be right for the real object.
- W: wrong relation.

| # | scene | R | V | B | W | notes |
|---|---|---|---|---|---|---|
| 1 | blocked_1 | 5 | 0 | 0 | 0 | |
| 2 | blocked_2 | 1 | 0 | 2 | 1 | "cell phone" = TV remote; "cup above cup" wrong |
| 3 | blocked_3 | 3 | 0 | 0 | 1 | "laptop resting on keyboard" inverted |
| 4 | blocked_4 | 1 | 0 | 0 | 0 | |
| 5 | empty_1 | 4 | 0 | 0 | 1 | bottle resting on bottle; 4 left/right lines trivial |
| 6 | empty_2 | – | – | – | – | none above threshold |
| 7 | empty_3 | 0 | 1 | 0 | 0 | |
| 8 | empty_4 | 1 | 0 | 0 | 0 | |
| 9 | person_facing_060 | – | – | – | – | none |
| 10 | person_facing_100 | – | – | – | – | none |
| 11 | person_facing_150 | 4 | 0 | 0 | 1 | couch in front of laptop |
| 12 | person_facing_200 | 3 | 0 | 0 | 1 | person holding person (YOLO double box) |
| 13 | person_facing_300 | 5 | 0 | 0 | 0 | |
| 14 | sideon_040 | 2 | 0 | 0 | 1 | person leaning against person (double box) |
| 15 | sideon_060 | 0 | 1 | 2 | 1 | "frisbee" = lamp; looking at TV wrong |
| 16 | sideon_080 | 2 | 1 | 0 | 0 | |
| 17 | sideon_100 | 3 | 1 | 0 | 1 | power strip "resting on couch" |
| 18 | sideon_150 | 2 | 0 | 0 | 0 | |
| 19 | sideon_200 | 2 | 0 | 0 | 0 | |
| 20 | sideon_300 | 2 | 0 | 0 | 3 | wearing couch; looking at plants x2 |
| 21 | bed_2 | 0 | 0 | 1 | 0 | bedding labelled bed |
| 22 | bed_3 | 1 | 0 | 0 | 0 | |
| 23 | couch_1 | 5 | 0 | 0 | 0 | 4 left/right lines trivial |
| 24 | couch_2 | 2 | 0 | 0 | 0 | |
| 25 | couch_3 | – | – | – | – | none |
| 26 | fridge_1 | 5 | 0 | 0 | 0 | |
| 27 | fridge_2 | 5 | 0 | 0 | 0 | |
| 28 | fridge_3 | 5 | 0 | 0 | 0 | |
| 29 | toilet_1 | 0 | 0 | 0 | 4 | bottle on floor called "resting on sink/toilet" |
| 30 | toilet_2 | 0 | 2 | 1 | 2 | washer = "microwave", toilet = "sink" |
| 31 | tv_3 | – | – | – | – | none |
| 32 | PXL_213419918 | 0 | 1 | 0 | 0 | |
| 33 | PXL_213509269 | 2 | 0 | 0 | 0 | trivial above/below |
| 34 | PXL_213514191 | 2 | 0 | 0 | 1 | wearing potted plant |
| 35 | PXL_213532927 | 0 | 0 | 2 | 0 | coats detected as person |
| 36 | abdel_close_back | 1 | 0 | 0 | 2 | watching TV; wearing chair |
| 37 | abdel_close_front | 1 | 0 | 0 | 1 | wearing couch |
| 38 | abdel_close_left | 2 | 1 | 1 | 0 | person on TV screen |
| 39 | abdel_close_right | 1 | 0 | 1 | 3 | wearing clock; looking at; watching TV |
| 40 | abdel_far_back | 2 | 0 | 0 | 1 | wearing couch |
| 41 | abdel_far_front | 2 | 1 | 0 | 1 | wearing couch |
| 42 | abdel_far_right | 1 | 0 | 0 | 1 | wearing couch |
| 43 | chiara_close_back | 5 | 0 | 0 | 0 | |
| 44 | chiara_close_front | 1 | 2 | 0 | 2 | phone double box |
| 45 | chiara_close_left | 2 | 0 | 0 | 1 | leaning against far couch |
| 46 | chiara_close_right | 3 | 1 | 0 | 0 | |
| 47 | chiara_far_back | 2 | 0 | 0 | 1 | chair resting on table |
| 48 | chiara_far_left | 0 | 0 | 0 | 1 | |
| 49 | chiara_far_right | 3 | 1 | 0 | 0 | |
| 50 | scene_test | 3 | 0 | 0 | 1 | leaning against couch |

**Totals over 152 lines:**
- R: 96 (63 %)
- V: 13 (9 %)
- B: 10 (7 %)
- W: 33 (22 %)

Strictly right: 63 %. Right or vague-but-true: 71 %.

**Pass bar (≥ 70 % strictly right): NOT MET.**

**Patterns:**
- Systematic errors:
  - "wearing <furniture>": 8 of 8 wrong.
  - "watching / looking at": mostly wrong.
  - Inverted pairs.
  - "resting on" when the object is only overlapping in 2D (toilet scenes).
- Strong predicates:
  - "resting on dining table"
  - "contained in fridge / vase"
  - "sitting on"
  - "person holding bowl / plant"
- Many right lines are trivial left/right/above lines that YOLO positions already give.
- Redundant pairs use up slots, e.g. "plant in vase" plus "vase holding plant".

## B. Gemma A/B (blind, X vs Y)
Unique scenes, grouped by verdict. "f" marks a false statement.

**X better (27):** scenes 1, 3, 6, 7 (X f1), 8 (X f1), 9, 10, 15, 16, 17, 22, 25, 26, 27, 28, 29, 32, 33, 34, 36, 38, 40 (Y f1), 41 (Y f1), 44, 46, 47, 50 (X f1).

**Y better (13):** scenes 2, 4, 5, 11, 13, 14, 19, 21, 23, 37 (X f1), 42 (X f1), 45 (X empty), 48.

**Tie (10):** scenes 12 (Y f1), 18, 20 (X f1, Y broken), 24, 30, 31, 35 (X f1), 39 (X f1), 43, 49.

**Caveats:**
- Many replies are broken in both variants. They repeat "What is in front of the robot?", echo the object list, get cut off at 96 tokens, and one is empty.
- This looks like a prompt/template defect in the harness, not a model property.
- Blindness is weak: replies containing relation words ("resting on", "wearing", "contained in") reveal the M2 variant.
- **Validity:** A/B is INDICATIVE, not decisive.

## B2. Unblinded with key.json (added after grading; the grades above are unchanged)
key.json sha256 4ee9…eea (as in the Coder report). Results over the 50 unique scenes:
- **M2 better:** 34 scenes. Four of these (6, 9, 10, 25) had "none above threshold", so M2 added nothing; those wins are noise.
- **YOLO-only better:** 6 scenes (8, 22, 37, 40, 41, 42).
- **Tie:** 10 scenes.
- **Excluding the scenes where M2 showed nothing** (6, 9, 10, 25, 31): M2 30, YOLO-only 6, tie 9.

False statements:
- M2 side: 10 (scenes 7, 12, 20, 35, 37, 39, 40, 41, 42, 50).
- YOLO-only side: 1 (scene 8).
- At least 7 of the 10 M2-side falses repeat a wrong M2 line ("wearing couch" in 5 scenes, "wearing clock", "leaning against").
- Many M2 wins happened because the YOLO-only reply was broken (repetition or echo). M2 text may mainly change the reply format, not add real understanding. Scene 29 is an example: every M2 line was wrong, yet the M2 reply was judged better.

Pass rule:
- "No more false statements than YOLO-only": **FAILED** (10 vs 1).
- "More specific on ≥ 1/3": met.

**Overall: M2 as built FAILS.**

Post-hoc filter (INDICATIVE only; chosen after seeing the data):
- Drop the predicates wearing, watching, looking at and leaning against. This removes the 17 wrong lines counted above.
- Strictly right then rises to about 71 % (96 of 135 lines). R/V lines with these predicates were not recounted, so the true figure may differ by a point.
- This must be confirmed on new robot-height frames with a fixed Gemma prompt before it counts.

## C. Report defects found
- Report section 6 lists the < 2-box photos differently from the PDF summary page. For example, the report gives bench empty_1 as 0 boxes, but the PDF shows it with 3 boxes; the PDF's list instead includes empty_5, person_facing_040cm, person_facing_080cm, and others.
- **That report list is NOT VALID.** The PDFs/results.json are the evidence.
