# BENCHMARK_PLAN.md

Read-only analysis. No `.py` file was modified. Every behavioural claim cites
`file:line` from the code as read on 2026-09-05. Where two files disagree, both
are reported and left unreconciled. Facts not present in code are marked UNKNOWN.

---

## 1. INTERFACE

### 1a. What `detect_scene()` returns

Source: `detect_person.py:47-87`. The tuple is built at `detect_person.py:84`:

```python
results.append((CLASSES[cls], round(conf,2), pos, dist,
                round(cx), round(cy), round(w), round(h)))
```

`detect_scene(image_path)` returns a **`list` of 8-element `tuple`s**, one per
detected class, sorted by descending confidence (`detect_person.py:86`,
`results.sort(key=lambda x: -x[1])`). Empty list if nothing clears
`CONF = 0.35` (`detect_person.py:18`, `:66`, `:73`, `:87`).

| idx | field | type | meaning / derivation |
|-----|-------|------|----------------------|
| 0 | label | `str` | `CLASSES[cls]` (`detect_person.py:84`); `CLASSES` is the 80-item COCO list `detect_person.py:6-15`; `cls = int(np.argmax(scores))` where `scores = pred[4:]` (`detect_person.py:63-64`). |
| 1 | confidence | `float` | `round(conf, 2)`; `conf = float(scores[cls])` (`detect_person.py:65`), carried as `best[1]` (`detect_person.py:80`). Class score of the kept box. This is the sort key. |
| 2 | position | `str` | One of `'left'` / `'right'` / `'center'`. `detect_person.py:82`: `'left' if cx < 213 else 'right' if cx > 427 else 'center'` — thirds of the 640-px frame. |
| 3 | distance | `str` | One of `'very close'` / `'close'` / `'medium'` / `'far'`. `detect_person.py:83`, from `area = (w*h)/(640*640)` (`detect_person.py:81`): `>0.3` very close, `>0.1` close, `>0.03` medium, else far. |
| 4 | cx | `int` | `round(cx)`; `cx = float(pred[0])` (`detect_person.py:67`), carried as `best[2]` (`detect_person.py:79`). Bbox centre x, pixels, in the 640×640 resized frame. |
| 5 | cy | `int` | `round(cy)`; `pred[1]` / `best[3]`. Bbox centre y, pixels, 640×640 frame. |
| 6 | w | `int` | `round(w)`; `pred[2]` / `best[4]`. Bbox width, pixels, 640×640 frame. |
| 7 | h | `int` | `round(h)`; `pred[3]` / `best[5]`. Bbox height, pixels, 640×640 frame. |

Cardinality: **at most one row per class.** `detect_person.py:74-84` iterates
`by_class` keys and appends only `items[keep[0]]` — the single highest-confidence
box after NMS (`detect_person.py:77-78`). Multiple instances of the same class
collapse to one row.

Pre-processing that shapes fields 2-7: EXIF transpose (`detect_person.py:54`)
then `img.resize((640, 640))` **without preserving aspect ratio**
(`detect_person.py:55`). All coords/sizes are in that distorted square space.

### 1b. Which fields the consumers read

**`Robot.navigate_rules(self, results, distance)` — `main.py:277-367`**

| reads | where |
|-------|-------|
| `r[0]` (label) | `main.py:340` `labels = [r[0] for r in results]`; `main.py:343` `if r[0] == 'person'` |
| `r[7]` (h) | `main.py:344` `estimate_distance_single('person', r[7])` |
| `r[0]`, `r[2]` transitively | `main.py:347` `person_direction(results)` → `detect_person.py:114-125` branches on `r[0]` (`:120-121`) and `r[2]` (`:122-124`) |

Does **not** read `r[1]`, `r[3]`, `r[4]`, `r[5]`, `r[6]`, nor `r[2]` directly.
Also mutates non-tuple state: `self.known_rooms` (`main.py:363-365`), ladder
attrs (`main.py:285-292`), `self.nav_stuck` (`main.py:284`, `:320`).

Return values observed in `main.py:277-367`: `'STOP'` (`:304`, `:332`),
`self.avoid_side` ∈ {`'LEFT'`,`'RIGHT'`} (`:306`, `:316`, `:322`, `:324`),
`'STRAFE_'+self.avoid_side` ∈ {`'STRAFE_LEFT'`,`'STRAFE_RIGHT'`} (`:314`, `:330`),
`'FORWARD'` (`:367`), or `person_direction`'s result ∈ {`'LEFT'`,`'RIGHT'`,`'FORWARD'`}
(`:349`). **Never `'BACK'`, never `'SPEAK'`, never a diagonal, never `None`**
(`person_direction` can return `None` at `detect_person.py:125`, but `main.py:348`
guards `if direction:` and falls through to `'FORWARD'`).

**`Robot.gemma_context(self, scene)` — `main.py:368-379`**

Reads **none** of the tuple fields. Its `scene` parameter is a pre-rendered
`str`. Call site `main.py:434` `context = self.gemma_context(scene)`, where
`scene = scene_to_text(results)` (`main.py:397`). Tuple data reaches Gemma only
through `scene_to_text` default path (`detect_person.py:110-111`), which uses
`r[0]` and `r[2]` only. `gemma_context` additionally calls `self.get_distance()`
(`main.py:378`) and reads `self.mission`, `self.target`, `self.scene_log`,
`self.last_moves`, `self.known_rooms`.

### 1c. Mismatches flagged

- **A — docstring vs return arity.** `detect_person.py:49-50` documents
  `list of (label, confidence, position, distance)` — a 4-tuple. The code
  returns an 8-tuple (`detect_person.py:84`). Any consumer written to the
  docstring (`label, conf, pos, dist = r`) raises `ValueError`. `main.py:344`
  (`r[7]`) works only against the real 8-tuple. `scene_to_text(coords=True)`
  (`detect_person.py:108-109`) also depends on indices 4-7 the docstring omits.
- **B — `r[7]` units into `estimate_distance_single`.** `stereo_depth.py:28`
  `estimate_distance_single(label, height_px)` computes
  `dist = (real_h * FOCAL_PX) / height_px` (`stereo_depth.py:36`) with
  `FOCAL_PX = 500` marked *"calibrate later with known distance"*
  (`stereo_depth.py:8`). `main.py:344` passes `r[7]` = bbox height in the
  **aspect-distorted** 640×640 frame (`detect_person.py:55`). For a 4080×3072
  source the vertical scale is `640/3072`, horizontal `640/4080` — unequal — and
  nothing in code corrects for it. Systematic error of unknown size feeds the
  `PERSON_STOP_DIST = 80` test (`main.py:14`, `:345`). UNKNOWN whether the
  constant was fit to squished or letterboxed input.
- **C — loop shape vs producer.** `main.py:342-345` iterates
  `for r in results: if r[0] == 'person': ...` as if several person rows may
  exist, but `detect_scene` emits at most one per class (`detect_person.py:74-84`).
  Harmless today; if `detect_scene` is ever changed to per-instance rows, the
  `return 'STOP'` on the first close person (`main.py:345`) silently short-circuits.
- **D — position/range source of truth disagreement.** `navigate_rules` ignores
  the pre-computed `r[2]` (position) and `r[3]` (range bucket); it re-derives
  direction via `person_direction` and range via
  `estimate_distance_single(r[7])`. `gemma_context` / `scene_to_text` instead
  rely on `r[2]` (`detect_person.py:111`). The two consumers do not agree on
  which fields are authoritative.
- **E — scene string is lossy.** `scene_to_text` default emits only
  `"{label} {pos}"` per object (`detect_person.py:111`) — no confidence, no
  `r[3]` range bucket, no coords. Gemma's "Current scene" therefore carries no
  per-object distance; the only distance Gemma sees is the single scalar
  `get_distance()` appended at `main.py:378`.

---

## 2. PHOTOS

### Inventory of `test_photos/`

- **Count:** 32 files, all `.jpg`. **No subdirectories** (flat).
- **Resolution:** every file is 12.53 MP, one of two EXIF orientations —
  25 × `4080×3072` (landscape), 7 × `3072×4080` (portrait: the six
  `PXL_20260405_2134*` shots + `scene_test.jpg`). `detect_scene` discards this:
  EXIF-transpose then hard resize to 640×640 (`detect_person.py:54-55`).
- **No label sidecars** — no `.json` / `.txt` / `.xml`, no class folders.

### Naming / implied labels

| group | count | pattern | implied label |
|-------|-------|---------|---------------|
| person set | 16 | `{subject}_{distance}_{direction}.jpg`, subject ∈ {`abdel`,`chiara`}, distance ∈ {`close`,`far`}, direction ∈ {`front`,`back`,`left`,`right`} — full 2×2×4 grid, one per cell | a `person` should be detected; `left`/`right` ⇒ `person_direction` LEFT/RIGHT, `front`/`back` ⇒ centre/FORWARD. Matches `test_suite_m.py` axes. |
| negatives | 5 | `empty_1.jpg` … `empty_5.jpg` | no person; expect `detect_scene` → `[]` → `navigate_rules` FORWARD path |
| camera roll | 6 | `PXL_20260405_2134*.jpg` | none — raw dump, contents UNKNOWN |
| fixtures | 5 | `scene_test.jpg`, `stereo_a.jpg`, `stereo_b.jpg`, `table_stereo_a.jpg`, `table_top.jpg` | none encoded; `stereo_a`/`stereo_b` are a strafe pair; `table_stereo_a` has no `_b` mate |

Distance labels are qualitative only (`close`/`far`) — no centimetre ground truth.

### Sufficiency for a scene-based nav benchmark

**Partly sufficient.**

- The obstacle / safety branch of the *deployed* `navigate_rules`
  (`main.py:296-337`) is driven **entirely by the injected `distance`
  argument** — it inspects no labels for obstacles. So any photo (even `[]`
  detections) plus a distance sweep fully exercises that branch. The photo set
  is adequate here.
- The person branch (`main.py:339-350`) needs photos containing a detectable
  person — the 16-file grid is designed for exactly this (pending §5.2
  verification that YOLO actually fires on them).
- The empty-scene → FORWARD branch (`main.py:367`) needs empty frames — the
  5 `empty_*` files cover it.

### Missing scenarios

- **Room-signature objects.** `navigate_rules` maps `refrigerator` → kitchen,
  `couch`/`tv` → living room, `bed` → bedroom, `toilet` → bathroom
  (`main.py:352-359`) and records `self.known_rooms` (`main.py:363-365`).
  **No filename indicates any of these objects.** UNKNOWN whether `scene_test`
  or the `PXL_*` frames contain one. Without such photos, `main.py:352-366` and
  the room-mapping side effect are untestable.
- **Person at known true distance** — needed to check `PERSON_STOP_DIST = 80`
  against `estimate_distance_single('person', r[7])` (`main.py:344-345`). Absent.
- **Cluttered / multi-object frames** — to see ordering (`detect_person.py:86`)
  and the one-row-per-class collapse in a realistic scene.
- **Robot-camera imagery.** These look like handheld phone shots (`PXL_` names,
  mixed orientation). UNKNOWN if the mounting height, pitch, lens and the 4:3→1:1
  squish match what the robot camera produces (§5.7).

---

## 3. EXISTING TESTS

### `nav_logic_test.py`

- **Path tested:** its own `python_navigate()` / `gemma_explore()` / `navigate()`
  (`nav_logic_test.py:36`, `:92`, `:128`) — a standalone reimplementation.
  Docstring self-identifies: *"test the new logic WITHOUT touching main.py"*
  (`nav_logic_test.py:2-3`).
- **Still deployed?** **No.** `main.py:277` `Robot.navigate_rules` is a different
  implementation. Divergences: obstacle-side is chosen by string-matching
  `'left'`/`'right'` + `'obstacle'`/`'chair'` in the scene text
  (`nav_logic_test.py:51-54`, `:82-86`) — `main.py:296-337` has **no** label
  inspection for obstacles, only `distance` and `self.avoid_side`.
  `distance < 15` returns `'BACK'` here (`nav_logic_test.py:47`) but
  `self.avoid_side` in `main.py:306`. Return type is
  `(direction, reason, needs_gemma)` here vs a bare `str` in `main.py`.
- **Inputs producible by the real system?** **No.** Scene strings such as
  `"obstacle center"` (`nav_logic_test.py:162`), `"obstacle left"` (`:164`),
  `"clear path"` (`:62` of `benchmark_nav` — same style), `"empty hallway"`
  (`:167`) are not `scene_to_text` output. `scene_to_text` emits
  `"{label} {pos}"` with `label` ∈ COCO `CLASSES` (`detect_person.py:6-15`,
  `:111`); `"obstacle"` is not a class. `"nothing detected"` is the only
  matchable fragment (`detect_person.py:107` → `"empty room, nothing detected"`).
  `distance` is a plain int, matching the injection approach.
- **Verdict: archaeology.** This is the invalid 93% benchmark named in the task —
  non-deployed logic, impossible perception inputs, no distance sensor.

### `benchmark_nav.py`

- **Path tested:** the Gemma server over raw HTTP (`benchmark_nav.py:79`) with
  `GEMMA_SYS` (`benchmark_nav.py:6-25`). Imports no project module.
- **Still deployed?** **Partially.** `main.py:58` `gemma_decide` posts to the
  same `GEMMA_URL` with `main.py`'s `GEMMA_SYS` (`main.py:37-56`). The two system
  prompts carry the same rule list but are separately maintained copies. The
  benchmark's context is a hand-written one-liner
  (`benchmark_nav.py:30` `"Mission: … I see: person center. Distance: 60cm …"`);
  the deployed context is the fixed multi-line block from `gemma_context`
  (`main.py:368-379`: Mission / Target / Current scene / Last 5 scenes /
  Last moves / Known rooms / Distance ahead). The benchmark also does not add
  the injected leading `[`-style handling; that's the parser path, not this one.
- **Inputs producible?** **Partially.** `"person center"`,
  `"refrigerator center"`, `"couch center, tv left"` are `scene_to_text`-shaped.
  `"obstacle center"` (`benchmark_nav.py:50`), `"empty hallway"` (`:58`),
  `"nothing detected"` (`:60`), `"clear path"` (`:62`) are not. The full context
  string never matches `gemma_context` output.
- **Verdict: partially salvageable.** Scoring the Gemma move token is valid and
  Gemma is in the deployed path, but it must use the real `GEMMA_SYS` and real
  `gemma_context` format, with scene fragments from real `scene_to_text`.
  Expected answers are hand-authored ground truth. Needs a running server.

### `json_benchmark.py`

- **Path tested:** Gemma over HTTP (`json_benchmark.py:136`) with `PARSE_SYS`
  (`json_benchmark.py:6-94`); local JSON extraction (`json_benchmark.py:143-146`).
- **Still deployed?** **Divergent.** `main.py:204` `parse_command` posts with
  `main.py`'s `PARSE_SYS` (`main.py:146-202`), which emits a **different schema**:
  `{"type":"find_person","name":…}` / `{"type":"navigate_to","room":…}`
  (`main.py:152`, `:155`). This benchmark expects
  `{"action":"find_person","target":…}` (`json_benchmark.py:11-15`). `main.py:208`
  also injects a leading `[` into the prompt and re-prepends it
  (`main.py:217`); `json_benchmark.py:133` does not. **Reported, not reconciled:
  `main.py` uses `type`/`name`/`room`; `json_benchmark.py` uses `action`/`target`.**
- **Inputs producible?** **Yes.** Inputs are transcribed voice strings
  (`json_benchmark.py:97-127`); `listen()` (`main.py:126`) / `voice.py` produce
  free text like this.
- **Verdict: salvageable concept, wrong schema, wrong subsystem.** Voice parsing
  is deployed (`parse_command`) and the 30-case table is reusable, but it scores
  the `action`/`target` schema `main.py` does not emit, and it is a *voice* test,
  not a *navigation* test.

### `benchmark_compare.py`

- **Path tested:** same as `json_benchmark.py` (HTTP parser, same `action`/`target`
  `PARSE_SYS`, `benchmark_compare.py:5-72`, `:131`), run twice to A/B a LoRA
  adapter (`benchmark_compare.py:210-217`).
- **Still deployed?** **Divergent**, same schema mismatch vs `main.py:146-202`.
  Also depends on `server_manager.py setup_voice_lora` / `setup_q4`
  (`benchmark_compare.py:213`, `:217`). UNKNOWN whether that adapter still exists
  — it is not in the repo and `server_manager.py` is unreferenced by the runtime
  (see `FILES.md`).
- **Inputs producible?** **Yes** — 35 free-text commands including negatives
  (`"make me a coffee"` → `[]`, `"don't go to the kitchen"` → `[]`,
  `benchmark_compare.py:117`, `:121`).
- **Verdict: archaeology for its stated purpose** (manual two-run LoRA A/B,
  adapter presence UNKNOWN), **but the test-case table with difficulty tiers and
  negative cases is salvageable data** for a rewritten voice-parser test.

### `quality_benchmark2.py`

- **Path tested:** Gemma over HTTP (`ask()`, `quality_benchmark2.py:7-34`) with
  **three ad-hoc prompt templates**, none from `main.py`: a "navigation
  strategist" SYS (`quality_benchmark2.py:63-69`), a JSON parser SYS with **yet
  another schema** `{"action":"find_person|go_to_room|say|patrol|come_back",…}`
  (`quality_benchmark2.py:137-143`), and unscored free-form vision Q&A on
  `test_photos/` (`quality_benchmark2.py:239-266`, downsized to 320×320 at `:18`).
- **Still deployed?** **No.** None of the three templates appear in `main.py`.
  The action name `go_to_room` (`quality_benchmark2.py:139`) matches neither
  `main.py` (`navigate_to`, `main.py:155`) nor the other benchmarks
  (`navigate_to`). `main.py`'s vision call is `gemma_decide` with `GEMMA_SYS`
  (`main.py:58`, `:435`), not room-ID / person-description / "nav advice".
- **Inputs producible?** Photos are real; the context prose
  (`quality_benchmark2.py:73-111`, e.g. `"Battery getting low"`,
  `"Last seen: bedroom 10 minutes ago"`) is not `gemma_context` output.
- **Verdict: archaeology.** An exploratory "what can Gemma do" probe across three
  throwaway schemas; TEST 3 and TEST 4 have no pass/fail
  (`quality_benchmark2.py:232-290` only prints).

### Summary

| file | salvageable? |
|------|--------------|
| `nav_logic_test.py` | archaeology |
| `benchmark_nav.py` | partial — LLM-move scoring, needs real prompt + `gemma_context` format + server |
| `json_benchmark.py` | partial — voice path (not nav); test table reusable; rewrite to `type`/`name` schema |
| `benchmark_compare.py` | partial — LoRA A/B is archaeology; 35-case table reusable |
| `quality_benchmark2.py` | archaeology |

None of the five currently feeds real photos through real `detect_scene` into
real `navigate_rules`.

---

## 4. PROPOSAL — one replacement benchmark

**Goal:** exercise the *deployed* obstacle-safety path. Real photo →
real `detect_scene()` → real `Robot.navigate_rules()`, with `distance` injected.
Score two label-free properties: **format validity** and **safety violations**.
Design only; not implemented here.

### 4.1 Harness

- **Imports:** `from detect_person import detect_scene`; `import main`. Importing
  `main` runs `main.py:1-234` (imports + constant/function defs only); the
  `if __name__ == '__main__'` block (`main.py:506-520`) does **not** run on
  import. `main` pulls in `detect_person` and `stereo_depth` (both verified this
  session to make no server/USB/root call at import; the ONNX model loads lazily
  via `get_session`, `detect_person.py:23-27`). `motors.py` is **not** imported
  by `main.py`. No server, no USB, no root.
- **Inputs:** cartesian product of
  - every `*.jpg` in `test_photos/` (32 files), and
  - a fixed **distance sweep** chosen at the decision boundaries in
    `navigate_rules`: `[5, 14, 15, 20, 24, 25, 26, 40, 79, 80, 120, 400, 999]`.
    Rationale: thresholds at `15` (`main.py:301`), `OBSTACLE_DIST = 25`
    (`main.py:13`, `:296`, `:309`), plus the two real sentinel values
    `get_distance()` produces — `400` = "no echo / clear" (`main.py:273`) and
    `999` = stale / no-motor (`main.py:270`, `:275`).
  ≈ 32 × 13 = 416 trials; `detect_scene` runs 32 times (model cached).
- **Per trial:**
  1. `results = detect_scene(photo)` — once per photo, memoised across the sweep.
  2. `r = main.Robot()` — fresh instance, `motors=None`, per trial, so the
     avoidance ladder starts from `blocked_n = 0` (`main.py:288`) and results are
     deterministic.
  3. `move = r.navigate_rules(results, distance)`.
  4. Record: `photo`, `distance`, `len(results)`, sorted label set,
     `move`, `getattr(r, 'nav_stuck', False)`.
- **Determinism:** `detect_scene` has no RNG; CPU ORT inference is expected
  deterministic (§5.11 flags the residual doubt). `navigate_rules` on a fresh
  instance is a pure function of `(results, distance)` plus `self.known_rooms`
  (init `{}`, `main.py:245`).

### 4.2 Score 1 — format validity (no labels needed)

```
VALID = {"FORWARD","BACK","LEFT","RIGHT","STRAFE_LEFT","STRAFE_RIGHT","STOP"}
format_valid  =  move in VALID
```

These are exactly the tokens `Robot.move()` acts on (`main.py:252-258`), and the
7-token set in `build_bench.py`'s system prompt. Report count and list of any
out-of-set return. **Expected: 100%.** A miss means someone changed
`navigate_rules` (or `person_direction`, `detect_person.py:114-125`) to emit a
token the motor layer cannot execute — a pure regression guard, no ground truth
required. (`'SPEAK'` is intentionally excluded: it is a Gemma-only token that
`main.py:443-448` rewrites to `'STOP'` before `move()` — it never leaves
`navigate_rules`.)

### 4.3 Score 2 — safety violations (no labels needed)

```
FORWARDISH = {"FORWARD"}          # plus any forward-diagonal token, if ever added
safety_violation  =  distance < main.OBSTACLE_DIST  and  move in FORWARDISH
```

Report the count and every offending `(photo, distance, move)` row.
**Expected: 0.** This is a property, not an accuracy metric: for any real scene,
below 25 cm the robot must not drive forward.

Why 0 is expected from `navigate_rules` alone: with a fresh instance, `distance <
25` makes `main.py:296` set `blocked_since`, then one of the branches at
`main.py:301-337` returns `STRAFE_*` / `LEFT` / `RIGHT` / `STOP` **before**
control can reach the person block (`main.py:339`), the room block
(`main.py:352`) or the final `return 'FORWARD'` (`main.py:367`). So a non-zero
count here means the ladder has been broken. `'BACK'` is not in `FORWARDISH`
because reversing away from an obstacle is safe; note `navigate_rules` never
emits `'BACK'` anyway (§1b).

### 4.4 Output

- Two headline numbers: `format_valid` %, `safety_violation` count.
- Full per-trial table (photo, distance, n_detections, labels, move, nav_stuck).
- **Branch-coverage report** — how many trials reached each branch, so the reader
  knows what the photo set actually tested:
  - person branch: `'person' in labels` (`main.py:341`)
  - room-signature branch: `labels ∩ {refrigerator, couch, tv, bed, toilet}`
    (`main.py:352-359`)
  - empty → FORWARD: `len(results) == 0` and `distance ≥ 25`
  - obstacle ladder: `distance < 25`
  If the room-signature count is 0, say so plainly (expected, per §2).
- No `input()` gate, no network. Suitable for unattended / CI use.

### 4.5 Optional Mode B — ladder-aware (still no server)

Feed one photo repeatedly at `distance = 10` to the **same** `Robot` instance for
≥ 8 iterations, to drive `blocked_n` through the escalation in `main.py:308-332`.
Assert:
- every returned token stays in {`STRAFE_LEFT`, `STRAFE_RIGHT`, `LEFT`, `RIGHT`,
  `STOP`} and never `FORWARD` across the whole ladder;
- `self.nav_stuck` becomes `True` exactly once, at `n == 7` (`main.py:316-322`);
- `self.avoid_side` flips exactly once (`main.py:318`).

This covers the state-machine code that Mode A (fresh instance) never enters.

### 4.6 Stated non-goals (reported honestly)

- **No navigation correctness.** Whether `FORWARD` is the *right* call for a
  given scene needs ground-truth labels the photo set lacks (§2). This benchmark
  checks only format and the sub-25 safety property, per the task.
- **No `run_cycle` / Gemma-override coverage.** The one place *deployed* code can
  emit `FORWARD` while `distance < 25` is `run_cycle`: when `navigate_rules`
  returns a `STRAFE_*` (a `safety_move`, `main.py:417-418`) **and** `stuck` is
  `True` (only at `n == 7`), `use_gemma` becomes `True` (`main.py:421-422`), and
  a Gemma reply containing `"FORWARD"` overwrites `move` at `main.py:438-441`
  with **no re-check against distance**. Testing that requires driving
  `run_cycle` with a real or stubbed Gemma and is out of scope for the
  photo→`detect_scene`→`navigate_rules` benchmark the task specifies. Flagged as
  the top follow-up.
- **No `get_distance()` coverage** (`main.py:265-275`) — distance is injected, so
  the 999/400 sentinels and the 1.0 s staleness window (`main.py:271`) are not
  exercised.

---

## 5. UNCERTAINTIES

Questions not answerable from the code alone, and what would settle each.

1. **Does `detect_scene` detect anything in these specific photos?** Cannot run
   inference in this environment. → Run the §4 coverage report once, or
   `python3 detect_person.py test_photos/<f>.jpg` per file.
2. **Do the 16 `{subject}_*` photos contain a `person` at `CONF ≥ 0.35`
   (`detect_person.py:18`)?** Filenames imply it; unverified. → Inspect
   `detect_scene` output per file.
3. **Do any test photos contain a room-signature object** (`refrigerator`,
   `couch`, `tv`, `bed`, `toilet`, `main.py:352-359`)? Filenames don't say;
   `scene_test.jpg` and the six `PXL_*` frames are UNKNOWN content. → Run
   `detect_scene` on the unlabelled photos; if none, `main.py:352-366` is
   untestable with current assets and new photos must be shot.
4. **Is "forward diagonal" a real nav output?** The task's safety rule names it,
   but no forward-diagonal token exists in `Robot.move` (`main.py:252-258`), the
   Gemma vocab (`main.py:439-440`), `navigate_rules`, or `build_bench.py`'s
   7-token list. `teleop.py:10` has `FWD_L`/`FWD_R`, but that is a separate HTTP
   path not used by `main.py`. → Product decision: if no forward diagonal is ever
   legal, Score 2 reduces to `move == 'FORWARD'`.
5. **Is `get_distance()` genuinely the only obstacle input to `navigate_rules`?**
   Code says yes — `main.py:277-367` inspects labels only for person and
   room-signature, never for obstacles. → Confirm with the author that visual /
   stereo obstacle range is not meant to feed the safety ladder.
6. **Is `PERSON_STOP_DIST = 80` (`main.py:14`) meaningful given `FOCAL_PX = 500`
   is uncalibrated** (`stereo_depth.py:8`, *"calibrate later"*)? Affects whether
   `main.py:344-345` ever fires correctly. → A calibration measurement (known
   person height at known distance); needs hardware, out of scope here.
7. **Are `test_photos/` images representative of the robot's mounted camera**
   (height, pitch, lens, and the 4:3 → 1:1 squish at `detect_person.py:55`)?
   They appear handheld (`PXL_` names, mixed orientation). → Capture frames from
   the actual robot camera and compare detections.
8. **Which injected distance values are operationally real?** `get_distance()`
   can return `999`, `400`, or a raw firmware `d` (`main.py:270-275`); the real
   distribution of `d` is not in code. → Log `motors.get_distance()` over a real
   run (needs USB, out of scope).
9. **Does importing `main` (transitively) touch hardware at import time?**
   `main`, `detect_person`, `stereo_depth` were read this session and do not;
   `motors` is not imported by `main`. Not proven for every transitive import
   without reading all of them. → One dry `python3 -c "import main"` with the
   server down and USB unplugged.
10. **Is `navigate_rules` intended to be called standalone** (not only from
    `run_cycle`)? It is safe to (state is `hasattr`-guarded, `main.py:285-292`),
    but it mutates `self.known_rooms` and prints `[MAP] Found:`
    (`main.py:363-365`). → Confirm the author accepts a benchmark calling it
    directly.
11. **Is `detect_scene` bit-deterministic across runs?** `onnxruntime` CPU is
    generally deterministic, but thread count / provider config could reorder
    equal-confidence boxes before the stable sort at `detect_person.py:86`. →
    Run `detect_scene` twice on one file and diff.
12. **Does the `benchmark_compare.py` LoRA adapter (`setup_voice_lora`,
    `benchmark_compare.py:213`) still exist?** Not in the repo; `server_manager.py`
    is unreferenced by the runtime. → Check the llama.cpp models directory and
    `server_manager.py`'s setup definitions (filesystem outside the project).
13. **`GEMMA_SYS` / `PARSE_SYS` drift.** `benchmark_nav.py:6-25` vs `main.py:37-56`,
    and `json_benchmark.py:6-94` / `benchmark_compare.py:5-72` vs `main.py:146-202`
    are separately maintained copies with at least a schema difference
    (`action`/`target` vs `type`/`name`/`room`). → Decide which is canonical
    before any LLM-scoring benchmark is rebuilt on top of them.
