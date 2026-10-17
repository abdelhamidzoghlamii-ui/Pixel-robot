# PICKER-1 — Coder report (stop and report; nothing committed)

Task: wire the Gemma picker (with RelateAnything M2, layout L2) and person-stop (#79) into the robot cycle.
Date 2026-10-15. Work folder `/termux-home/review/picker1/`. Candidate diff: `picker1_candidate_diff.txt` (same folder,
= `candidate_r4.diff`, 1823 lines, full diff vs 1969f68 incl. new files).

**Status: round 4 (owner-authorized, last) is VALID and APPROVED with no findings. Rounds 1–3 requested changes; all
findings were fixed. The working tree equals the round-4 frozen manifest. Nothing is committed; waiting for
"yes commit".**

Owner decisions 2026-10-15 (after the first report): (1) round 4 for exactly the two round-3 fixes — done;
(2) PERSON_STOP_BUCKETS stays ('close', 'very close'); (3) greeting text `Hello, nice to see you.` accepted.

## 1. Profiles

```text
EXECUTION PROFILE                         REVIEWER PROFILE (used, all 4 rounds)
ROLE: Coder                               ROLE: Reviewer (independent, read-only)
PLATFORM: Claude Code CLI (proot Debian)  PLATFORM: Codex CLI 0.162.1 (codex exec, isolated copy)
MODEL: claude-opus-5-5                    MODEL: gpt-6.1-sol
EFFORT: high (session later switched      EFFORT: medium
  by the owner's /model to medium)        PONYTAIL: off (stated in every request)
PONYTAIL: lite
```
No fallback platform was used. Note: in round 3 the reviewer listed
`/root/.codex/plugins/cache/ponytail/ponytail/5.2.0/skills/ponytail-review/SKILL.md` among files read; its findings
are correctness findings, not simplifications.

## 2. Base check

- `git rev-parse HEAD origin/main` = `git ls-remote origin refs/heads/main` = 1969f68f8c53c1dbffc72b64dafd41295aafaddf.
- `git status --porcelain=v1 -uall` empty at start; `git stash list` empty (still empty now).
- Read: HANDOFF, WORKFLOW, CODER, docs/CLAUDE, STATUS, LICENSES, DECISION_INDEX; DECISIONS #14, #78–#80, #104, #109,
  #111, #116, #119, #125, #127, #129–#132, #135–#137 in full; the listed code files.

## 3. Facts F1–F5

**F1 — campaign picker.** `benchmark/coresidency/coresidency.py:479` `def make_selector(post=None):` imports
`from adapters import S1O` (:484) and `from s1.schema import Example, Q, LETTERS, build` (:485) with
`S1O_SRC = HOME / 'sel-candidates/s1o/src'` (:66, outside the repo). `select()` (:511):
`dist = sel.decide(case['situation'], case['options'], case['instruction'])` (:513),
`choice = max(keys, key=lambda k: dist[k])  # ties: first offered, as in ladder.py` (:516).
S1O scoring, `benchmark/strategic_selector/v3/adapters.py:182`:
`body = {"prompt": ids, "n_predict": 1, "n_probs": self.N_PROBS, "post_sampling_probs": False,` with
`N_PROBS = 1000` (:139), renormalised `return {k: v / mass for k, v in raw.items()}` (:207).
Campaign: phase1 `bounded_selector` wraps `cr.make_selector(post)` with a 15 s urlopen timeout; the L2 selector
text is `context(shared['triplets'])` appended to a ladder case's situation. Measured medians (#131,
`docs/DECISIONS.md:1454`): `- L0 no M2: 4.10 W, picker median 1733 ms (VALID).`; L2 picker median 2129 ms
(NOT COMPARABLE — START TEMP).

**F2 — M2 runtime.** `benchmark/relate_anything/speed1/variants.py:14`
`CLUSTERS = {'MID': ({4, 5}, 2), 'BIG': ({6, 7}, 2), 'LITTLE': ({0, 1, 2, 3}, 4)}`; `build()` (:66) →
`return sb.build(NAME, cpus, threads, lambda name, threads: make_head(variant, threads))` (:68), where `sb` is
`benchmark/relate_anything/desk2/speed_block.py` (`build` at :153: ORT-worker thread diff, pin, pinned caller).
Box rule `desk_check.py:37` `raise ValueError('need 2..32 xyxy boxes; never silently truncate detections')`;
parity gate `desk_check.py:50` `def require_parity(name):` / :57
`raise RuntimeError('STOP M2: graph SHA-256 differs from tested parity graph')`. Parity record
`desk2/parity_valid_relsgg-vits16.json`: passed true, graph_sha256 24a2e2a2…98d5 (ORT 1.25.1).
Assets stay outside the repo; weight licence for shipped use NOT evaluated (unchanged).

**F3 — base person branch.** `git show 1969f68:main.py` line 335
`dist_est = estimate_distance_single('person', r[7])`, line 336 `if dist_est and dist_est < PERSON_STOP_DIST:`,
line 16 `PERSON_STOP_DIST = 80`. Legacy r[7] is at most 640, so the estimate is at least 170·500/640 ≈ 133 cm > 80:
the branch can never fire (consistent with #78).

**F4 — server start.** Campaign: phase1 preflight `server=cr.Server(...)` (`benchmark/campaign/phase1.py:851`; phase23
reuses it, `phase23.py:658`), whose command is `coresidency.server_cmd()` — docstring
`benchmark/coresidency/coresidency.py:404`: "No taskset: the robot does not pin the server." Archived command
(`benchmark/campaign/runs/owner_adaptive_p23_01.json`): llama-server b1609-dotprod, gemma-4-E2B-it-Q4_0, ctx 2048,
`--threads 4 --threads-batch 4 --parallel 1 --swa-full --cache-ram 0 --host 127.0.0.1` — the same flags as
`server_manager.start_server` for setup_q4 (`server_manager.py:47-61`, `os.system(cmd)` with `" 2>/dev/null &"`,
no taskset). Difference: the campaign Popen'd it from a process whose main thread had just been set to all CPUs
(`benchmark/campaign/runtime.py:283-284` `requested = set(range(os.cpu_count()))` / `os.sched_setaffinity(0, requested)`)
with `start_new_session=True`; the server inherits that mask subject to Android's cpuset. The server's actual
mask during the campaign is not recorded in the files read: UNKNOWN. setup_q4 run by the owner inherits the
launching shell's mask. server_manager.py unchanged.

**F5 — licence.** `~/sel-candidates/s1o/src/LICENSE`: MIT License, "Copyright (c) 2026 Rami Mithalouni"
(plus a note that s1/demo_presets.py, not copied, adapts MIT code by rorshopping). Repo metadata: git
`77f1f7cccf8aa752e0ed7edcc8d2094bac707bdc` (2026-09-17), origin https://github.com/mithalouni/system-one-open.git.
Permissive → copied the needed parts of s1/schema.py (LETTERS, render_option, truncate_tokens, build — trimmed to one
choice question) into `picker.py` with the full MIT notice and attribution. No runtime import from outside the repo.
The reviewer could read s1/schema.py but not the root LICENSE (its rounds confirm the embedded notice only).

## 4. Situation template and options

```text
Mission: {mission}
Scene: {label} ({position}, {bucket}), ...            | "nothing detected"
Ultrasonic distance ahead: {d} cm                     | "no reading" (get_distance() 999)
Current action: {mode phrase} (for {s:.0f} s)
Last moves: {last 3 moves, comma-separated}            | "none"
Rooms seen: {known_rooms, comma-separated}             | "none"
M2 relations:
- {subject} [{i}] {predicate} {object} [{j}].        (phase1.context(); up to 5)
                                                     | "- none above threshold." | "- none (fewer than 2 objects)."
```
Instruction: `Choose the single best next action for the robot.` Options in this fixed order (A, B, …):
`keep exploring` (EXPLORE), `go toward the person` (APPROACH), `stop and greet the person` (GREET),
`look around in place` (LOOK); APPROACH and GREET only when a person is in the latest detections.
Prompt = s1 build: `<bos><state>\n{state}\n</state>\n\nQuestion (choice): {instruction}\nOptions:\n(A) …\nAnswer: (`;
scoring = one /completion, n_predict 1, n_probs 1000, temperature 0, post_sampling_probs false, cache_prompt false.
Greeting (fixed): `Hello, nice to see you.`

## 5. Design summary

- **picker.py** (new): `eligible()`, `situation()`, `http()` (urllib, 10 s per request), `Gemma` (health check,
  S1O letter scoring, `request()` runs a per-pick cancellation check before every HTTP request), `Cadence` (D2),
  `Picker` (one daemon worker thread = M2 calling thread, pinned to 6–7; plain-data request in, result dict out;
  `allow()`/`forbid()` gate with a generation number so a cancelled pick is never revived; `stop()`).
- **relate.py** (new): M2 head copied from desk_check (SHA-256 of all three files checked first = parity gate bound to
  the graph hash, then np.load(allow_pickle)), `Relate` (thread diff → 1 ORT worker pinned to {6,7}; the caller pins
  itself; affinity()/check_affinity() borrowed from PinnedDetector), `relations()` (<2 boxes: no M2; >32: top-32 by
  confidence in detector order), `context()` from phase1.
  Paths: `/data/data/com.termux/files/home/models/relate_anything/relsgg-vits16/{relateanything.onnx,
  relateanything.json, predicate_bank.npz}`; server `http://127.0.0.1:8080`.
- **main.py**: `PERSON_STOP_DIST` and the estimate_distance_single import removed; `PERSON_STOP_BUCKETS =
  ('close', 'very close')`, `FORWARD_MOVES = ('FORWARD', 'FWD_L', 'FWD_R')`, `LOOK_CYCLES = 4`, `GREETING`.
  `navigate_rules(..., approach=True)`: approach=False skips only the person branch. New move `HOLD`
  (motors.stop + sleep CYCLE_MOVE_TIME; never ends the mission). `choose_move()`: GREET → HOLD (+ greeting once per
  GREET, no navigate_rules); LOOK → navigate_rules(approach=False) with its clear-path FORWARD turned into LEFT, EXPLORE
  after 4 cycles; EXPLORE/APPROACH/None(--no-picker) → navigate_rules; then person-stop over every person detection
  (legacy tuples + each detector dict bucketed through scene_from_detections) turns forward moves into HOLD.
  `request_pick()` (ACTIVE cycles only; forbid/allow by the monitor's latest status; cadence; one in flight; needs
  this slot's 640 frame), `collect_pick()` (every loop iteration in every phase, start of run_cycle, mission end;
  counts/applies only for the current open slot, not cancelled, below SEVERE now), `set_mode()`, `severe_now()`,
  `trace()`. last_640 kept per slot. Picker forbidden before the camera stops on ACTIVE→PAUSE/HOLD, allowed on ACTIVE.
  `active_slot` gains picks, picker_ms, m2_ms (median/max), fallbacks.
- **run_mission.py**: `--no-picker`; `build_picker()` after the detector and before motors/heat threads (thread diff);
  server/M2 failure → print `[PICKER] disabled: … — the mission continues in EXPLORE`, picker_config enabled false;
  finally: camera stop → picker.stop() → collect last pick → heat stop.
- Trace events: picker_config, pick (slot, trigger, eligible, mode, probs per phrase in letter order, m2_ms, m2_boxes,
  m2_relations, m2_repinned, picker_ms, prompt_tokens, fallback, cancelled, applied), mode (from, to, reason,
  after_s), person_stop (move, mode, bucket), picker_severe (severe, mode), active_slot extended.
- Unchanged: heat_control.py, SizePolicy, CAMERA_RATE, stale-frame STOP, MOTOR_SPEED, CYCLE_MOVE_TIME,
  server_manager.py, pause_cycle.
- Owner answer 2026-10-15 (asked mid-task): person-stop holds and the mission continues, in every mode incl.
  --no-picker.

## 6. Checks (native Termux Python 3.13.13 from proot; python3 -B; compile())

| Suite | Base 1969f68 | Candidate (r4) |
|---|---|---|
| run_cycle_safety_test | 9 OK | 22 OK |
| test_detector_robotcam | 17 OK | 17 OK |
| test_heat_control | 20 OK | 29 OK |
| test_run_mission | 5 OK | 10 OK |
| test_chat_mtp | 10 OK | 10 OK |
| chat_mtp/test_chat_mtp_runner | 32 OK | 32 OK |
| test_picker (new) | — | 29 OK |
| nav_test.py stdout | 41 lines, md5 fbbfe5b3… | 41 lines, md5 fbbfe5b3248f1c930ac7c0d5fc67d4c6 |

Base run on an archive of 1969f68 (`checks_base.txt`; test_picker absent there by design). The reviewer reran the
candidate checks in every round with the same counts. Existing tests changed only where the interface changed
(navigate_rules approach kwarg, active_slot keys, run_mission fakes patch build_picker).
Mutation checks (scratch copy): each of 16 distinct reverted safety/gating lines (person-stop off, tuples-only person-stop,
EXPLORE approaching, picks at SEVERE, no forbid at SEVERE, stale slot applied, gate not closed before camera stop,
M2 on 1 box, no per-request check, no check after /completion, last-cycle SEVERE flag, counting in any slot, no
shutdown trace, apply regardless, no early forbid at SEVERE, no check after the affinity check …) makes at least one
test fail. The two round-4 tests passed 10/10 repeated runs.

**nav_test changes: none.** nav_test feeds `results=[]` only, so person-stop never applies there.

## 7. D4 photo table (owner decides the threshold; default unchanged)

Method (`photo_table.py`): every bench photo with a person (14 person rows + 3 couch rows with a person, no labelled
distance), EXIF-upright, resized to the RobotCam frame size keeping orientation (photos are portrait 3072×4080 →
480×640; RobotCam is 640×480 landscape), detected at 320 and 640; bucket of the person tuple navigate_rules sees.
Labelled distances are visual estimates (±30–50 cm, subject mostly reclining; labels.csv, #78).

| file | labelled cm | source size | size | person area | bucket | STOP at close | STOP at very close |
|---|---|---|---|---|---|---|---|
| roomsig_couch_1.jpg | - | 3072x4080 | 320 | 0.126 | close | yes | no |
| roomsig_couch_1.jpg | - | 3072x4080 | 640 | 0.123 | close | yes | no |
| roomsig_couch_2.jpg | - | 3072x4080 | 320 | 0.070 | medium | no | no |
| roomsig_couch_2.jpg | - | 3072x4080 | 640 | 0.069 | medium | no | no |
| roomsig_couch_3.jpg | - | 3072x4080 | 320 | 0.054 | medium | no | no |
| roomsig_couch_3.jpg | - | 3072x4080 | 640 | 0.054 | medium | no | no |
| person_facing_040cm.jpg | 45 | 3072x4080 | 320 | 0.098 | medium | no | no |
| person_facing_040cm.jpg | 45 | 3072x4080 | 640 | 0.101 | close | yes | no |
| person_sideon_040cm.jpg | 55 | 3072x4080 | 320 | 0.239 | close | yes | no |
| person_sideon_040cm.jpg | 55 | 3072x4080 | 640 | 0.266 | close | yes | no |
| person_facing_060cm.jpg | 180 | 3072x4080 | 320 | 0.154 | close | yes | no |
| person_facing_060cm.jpg | 180 | 3072x4080 | 640 | 0.148 | close | yes | no |
| person_sideon_060cm.jpg | 50 | 3072x4080 | 320 | 0.388 | very close | yes | yes |
| person_sideon_060cm.jpg | 50 | 3072x4080 | 640 | 0.385 | very close | yes | yes |
| person_facing_080cm.jpg | 95 | 3072x4080 | 320 | 0.279 | close | yes | no |
| person_facing_080cm.jpg | 95 | 3072x4080 | 640 | 0.356 | very close | yes | yes |
| person_sideon_080cm.jpg | 75 | 3072x4080 | 320 | 0.746 | very close | yes | yes |
| person_sideon_080cm.jpg | 75 | 3072x4080 | 640 | 0.717 | very close | yes | yes |
| person_facing_100cm.jpg | 120 | 3072x4080 | 320 | 0.228 | close | yes | no |
| person_facing_100cm.jpg | 120 | 3072x4080 | 640 | 0.213 | close | yes | no |
| person_sideon_100cm.jpg | 130 | 3072x4080 | 320 | no person | - | no | no |
| person_sideon_100cm.jpg | 130 | 3072x4080 | 640 | 0.147 | close | yes | no |
| person_facing_150cm.jpg | 170 | 3072x4080 | 320 | 0.259 | close | yes | no |
| person_facing_150cm.jpg | 170 | 3072x4080 | 640 | 0.263 | close | yes | no |
| person_sideon_150cm.jpg | 150 | 3072x4080 | 320 | 0.308 | very close | yes | yes |
| person_sideon_150cm.jpg | 150 | 3072x4080 | 640 | 0.322 | very close | yes | yes |
| person_facing_200cm.jpg | 215 | 3072x4080 | 320 | 0.106 | close | yes | no |
| person_facing_200cm.jpg | 215 | 3072x4080 | 640 | 0.264 | close | yes | no |
| person_sideon_200cm.jpg | 205 | 3072x4080 | 320 | 0.095 | medium | no | no |
| person_sideon_200cm.jpg | 205 | 3072x4080 | 640 | 0.094 | medium | no | no |
| person_facing_300cm.jpg | 235 | 3072x4080 | 320 | 0.130 | close | yes | no |
| person_facing_300cm.jpg | 235 | 3072x4080 | 640 | 0.129 | close | yes | no |
| person_sideon_300cm.jpg | 260 | 3072x4080 | 320 | 0.212 | close | yes | no |
| person_sideon_300cm.jpg | 260 | 3072x4080 | 640 | 0.203 | close | yes | no |

Observations: 'close' (default) stops in 26/34 rows, including subjects labelled 170–260 cm; 'very close' stops only
in 7/34 and misses both 40 cm-file photos (45/55 cm). At 45 cm (person_facing_040cm) the 320 frame gives 0.098 —
just under 'close' — so neither threshold stops on that frame (the 640 frame gives 0.101 'close'). The area bucket
does not track distance well on this set. The 25 cm ultrasonic rules are unchanged and still apply.

## 8. Phone check (untimed, agent resident, proot, no camera/motors) — run on the round-3 code

Not rerun after round 4 (its change: one early forbid() at SEVERE in run_mission and one cancellation check in the
worker; neither path runs in this check).

`setup_q4` started via server_manager (MemAvailable 2.52 GB with the server up), picker + M2 on
bench_photos/roomsig_couch_1.jpg, 3 picks, server stopped afterwards (verified no llama-server).

```text
photo roomsig_couch_1.jpg frame (480, 640) boxes 4 scene [('person', 0.92, 'left', 'close'), ('potted plant', 0.78, 'center', 'medium'), ('couch', 0.53, 'center', 'very close')]
main thread affinity [0, 1, 2, 3, 4, 5, 6, 7] pid 19073
M2 affinity {19082: [6, 7], 19080: [6, 7]} workers [19080] caller 19082
--- situation (pick 1, without M2 block) ---
Mission: explore and map the rooms
Scene: person (left, close), potted plant (center, medium), couch (center, very close)
Ultrasonic distance ahead: 400 cm
Current action: keep exploring (for 0 s)
Last moves: FORWARD
Rooms seen: living room
pick 1: trigger=start options=['keep exploring', 'go toward the person', 'stop and greet the person', 'look around in place'] probs={'keep exploring': 0.966, 'go toward the person': 0.018, 'stop and greet the person': 0.007, 'look around in place': 0.009} mode=EXPLORE picker_ms=3151 m2_ms=653 m2_boxes=4 m2_relations=5 tokens=202 fallback=None
pick 2: trigger=person_appeared options=['keep exploring', 'go toward the person', 'stop and greet the person', 'look around in place'] probs={'keep exploring': 0.977, 'go toward the person': 0.012, 'stop and greet the person': 0.005, 'look around in place': 0.006} mode=EXPLORE picker_ms=2994 m2_ms=784 m2_boxes=4 m2_relations=5 tokens=205 fallback=None
pick 3: trigger=heartbeat options=['keep exploring', 'go toward the person', 'stop and greet the person', 'look around in place'] probs={'keep exploring': 0.063, 'go toward the person': 0.03, 'stop and greet the person': 0.011, 'look around in place': 0.896} mode=LOOK picker_ms=3130 m2_ms=800 m2_boxes=4 m2_relations=5 tokens=207 fallback=None
--- M2 relations text ---
M2 relations:
- person [0] behind couch [3].
- potted plant [2] to the right of potted plant [1].
- potted plant [1] to the right of person [0].
- potted plant [1] behind couch [3].
- person [0] to the left of potted plant [1].
M2 affinity after {19082: [6, 7], 19080: [6, 7]}
stopped True
```
The same run on the round-1 code gave identical probabilities and choices (`phone_check_r1code.txt`). Pick 3 offered
"Current action: look around in place" and the model chose LOOK (0.896): the picker may tend to repeat the current
action. Picker quality is NOT evaluated. Times are not measurements (agent resident).

## 9. Reviews (verbatim) and validity

Procedure per round: frozen manifest, isolated copy (tracked files + candidate, no .git/ignored files), identity
snapshot before/after (SHA-256 of every file in copy and live tree excl. .git, HEAD, origin/main, ls-remote line,
stash, status --ignored), request with no-mutation rules and allowed checks, `codex --no-daemon exec
--skip-git-repo-check -s danger-full-access -m gpt-6.1-sol -c model_reasoning_effort=medium`, mechanical citation
check (`cite_check.py`). Diagnostics (stderr) kept separately in the work folder.

| Round | Started (UTC) | Exit | Identity | Citations | Validity | Verdict |
|---|---|---|---|---|---|---|
| 1 | 2026-10-15T17:32:59Z | 0 | unchanged | 18/19 exact (picker.py:104 quoted line 105) | INVALID (WORKFLOW step 7) | CHANGES REQUESTED |
| 2 | 2026-10-15T17:44:12Z | 0 | unchanged | 15/15 exact | VALID | CHANGES REQUESTED |
| 3 | 2026-10-15T17:52:24Z | 0 | unchanged | 14/14 exact | VALID | CHANGES REQUESTED |
| 4 | 2026-10-15T20:25:37Z | 0 | unchanged | 12/12 exact | VALID | APPROVED |

All round-1, round-2 and round-3 findings were accepted and fixed before the next round (requests r2/r3/r4 list
each change). Round 4 was authorized by the owner for exactly the two round-3 fixes. Round-4 change: main.py
run_mission calls picker.forbid() right after heat.decide() when severe_now(); picker.py checks cancellation after
the affinity check, before relations(); two tests; one existing test's fake result marked as finished before SEVERE
(test-only, declared in request_r4).

### Round 1 review (verbatim)

Changes are required. The passing tests miss multi-person safety and worker lifecycle cases.

1. **[P1] A second, closer person can bypass person-stop.**

Person-stop examines the legacy scene after it has discarded duplicate labels. With a higher-confidence far person followed by a lower-confidence close person, only the far person reaches this check. EXPLORE, APPROACH and `--no-picker` can therefore execute FORWARD despite a detected person exceeding the stop threshold.

main.py:474
```python
        close = [r[3] for r in results if r[0] == 'person' and r[3] in PERSON_STOP_BUCKETS]
```

detect_person.py:208
```python
            continue
```

Evaluate the safety condition against every person detection before label deduplication. Add a regression using two detection dictionaries through the real scene conversion; the current person-stop tests inject already-converted tuples.

2. **[P1] SEVERE blocks submissions but does not gate an existing worker or preserve the mode.**

If status becomes SEVERE while M2 runs, the worker can subsequently start Gemma: its gate tests ACTIVE alone, and SEVERE can remain ACTIVE. A completed pick can also change the mode at SEVERE because result application checks only the slot number.

picker.py:193
```python
            if not self.active.is_set():
```

main.py:506
```python
                                   ready=not (self.pick_inflight or severe or self.last_640 is None))
```

main.py:524
```python
        applied = r['slot'] == self.slot_n
```

Have the main thread publish whether optional work is permitted, and enforce it between worker stages and before applying results. Test a below-SEVERE submission followed by SEVERE during M2 and before result collection. The existing SEVERE test starts at SEVERE and submits only after recovery.

3. **[P1] Gemma can start new HTTP requests after PAUSE/HOLD begins.**

The worker checks ACTIVE before calling `decide()`, but `decide()` performs several tokenization requests followed by completion without another phase check. If PAUSE begins during tokenization, subsequent requests—including the expensive completion—still start while the camera is off. This exceeds the stated allowance for an already-sent request to finish.

picker.py:198
```python
            dist = self.gemma.decide(request['situation'] + '\n' + text, phrases)
```

picker.py:104
```python
        res = self.post('/completion', {'prompt': ids, 'n_predict': 1, 'n_probs': N_PROBS, 'post_sampling_probs': False,
```

Check cancellation before each HTTP request, using request/slot identity so a later ACTIVE phase cannot revive cancelled work. Add a fake transport test that changes phase during tokenization and verifies no completion starts.

4. **[P2] Worker shutdown can return while the worker is still running.**

Cleanup waits 15 seconds, but an ordinary pick contains five sequential HTTP requests, each with a 10-second timeout. Clearing ACTIVE during `decide()` does not interrupt that sequence. Several individually successful slow requests can therefore outlast cleanup; the daemon may continue issuing requests or be terminated without clean completion.

picker.py:219
```python
    def stop(self, timeout=TIMEOUT_S + 5):
```

picker.py:223
```python
        self.thread.join(timeout)
```

run_mission.py:93
```python
            cleanup_errors.append('[PICKER] worker still running after its timeout')
```

Make cancellation terminate the sequence after the current bounded request, and ensure cleanup observes worker termination. Test shutdown during an actual fake HTTP sequence; the existing worker-stop test stops an idle worker, while entry-point exit tests mock `stop()`.

5. **[P2] Picks crossing a phase boundary are omitted or charged to the wrong active slot.**

Results are collected only at the beginning of an ACTIVE cycle. A pick finishing during PAUSE is counted in the next slot when ACTIVE resumes, even though its result is rejected as belonging to an earlier slot. If the mission ends first, its result and fallback are never traced. Thus `pick` is not reliably emitted per call, and `active_slot` statistics can describe work from another slot.

main.py:386
```python
        self.collect_pick()  # a pick that finished meanwhile changes the mode from this cycle on
```

main.py:525
```python
        self.slot['picks'] += 1
```

Separate result tracing/accounting from mode application. Preserve the originating slot and handle terminal results during pauses and shutdown. Add a delayed-result test spanning ACTIVE → PAUSE → ACTIVE, plus mission termination with a pick outstanding.

The permitted check script completed successfully: candidate test counts were **21 / 17 / 25 / 10 / 10 / 25 / 32**, totaling **140**. Syntax checks passed; `nav_test.py` produced 41 lines and MD5 `fbbfe5b3248f1c930ac7c0d5fc67d4c6`. The supplied base counts total 93; I did not independently execute the base snapshot.

The remaining requested comparisons support these conclusions:

- **F1:** The copied completion payload, letter-probability normalization and first-offered tie handling match S1O/coresidency. The external prompt-builder parity test passed, including its long-state case.

benchmark/strategic_selector/v3/adapters.py:182
```python
        body = {"prompt": ids, "n_predict": 1, "n_probs": self.N_PROBS, "post_sampling_probs": False,
```

- **F2:** The benchmark variant delegates construction to `speed_block`. That file actually resides under `desk2/` in this copy; the `speed1/speed_block.py` provenance comment is inaccurate. The copied preprocessing, calibration, inference postprocessing, context formatting and highest-confidence box selection match the sources examined. The deliberate fewer-than-two-box behavior matches the task. The three embedded hashes match the campaign manifest, and the graph hash matches the parity-passed record. Hash validation precedes pickle-enabled bank loading.

benchmark/relate_anything/speed1/variants.py:68
```python
    return sb.build(NAME, cpus, threads, lambda name, threads: make_head(variant, threads))
```

relate.py:74
```python
            if sha256(os.path.join(directory, name)) != digest:
```

  M2 uses two threads on CPUs 6–7, checks its calling-thread affinity, and refuses main-thread pinning. Its asset directory is `/data/data/com.termux/files/home/models/relate_anything/relsgg-vits16`, containing `relateanything.onnx`, `relateanything.json` and `predicate_bank.npz`.

- **F3:** The diff confirms removal of the centimetre-based person check. With the detector’s clamped legacy height at most 640, the old estimate was at least approximately 133 cm, exceeding the old 80 cm threshold.

candidate.diff:101
```diff
-                    dist_est = estimate_distance_single('person', r[7])
```

- **F4:** The campaign uses the setup_q4 server command: four inference/batch threads, parallel 1, context 2048, SWA-full and cache RAM 0. It does not add a server taskset. `server_manager.py` is unchanged.

benchmark/coresidency/coresidency.py:404
```python
    `2>/dev/null &` replaced by a log file). No taskset: the robot does not pin the server."""
```

- **F5:** `picker.py` contains the MIT permission/warranty text and upstream attribution. The original external LICENSE and repository metadata were not furnished in this copy, so I could not independently establish their authenticity.

picker.py:12
```text
  Copyright (c) 2026 Rami Mithalouni
```

The situation template is:

```text
Mission: {mission}
Scene: {label} ({position}, {bucket}), ...
Ultrasonic distance ahead: {distance} cm
Current action: {mode phrase} (for {seconds rounded to zero decimals} s)
Last moves: {last three moves, comma-separated}
Rooms seen: {room names, comma-separated}
M2 relations:
{formatted relations}
```

Empty values use `nothing detected`, `no reading` and `none` as applicable. Options retain the required fixed order, and the instruction is exactly `Choose the single best next action for the robot.`

Python retains motor authority; the worker has no motor, camera or heat-state reference. Queue communication and immutable-by-use frame requests avoid an evident shared-data mutation race. Stale-frame STOP remains ahead of mode execution. HOLD preserves mission continuation. Ordinary request/M2/startup failures fall back to EXPLORE. The diff leaves heat thresholds/rules, SizePolicy, CAMERA_RATE, MOTOR_SPEED and CYCLE_MOVE_TIME unchanged. The `--no-picker` navigation path retains person approach, subject to finding 1.

Verification has an isolation limitation: the pre-existing absolute import path can resolve dependencies from the live tree. I did not independently verify the claimed live/copy byte identity. The new production modules contain no benchmark or external-schema import.

main.py:2
```python
sys.path.insert(0, '/data/data/com.termux/files/home/robot')
```

Line counts for every file inspected, including partially inspected sources:

| File | Lines |
|---|---:|
| docs/HANDOFF.md | 68 |
| docs/WORKFLOW.md | 303 |
| docs/REVIEWER.md | 15 |
| docs/CLAUDE.md | 127 |
| docs/STATUS.md | 449 |
| docs/DECISION_INDEX.md | 54 |
| docs/DECISIONS.md | 1669 |
| TASK.txt | 95 |
| candidate.diff | 1558 |
| check_output.txt | 185 |
| frozen_manifest.json | 42 |
| photo_table.md | 36 |
| phone_check.txt | 22 |
| main.py | 672 |
| run_cycle_safety_test.py | 409 |
| run_mission.py | 112 |
| test_heat_control.py | 543 |
| test_run_mission.py | 249 |
| picker.py | 232 |
| relate.py | 185 |
| test_picker.py | 332 |
| detect_person.py | 283 |
| heat_control.py | 345 |
| detector_size_policy.py | 32 |
| stereo_depth.py | 173 |
| server_manager.py | 116 |
| benchmark/strategic_selector/v3/adapters.py | 368 |
| benchmark/coresidency/coresidency.py | 1290 |
| benchmark/relate_anything/desk2/desk_check.py | 232 |
| benchmark/relate_anything/desk2/speed_block.py | 495 |
| benchmark/relate_anything/speed1/variants.py | 78 |
| benchmark/campaign/phase1.py | 1074 |
| benchmark/campaign/runtime.py | 723 |
| benchmark/campaign/expected_hashes.json | 9 |
| benchmark/relate_anything/desk2/parity_valid_relsgg-vits16.json | 7988 |
| ../checks.sh | 24 |

No files were edited. Photos, model assets and hardware behavior were not independently checked; the supplied phone output is an untimed functional observation.

VERDICT: CHANGES REQUESTED

### Round 2 review (verbatim)

Two remaining SEVERE boundary bugs require changes. No files were changed. All permitted checks passed, but the tests miss these interleavings.

1. [P1] **A completed pick can change the mode after Android reaches SEVERE.** `collect_pick()` uses `self.picker_severe`, which reflects the previous cycle’s `request_pick()`. If the monitor publishes SEVERE between cycles and a result is ready, collection applies that result before refreshing the flag. The second collection at the start of `run_cycle()` has the same ordering problem. This violates D2’s requirement to keep the current mode at SEVERE.

main.py:534
```python
        applied = own_slot and not r['cancelled'] and not self.picker_severe
```

main.py:579
```python
                self.collect_pick()  # also in PAUSE/HOLD: every pick is traced when it ends
```

main.py:508
```python
            self.picker_severe = severe
```

Refresh the thermal gate before collecting results, and validate the current thermal reading when deciding whether to apply them. Add a regression where a successful result is ready when the monitor first reports SEVERE; assert that the mode stays unchanged and the result is traced with `applied=False`.

2. [P1] **Cancellation during the final `/completion` request is lost.** Requests check permission before sending, but there is no generation check after `Gemma.decide()` returns. If `forbid()` happens while `/completion` is in flight, the worker still returns its chosen mode with `cancelled=False`. If SEVERE ends before that result is collected, the unchanged ACTIVE slot accepts it. A later `allow()` has therefore effectively revived cancelled work.

picker.py:220
```python
            dist = self.gemma.decide(request['situation'] + '\n' + text, phrases)
```

picker.py:225
```python
            out['mode'] = request['eligible'][phrases.index(choice)]
```

picker.py:229
```python
            out['cancelled'] = isinstance(e, Cancelled)
```

Check the captured generation after the final request and before publishing success. Add a real-Gemma/fake-transport test that blocks `/completion`, calls `forbid()` followed by `allow()`, then releases the response. The result must remain cancelled and must never change the robot mode.

The mission fake currently masks this distinction by directly rewriting its queued results when forbidden; the real picker does not do that.

test_heat_control.py:419
```python
            r.update(cancelled=True, mode='EXPLORE', fallback='M2: Cancelled: optional work not allowed')
```

The remaining review checks support these conclusions:

- Python retains motor authority. Modes shape Python rules; picker output never becomes a motor command. Person-stop checks both legacy tuples and every person detection, including a less confident second person. HOLD stops motors for the cycle and preserves mission continuation. Obstacle avoidance and stale-frame stopping remain ahead of forward execution. GREET’s stationary hold is consistent with the owner’s clarification.
- PAUSE/HOLD transitions forbid worker work before stopping the camera. The worker receives frame/detection data and returns result dictionaries; it does not access motors, camera controls or heat state. Queues provide the request/result handoff. Generation cancellation improves round 1, subject to finding 2.
- Startup and per-pick failures select EXPLORE without introducing a mission stop. `--no-picker` retains the original person-approach rules with the intended person-stop replacement; it constructs neither Gemma nor M2.
- The authoritative diff leaves heat thresholds, SizePolicy, CAMERA_RATE, stale-frame handling, MOTOR_SPEED and CYCLE_MOVE_TIME unchanged.

F1: The copied completion settings, letter-probability extraction, renormalisation and first-offered tie handling match S1O and coresidency. The prompt parity test passed, including truncation.

picker.py:110
```python
        res = self.request('/completion', {'prompt': ids, 'n_predict': 1, 'n_probs': N_PROBS, 'post_sampling_probs': False,
```

F2: M2 preprocessing, graph validation, calibration and relation extraction match `desk_check`; context formatting and confidence-based box selection match phase1. The implementation uses BIG cores 6–7 and two intra-op threads, with separate worker/caller pinning. All three constant hashes match `expected_hashes.json`; the graph hash matches the parity record with `passed=true`. Hash checks precede pickle-enabled bank loading.

relate.py:27
```python
CPUS = {6, 7}     # BIG (#131 L2)
```

relate.py:28
```python
THREADS = 2       # intra-op; ORT counts the calling thread as one of them
```

The requested `benchmark/relate_anything/speed1/speed_block.py` does not exist in this copy. `speed1/variants.py` delegates to the supplied `desk2/speed_block.py`, which is the source named by the candidate’s corrected provenance comment.

benchmark/relate_anything/speed1/variants.py:68
```python
    return sb.build(NAME, cpus, threads, lambda name, threads: make_head(variant, threads))
```

M2 assets are `relateanything.onnx`, `relateanything.json` and `predicate_bank.npz` beneath:

relate.py:20
```python
DIRECTORY = '/data/data/com.termux/files/home/models/relate_anything/relsgg-vits16'
```

F3: The base person-distance call is shown as removed in the authoritative diff, replaced by final-move area-bucket enforcement.

candidate.diff:101
```diff
-                    dist_est = estimate_distance_single('person', r[7])
```

F4: The supplied coresidency campaign server command follows `server_manager`: four generation/batch threads, no taskset. There is no demonstrated taskset difference between these two supplied implementations.

benchmark/coresidency/coresidency.py:404
```python
    `2>/dev/null &` replaced by a log file). No taskset: the robot does not pin the server."""
```

F5: `picker.py` includes attribution, source revision and the full MIT notice. No new production import targets benchmark modules or the external selector checkout. The external original LICENSE/repository metadata is absent from this copy, so its authenticity was not independently established here.

picker.py:12
```python
  Copyright (c) 2026 Rami Mithalouni
```

The situation template is exactly:

```text
Mission: {mission}
Scene: {label (position, bucket), ... | nothing detected}
Ultrasonic distance ahead: {distance cm | no reading}
Current action: {mode phrase} (for {seconds rounded to zero decimal places} s)
Last moves: {last three moves, comma-separated | none}
Rooms seen: {known room names, comma-separated | none}
M2 relations:
{formatted relations | - none above threshold. | - none (fewer than 2 objects).}
```

The instruction is `Choose the single best next action for the robot.` Options retain the fixed EXPLORE, APPROACH, GREET, LOOK order, excluding APPROACH/GREET when no person is visible.

I reran `sh ../checks.sh`: syntax checks passed; unit-test counts were **22 / 17 / 27 / 10 / 10 / 27 / 32**, all OK. Navigation stdout remained **41 lines**, MD5 **fbbfe5b3248f1c930ac7c0d5fc67d4c6**. Base counts remain supplied evidence, not an independently rerun base checkout. Existing tests cover substantial safety behavior; neither remaining SEVERE interleaving is covered.

Files inspected, with line counts (benchmark/documentation reads were targeted where applicable):

```text
TASK.txt — 95
candidate.diff — 1721
check_output.txt — 190
frozen_manifest.json — 42
photo_table.md — 36
phone_check.txt — 22
main.py — 684
picker.py — 255
relate.py — 185
run_mission.py — 118
run_cycle_safety_test.py — 425
test_picker.py — 382
test_heat_control.py — 586
test_run_mission.py — 251
detect_person.py — 283
heat_control.py — 345
detector_size_policy.py — 32
server_manager.py — 116
docs/HANDOFF.md — 68
docs/WORKFLOW.md — 303
docs/REVIEWER.md — 15
docs/STATUS.md — 449
docs/DECISION_INDEX.md — 54
docs/CLAUDE.md — 127
docs/DECISIONS.md — 1669
benchmark/strategic_selector/v3/adapters.py — 368
benchmark/coresidency/coresidency.py — 1290
benchmark/relate_anything/desk2/desk_check.py — 232
benchmark/relate_anything/desk2/speed_block.py — 495
benchmark/relate_anything/speed1/variants.py — 78
benchmark/campaign/phase1.py — 1074
benchmark/campaign/expected_hashes.json — 9
benchmark/relate_anything/desk2/parity_valid_relsgg-vits16.json — 7988
```

Additional files counted or exercised by the permitted checks:

```text
delta_r1_r2.diff — 529
../checks.sh — 24
nav_test.py — 35
test_detector_robotcam.py — 317
test_chat_mtp.py — 202
benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py — 468
benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py — 560
benchmark/llm_objective_setting/conversation/chat_mtp/oneshot.sh — 135
benchmark/llm_objective_setting/conversation/chat_mtp/run_chat_mtp.sh — 9
```

Not checked: live hardware, absent photos/assets, external licence metadata or live-tree identity. Risk: the two SEVERE interleavings remain despite passing tests.
VERDICT: CHANGES REQUESTED

### Round 3 review (verbatim)

Two remaining cancellation gaps require changes. The round-3 fixes are present, but they do not close these execution windows.

1. **[P1] SEVERE does not cancel the worker before vision runs.**

   When the next mission iteration observes SEVERE and remains ACTIVE, the worker stays allowed throughout camera reading, YOLO inference, and movement selection. Cancellation happens only in `request_pick()` afterward. An outstanding tokenization request can therefore finish and start further requests—including `/completion`—after the mission thread has observed SEVERE. This exceeds the permitted exception for an already-sent request.

main.py:584
```python
                target, reason = heat.decide(self.get_distance())
```

main.py:428
```python
                detections = self.detector.detect(frame, size)
```

main.py:453
```python
        self.request_pick(results, distance)  # never waits; M2 runs during the move below
```

main.py:505
```python
        (self.picker.forbid if severe else self.picker.allow)()  # also cancels a pick already running
```

   Close the worker gate when SEVERE is observed, before entering vision. Add a regression using the real Picker with fake transport: hold a tokenization request, publish SEVERE, block detector execution, then release tokenization. No subsequent HTTP request should start while detection remains blocked. Existing mission tests check eventual cancellation and mode retention, not this ordering.

2. **[P2] Cancellation during affinity verification still starts M2.**

   The worker checks permission before `check_affinity()`, which performs `/proc` reads and affinity syscalls. If the mission calls `forbid()` during those operations, the worker subsequently enters `relations()` without checking again. With at least two boxes, a new M2 inference starts after permission was withdrawn; cancellation is noticed only afterward.

picker.py:211
```python
            affinity = self.relate.check_affinity()
```

picker.py:215
```python
            text, out['m2_boxes'], out['m2_relations'], out['m2_ms'] = self.relate.relations(*request['frame_640'])
```

   Recheck cancellation after affinity verification and before starting M2. A deterministic regression can make the fake `check_affinity()` call `forbid()` before returning success, then assert that neither `relations()` nor Gemma runs. This finding is from static inspection; I did not run an additional, unauthorized reproduction.

The permitted `sh ../checks.sh` run passed: syntax checks, 147 tests across the seven suites, and nav_test’s unchanged 41-line output hash, `fbbfe5b3248f1c930ac7c0d5fc67d4c6`. Suite counts were 22 / 17 / 28 / 10 / 10 / 28 / 32. The reported base counts were not independently rerun because this copy contains the candidate.

The remaining safety paths inspected preserve Python’s motor authority. Person-stop considers every detected person, including a closer person discarded by label deduplication. Forward and diagonal-forward candidates become a non-ending HOLD. Obstacle navigation remains authoritative for moving modes; GREET stays stopped. Stale-frame handling precedes mode execution. The worker receives frame data and text and does not access motors, camera controls, or heat state. Startup and per-pick failure paths retain EXPLORE fallback behavior. `--no-picker` retains person approach and avoids constructing Gemma or M2. SizePolicy, CAMERA_RATE, MOTOR_SPEED, CYCLE_MOVE_TIME, and heat thresholds are unchanged in the authoritative diff.

The requested source comparisons established:

- **F1:** The copied prompt construction matches the single-question S1O builder, including truncation. Letter scoring preserves one-token completion, raw letter probabilities, renormalization, and first-option tie breaking. The original-builder parity test passed.

benchmark/strategic_selector/v3/adapters.py:194
```python
        return self.build(self.Example(state, [q]), self, shuffle_choice=False, train=False)["ids"]
```

benchmark/coresidency/coresidency.py:516
```python
    choice = max(keys, key=lambda k: dist[k])  # ties: first offered, as in ladder.py
```

- **F2:** Preprocessing, graph validation, calibration, relation decoding, context formatting, and confidence-based box selection agree with their copied sources. BIG uses two threads on CPUs 6–7. The three embedded asset hashes match the campaign manifest; the graph hash matches the parity-passed record. Hash validation precedes pickle-enabled bank loading. The requested `speed1/speed_block.py` path does not exist; `speed1/variants.py` delegates to `desk2/speed_block.py`.

benchmark/relate_anything/speed1/variants.py:14
```python
CLUSTERS = {'MID': ({4, 5}, 2), 'BIG': ({6, 7}, 2), 'LITTLE': ({0, 1, 2, 3}, 4)}
```

benchmark/relate_anything/desk2/desk_check.py:56
```python
        if parity.get('graph_sha256') != hashlib.sha256(graph.read_bytes()).hexdigest():
```

  Runtime M2 assets are `relateanything.onnx`, `relateanything.json`, and `predicate_bank.npz` beneath:

relate.py:20
```python
DIRECTORY = '/data/data/com.termux/files/home/models/relate_anything/relsgg-vits16'
```

- **F3:** The base person branch used the height estimate shown below. With valid detector coordinates, the minimum estimate is approximately 133 cm, above the old 80 cm threshold.

candidate.diff:101
```diff
-                    dist_est = estimate_distance_single('person', r[7])
```

- **F4:** Campaign server construction uses the setup_q4 binary and flags, with four generation and four batch threads. Neither command adds taskset; actual inherited scheduling remains an environment property.

benchmark/coresidency/coresidency.py:404
```python
    `2>/dev/null &` replaced by a log file). No taskset: the robot does not pin the server."""
```

- **F5:** `picker.py` includes attribution and the complete MIT permission and warranty text.

picker.py:12
```python
  Copyright (c) 2026 Rami Mithalouni
```

  I could inspect the external schema source, but its root LICENSE, LICENSE.md, LICENSE.txt, and pyproject.toml were absent. Thus the original licence declaration and claimed revision remain independently unverified; the embedded notice alone does not establish provenance. The new runtime modules contain no benchmark or external s1 imports. The pre-existing absolute repository path insertion remains, so some checks resolve dependencies from the live tree; live/copy byte identity remains coder-supplied evidence.

The exact situation-text shape is:

```text
Mission: {mission}
Scene: {label} ({position}, {bucket}), ...
Ultrasonic distance ahead: {distance} cm
Current action: {mode phrase} (for {rounded seconds} s)
Last moves: {last three moves}
Rooms seen: {room names}
M2 relations:
{phase1.context-formatted relation lines}
```

Empty values render as `nothing detected`, `no reading`, or `none`; fewer than two boxes produces `- none (fewer than 2 objects).` The instruction is `Choose the single best next action for the robot.` Eligible options retain the specified fixed order.

Files inspected, with total line counts; some supporting files were inspected only in part:

```text
docs/HANDOFF.md                                      68
docs/WORKFLOW.md                                    303
docs/REVIEWER.md                                     15
docs/STATUS.md                                      449
docs/DECISION_INDEX.md                               54
docs/CLAUDE.md                                      127
docs/DECISIONS.md                                  1669
docs/LICENSES.md                                     85
TASK.txt                                            95
candidate.diff                                    1768
delta_r2_r3.diff                                    134
check_output.txt                                   192
frozen_manifest.json                                42
photo_table.md                                      36
phone_check.txt                                     22
main.py                                            688
run_cycle_safety_test.py                            425
run_mission.py                                     118
test_heat_control.py                               601
test_run_mission.py                                251
picker.py                                          256
relate.py                                          185
test_picker.py                                     409
detect_person.py                                   283
heat_control.py                                    345
detector_size_policy.py                             32
server_manager.py                                 116
stereo_depth.py                                    173
nav_test.py                                         35
test_detector_robotcam.py                          317
test_chat_mtp.py                                   202
benchmark/strategic_selector/v3/adapters.py          368
benchmark/coresidency/coresidency.py                1290
benchmark/relate_anything/desk2/desk_check.py         232
benchmark/relate_anything/desk2/speed_block.py        495
benchmark/relate_anything/speed1/variants.py           78
benchmark/campaign/phase1.py                       1074
benchmark/campaign/expected_hashes.json                9
benchmark/relate_anything/desk2/parity_valid_relsgg-vits16.json  7988
benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py  468
benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py  560
benchmark/llm_objective_setting/conversation/chat_mtp/oneshot.sh  135
benchmark/llm_objective_setting/conversation/chat_mtp/run_chat_mtp.sh  9
../checks.sh                                        24
/data/data/com.termux/files/home/sel-candidates/s1o/src/s1/schema.py  153
/root/.codex/plugins/cache/ponytail/ponytail/5.2.0/skills/ponytail-review/SKILL.md  101
```

No files were changed. Hardware behavior, omitted photos/assets, original licence provenance, and live/copy hashes were not independently verified; owner threshold and greeting decisions remain pending.

VERDICT: CHANGES REQUESTED

### Round 4 review (verbatim)

No [P0]–[P3] findings. The two authorized round-4 fixes address the reported cancellation gaps; I found no additional correctness or safety defect requiring changes.

The mission closes the picker gate after the heat decision and before vision:

main.py:586
```
                    self.picker.forbid()  # before this cycle's vision: no further picker request at SEVERE
```

The worker checks cancellation again after accepting the affinity result, before entering M2:

picker.py:215
```
            check()  # a forbid() during the affinity check: no M2 run
```

Both have targeted regression tests. The SEVERE test uses the real Picker and Gemma with fake transport, releases an outstanding tokenization during blocked detection, and verifies cancellation without further HTTP requests. The affinity test verifies that neither relations nor Gemma runs after cancellation.

test_heat_control.py:541
```
    def test_severe_closes_the_real_worker_gate_before_vision(self):
```

test_picker.py:315
```
    def test_forbid_during_affinity_check_runs_no_m2(self):
```

Python retains movement authority. Person-stop checks every detected person, including a closer person discarded by label deduplication, and holds forward moves without ending the mission. Obstacle navigation remains authoritative for moving modes; GREET holds still. Stale-frame handling precedes mode execution. The worker receives frame/data references and never accesses motors, camera control, or heat state.

PAUSE/HOLD transitions forbid optional work before stopping the camera. Generation checks prevent a running cancelled pick from reviving after a later `allow()`. Results apply only to their still-open ACTIVE slot, when uncancelled and below the monitor’s latest SEVERE status:

main.py:538
```
        applied = own_slot and not r['cancelled'] and not self.severe_now()  # the reading now, not last cycle's
```

Startup and per-pick fallbacks continue in EXPLORE. `--no-picker` preserves the original person-approach rules except the authorized person-stop change. The candidate diff leaves heat thresholds, SizePolicy, CAMERA_RATE, stale-frame limits, MOTOR_SPEED and CYCLE_MOVE_TIME unchanged.

Copy-fidelity checks:

- **F1:** The campaign uses S1O’s schema builder and letter-probability scoring, then chooses the highest-probability offered option, with ties resolved by offered order. The copied request parameters, renormalization and choice match. Prompt-equivalence tests passed, including truncated state input.

benchmark/strategic_selector/v3/adapters.py:194
```
        return self.build(self.Example(state, [q]), self, shuffle_choice=False, train=False)["ids"]
```

benchmark/coresidency/coresidency.py:485
```
    from s1.schema import Example, Q, LETTERS, build
```

- **F2:** BIG means CPUs 6–7 and two threads. M2 preprocessing, calibration, decoding, relation formatting and highest-confidence box selection match their sources. The three constant hashes match the campaign manifest; the graph hash matches the passed parity record. Hash verification precedes pickle-enabled bank loading.

benchmark/relate_anything/speed1/variants.py:14
```
CLUSTERS = {'MID': ({4, 5}, 2), 'BIG': ({6, 7}, 2), 'LITTLE': ({0, 1, 2, 3}, 4)}
```

The asset directory contains `relateanything.onnx`, `relateanything.json` and `predicate_bank.npz`:

relate.py:20
```
DIRECTORY = '/data/data/com.termux/files/home/models/relate_anything/relsgg-vits16'
```

The requested `speed1/speed_block.py` is absent. `speed1/variants.py` imports the implementation from `desk2/speed_block.py`; I checked that implementation.

- **F3:** The removed centimetre check used the unchanged single-image formula. With person height 170, focal length 500 and a maximum 640-space box height of 640, the estimate cannot reach the former 80 cm threshold.

stereo_depth.py:36
```
    dist = (real_h * FOCAL_PX) / height_px
```

- **F4:** The campaign starts the same server binary/model and four-thread settings as `setup_q4`, without an explicit server taskset. Its measurement-thread affinity controls are separate. No server-manager change appears in the candidate.

benchmark/coresidency/coresidency.py:404
```
    `2>/dev/null &` replaced by a log file). No taskset: the robot does not pin the server."""
```

- **F5:** `picker.py` includes upstream attribution, pinned revision and the complete MIT notice. Production modules do not import benchmark code or the external S1O source; the external schema import is test-only.

picker.py:12
```
  Copyright (c) 2026 Rami Mithalouni
```

The situation template is:

```text
Mission: {mission}
Scene: {label} ({position}, {bucket}), ...
Ultrasonic distance ahead: {distance} cm
Current action: {mode phrase} (for {seconds rounded to an integer} s)
Last moves: {last three moves, comma-separated}
Rooms seen: {room names, comma-separated}
M2 relations:
{formatted relations}
```

Empty alternatives are `nothing detected`, `no reading`, and `none`. Fewer than two boxes produces `- none (fewer than 2 objects).` The instruction is exactly “Choose the single best next action for the robot.”

I reran `sh ../checks.sh` successfully: syntax checks passed; test counts were **22 / 17 / 29 / 10 / 10 / 29 / 32**, totaling **149**, all passing. Navigation output remained **41 lines**, MD5 **fbbfe5b3248f1c930ac7c0d5fc67d4c6**. Base counts are supplied evidence, not independently rerun here.

File line counts below include partially inspected sources and files compiled or checked by the permitted script.

| File | Lines |
|---|---:|
| docs/HANDOFF.md | 68 |
| docs/WORKFLOW.md | 303 |
| docs/REVIEWER.md | 15 |
| docs/STATUS.md | 449 |
| docs/DECISION_INDEX.md | 54 |
| docs/DECISIONS.md | 1669 |
| docs/CLAUDE.md | 127 |
| docs/LICENSES.md | 85 |
| TASK.txt | 95 |
| candidate.diff | 1823 |
| check_output.txt | 194 |
| frozen_manifest.json | 42 |
| photo_table.md | 36 |
| phone_check.txt | 22 |
| delta_r3_r4.diff | 104 |
| main.py | 690 |
| run_cycle_safety_test.py | 425 |
| run_mission.py | 118 |
| test_heat_control.py | 643 |
| test_run_mission.py | 251 |
| picker.py | 257 |
| relate.py | 185 |
| test_picker.py | 418 |
| detect_person.py | 283 |
| detector_size_policy.py | 32 |
| heat_control.py | 345 |
| robotcam_reader.py | 43 |
| stereo_depth.py | 173 |
| server_manager.py | 116 |
| test_detector_robotcam.py | 317 |
| test_chat_mtp.py | 202 |
| nav_test.py | 35 |
| ../checks.sh | 24 |
| benchmark/strategic_selector/v3/adapters.py | 368 |
| benchmark/coresidency/coresidency.py | 1290 |
| benchmark/relate_anything/desk2/desk_check.py | 232 |
| benchmark/relate_anything/desk2/speed_block.py | 495 |
| benchmark/relate_anything/speed1/variants.py | 78 |
| benchmark/campaign/phase1.py | 1074 |
| benchmark/campaign/runtime.py | 723 |
| benchmark/campaign/expected_hashes.json | 9 |
| benchmark/relate_anything/desk2/parity_valid_relsgg-vits16.json | 7988 |
| benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py | 468 |
| benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py | 560 |
| benchmark/llm_objective_setting/conversation/chat_mtp/oneshot.sh | 135 |
| benchmark/llm_objective_setting/conversation/chat_mtp/run_chat_mtp.sh | 9 |

Limits: hardware, photos, asset contents and upstream licence provenance were not independently rechecked. Existing absolute import paths make some checks depend on the live tree’s supplied byte-identity assurance. The round-3 phone output establishes neither round-4 hardware validation nor timing. This approval does not authorize a commit.

VERDICT: APPROVED

## 10. Frozen manifest (round 4 = current working tree, verified after the round)

| path | SHA-256 | lines |
|---|---|---|
| main.py | 6b87e1c5377667e9f29a0430a44effd2ecbfe71508edb21514aa974b99ef87d3 | 690 |
| run_cycle_safety_test.py | 0ef59b59695ce7a1066aaabb8acdd3c35570e42205b45f81cfb9cac36afcba60 | 425 |
| run_mission.py | 8a4b513509cb019067016c021f6cb3c0073d554ca053e3b94be51fba71a510dd | 118 |
| test_heat_control.py | fa8a512aec673bb71b42ab952e4ee09486f161eb87bff6b51c86c5fba241ff04 | 643 |
| test_run_mission.py | ce1d9e4069cf53ecaf9210c91ca653846e6df8ac8e00bec61b4776f64ce649b6 | 251 |
| picker.py | b152947be98f8bde83ad0d6e34118c999f90b59025308cef2f3ebc3cf745df66 | 257 |
| relate.py | 0f4790bd9f8822f2131adcafa6813983a2b1d5b8a76abaaea074a44ea258ad20 | 185 |
| test_picker.py | ac8f9ecc3d911763970552cb298b7cd2dcaf2558487259187d991869df2b3b44 | 418 |

## 11. Open items, risks and owner decisions

**Owner decisions (resolved 2026-10-15)**
1. Round-3 findings: round 4 authorized and done (APPROVED).
2. PERSON_STOP_BUCKETS = ('close', 'very close') kept. Known limit from section 7: at 45 cm the 320 frame gave 0.098
   ('medium'), so a single frame can miss a close person; the 640 frame gave 'close'; ultrasonic 25 cm still applies.
3. Greeting text accepted.

**Implemented choices (reviewed, recorded for Local AI)**
- Person-stop holds and continues (owner answer 2026-10-15); GREET never calls navigate_rules (holds still, avoids
  the 60 s escape-timeout STOP when someone stands close); LOOK keeps obstacle-ladder moves.
- Cadence: "ultrasonic blocks" = transition into < 25 cm; events in the 5 s gap stay pending (latest wins); first
  call at once; no call before this slot's first 640 frame.
- 10 s timeout per HTTP request; a pick is 4 /tokenize + 1 /completion (as S1O), so a hung server costs ≤ 10 s per
  request before fallback; stop() joins ≤ 15 s.
- M2 parity gate = SHA-256 constants (copied from expected_hashes.json / the parity record), no read of benchmark/.
- A pick computed while a person was visible may be applied after the person left (APPROACH then acts as EXPLORE;
  GREET holds until the person-left event triggers the next pick ≥ 5 s later).

**UNKNOWN / not measured**
- Any `run_mission.py --dry` with the picker (no mission run was done), real cycle period with picks, heat effect
  (the #135 dry run had no picker/M2; #132 measured L2 under the campaign).
- Picker accuracy for these four modes (NOT EVALUATED); M2 quality (NOT EVALUATED, owner knows).
- RAM with drive loop + Gemma + M2 in a mission (campaign co-residency measured, not this code).
- The campaign server's actual CPU mask (F4).
- Motors-live behaviour of HOLD/GREET/LOOK; FWD_L/FWD_R exist only as firmware/teleop names (Robot.move has no
  diagonals), covered by unit tests only.

**DOC DIFF items for Doc Keeper (later, after a commit)**
- LICENSES.md, "Copied source": system-one-open s1/schema.py (prompt builder) — MIT, Copyright (c) 2026 Rami
  Mithalouni, github.com/mithalouni/system-one-open @ 77f1f7cc; copied into picker.py with the full MIT notice;
  verified 2026-10-15 by reading LICENSE at that checkout.
- STATUS: PERSON_STOP_DIST removed → PERSON_STOP_BUCKETS; picker/M2 wired (run_mission default; --no-picker);
  #79 implemented (threshold pending); Gemma's role row; RelateAnything "not deployed" line.
