CHANGES REQUESTED

This is a source-only static review of the supplied packet. No tools, shell commands, file reads, edits, web access or sub-agents were used. As specified, Codex’s read-only sandbox cannot execute commands in proot (#144 R3). All test results, smoke artifacts, hashes and repository-status statements are **supplied evidence**, not independently verified runtime results.

The candidate handles much of the requested machinery, but cancellation, scheduler initialization, resume timing, deadline enforcement and heat accounting need changes before the timed run.

1. **P1 — A decision forbidden during its final request can still count as served.**

   `benchmark/mind_picker/b5_mission/b5.py:250` — **808 lines**:
   > `r = b2.score(rec, it, 'V3', 'FWD', prompt)`

   `benchmark/mind_picker/b5_mission/b5.py:223` — **808 lines**:
   > `return super().__call__(path, body)`

   `Checked` checks permission only before sending a request. If PAUSE or SEVERE forbids the picker while the final `/completion` is running, its response can return, `b2.score` can finish, and the decision gets `top` and fate `served`. The generation change is never checked again. This differs from deployed `Picker.pick`, which checks after Gemma returns.

   `picker.py:222` — **257 lines**:
   > `check()  # a forbid() while /completion was in flight cancels it too`

   **Fix:** Recheck the captured generation and permission after each HTTP response, and after scoring before accepting the result. Route cancellation through the existing requeue/fate handling. Add tests that forbid during the final completion, including forbid followed by allow before the response returns.

2. **P1 — A camera cycle can start after the deadline when a heat transition or reading crosses it.**

   `benchmark/mind_picker/b5_mission/b5.py:296` — **808 lines**:
   > `while clock() < m.deadline:`

   `benchmark/mind_picker/b5_mission/b5.py:343` — **808 lines**:
   > `r = main.read_frame(session=robot.camera_session)`

   The only loop deadline check precedes `heat.decide`, camera restart and the required pause-end reading. Those operations can cross the deadline, after which execution still reaches `read_frame` and YOLO. A real event can then also be offered after mission time ends. An overrun below 15 seconds would not invalidate this prohibited extra cycle.

   **Fix:** Recheck the deadline immediately before beginning each camera cycle. A transition already underway may finish and receive its required reading, but must not admit a subsequent cycle. Define and test whether detections from an admitted cycle finishing after the deadline may offer real events; do not silently serve them as mission-time events.

3. **P1 — Arm A can consume its first mark before the picker has been initialized as allowed (O1).**

   `benchmark/mind_picker/b5_mission/b5.py:127` — **808 lines**:
   > `self.ok, self.gen, self.k_offer, self.k_tick, self.n_started = False, 0, 0, 0, 0`

   `benchmark/mind_picker/b5_mission/b5.py:477` — **808 lines**:
   > `th.start()`

   The scheduler starts before `mission_loop` performs its first heat assessment and `set_ok`. Its first poll can offer item 1 and permanently classify the zero-second mark as paused. Supplied S2 evidence shows this happened: A first served item 4 at 30 seconds.

   **O1 is confirmed as a startup race.** “Systematically” losing exactly one mark is stronger than the source establishes: thread scheduling could let the mission thread initialize first.

   **Fix:** Establish the initial heat/SEVERE admission state before starting the scheduler and consuming any marks, with a defined mission start barrier. Test that an initially ACTIVE, non-SEVERE arm serves its zero-second mark, and that an initially forbidden arm does not.

4. **P1 — Resume keeps the picker forbidden during the required reading, changing the scheduler’s result (O2).**

   `benchmark/mind_picker/b5_mission/b5.py:331` — **808 lines**:
   > `if not read('pause_end'):`

   `benchmark/mind_picker/b5_mission/b5.py:335` — **808 lines**:
   > `m.set_ok(heat.phase == 'ACTIVE' and not sev)`

   Permission is restored only after `pause_end` returns. Deployed `main.py` allows the picker directly after entering ACTIVE:

   `main.py:615` — **690 lines**:
   > `if self.picker: self.picker.allow()`

   This extra forbidden interval can eliminate an entire A mark and delay B. Supplied S2 evidence shows A’s 90-second mark was lost around the restart, while B’s pending event waited until 124.700 seconds.

   **O2’s ordering defect is confirmed; its timing explanation needs correction.** The heat row records the restart attempt’s beginning, not completed camera start. Also, `b4.reading` records `mono` before its root call:

   `benchmark/mind_picker/b4_baseline/b4.py:166` — **1031 lines**:
   > `c = {'utc': cr.utc(), 'mono': round(time.monotonic(), 3), 'caps': caps()}`

   Thus `pause_end@90.585` is not the reading’s return time. The supplied trace places successful camera start near that timestamp; the following reading adds further delay.

   **Fix:** Restore permission after successful camera start and ACTIVE entry, using a fresh SEVERE assessment, before the reading. If the reading triggers battery stop or raises, immediately forbid and stop the arm. Test with a reading that advances time across an A mark and with B pending at resume.

5. **P1 — Reported pause time excludes camera restart, overstating ACTIVE share.**

   `benchmark/mind_picker/b5_mission/b5.py:305` — **808 lines**:
   > `add({'kind': 'heat', 'from': heat.phase, 'to': target, 'reason': reason, 't': m.rel(clock())})`

   `benchmark/mind_picker/b5_mission/b5.py:549` — **808 lines**:
   > `out.append((began, r['t']))`

   The ACTIVE transition row is emitted **before** `start_camera`. `pauses` closes the pause at that row, even though the camera is still restarting and heat phase remains PAUSE. Restart failure would also be counted as a resume. The smoke report consequently understates unavailable time and overstates ACTIVE share.

   Supplied S2 A evidence has the resume row at 88.778 seconds, with successful camera start roughly 1.8 seconds later. B shows the same pattern.

   **Fix:** Record transition intent separately from successful completion. Close pause accounting and count a resume only at successful ACTIVE entry. State explicitly how stop/start overhead and reading overhead contribute to phase time and picker availability. Add fake-clock tests with nonzero restart duration and failed restart.

6. **P2 — The initial agent/core refusal produces an arm record that the report cannot read.**

   `benchmark/mind_picker/b5_mission/b5.py:428` — **808 lines**:
   > `return rec, rows`

   `benchmark/mind_picker/b5_mission/b5.py:626` — **808 lines**:
   > `r, x = final[k], final[k]['metrics']`

   The early refusal path returns without `metrics`. `session_loop` saves that INVALID record, then final reporting raises `KeyError: 'metrics'`. This prevents normal report and artifact-manifest completion and can recur on later sessions containing that record.

   **Fix:** Give every returned arm record the same schema, including empty metrics, or make reporting explicitly handle an arm refused before workload setup. Test the real early-refusal path through serialization and report generation.

7. **P2 — Tests cover the requested categories but miss the integration boundaries exposing these defects.**

   `benchmark/mind_picker/b5_mission/test_b5.py:27` — **729 lines**:
   > `m.set_ok(True)`

   `benchmark/mind_picker/b5_mission/test_b5.py:698` — **729 lines**:
   > `self.assertLessEqual(len(dec), 9)  # one per 0.5 s mark at most`

   Scheduler unit tests pre-enable permission, hiding startup ordering. The integrated A test asserts only an upper bound, allowing a lost initial mark. Fake loop readings return immediately, restart does not advance fake time, and cancellation tests do not forbid during the final response. Definition testing exercises the diff helper rather than the actual resume-refusal entry path.

   **Fix:** Add deterministic integration tests for findings 1–6, including exact eligible A marks, timed restart/read operations, post-response cancellation, transition crossing the deadline, and refusal-to-report completion. Retain the existing queue, scheduler, latency and label tests.

8. **P3 — Start-reading relative time is absent (O3).**

   `benchmark/mind_picker/b5_mission/b5.py:436` — **808 lines**:
   > `c['t'] = m.rel(c['mono']) if m else None`

   The start reading precedes `Mission` creation, so `start@None` is expected. **O3 is confirmed and cosmetic** for the current metrics; absolute monotonic time remains available.

   **Fix:** Once mission `t0` exists, backfill the reading’s actual relative timestamp, usually negative, or render it explicitly as a setup reading. Do not label it zero unless that is its actual timestamp.

The B4 checklist assessment below concerns the stated original findings; it does not imply that the new mission-specific behavior is correct.

| B4 checklist row | Status | Assessment |
|---|---|---|
| R1-1 Camera start failure leaves capture running | Handled | Ownership precedes initial/restart attempts; cleanup attempts stop. |
| R1-2 Work ceiling not enforced | Partly | Decision admission and overrun checks exist; post-transition camera admission still violates the deadline, finding 2. |
| R1-3 Camera failures could be VALID | Handled | Camera failures generate invalidating errors; “no new” retains B4’s bounded allowance. Successful-resume reporting needs finding 5 fixed. |
| R1-4 Resume did not freeze all inputs | Handled | Imported repository modules, launcher, TEST set, record and server command enter the definition; resume compares it. Import closure outside the excerpts remains unverified. |
| R1-5 Endpoint-only agent checks | Handled | Imported watcher scans throughout the arm, with the documented 30-second blind spots. |
| R1-6 Incomplete readings could pass | Handled | Existing readings use B4 validation and camera-dependent memory requirements; missing start/end invalidate. |
| R1-7 Thread scheduling not validated | Handled | Watcher scheduling is validated through imported B4 helpers; no optional sampler is added. |
| R1-8 Tests without a clock | Partly | Explicit/fake time exists, but critical integrated timing boundaries are missed, finding 7. |
| R1-9 Cleanup not independently protected | Handled | Cleanup actions have separate protection. The acknowledged inherited interrupt limitation remains. |
| R2-1 Metadata before stop could skip it | Handled | Imported stop helper isolates affinity metadata from the stop attempt. |
| R2-2 Restore crossing deadline | Handled under accepted choice 6 | Restore belongs to an admitted decision and contributes to wall/overrun. No B4 restore-only row is required here. |
| R2-3 Memory summed before validation | Handled | Imported parser retains per-PID missing-field information. |
| R2-4 Only server leader affinity checked | Handled | Imported watcher enumerates server task threads. |
| R3 Vanished threads | Handled | Imported watcher counts disappearing threads separately from hits. |
| R3 Restore-only statistics omitted | Handled / inapplicable | B5 has no restore-only rows; restore wall is included in its decision. |

| Coder observation | Assessment | Rating |
|---|---|---|
| O1 | Confirmed startup race; S2 demonstrates it, but universal/systematic occurrence is unproved. | P1, finding 3 |
| O2 | Confirmed permission-order defect; reported timestamps conflate restart intent, successful start and reading completion. | P1, finding 4 |
| O3 | Confirmed missing relative setup timestamp; current calculations do not depend on it. | P3, finding 8 |

The **supplied evidence** reports 41 passing tests. Source inspection confirms coverage of both schedulers, queue replacement/drop counts, cancellation/requeue, latency arithmetic, pause/resume, labels, battery thresholds and session ordering. It does not support treating those categories as exhaustively verified.

The **supplied S2 smoke evidence** supports the owner’s prerequisite: both arms show pauses, successful camera restarts and subsequent camera cycles. It also reports clean watcher scans, server exit zero, restored screen timeout and passing artifact checks. Its event totals reconcile: A has 18 offers = 2 served + 15 dropped + 1 unserved; B has 18 = 12 served + 6 dropped. However, `problems: []` does not detect findings 1–5. S2 directly exposes the lost A marks and transition-accounting problem. With zero real events and zero cancelled attempts, it provides no phone evidence for those paths.

On other scoped requirements, the visible source implements B’s strict `<36.0` admission rule, one pending event shared across sources, original-offer latency after requeue, 85% first-start / 50% resume thresholds, and the 30% cutoff at readings and before arms. Smoke threshold mutation is guarded by `ctx['smoke']`; the CLI rejects smoke plus resume, so no ordinary timed CLI path to lowered thresholds is visible.

The visible dry-run path constructs `Robot(None)`, imports no motor entry point, and checks `sys.modules` at arm boundaries. That supports the design claim; module-name checks alone are not proof that no serial port was opened. The launcher comparison supports the claimed path/header/runner adaptation from B4.

This static review cannot establish actual runtime behavior, hardware safety, complete transitive import behavior, native affinity, capture cessation, thermal response, battery behavior between readings, or 30-minute sustainability. It cannot independently verify HEAD/origin equality, hashes, untracked-only repository status, unchanged deployed files or the Coder’s unarchived mutation checks. No commit, deployment or timed run is approved by this verdict.