# MIND-B5 Coder self-check against every MIND-B4 review finding (rounds 1-3)

Paths: `b5.py` = benchmark/mind_picker/b5_mission/b5.py. "Reused" = imported from b4.py unchanged.

| B4 finding | B5 handling | Where / test |
|---|---|---|
| R1-1 P1 camera start failure leaves capture running | `cam['on'] = True` set before every start attempt (arm start and each pause-end restart); cleanup runs `b4.stop_camera(robot)` whenever `cam['on']`; a failed restart ends the arm with the camera still owned | `run_arm`, `mission_loop`; test `Loop.test_camera_start_failure_ends_and_leaves_camera_owned` |
| R1-2 P1 work ceiling not enforced | absolute deadline `Mission.deadline`; `Mission.start` refuses at/after it; scripted offers only for k*10 s < 1800; mission loop starts no cycle after it and cuts the last move; the decision in flight may finish; overrun = max(0, after scheduler join - deadline), INVALID > 15 s; request timeout 60 s (`b4.post`) | `Mission.start/offer_due`, `mission_loop`, `run_arm`, `arm_problems`; tests `Offers`, `Schedulers.test_nothing_starts_after_the_deadline`, `Labels.test_invalid` (overrun 15.5) |
| R1-3 P1 camera failures could be VALID | any stop other than 'no new' ends the arm and is INVALID; 'no new' re-read after 0.1 s, >3 INVALID; camera start/stop failure at a heat transition INVALID | `mission_loop`, `arm_problems`; tests `Loop.test_no_new_frames_limit`, `Labels.test_invalid` |
| R1-4 P1 resume did not freeze all inputs | `definition()`: SHA-256 of every imported module under the repo (b4, bc, b2, main, heat_control, detect_person, ...), `oneshot.sh`, the B2 run of record, the TEST set, and the full server command; `--resume` refuses any `b4.definition_diff` (reused). No frames used | `definition`, `main_`; test `Sessions.test_definition_and_resume_refusal` |
| R1-5 P1 endpoint-only agent checks | initial agent/cores refusal before any workload; B4's watcher (reused `b4.Watch`/`watch_check`: agents, runner cores 4-7, every llama-server thread 4-7, detector threads 4-5, 30 s resolution) during the whole arm | `run_arm`; tests `Labels.test_invalid` (watch hit, agents, cores) |
| R1-6 P2 incomplete readings could pass | every reading (start, pause_start, pause_end, end) checked with `b4.reading_problems` (section rc, required fields, per-PID Pss/VmHWM); need = runner + llama_server, + robotcam while the camera is on; a missing start/end reading is INVALID | `need`, `arm_problems`; test `Labels.test_robotcam_memory_needed_while_the_camera_is_on` |
| R1-7 P2 thread scheduling not validated | no extra sampler (task: start/end + pause readings only); the watcher's cores 0-3 / nice 10 checked by `b4.watch_problems` -> `thread_problems` (reused) | `arm_problems` |
| R1-8 P2 tests without a clock | Mission tests use explicit times; mission-loop tests use B4's `FakeClock` advanced by detect and pauses; one real-time arm with every resource faked | `test_b5.py` |
| R1-9 P3 cleanup not independently protected | each cleanup action (scheduler join, watch, camera, heat control, server) through `b4.cleanup` (reused); a failing one never skips the others; carried-over interrupt limitation stated in README | `run_arm` finally |
| R2-1 P1 metadata before the stop could skip it | `b4.stop_camera` reused at every pause start and at cleanup (affinity read in its own try; stop always attempted; capture shown stopped within 10 s) | `mission_loop`, `run_arm` |
| R2-2 P1 restore crossing the deadline | B5's restore is part of a decision that `Mission.start` admitted before the deadline (checked before the restore); it counts in that decision's wall and in the overrun. No decision or restore can start after the deadline. Deliberate difference from B4: no restore-only row | `decide`, `Mission.start` |
| R2-3 P2 memory summed before validation | `b4.reading` / `mem_section` reused (per-PID `missing`) | `arm_problems` |
| R2-4 P2 only the server leader's affinity | `b4.watch_check` reused (every `/proc/<pid>/task` thread) | watcher |
| R3 vanished threads | reused (`vanished` counted, not a hit) | watcher |
| R3 P3 restore stats omit restore-only rows | not applicable (no restore-only rows); restores are inside decision rows (`restore` field, wall included) | `decide` |

Other B4-run notes carried into B5: stale RobotCam process (stop stops capture, not the process): RobotCam memory is
required only while the camera is on; server exit != 0 after SIGTERM reported as FINDING (#145), not a validity input.
