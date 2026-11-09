# MIND-B4 — timed baselines: picker, vision call, RobotCam+YOLO, idle, monitoring cost

Research only. Measured alone, with no agent running: how long one picker decision takes, what one vision call
costs, how RobotCam+YOLO behave, how fast each heats the phone, what RAM each uses, and how much 5 s root monitoring
slows the picker. Timings count only from the owner's `oneshot.sh` run in native Termux with no agent resident.
No motors (`motors.py` is never imported), no serial port, no mission, no MTP.

- Server (P, P0, PT, V, PY): BC's b11716 command, `bc.command(<run>/slots)` (setup_q4 flags, b11716 `llama-server`,
  Q8_0 projector, `--image-min/max-tokens 70`, `--slot-save-path <run>/slots`), pins `bc.PINS`; one launch per block.
- Picker: B2 V3 unchanged (`b2.preflight` / `b2.score`), no memory, B2 TEST items 1–20 FWD, repeated; system prefix
  sent once and slot 0 saved before the block (not timed); slot restored before the 5th, 10th, … decision.
- Vision: BV ids 1, 5, 6, 7, 9, 10, 16, 24, 30, 33 (frame SHA-256 checked), `bv.body('V70')`, repeated.
- Camera: `main.Robot(motors=None).start_camera()` / `stop_camera()` (rate 1, PinnedDetector cpus 4–5, #136); each
  cycle is `run_cycle`'s vision step plus the dry run's 1.5 s move time; no navigation, picker or heat control.
- Blocks I, P, P0, PT, V, Y, PY (at most 180 s of work each; P0 = two cold decisions, seconds only; PT = P + root
  sampler every 5 s on cores 0–3 at nice 10; PY = a P phase, then right away a Y phase with the server idle). Before
  each: cores 4–7, agent check, MEM-1 skin gate (launch-start skin + 1.0 °C, at most 8 min), status < 3, battery
  < 45 °C; a gate not passed, an agent or lost cores ends the session before the block; an INVALID block ends the
  session and is redone in the next. Details and labels: docstring of `b4.py`.
- Labels: block VALID / INVALID; P vs PT COMPARABLE only if both VALID, start skin within 1.0 °C and battery within
  10 %; other cross-block numbers descriptive; energy from the charge counter INDICATIVE; server exit code ≠ 0 is a
  FINDING (#145).
- Run (native Termux, no agent): `bash ~/robot/benchmark/mind_picker/b4_baseline/oneshot.sh`, then
  `… oneshot.sh --resume <UTC>` until the session ends with `all blocks done`. `--smoke`: 20 s blocks, no gate,
  checks not enforced, SMOKE - NOT A MEASUREMENT (an agent may stay resident).
- Writes `~/robot_runs/mind_b4/<UTC>/`: run.json, sessionNN.json, sessionNN_end.json, blocks.jsonl, requests.jsonl,
  NN_<block>_server.log, slots/, report.txt, artifacts.sha256 (the last two rewritten at each session end); the
  launcher's console logs and `thermal.log` are in `~/robot_runs/mind_b4/`.
- Offline tests: `python -B test_b4.py` (fakes; no server, camera, root or phone).
- Known limitation carried over from B0: an interrupt or a server-stop error can leave cleanup incomplete.
