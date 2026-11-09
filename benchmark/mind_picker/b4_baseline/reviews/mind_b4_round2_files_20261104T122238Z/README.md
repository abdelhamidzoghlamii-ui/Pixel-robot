# MIND-B4 — timed baselines: picker, vision call, RobotCam+YOLO, idle, monitoring cost

Research only. Measured alone, with no agent running: how long one picker decision takes, what one vision call
costs (70 and 140 image tokens), how RobotCam+YOLO behave, how fast each heats the phone, what RAM each uses, and how
much 5 s root monitoring slows the picker. Timings count only from the owner's `oneshot.sh` run in native Termux with
no agent resident. No motors (`motors.py` is never imported), no serial port, no mission, no MTP.

- Server (P, P0, PT, V70, V140, PY): BC's b11716 command, `bc.command(<run>/slots)` (setup_q4 flags, b11716
  `llama-server`, Q8_0 projector, `--image-min/max-tokens 70`, `--slot-save-path <run>/slots`); V140 changes only
  `--image-min-tokens` and `--image-max-tokens` to 140. Pins `bc.PINS`; one launch per block; request timeout 60 s.
- Picker: B2 V3 unchanged (`b2.preflight` / `b2.score`), no memory, B2 TEST items 1–20 FWD, repeated; the common
  prompt prefix sent once and slot 0 saved before the block (not timed); slot restored before the 5th, 10th, …
  decision.
- Vision: BV ids 1, 5, 6, 7, 9, 10, 16, 24, 30, 33 (frame SHA-256 checked), `bv.body`, repeated; V70 and V140 use the
  same frames, prompt and settings.
- Camera: `main.Robot(motors=None)` is created first, then `start_camera()`; `stop_camera()` is attempted whenever a
  start was attempted (rate 1, PinnedDetector cpus 4–5, #136). Each cycle is `run_cycle`'s vision step plus the dry
  run's 1.5 s move time; no navigation, picker or heat control. "no new" frames: at most 3 per block; any other camera
  stop ends the phase and makes the block INVALID.
- Blocks I, P, P0, PT, V70, V140, Y, PY. Each work phase has an absolute 180 s deadline: nothing new starts after it,
  the operation in flight may finish, the overrun is recorded and must be ≤ 15 s (P0, two cold decisions, has no
  deadline). PT = P + root sampler every 5 s (its thread must get cores 0–3 at nice 10); PY = a P phase, then right
  away a Y phase with the server idle (hand-off reading after the camera start).
- Before each block: cores 4–7, agent check, MEM-1 skin gate (launch-start skin + 1.0 °C, idle wait at most
  30 min per block, recorded as cooling data), status < 3,
  battery < 45 °C; a gate not passed, an agent or lost cores ends the session before the block, and the block itself
  checks agent and cores again before any setup. During every block a watcher (cores 0–3, nice 10) scans `/proc`
  every 30 s plus at the work start and end for claude/agy/codex/node processes and the affinities (runner cores
  4–7, llama-server 4–7, detector threads 4–5); any hit makes the block INVALID. Resolution 30 s: an agent that lives
  entirely between two scans is not seen. Every reading's section return codes and required fields are checked.
  An INVALID block ends the session and is redone in the next. Details: docstring of `b4.py`.
- Frozen definition: the first session records the SHA-256 of `oneshot.sh` and every imported module file under the
  repo, every server block's full command and the SHA-256 of the 10 frame files; `--resume` refuses any difference.
- Labels: block VALID / INVALID; P vs PT COMPARABLE only if both VALID, start skin within 1.0 °C and battery within
  10 %; other cross-block numbers descriptive; energy from the charge counter INDICATIVE; server exit code ≠ 0 is a
  FINDING (#145).
- Run (native Termux, no agent): `bash ~/robot/benchmark/mind_picker/b4_baseline/oneshot.sh`, then
  `… oneshot.sh --resume <UTC>` until the session ends with `all blocks done`. `--smoke`: every block once for 20 s,
  gate before the first block only (MEM-1's default wait; later blocks ungated), agent/cores checks recorded but not enforced, SMOKE - NOT A MEASUREMENT.
- Writes `~/robot_runs/mind_b4/<UTC>/`: run.json, sessionNN.json, sessionNN_end.json, blocks.jsonl, requests.jsonl,
  NN_<block>_server.log, slots/, report.txt, artifacts.sha256 (the last two rewritten at each session end); the
  launcher's console logs and `thermal.log` are in `~/robot_runs/mind_b4/`.
- Offline tests: `python -B test_b4.py` (fakes; no server, camera, root or phone).
- Known limitation carried over from B0: an interrupt during resource start can leave cleanup incomplete; each
  cleanup action (watcher, sampler, camera, server) is protected separately.
