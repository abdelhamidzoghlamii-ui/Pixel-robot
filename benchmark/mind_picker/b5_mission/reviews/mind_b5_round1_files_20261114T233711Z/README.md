# MIND-B5 — 30-minute motors-off mission: sustainable picker rate with camera, YOLO and heat pauses

Research only. With RobotCam + YOLO running (CAM1-MID, #136) and the deployed adaptive heat pause in force (37 °C
pause / 35 °C resume, #135), how many B2 V3 picker decisions does the phone serve in 30 minutes, how fast after an
event, how much time do heat pauses take, and does throttling change the picker's choices? Timings count only from
the owner's `oneshot.sh` run in native Termux with no agent resident.

- **Motors off.** `motors.py` and pyserial are never imported (`main.py` imports neither; `run_mission.py`, which
  imports `motors` without `--dry`, is not imported); `main.Robot(None)` has no motors (distance 999 cm, moves are
  sleeps). An arm refuses to start, and is INVALID, if `motors` or `serial` is in `sys.modules` at its start or end.
  No serial port, no ESP32, no MTP, no memory notes, no vision calls.
- **Setup per arm (not timed).** `main.get_detector(pinned=True)` first (run_mission's order); B4's server
  (`b4.server_command(<run>, 'P')`: b11716, setup_q4 flags, `--slot-save-path <run>/slots`, Q8_0 projector at 70
  image tokens; `bc.PINS`); the 40 B2 TEST prompts (V3 FWD, `b2.preflight`), the common prefix sent once and slot 0
  saved (`b4.warm`); `heat_control.HeatControl(HeatMonitor())` started (its own 1 s / 5 s root readings: essential
  safety monitoring; trace in the run folder); `robot.start_camera(heat)`; start reading; B4's 30 s watcher.
- **Mission loop** = `Robot.run_mission` motors-off: heat transitions exactly as `main.py` (picker forbidden, camera
  stopped and shown stopped, `heat.enter`; resume: `start_camera(heat)`, `heat.enter('ACTIVE', since=before the
  start)`); HOLD sleeps 1 s, PAUSE the move time; ACTIVE runs B4's camera cycle (new frame, SizePolicy, detect) then
  the move time. Picker allowed only while ACTIVE and Android status < SEVERE; a forbid cancels a running decision
  before its next HTTP request, for good. Heat END, a camera start/stop failure or an unusable frame ends the arm.
- **Events.** Scripted: the 40 B2 TEST items in id order, cycled, one every 10 s (180 per arm). Real: the most
  confident person's area bucket per ACTIVE cycle; appear / leave / bucket change → an event line in the B2 state
  format (mission M1, current activity exploring, memory none, all five actions; served, unscored). Pending queue of
  one: a newer event replaces an unserved one (dropped).
- **Arms** (30 min of mission time each, order A then B): **A FIXED** at each 30 s mark, if allowed and idle, decide
  on the pending event; **B ADAPTIVE** as soon as an event is pending, the previous decision finished, the picker is
  allowed and the heat control's latest VIRTUAL-SKIN < 36.0 °C. A decision: slot 0 restored before every 5th, then
  `b2.score` unchanged (a real event's prompt is rendered first, inside its wall). A cancelled decision's event goes
  back to pending unless a newer one is waiting. The scheduler polls every 0.1 s (latency resolution).
- **Deadline.** Nothing starts after it; the decision or camera cycle in flight finishes; overrun ≤ 15 s.
- **Readings** (`b4.reading`, per-PID memory checked): start, end, one at each heat-pause start (after the camera
  stop) and end (after the camera start). RobotCam memory required while the camera is on.
- **Gates.** Before every arm: cores 4–7, agent check, MEM-1 skin gate (launch-start skin + 1.0 °C, wait ≤ 30 min),
  status < 3, battery < 45 °C and ≥ 30 %. A timed run needs battery ≥ 85 % at its first start and ≥ 50 % at each
  `--resume`. Battery < 30 % at any reading ends the arm (INVALID) and the session.
- **Labels.** Arm VALID / INVALID by B4's rules (watcher hit, cores, missing or incomplete reading, camera failure,
  agent, request errors, overrun > 15 s) plus heat END and battery stop; an INVALID arm ends the session and is redone
  with `--resume`. A vs B descriptive (no matching claim). Energy INDICATIVE.
- **Metrics per arm:** decisions served; offered / served / dropped / cancelled / unserved (scripted and real);
  event→decision latency and decision wall median / p95; A's ticks; heat pauses, time to the first, total pause time,
  ACTIVE share; skin start / max (readings and heat trace) / end; minimum CPU caps at readings; camera cycle period
  (within one ACTIVE slot), detect ms 320 / 640; MemAvailable minimum; energy; agreement of served scripted choices
  with the B2 run of record (`b2_rules/runs/20261102T143251Z`, TEST V3 FWD, SHA-256 pinned).
- **Frozen definition:** SHA-256 of `oneshot.sh`, the run of record, the TEST set, every imported module under the
  repo, and the server command; `--resume` refuses any difference.
- **Run** (native Termux, no agent): `bash ~/robot/benchmark/mind_picker/b5_mission/oneshot.sh`, then `… oneshot.sh
  --resume <UTC>` until `all arms done`. `--smoke`: each arm 180 s, gate before the first arm only, agent/cores
  checks recorded but not enforced, SMOKE - NOT A MEASUREMENT. Smoke only: the deployed HeatControl runs with
  lowered thresholds set per arm from that arm's start reading (after server load and camera start): pause at arm-
  start skin + 1.0 °C, resume at pause − 0.5 °C; its 60 s active / 30 s pause rules unchanged; set at runtime,
  `heat_control.py` unchanged. The smoke report gives each arm's thresholds and its pauses (camera stop) and resumes
  (camera start), and says when no pause or no resume happened. Timed runs use 37/35 °C. `oneshot.sh` is B4's
  launcher with only its paths, runner name and header changed (a shell script cannot import B4's).
- Writes `~/robot_runs/mind_b5/<UTC>/`: run.json, sessionNN.json, sessionNN_end.json, arms.jsonl, requests.jsonl,
  NN_<arm>_server.log, heat_trace_<UTC>.jsonl, slots/, report.txt, artifacts.sha256.
- Offline tests: `python -B test_b5.py` (fakes and a fake clock; one short real-time arm with the deployed
  HeatControl over a fake monitor; imports B4's test fixtures).
- Known limits: battery % is seen only at readings (no extra sampler); a person flickering between frames produces
  several real events (the queue of one absorbs them); B4's carried-over interrupt/cleanup limitation.
