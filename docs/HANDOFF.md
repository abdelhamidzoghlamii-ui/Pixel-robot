# HANDOFF — Pixel Robot

Self-contained brief for continuing this work with any assistant (Claude,
Codex, ChatGPT). Assumes no prior conversation. Written 2026-09-05.

---

## 0. What this project is

An offline autonomous robot. A rooted Pixel 7 runs the AI; an ESP32 drives
four mecanum wheels.

```
Pixel 7 (rooted, Termux, Android 17)
  llama.cpp server on :8080  — Gemma 4 E2B Q4_K_M, 3.3 GB, ~11.5 tok/s
  YOLO yolo11m.onnx          — object detection, every cycle
  whisper.cpp                — voice in
  main.py                    — the robot brain (perceive → decide → act)
      |
      | USB serial, CP2102, 115200 8N1
      v
ESP32 NodeMCU + 2x MX1508 + 4 mecanum motors + HC-SR04 ultrasonic
```

Two modes exist. **Mode 1 teleop** works: drive from a phone browser.
**Autonomous** runs the full loop but has never driven with motors live.

---

## 1. Read these first — they are canonical

In the project files:

| File | What it is |
|---|---|
| `STATUS.md` | Current state, config values, what works vs pending |
| `DECISIONS.md` | Append-only log of settled tradeoffs, numbered |
| `COMMANDS.md` | Every command that runs on the phone |
| `CLAUDE.md` | Coding guidelines for this repo |
| `FILES.md` | Dependency map — which files are live, which are dead |
| `BENCHMARK_PLAN.md` | Design for the replacement nav benchmark |
| `WORKFLOW.md` | Provider-independent roles, reviewer procedure, and human commit gate |
| `CODER.md` / `REVIEWER.md` | Execution and independent-review roles |

**Config values come from STATUS.md, not from memory.** If STATUS and a
claim disagree, flag it — do not silently pick one.

---

## 2. Doc discipline — non-negotiable

Docs update via **DIFFS**, never full rewrites. No working thread
regenerates STATUS.md; that causes section-clobbering.

- `DECISIONS.md` is **append-only**. Never edit an existing entry — supersede
  it with a new numbered one that references the old.
- `STATUS.md` has ONE writer (the Doc Keeper thread). Working threads emit
  targeted line-level changes.
- At the end of a working session, emit a `DOC DIFF` block:

```
DECISIONS: append #NN — <one line + why>
STATUS: <section> — change "<old>" to "<new>"
```

Read the last entry in `DECISIONS.md` directly to determine the next number. If a number is uncertain, flag it — do not guess.

---

## 3. Hard environment facts

**Root is required for all USB/motor work.** `/dev/bus/usb` is not readable
by Termux's app UID, and root does not inherit Termux's environment:

```bash
su -c "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib \
  /data/data/com.termux/files/usr/bin/python \
  /data/data/com.termux/files/home/robot/<script>.py"
```

Note the interpreter is `.../usr/bin/python`, not `python3`, in the root form.

**Port 8080 is contested.** llama-server and `teleop.py` both bind it and
cannot run together. Autonomous is unaffected (`run_mission.py` is a client).

**`pkill -f <pattern>` kills your own shell** in Termux. Find the PID with
`su -c "ps -A | grep python"` and kill by number.

**The board is CP2102 (`10c4:ea60`), not CH340.** Endpoints 0x01 OUT /
0x81 IN. `usb.util.claim_interface()` is mandatory — without it every bulk
read times out silently. DTR/RTS must be left untouched; asserting them
resets the ESP32.

**Thermal:** BIG cores run 97-101 °C under inference, sitting exactly on the
kernel's 100 °C passive trip point. That is normal, not a fault. Idle is
31-38 °C. Die temperature is not skin temperature.

---

## 4. State: what works, what doesn't

### Working

- LLM server stable, prompt-cache fix (`--swa-full`) landed and verified
- YOLO scene detection
- Voice pipeline end to end
- Mode 1 teleop from the Pixel (`teleop.py`) — web UI, diagonals, speed
  slider, live distance readout, forward blocked under 25 cm
- Serial link live end to end: Pixel → CP2102 → ESP32 → wheels
- Distance sensor live, `DIST:` flowing and verified
- Autonomous **dry run** only: 24 cycles, camera → YOLO → rules → Gemma

### Not working / not done

- **Autonomous has never run with motors live.**
- **OPEN SAFETY DEFECT (DECISIONS #81)** — see section 5.
- Rotation uncalibrated — nobody knows the degrees per command
- Forward motion uncalibrated — tooling written, never run against a wall
- `PERSON_STOP_DIST` unreachable as coded (#78) — decision made to move to
  the area bucket (#79), not implemented
- `BACK_R` diagonal drove one wheel instead of two; not retested since the
  corner-map fix
- Nav logic **unbenchmarked** — the old 93% figure does not describe live
  code (#55)

---

## 5. The open safety defect — read this before touching nav code

**DECISIONS #81.** In `main.py: run_cycle`, the guard is:

```python
use_gemma = (... or stuck) and not (safety_move and not stuck)
```

At escalation ladder step `n == 7`, `nav_stuck` is set. So `safety_move` and
`stuck` are both true, the guard admits the Gemma call, and the parse loop
overwrites `move` with the model's output — including `FORWARD` — with no
re-check afterwards.

Exposed window: distance under 25 cm, robot boxed in, both sweep directions
exhausted. Exactly the case DECISIONS #53 was written to prevent.

**The fix must re-check the move after the Gemma block.** It must be
verified against `nav_test.py` AND a `run_cycle`-level test that does not
yet exist — `nav_test.py` exercises `navigate_rules` in isolation and
structurally cannot reach this bug.

Do not run motors-live autonomous until this is fixed.

---

## 6. Work queue, in priority order

### A. Fix DECISIONS #81 (software, no hardware)
Highest priority. Blocks the first motors-live run.

### B. Implement #79 — person-stop via area bucket (software)
`navigate_rules` still calls `estimate_distance_single`, which #78 proved
unusable. Replace with the area-ratio bucket (field `r[3]`, DECISIONS #14).
The threshold ('very close' vs 'close') is not yet chosen and must be picked
against `bench_photos/`.

### C. Build the replacement nav benchmark (software)
Design is in `BENCHMARK_PLAN.md`. Must feed real photos through real
`detect_scene()` into real `navigate_rules()`, distance injected as a
parameter. Score two things separately, both label-free:
- format validity: is the move one of the 7 valid tokens
  (FORWARD, BACK, LEFT, RIGHT, STOP, STRAFE_LEFT, STRAFE_RIGHT)
- safety violations: `distance < 25 and move == 'FORWARD'`

This unblocks any model-swap decision, since the accuracy gate is currently
undefined.

### D. Forward-motion calibration (NEEDS ROBOT)
Robot square-on to a flat wall 2-3 m back, clear floor:
```bash
cd ~/robot
su -c "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib \
  /data/data/com.termux/files/usr/bin/python \
  /data/data/com.termux/files/home/robot/log_run.py 130 run130.csv"
python3 identify_it1.py run130.csv 130
```
Model is IT1 with dead time: `G(s) = K_I/(s(T1·s+1))·e^(−Tt·s)`. The fitter
is validated on synthetic data. Sanity check: if K says 40 cm/s, a 2 s run
should cover ~80 cm.

### E. Rotation calibration (NEEDS ROBOT)
No method yet. The ultrasonic cannot measure angle — needs a protractor or
overhead video. Matters because the escalation ladder's "4 rotations one
way, then sweep past centre" assumes a step is near 90°/4, which nobody has
measured.

### F. First motors-live autonomous run (NEEDS ROBOT, gated on A)
Wheels on a stand. This also produces the loaded thermal curve needed to
set the pause threshold properly (see #75).

---

## 7. Deferred / blocked

- **QAT model evaluation (#71, #72)** — blocked on C. The ~1 GB footprint
  claim is unsupported for a llama.cpp deployment; the format (`wNa8o8`) is
  not known to load in GGUF. Do not swap the default on footprint grounds.
- **llama.cpp build 2233** — built and SWA-verified in
  `~/llama.cpp-upstream`, NOT deployed. `~/llama.cpp` (b1609) remains the
  running server. Deleting the upstream tree loses nothing.
- **Mode 2 (video/audio teleop)** — parked indefinitely, thermal cost.

---

## 8. Recovery points

```
~/build-b1609.tar.gz            verified archive of the working llama.cpp build
                                restore: cd ~/llama.cpp && rm -rf build &&
                                         tar xzf ~/build-b1609.tar.gz
~/robot-preclaudecode.tar.gz    ~/robot before any agent touched it
~/robot                         git repo, working state at commit fe14be2
                                revert edits: git checkout -- .
```

**Note:** `~/robot` contains ~420 MB of unused ONNX models. GitHub rejects
files over 100 MB, so `yolo11x.onnx` (218 MB) and `yolov8m.onnx` (100 MB)
must be gitignored or removed from history before any push.

---

## 9. Working style that has worked

- One instruction at a time; wait for output before the next.
- Paste real terminal output, not summaries. Terminal rendering sometimes
  runs lines together — if `python -m py_compile` passes, mangled-looking
  output is display, not a real error.
- Heredocs (`cat > file <<'EOF'`) for writing files. Pasting raw Python into
  bash executes it as shell commands and produces pages of errors.
- Verify before trusting: an untested backup is not a backup.
- If a fact is not in the code or in a doc, say UNKNOWN. Do not infer.
  The invalid 93% benchmark exists because someone tested an interface that
  did not exist.

---

## 10. On-device coding agent

Claude Code 2.1.261 runs in proot-distro Debian:

```bash
proot-distro login debian --bind /data/data/com.termux/files/home:/termux-home
cd /termux-home/robot && claude
```

**It can:** read and edit files, run non-root Python, inspect the environment.
**It cannot:** use `su` or `/dev/bus/usb`, so `motors.py`, `teleop.py`,
`run_mission.py`, `log_run.py`, `dist_raw.py`, `cp2102_test.py` stay manual.

**Its limit:** it reasons from the files it is given. It reported
`Robot.run_mission()` as dead code because nothing in `main.py` calls it —
`run_mission.py` does. Scope cross-file questions explicitly.

AVF / Android Linux Terminal was rejected: no USB host controller, and it
shares only `/mnt/shared`, so it can reach neither the ESP32 nor `~/robot`.


## 11. Workflow and fresh-session starters

Read `WORKFLOW.md` before assignment; record selected reviewer/model at handoff.

**Local AI:** `Act as Pixel Robot Local AI. Read HANDOFF.md, WORKFLOW.md,
LOCAL_AI.md, STATUS.md and DECISIONS.md. Define a bounded task and request phone
evidence rather than assuming it.`

**Coder:** `Act as Pixel Robot Coder. Read AGENTS.md, HANDOFF.md, WORKFLOW.md,
CODER.md, STATUS.md and DECISIONS.md. Implement only the approved task, verify it,
and automatically obtain independent review. Do not commit.`

**Reviewer:** `Act as a fresh Pixel Robot Reviewer. Read AGENTS.md, HANDOFF.md,
WORKFLOW.md and REVIEWER.md. Review the supplied frozen candidate and return the
complete final review. Do not edit, commit, push, run hardware, or approve commit.`

**Doc Keeper:** `Act as Pixel Robot Doc Keeper. Read HANDOFF.md, WORKFLOW.md,
DOC_KEEPER.md, STATUS.md and DECISIONS.md. Apply only evidence-backed DOC DIFFs,
keep decisions append-only, and record the next-session allocation.`
