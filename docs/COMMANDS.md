# COMMANDS — Pixel 7 AI stack

Quick reference for everything that runs on the phone. All paths assume `~/robot`.

**Legend:** ✅ verified in a live session · ⚠️ exists but not exercised recently — confirm before relying on it

Run `ls ~/robot/*.py | xargs -n1 basename` to confirm what's actually present.

---

## 0. Read this first — two things that will bite you

### Motor control requires root

`/dev/bus/usb` is not readable by Termux's app UID, so **anything that touches
the ESP32 must run under `su`**, and `LD_LIBRARY_PATH` must be set explicitly
because root does not inherit Termux's environment.

```bash
su -c "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib \
  /data/data/com.termux/files/usr/bin/python \
  /data/data/com.termux/files/home/robot/<script>.py"
```

Note the interpreter is `.../usr/bin/python`, not `python3`, in the root form.

Without root: `usb.core.NoBackendError: No backend available`.
Without `LD_LIBRARY_PATH`: the same error, from a different cause.

Applies to `motors.py`, `teleop.py`, `run_mission.py`, `log_run.py`,
`dist_raw.py`, `cp2102_test.py` — **not** to the LLM server, benchmarks, vision
scripts, `nav_test.py`, or `identify_it1.py`.

> `su -c` does not change directory. Scripts that write output files (notably
> `log_run.py`) drop them in whatever the current directory is, so `cd ~/robot`
> first.

### Port 8080 is contested

The llama.cpp server and `teleop.py` both bind 8080. **They cannot run at the
same time.**

| Process | Port | Notes |
|---|---|---|
| llama-server | 8080 | `127.0.0.1:8080/health` |
| `teleop.py` | 8080 | binds `0.0.0.0` — collides |

Autonomous mode is unaffected — `run_mission.py` is a *client* of the server, it
binds nothing. If both are ever needed at once, change `PORT` in `teleop.py`.

A leftover `teleop.py` holding 8080 makes `server_manager.py` report
`failed to load in 60s`, which looks nothing like a port problem. **Check first:**

```bash
python3 -c "
import socket
s = socket.socket()
try: s.bind(('0.0.0.0', 8080)); print('8080 FREE')
except OSError as e: print('8080 IN USE:', e)
finally: s.close()"
```

If it's in use, find the holder and kill it **by PID** — see §8.

---

## 1. Server control — start this first for autonomous mode

The LLM server must be running before `run_mission.py` or `chat.py`. The nav loop
consults Gemma on triggers; without the server those calls fail.

| Command | What it does |
|---|---|
| `python3 server_manager.py setup_q4` ✅ | **Robot default.** Gemma 4 E2B Q4_K_M, 3.3 GB, ~11-12 tok/s. Ready in ~6 s. |
| `python3 server_manager.py setup_e4b` ⚠️ | Gemma 4 E4B Q4_K_M, 5.0 GB, ~7.2 tok/s. Chat only — too slow for the control loop. |
| `python3 server_manager.py setup_qwen3b` ⚠️ | Qwen 2.5 3B fallback. |
| `python3 server_manager.py setup_qwen1b` ⚠️ | Qwen 2.5 1.5B — fastest, lowest quality. |
| `python3 server_manager.py stop` ✅ | Kills all llama-server processes. |

All setups launch with the benchmarked flags:
`--threads 4 --threads-batch 4 --parallel 1 --swa-full --ctx-size 2048`

**Standard startup:**
```bash
cd ~/robot
python3 server_manager.py setup_q4
curl -s http://127.0.0.1:8080/health && echo " — server up"
```

> ⚠️ **`server_manager.py` hides the real error.** It reports only
> `failed to load in 60s`. A port collision, a missing model, and a bad binary
> all look identical. To see the actual cause, run the server by hand:
> ```bash
> ~/llama.cpp/build/bin/llama-server -m ~/models/gemma-4-e2b-it-q4_k_m.gguf \
>   --threads 4 --threads-batch 4 --parallel 1 --swa-full --ctx-size 2048 \
>   --port 8080 2>&1 | tail -25
> ```
> A port collision shows as `couldn't bind HTTP server socket`.

> ⚠️ `~/models/gemma-4-e4b-it-q3_k_m.gguf` is **0 bytes** — a failed download.
> Harmless unless something tries to load it. Q3 was rejected anyway (#3).

---

## 2. Running the robot

| Command | Root? | What it does |
|---|---|---|
| `run_mission.py "<mission>"` | yes | **Autonomous entry point.** `--dry` runs the full loop with no motor commands and needs no root. |
| `teleop.py` ✅ | yes | **Mode 1 teleop.** Pixel hotspot + web UI; drive from a second phone's browser. Diagonals, speed slider, live HC-SR04 readout. Needs neither the server nor the AI stack. |
| `motors.py` ✅ | yes | Serial self-test: connect, PING, distance, forward 1 s, rotate left 0.5 s. |
| `dist_raw.py` ✅ | yes | Dumps raw serial lines for 5 s. Use to see `DIST:` exactly as the firmware sends it. |
| `cp2102_test.py` ✅ | yes | Raw transport test: claims the interface, drains idle output, PINGs, expects `ALIVE`. Use when `motors.py` won't connect. |
| `detect_person.py` ⚠️ | no | YOLO detection module. Run directly to test vision alone. |
| `stereo_depth.py` ⚠️ | no | Stereo depth via mecanum strafe as baseline. `FOCAL_PX` uncalibrated. |
| `nav_sim.py` ⚠️ | no | Standalone older simulation (Qwen 2.5 3B, own thermal limits). Not the live path. |

> **`main.py` does not run missions.** Its `__main__` builds `Robot()` with no
> motors, runs one cycle against `test_photos/scene_test.jpg` if present, prints,
> and exits. Use `run_mission.py`.

**Autonomous — dry run** ✅ *(verified: 24 cycles, camera → YOLO → rules → Gemma)*
```bash
cd ~/robot
python3 run_mission.py --dry "explore and map the rooms"
```
In `--dry`, `self.motors` is None, so `get_distance()` returns the 999 sentinel
and **every obstacle branch is unreachable**. A dry run does not test the safety
ladder — use `nav_test.py` (§6) for that.

**Autonomous — motors live** ⚠️ *NEVER RUN. Wheels on a stand for the first attempt.*
```bash
cd ~/robot
su -c "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib \
  /data/data/com.termux/files/usr/bin/python \
  /data/data/com.termux/files/home/robot/run_mission.py 'explore and map the rooms'"
```

**Teleop startup:**
```bash
python3 server_manager.py stop        # port 8080 collision
su -c "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib \
  /data/data/com.termux/files/usr/bin/python \
  /data/data/com.termux/files/home/robot/teleop.py"
```
It prints every interface it's listening on. Open the **hotspot** address on the
driving phone — the one on `ap_br_wlan2`, not `wlan0`.

> Android reassigns the hotspot subnet on restart (seen: `10.98.246.229`, then
> `10.98.123.14`). The control server binds `0.0.0.0` so it always works, but the
> camera URL inside `teleop.py` is hardcoded. Find the current address with:
> ```bash
> su -c "/system/bin/ip -4 addr" | grep -A2 ap_br
> ```

### Reading the nav log

`run_cycle` prints one block per cycle. `[NAV]` is the Python rule decision;
`[GEMMA]` only appears when a trigger fires. Safety moves bypass Gemma entirely
unless the robot is stuck, so seeing `[NAV] LEFT` with no `[GEMMA]` line is
correct, not a missing call.

Blocked-path escalation, in order: 2 strafes → 4 rotations one way → flip side
(Gemma consulted once here) → 6 rotations back through centre → restart. After
60 s of continuous blockage it returns `STOP`, which ends the mission. **The
robot never reverses to avoid an obstacle** — the sensor faces forward, so BACK
would drive blind.

---

## 3. Serial / hardware link

**Board is CP2102 (`10c4:ea60`) — not CH340.** The original `motors.py` transport
was a CH340 vendor-register sequence and could never have worked with this board.
Rewritten; backup at `motors.py.ch340.bak`.

| Fact | Value |
|---|---|
| VID:PID | `10c4:ea60` (Silicon Labs CP2102) |
| Endpoints | `0x01` OUT, `0x81` IN |
| Baud | 115200, set explicitly via CP210x `SET_BAUDRATE` (0x1E) |
| Init | `IFC_ENABLE` (0x00) → `SET_BAUDRATE` (0x1E) → `LINE_CTL` 8N1 (0x03) |
| DTR/RTS | **left untouched** — asserting them resets the ESP32 |
| `claim_interface` | **mandatory** — without it every bulk read times out silently |

**Is the board there?**
```bash
su -c "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib lsusb"
# want: ID 10c4:ea60
```
Only `1d6b:...` entries means nothing is attached — check the cable first. A
charge-only cable powers the ESP32 (LED lights) but carries no data, which looks
exactly like a dead board.

**`Errno 16 Resource busy`** on connect = a previous process still holds the
interface. `motors.py` registers `disconnect()` via `atexit`, so this should
self-clear; if it doesn't, unplug and replug the ESP32.

**`NoBackendError`** = not running under root, or `LD_LIBRARY_PATH` unset. See §0.

### Serial protocol (firmware `mode2_auto.ino`, not tracked in this repository)

```
in :  FORWARD:<spd>  BACK:<spd>  LEFT:<spd>  RIGHT:<spd>      (LEFT/RIGHT = strafe)
      ROTATE_L:<spd> ROTATE_R:<spd>
      FWD_L:<spd> FWD_R:<spd> BACK_L:<spd> BACK_R:<spd>       (diagonals)
      STOP  SERVO:<deg>  PING
out:  READY (boot)  ALIVE (PING reply)  DIST:<cm>   (-1 = no echo, i.e. clear)
```

> ⚠️ **Naming inverts between layers.** In the firmware `LEFT`/`RIGHT` are
> strafes. In `main.py: Robot.move()` they are *rotations*, and strafing is
> `STRAFE_LEFT`/`STRAFE_RIGHT`. `navigate_rules` must emit the `move()`
> vocabulary, not the wire vocabulary.

> A diagonal drives **two wheels, not four**. That is correct, not a fault.

**`DIST:` sanitising.** The firmware sends `-1` for no echo within ~4 m.
`Robot.get_distance()` maps `-1` → 400 (clear to sensor max) and any reading
older than 1.0 s → 999 (unusable, treated as clear). Without this, `-1 < 15` is
true and an open corridor triggers avoidance on every cycle.

**Watchdog:** the ESP32 stops all motors after 1000 ms with no serial line.
`motors.py`'s duration helpers re-send the command every 200 ms via `_hold()`, so
longer moves are not cut short (verified: a 3.0 s forward runs 3.1 s,
continuous). Any new code that issues a move and then waits must use those
helpers or re-send itself.

**Motion speeds** are three separate values, and only one is a named constant:

| Move | Speed | Where |
|---|---|---|
| FORWARD / BACK | 130 | `MOTOR_SPEED` in `main.py` |
| rotations | 120 | hardcoded literal in `Robot.move()` |
| strafes | 100 | hardcoded literal in `Robot.move()` |

---

## 4. Chat mode ⚠️

`python3 chat.py` — interactive chat with the local model. In-chat commands:

| Command | What it does |
|---|---|
| `/switch` | Switch model without leaving chat |
| `/photo` | Attach a photo (Gemma vision) |
| `/camera` | Take a photo with the phone camera and attach it |
| `/clear` | Clear conversation history |
| `/history` | Show the conversation so far |
| `/temp` | Show SoC temperatures |
| `/kill` | Kill the server |
| `/quit` | Exit |

---

## 5. Benchmarks ⚠️

| Command | What it measures |
|---|---|
| `python3 json_benchmark.py` | Voice-command → JSON parsing accuracy. |
| `python3 nav_logic_test.py` | Old dual-rate nav logic. **Its 93% does not describe the live code** — see below. |
| `python3 benchmark_nav.py` | Navigation decisions through the LLM prompt path. |
| `python3 benchmark_compare.py` | A/B comparison harness (used for the LoRA tests). |
| `python3 full_benchmark.py` | Nav + voice + thermal in one run, with summary. |
| `python3 system_benchmark.py` | 10 inference cycles: tok/s, RAM before/after, throttle count. |
| `python3 swa_benchmark.py` | 8-cycle simulation — proves the prompt-cache fix. |
| `python3 thermal_real.py` | Sustained-load thermal sweep at 0/1/2 s rest. Needs root. |

> **`nav_logic_test.py`'s 93% does not transfer to `main.py`.** It matched the
> literal string `'obstacle'` in synthetic scene text — a label YOLO cannot emit
> — and ran with no distance sensor. The live rules take obstacles from
> `get_distance()`. `main.py`'s nav logic is a rewrite against real inputs and is
> **unbenchmarked**. Use `nav_test.py` (§6) to exercise it.

> **`json_benchmark.py` schema mismatch stands.** `main.py: PARSE_SYS` emits
> `{"type", "name", "room"}`; the benchmark scripts used `{"action", "target"}`.

**Quick speed check (no script):**
```bash
curl -s http://127.0.0.1:8080/completion \
  -d '{"prompt":"<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n","n_predict":40}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"timings\"][\"predicted_per_second\"]:.1f} tok/s')"
```

---

## 6. Nav rules & calibration

### `nav_test.py` ✅ — no hardware, no root, no server

Calls `navigate_rules` directly with synthetic distances. This is the **only**
way to exercise the obstacle branches: in `run_mission.py --dry` distance is
pinned at 999, so every safety path is dead code.

```bash
cd ~/robot && python3 nav_test.py
```

Three blocks: single-shot distance → move; a 16-cycle persistent blockage; and
blocked-then-cleared. Expected behaviour — thresholds exact at `<15` / `<25` /
`>=25`, `stuck=True` on exactly **one** cycle per episode, no LEFT/RIGHT
alternation. Repeated `stuck=True` or alternating rotations is the oscillation
bug this harness was written to catch.

### Forward-motion calibration

Forward motion is modelled as **IT1 with dead time** — velocity responds to a PWM
step as PT1 (drivetrain inertia), position is its integral:

```
G(s) = K_I / (s*(T1*s + 1)) * e^(-Tt*s)
x(t) = K*[(t-Tt) - T1*(1 - exp(-(t-Tt)/T1))]   for t >= Tt
```

Feedforward only — there is no odometry, so the loop cannot be closed during
normal operation.

**Step 1 — log a run** (⚠️ written and syntax-checked, never run against hardware):

Robot square-on to a flat, wide wall 2–3 m back, clear floor. The HC-SR04 beam is
a ~15° cone, so at 2 m it spans ~50 cm — nothing else may be inside it. Any yaw
during the run corrupts the measurement, since travel is computed as `d0 - d`.

```bash
cd ~/robot
su -c "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib \
  /data/data/com.termux/files/usr/bin/python \
  /data/data/com.termux/files/home/robot/log_run.py 130 run130.csv"
```
Drives forward, stops at 30 cm, writes `time_s,travel_cm`. It dedupes on
`dist_at`, so the CSV holds true sensor sample times rather than Python loop
timing.

**Step 2 — fit it** ✅ *(fitter validated on synthetic data: recovered
42.06 / 0.1788 / 0.0953 against true 42 / 0.18 / 0.09)*

```bash
cd ~/robot && python3 identify_it1.py run130.csv 130
```
Prints T1, Tt, steady-state cm/s, K_I in (cm/s)/duty, the residual range, and a
table of commanded duration per target distance.

**Self-test the fitter** (regenerates the synthetic run and fits it):
```bash
cd ~/robot && python3 -c "
import numpy as np, sys
sys.path.insert(0, '.')
from identify_it1 import step_response
t = np.arange(0, 2.0, 1/240)
x = step_response(t, 42.0, 0.18, 0.09) + np.random.default_rng(1).normal(0, 0.3, t.size)
np.savetxt('synth_test.csv', np.column_stack([t, x]), delimiter=',', fmt='%.4f')
print('synthetic run written')" && python3 identify_it1.py synth_test.csv 130
```
The `K_I` line is meaningless on synthetic data — it has no duty. Ignore it here.

**Sampling rate.** `DIST_EVERY_MS` 200 gives ~5 Hz, which fits K_I but **not**
T1/Tt: the whole transient is 2–3 samples and Tt's quantisation error exceeds Tt
itself. For the transient, reflash with `DIST_EVERY_MS = 50` and
`ECHO_TIMEOUT = 12000`. `log_run.py` warns when the interval is too coarse.

> ⚠️ **`ECHO_TIMEOUT = 12000` is a CALIBRATION BUILD ONLY. Restore 25000 before
> driving.** At 12000 µs the sensor's max range is ~2 m, so `DIST:-1` means
> "≥2 m" while `get_distance()` maps it to 400 cm — asserting 4 m of clearance on
> 2 m of evidence. Not hazardous at the 25 cm threshold, but it feeds a false
> number to `GEMMA_SYS`.

**Sweep duties** 100 / 130 / 160 / 200 for the gain curve; there is a stiction
floor at low duty, so it will not extrapolate to zero. K_I also drifts as the
pack sags from 8.2 V toward cutoff — unmeasured.

### Rotation calibration — no method yet

The ultrasonic cannot measure angle, so `log_run.py` does not apply. Needs a
protractor or overhead video. Rotation is also IT1 (angular velocity is PT1,
angle is its integral), so `identify_it1.py` fits it unchanged once `(t, angle)`
pairs exist — the units simply become deg and deg/s.

**This matters for safety**: the escalation ladder's "4 rotations one way, then
sweep past centre" assumes a rotation step is somewhere near 90°/4. Nobody has
measured it. The coverage claim is a guess until this is done.

---

## 7. Thermal & system ⚠️

| Command | What it does |
|---|---|
| `python3 get_temp.py` | BIG / MID / LITTLE / GPU / TPU / battery temps + CPU hotspot. Needs root. |
| `python3 thermal_guard.py` | Orphaned module; its warn 82 °C / critical 86 °C values are not live runtime thresholds (Decision #75). |
| `su -c "cat /sys/class/thermal/thermal_zone9/temp"` | Raw BIG-core temp, millidegrees. zone9 = BIG, 10 = MID, 11 = LITTLE, 12 = GPU, 14 = TPU, 22 = battery. |
| `free -h \| grep -E "Mem\|Swap"` | RAM and swap. Models are large and swap kills speed. |
| `termux-battery-status` | Battery temp/level (no root needed). |

**Sample the BIG core over a minute:**
```bash
su -c 'i=0; while [ $i -lt 12 ]; do cat /sys/class/thermal/thermal_zone9/temp; sleep 5; i=$((i+1)); done'
```
> Use the `while` form. `su -c "for i in $(seq 1 12); ..."` fails — the outer
> shell expands `$(seq)` before `su` sees it and Android's `sh` rejects it.
>
> Historical idle readings of 52–67 °C do not establish the loaded ceiling.

> ⚠️ **Thermal configuration remains unresolved.** Only `main.py`'s
> `run_mission` pause above **80 °C** is currently live. `thermal_guard.py` is
> orphaned; its **82/86 °C** values are not competing live thresholds (#75).
>
> Existing measurements are historical observations from different workloads:
> the 24-cycle dry run peaked at 67 °C with YOLO and occasional Gemma calls;
> #74 records 97–101 °C under sustained llama.cpp inference and 31–38 °C idle,
> superseding the older unsourced 79–82 °C sustained-inference claim.
> These observations do not establish the correct runtime threshold.
>
> **Do not select or change a thermal threshold until the planned definitive
> benchmark exercising the real `run_cycle` has been completed and reviewed.**
> Use its per-cycle temperature evidence; retain the existing hardware
> restrictions and human approval gates.

---

## 8. Process management

> ⚠️ **`pkill -f <pattern>` kills your own shell.** In Termux the pattern
> routinely matches the invoking `su -c` command line, so you get
> `Terminated` / `Killed` and no information about whether the target died.
> This bit twice in one session.

Find the PID, then kill it by number:
```bash
su -c "ps -A | grep -E 'python|llama'"
su -c "kill <PID>"
```
Then confirm with the port check in §0 — do not trust the kill itself.

---

## 9. Housekeeping and recovery

```bash
ls -lh ~/models/*.gguf | awk '{print $5, $9}'
ls ~/robot/*.py | xargs -n1 basename
df -h /data | tail -1
git -C ~/llama.cpp rev-parse --short HEAD
git -C ~/robot rev-parse HEAD
git -C ~/robot status --short --branch
```

Git commit and push operations follow [WORKFLOW.md](WORKFLOW.md): stage only the
approved files, obtain the required independent review and human commit decision
for code, and request separate push authorization. Remote-tracking refs are local
records, not proof that the remote has not changed.

### Historical recovery provenance

These notes preserve the 2026-09-05 HANDOFF and DECISIONS #73 evidence. The phone
archives have not been freshly inspected or verified in this documentation cleanup.

| Recovery item | Historical provenance and limitation |
|---|---|
| `~/build-b1609.tar.gz` | DECISIONS #73 records a 49 MB archive of llama.cpp commit `e1a1abb7`, version 1609, with all four critical files present and gzip integrity checked at that time. Restore the whole build tree; present archive integrity is unverified here. |
| `~/robot-preclaudecode.tar.gz` | The old handoff describes this as `~/robot` before any agent touched it. Exact contents, corresponding commit, and present integrity are UNKNOWN; no new verification is claimed. |
| Git commit `fe14be2` | Historical CP2102/nav/calibration snapshot, not the current working state and not an asserted identity for either archive. Read `git rev-parse HEAD` for the current checkout. |

DECISIONS #73 supersedes the old single-launcher rollback instructions:
`llama-server.swafix` was a 5.9 KB fragment that depended on shared libraries.
`llama-server.working` was a genuine earlier 12 MB static fallback, but predates
the SWA fix (#6); using it loses that fix. Another documented recovery route is
rebuilding llama.cpp from commit `e1a1abb7`.

Historical whole-tree restoration command from #73, for native Termux only after
verifying the archive and obtaining human authorization for replacing the build:

```bash
cd ~/llama.cpp && rm -rf build && tar xzf ~/build-b1609.tar.gz
```

The old handoff's `git checkout -- .` discarded unstaged tracked-file edits; it
did not restore the historical snapshot or recover ignored models. It is not a
routine recovery step. Identify and preserve current work and agree exact recovery
targets before any destructive Git operation.

The earlier handoff recorded about 420 MB of unused ONNX files, including
`yolo11x.onnx` (218 MB) and `yolov8m.onnx` (100 MB). Those are historical size
observations, not a fresh inventory. See [STATUS.md](STATUS.md) and `.gitignore`
for model/capture exclusions; adding ignore rules does not remove files already
in Git history.

---

## 10. Firmware

Compiling requires the laptop — the ESP32 toolchain has no Android build, so
Termux cannot produce a `.bin`. A native flasher app on the Pixel saves nothing,
since the compile still happens on the laptop. **OTA would remove the cable
entirely and has not been set up.**

Before flashing: **switch off the motor battery rail.** GPIO pins float during
reset and can twitch a motor. The ESP32 runs on USB, so it flashes fine with the
motor supply off. Signal wires may stay connected — none of the motor pins
(16/17/18/19/21/22/23/25) are strapping pins. What *does* break flashing is
having buck 5 V on VIN while USB is also connected.

Verify a flash in the Arduino serial monitor at **115200**: press EN, expect the
ROM bootloader banner then `READY`; send `PING` with line ending set to Newline,
expect `ALIVE`. If the banner appears but `READY` does not, the sketch is not the
one you think you flashed.

**Current firmware constants** (`mode2_auto.ino`, not tracked in this repository):
```
BAUD            115200
WATCHDOG_MS     1000        paired with motors.py's 200ms keepalive
MAX_SPEED       200         speeds above this are silently clamped
HAS_ULTRASONIC  1           TRIG P27, ECHO P26 via 1k/2k divider
DIST_EVERY_MS   200         50 for calibration builds only
ECHO_TIMEOUT    25000 µs    ~4m. 12000 for calibration builds only
corner map      RR=0 FR=1 FL=2 RL=3, INVERT indices 2 and 3 true
```

Sensor VCC must come from the ESP32 **5 V/VIN** pin. Wired to GND instead, the
sensor is unpowered and the trigger pulse couples through it, producing a
constant `DIST:0` — which looks like a working sensor reading zero, not a wiring
fault.

---

## Typical session — autonomous

The live step below is gated on the safety fix and verification in
[STATUS.md](STATUS.md#work-priorities). It requires human control and wheels on a
stand for the first attempt; this example is not authorization to run motors.

```bash
cd ~/robot
python3 -c "
import socket
s = socket.socket()
try: s.bind(('0.0.0.0', 8080)); print('8080 FREE')
except OSError as e: print('8080 IN USE:', e)
finally: s.close()"
python3 server_manager.py setup_q4          # start server (port 8080)
curl -s http://127.0.0.1:8080/health        # confirm it's up
python3 nav_test.py                         # safety rules still sane?
python3 run_mission.py --dry "explore"      # dry run — no motors
# wheels on a stand for the first live run
su -c "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib \
  /data/data/com.termux/files/usr/bin/python \
  /data/data/com.termux/files/home/robot/run_mission.py 'explore'"
python3 server_manager.py stop
```

## Typical session — Mode 1 teleop

```bash
python3 server_manager.py stop              # port 8080 collision
# hotspot on, ESP32 on phone USB, motor rail on
su -c "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib \
  /data/data/com.termux/files/usr/bin/python \
  /data/data/com.termux/files/home/robot/teleop.py"
# open the printed ap_br_wlan2 address on the driving phone
```

## Typical session — calibration

```bash
cd ~/robot
python3 server_manager.py stop              # free the CPU and the port
# robot square-on to a wall, 2-3m back, clear floor
su -c "LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib \
  /data/data/com.termux/files/usr/bin/python \
  /data/data/com.termux/files/home/robot/log_run.py 130 run130.csv"
python3 identify_it1.py run130.csv 130
# sanity-check: if K says 40 cm/s, a 2s run should have covered ~80cm
```


## 11. Agent role sessions

Read `AGENTS.md`, `WORKFLOW.md`, and the role document first. From native Termux:
```bash
proot-distro login debian --bind /data/data/com.termux/files/home:/termux-home
```
Inside Debian, choose one session: `cd /termux-home/robot && codex`,
`cd /termux-home/robot && /root/.local/bin/agy`, or
`cd /termux-home/robot && claude`. They are alternatives: use another terminal or
exit the current agent first. The two repository paths are the same bind-mounted
files. AGY 1.1.27 and `gemini-3.1-pro-high` started here; see `WORKFLOW.md` for
headless review and its current read-only limitation.

The prior Claude Code 2.1.261 Debian session could read/edit files and run non-root
Python, but could not use `su` or `/dev/bus/usb` (DECISIONS #82). The
`motors.py`, `teleop.py`, `run_mission.py`, `log_run.py`, `dist_raw.py`, and
`cp2102_test.py` hardware operations remain human-controlled in native Termux.
That session incorrectly called `Robot.run_mission()` dead after reading only
`main.py`; `run_mission.py` calls it. Scope cross-file investigations explicitly.
#82 also records why AVF was rejected: no USB host controller and only
`/mnt/shared` shared, rather than access to the robot checkout.

### Terminal evidence and editing practice

Preserved from the old handoff: when the human executes phone commands, send one
instruction at a time and wait for its output. Return exact requested terminal
evidence, not a paraphrase. Display rendering can join lines; a syntax check may
distinguish a display problem from a real syntax error, but does not prove runtime
correctness. Do not paste raw Python into a shell. Use the agent's supported patch
tool for edits; for manual multiline shell input, a correctly quoted heredoc keeps
content from being interpreted as commands. No editor choice grants write authority.

Verify before trusting a backup. Mark unsupported facts UNKNOWN. In particular,
the invalid 93% navigation result tested an interface that did not exist.
