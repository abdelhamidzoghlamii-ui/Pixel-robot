# APP_STATUS

State of the native Android app. Mirrors STATUS.md's rule: values here are read from
the app as it exists, not from plans, estimates, or the Termux prototype. Anything
unmeasured is marked PENDING.

Single writer: Doc Keeper. The APP thread emits DOC DIFFs against this file.

## Architecture

**Target device:** Pixel 7 (Tensor G2), Android 17, **unrooted** — normal app
sandbox. No Termux, no `su`, no `LD_LIBRARY_PATH`. This is the defining difference
from the prototype in STATUS.md, which is root-dependent throughout.

**Body:** unchanged — ESP32 NodeMCU + 2x MX1508 + 4 mecanum motors. The app replaces
the phone side only. Firmware (`mode2_auto.ino`) and its serial contract are
inherited as-is and stay owned by the Hardware thread.

**Transport:** Android USB Host API via `usb-serial-for-android`, CP2102 at
115200 8N1. Replaces `motors.py`'s rooted PyUSB path (DECISIONS #44, #46).

**Layer split** (non-negotiable):

```
execution layer   drive, obstacle stop, emergency stop    must run with the
                                                          reasoning layer disabled,
                                                          crashed, or absent
reasoning layer   Gemma via llama.cpp JNI,                strictly additive
                  voice -> JSON action
```

The firmware watchdog (1000 ms, DECISIONS #47) sits below the app entirely and is
the hard safety floor. No app-layer change may weaken or bypass it.

**Planned per-command flow** (Phase B scope, not yet built):

```
voice in (whisper.cpp) -> Gemma PARSE_SYS -> JSON action
                       -> execution layer -> USB serial -> ESP32
voice out via Android TextToSpeech
```

## Blocker status

Both spikes passed 2026-08-17 (DECISIONS #67, #68). Phase B is unblocked. Both
spikes were throwaway: the deliverable was a yes/no, not software.

| Spike | Answers | Status |
|---|---|---|
| A1 | USB Host serial to the ESP32 without root | **PASS** — 2026-08-17, DECISIONS #67 |
| A2 | Gemma sustained throughput inside a real app process | **PASS** — 2026-08-17, DECISIONS #68 |

**A1 — USB Host serial, root-free.** No code. *Serial USB Terminal* (kai-morich),
ESP32 over USB-C OTG, 115200 8N1, send `PING`. Pass = `ALIVE` returns. Rule out
before concluding anything: power-only cable with no data lines, USB permission
dialog never granted, wrong baud. In the passing run the dialog never appeared
because permission had already been granted — its absence is not a failure signal.
`USB Device Info` lists enumerated devices independently of any serial library and
separates "phone cannot see the board" from "app cannot open it".

```
result            PASS — PING -> ALIVE, 5 ms round trip
board tested      ESP32-S3, CH343 bridge (1a86:55d3), 115200 8N1
CP2102 unrooted   NOT YET EXERCISED — follow-up, not a blocker
```

**A2 — Gemma sustained throughput.** No code. Off-the-shelf llama.cpp-backed GGUF
runner (PocketPal AI or Maid), the exact `gemma-4-e2b-it-q4_k_m.gguf` the robot
uses, 10 minutes continuous generation, **screen on**.

```
sustained tok/s          7.4-7.6    (Termux baseline 11-12, STATUS.md)
degradation over 10 min  NONE — stable across the full run
zone9 peak under load    82 C, median 66 C
throttle onset           NOT REACHED — no throttle collapse observed
config used              4 threads, ctx 2048, PocketPal AI, screen on
```

Sustained, not first-token, not a single completion. A zone9 trace is taken during
the same run. It measures the app runtime — unrooted sandbox, screen forced on,
third-party runner, none of the prototype's server flags — so it does **not** fill
STATUS.md's outstanding loaded thermal measurement, which COMMANDS.md §7 requires be
taken from `run_cycle`'s own per-cycle print. It is routed to the **Local AI thread**
as corroborating evidence, not as the missing number.

**Gate:** both pass -> Phase B unblocked, numbers recorded above. A1 fails -> the
wired root-free premise is in question, escalate to Review & Research before
designing around it. A2 badly under -> tiering may need a smaller default model,
escalate.

## Known-good config (as coded)

```
NONE — no app code exists yet.
```

This section stays empty until code is written. Values from the Termux prototype are
**not** inherited into it; they were measured on a different runtime.

**Inherited contracts** — not app config, owned by the Hardware thread, listed so the
app does not reinvent them:

```
board       CP2102, VID:PID 10c4:ea60, endpoints 0x01 OUT / 0x81 IN
serial      115200 8N1
in          FORWARD:<spd>  BACK:<spd>  ROTATE_L:<spd>  ROTATE_R:<spd>  STOP  PING
out         READY (boot)   ALIVE (PING reply)   DIST:<cm>   (-1 = no echo, ~4 m)
watchdog    1000 ms — a held move must be re-sent, not awaited
MAX_SPEED   200, clamped in firmware
```

Strafe, diagonals and `SERVO:` exist in the firmware but are out of Phase B scope.

## Working now

```
NOTHING — no app code exists yet.
```

## Pending / untested

- **Phase B gate cleared.** Both blockers passed 2026-08-17 (DECISIONS #67, #68).
  One prerequisite remains and it is not a blocker: the Termux prototype must reach a
  reliable state first, meaning at minimum STATUS.md's open items closed — the first
  motors-live autonomous run executed, forward and rotation calibration taken, and
  the benchmark figures re-measured against deployed code. Phase A was allowed to run
  in parallel with V1 work because it touches no robot code; Phase B is not.
- **In-app E2B throughput is near the rate #2 rejected.** A2 measured 7.4-7.6 tok/s;
  DECISIONS #2 parked E4B as too slow for the control loop at 7.2. The `--swa-full`
  recovery predicted in #68 is reasoning, not measurement — nothing has yet run E2B
  with that flag inside an app process. Note also that #2's rejection was made on
  warm cycle time (9.9 s vs 5.1 s), not tok/s, and in-app cycle time is unmeasured.
  First build measures both before the default model is settled.
- **CP2102 not yet exercised unrooted.** A1 passed on an ESP32-S3 / CH343. The
  robot's CP2102 (`10c4:ea60`) is proven on this phone only with root (#44). The
  app's manifest VID:PID filter and driver selection must be checked against it at
  first build. Follow-up, not a blocker.
- **JSON schema not chosen.** Two are in circulation in the prototype:
  `{"type","name","room"}` (`main.py: PARSE_SYS`) and `{"action","target"}`
  (benchmark scripts). The app picks one and it becomes canonical for the app. A
  DECISIONS entry is emitted when it is chosen. Both are not carried forward.
- **Android / NDK targets unset.** compileSdk, targetSdk, minSdk, NDK, Kotlin and
  AGP versions are PENDING. They get recorded once the toolchain is installed, not
  guessed and not copied from a tutorial.
- **Foreground service is a liveness requirement, observed not predicted
  (DECISIONS #69).** During A2, inference stopped the moment the app lost focus;
  Termux's concurrent logger survived because it holds a foreground service. Without
  one the reasoning layer dies on any notification, call or app switch. A partial
  wake lock additionally keeps the CPU running with the display off. In scope for the
  first build. Screen-off throughput under a wake lock is PENDING — A2's 7.4-7.6 was
  measured screen-on and equal performance is not assumed.
- **Module layout is PROPOSED, not settled** — `:app` / `:transport` / `:llm` /
  `:voice`, with `:transport` required to build and run with `:llm` absent. Becomes
  a DECISIONS entry once the first build exists.
- **`SimpleUsbTerminal` licence unverified.** It is starting material for code
  reuse, so its licence attaches to anything copied. See LICENSES.md.
- **Gemma GGUF provenance unrecorded.** The shipped artifact is a community requant;
  the redistributor's terms attach to the file, not Google's release page. See
  LICENSES.md note 1.
- **`BACK_R` single-wheel fault (STATUS.md) is inherited but not exposed.** The
  first build ships forward / back / rotate only, which avoids the strafe and
  diagonal code path entirely. Avoided, not fixed.
- **Out of scope for the first build**, listed so it is not mistaken for missing:
  vision, strafe, diagonals, phone-tier detection, model tiering, module detection,
  disclaimer table, calibration engine, settings.
