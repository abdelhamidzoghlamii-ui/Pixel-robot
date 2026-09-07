# CLAUDE.md — Pixel Robot Android app

Context file for Claude Code running in this repo. Written to be read by an agent.
Rules here are binding, not advisory.

This is **not** the robot prototype's `CLAUDE.md`. That file governs the Termux
Python stack and does not apply here. Do not merge or conflate the two.

---

## 0. Hard rules

Breaking any of these is a defect regardless of whether the code works.

### 0.1 No root

The app runs in a normal Android sandbox. No `su`, no `LD_LIBRARY_PATH` shims, no
direct `/dev/bus/usb`. USB access is the Android USB Host API only.

If a design appears to need root, the design is wrong. Stop and say so.

### 0.2 The execution layer never depends on the reasoning layer

Drive, obstacle stop and emergency stop must work with the LLM disabled, crashed,
absent, or running the smallest model tier. Gemma is strictly additive.

Concretely:

- No motor command path may block on, await, or branch on an LLM result.
- The transport module must compile and run with the LLM module absent.
- A reasoning-layer failure degrades capability. It never degrades safety.

### 0.3 The firmware watchdog is the safety floor

The ESP32 stops all motors after 1000 ms with no serial line. It sits below the app
entirely. No app-layer change may weaken, extend or bypass it.

A move held longer than the watchdog must be re-sent on a timer. Code that issues a
move once and then sleeps is a bug, not a shortcut.

### 0.4 Any code that can move a wheel gets human review

Plus a second-model review. Not vibe-coded, not auto-approved, not delegated to a
subagent unattended.

Precedent: four bare `except:` clauses in the prototype silently swallowed
KeyboardInterrupt and broke the emergency stop (DECISIONS #56). In a shipped product
that class of bug is a physical-world event.

Therefore, in this repo:

- Never write a bare `catch (Exception)` / `catch (Throwable)` in a motor path.
- Never widen an existing catch in a motor path.
- When you touch code that can move a wheel, say so explicitly in your output and
  flag that it needs the two reviews. Do not let it pass silently.

### 0.5 Licence-clean by construction

Nothing enters the app whose licence is incompatible with a closed-source commercial
product.

```
allowed     MIT, Apache-2.0, BSD-2/3, Android platform APIs
forbidden   AGPL (any version), GPL
review      LGPL — case by case, never assumed acceptable
```

This allowlist is a copy, kept here so an agent never has to open another file to
learn what is forbidden. `LICENSES.md` is canonical (DECISIONS #66). If the two
disagree, `LICENSES.md` is right and this block is stale — say so and stop. Do not
reconcile it yourself.

**Ultralytics YOLO (AGPL-3.0) is specifically excluded.** Permissive alternatives
only: YOLOX, NanoDet, TFLite EfficientDet-Lite.

Every new dependency is recorded in `LICENSES.md` **before** it is added, not after.
Adding a dependency without its row is a defect. If you cannot determine a licence,
stop and ask — do not add it and note it later.

---

## 1. Think before coding

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them. Do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

## 2. Simplicity first

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios — but rule 0.4 overrides this in motor
  paths. Nothing there is an impossible scenario.
- If you write 200 lines and it could be 50, rewrite it.

## 3. Surgical changes

- Touch only what you must.
- Do not improve adjacent code, comments or formatting.
- Do not refactor what is not broken.
- Match existing style even if you would do it differently.
- If you notice unrelated dead code, mention it. Do not delete it.
- Remove only the imports and symbols *your* change orphaned.

Test: every changed line traces directly to the request.

## 4. Goal-driven execution

Turn tasks into verifiable goals. State a plan with per-step verification before any
multi-step change:

```
1. [step] -> verify: [check]
2. [step] -> verify: [check]
```

"Make it work" is not a goal.

**Hardware-verified steps are gated by a human with a robot plugged in.** No amount
of agent parallelism compresses that loop; do not plan as if it does. When a step
needs hardware, stop and hand off. Never simulate it and report success.

## 5. Context discipline

- Do not re-read a file already open in this session.
- Benchmarks as one row: `E2B Q4_K_M | 4 threads | 11-12 tok/s | thermal=PENDING`
- Stack traces as the failing line and exception type only.
- Terminal output: the result, not the scrollback. Gradle sync logs are noise.
- Any command producing more than 20 lines gets piped: `| tail -5`, `| grep -i error`.

---

## 6. Project shape

**Target:** Pixel 7 (Tensor G2), Android 17, unrooted.
**Stack:** Kotlin, JNI/NDK for natives, Gradle.

```
compileSdk    PENDING
targetSdk     PENDING
minSdk        PENDING
NDK version   PENDING
Kotlin / AGP  PENDING
```

Recorded once the toolchain is installed. Do not guess these and do not copy them
from a tutorial.

**Module layout — PROPOSED, not settled:**

```
:app         UI, wiring, lifecycle
:transport   USB Host serial to the ESP32.  Must build and run with :llm absent.
:llm         llama.cpp via JNI
:voice       whisper.cpp in, Android TextToSpeech out
```

Dependency direction: `:app` depends on everything. `:transport` depends on nothing
else in this repo. Nothing depends on `:llm` except `:app`. That shape is what
enforces rule 0.2 at build time rather than by convention.

## 7. First build scope

**In:** USB Host serial link, forward / back / rotate only, Gemma via llama.cpp JNI,
voice in (whisper.cpp), voice out (Android `TextToSpeech`), voice -> JSON action ->
motor command, minimal UI, foreground service (`connectedDevice`) plus partial wake
lock.

The foreground service is not an optimisation. Without it the reasoning layer dies on
any notification, call or app switch (DECISIONS #69).

**Out:** vision, strafe, diagonals, phone-tier detection, model tiering, module
detection, disclaimer table, calibration engine, settings.

Strafe and diagonals are the code path carrying the unresolved `BACK_R` fault.
Forward / back / rotate avoid it entirely. This is deliberate — do not add them back
for completeness.

## 8. Inherited hardware contract

Owned by the Hardware thread. Do not change it from here. If it looks wrong, say so
and stop.

```
board       CP2102, VID:PID 10c4:ea60, endpoints 0x01 OUT / 0x81 IN
serial      115200 8N1
in          FORWARD:<spd>  BACK:<spd>  ROTATE_L:<spd>  ROTATE_R:<spd>  STOP  PING
out         READY (boot)   ALIVE (PING reply)   DIST:<cm>   (-1 = no echo, ~4 m)
watchdog    1000 ms
MAX_SPEED   200, clamped in firmware
```

Naming inverts between layers in the prototype: in the firmware, `LEFT` and `RIGHT`
are **strafes**, not rotations. The app uses `ROTATE_L` / `ROTATE_R` for rotation and
emits no strafes at all in the first build.

`DIST:-1` means no echo within about 4 m — that is "clear", not "unknown". Do not
compare a raw `-1` against a threshold.

## 9. Starting material

Do not start from an empty project.

- `llama.cpp/examples/llama.android` — official JNI wrapper. The Gemma-in-app
  problem is already solved there.
- `SimpleUsbTerminal` (kai-morich) — working CP2102 serial code by the author of the
  transport library. **Licence unverified. Check `LICENSES.md` before copying any
  of it.**
- `PARSE_SYS` from the prototype — port the prompt and the JSON contract. Do not
  reinvent them.
- OpenBot's Android app — read for control-loop and camera-capture architecture
  only. **No code reuse.** That is also why no licence obligation attaches.

## 10. Docs

- `APP_STATUS.md` has exactly one writer and it is not you. Propose changes as a
  DOC DIFF block and hand it off.
- `LICENSES.md` is updated **before** a dependency is added.
- Decisions belong in the wider project's single append-only `DECISIONS.md`, which
  lives outside this repo. Hand the numbered entry to the human. Do not start a
  second decision log here.
- If a claim and a doc disagree, flag it. Never silently pick one.

DOC DIFF format:

```
DECISIONS: append #NN — <one line + why>
APP_STATUS: <section> — change "<old>" to "<new>"
```


## 11. Workflow and commit gate

Read `AGENTS.md`, `docs/WORKFLOW.md`, and the assigned role file before work. The
Coder obtains a fresh independent review and waits for the human's explicit commit
decision after Local AI discussion. A pass is not commit permission; push needs
separate authorization. This supplements, and does not weaken, §0.4's wheel-motion
review requirement.
