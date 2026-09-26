# Coder reports, verbatim

Extracted unchanged from Claude Code session transcript `b32f7c1a-ea17-4af1-98e0-189c71dbb218.jsonl` (assistant text messages at the UTC timestamps below). Only this header and the separators were added.

---

## Message at 2026-09-26T10:16:21Z

I rebuilt the robot's llama.cpp at the same commit with dotprod enabled and pointed `server_manager.py` at it. The rebuild is not bit-identical in behaviour: 2 of 120 s1o decisions changed, both from a wrong answer to the correct one, and voice parsing is identical. The review came back **APPROVE WITH NOTES**. Nothing is committed and the old binary is untouched.

## 1. Build
- **Commit:** the old binary reports `version: 1609 (e1a1abb7)`. Its source tree is clean at `e1a1abb78746c025f5e9039f590e37ccdb758ae7` (tag b10194).
- **New directory:** `/data/data/com.termux/files/home/llama.cpp-b1609-dotprod`, a clone checked out at the same commit. It uses the native path, so the robot can run it directly from Termux.
- **Toolchain:** the same Termux clang 21.1.8.
- **Flags:**
```
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/data/data/com.termux/files/usr/bin/cc -DCMAKE_CXX_COMPILER=/data/data/com.termux/files/usr/bin/c++ \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
make -C build -j6 llama-server      # 7 min
```
- **Only difference:** comparing the cmake settings with the old build, only `GGML_CPU_ARM_ARCH` changed (empty → `armv8.2-a+dotprod+fp16`); build type and C/C++ flags are identical.
- **Web UI:** the browser UI bundle for b1609 couldn't be downloaded, so the build used the "latest" one. The robot only uses `/completion`, so this doesn't affect it.

## 2. Dotprod evidence
- **system_info:**
  - old: `NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1`
  - new: `NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | DOTPROD = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1`
- **`objdump -d | grep -c sdot`:** `llama-server` itself is a 6 KB launcher with 0 in both builds. The kernels are in `libggml-cpu.so`: **0 → 1045** sdot, and 0 → 456 fp16 `fmla`. All other libraries have 0.
- Each server loads its own `libggml-cpu`, confirmed via `/proc/<pid>/maps` with no `LD_LIBRARY_PATH` set.

## 3. Behaviour checks
**a. s1o, full ladder** (both orders, 4 threads, same adapter and server flags). The old reference is the b1609 main block from your speed run.

| | old | new |
|---|---|---|
| correct / acceptable | 92/120 (77%) / 97 | 94/120 (78%) / 99 |
| order flips | 6/60 | 8/60 |
| median / P95 | 6650 / 13665 ms | **2478 / 4605 ms** |

- **Agreement:** 118 of 120 decisions match.
  - L3_01 reversed: old chose kitchen, which is wrong (p 0.56); new chose living room, which is correct (p 0.58).
  - L3_12 written: old chose bedroom, which is wrong (p 0.76); new chose living room, which is correct (p 0.64).
  - Max probability difference: **0.40**, on L3_12.
- **The new build is deterministic:** an L3 rerun matched 24/24 with a probability difference of 0.0.
- **The cause is dotprod, not fp16.** I built a dotprod-only variant (`llama.cpp-b1609-dotprod-nofp16`) to check. It flips the same two cases and matches the dotprod+fp16 build on 120/120 decisions, within 0.0001. So the dotprod kernels round differently from the old NEON path.
- For scale, b2351 differs from the old build in 4 of 120 decisions.

**b. Voice parsing:** `main.py`'s own `parse_command` and `PARSE_SYS`, with the server started through `server_manager.start_setup('setup_q4')`, gives **15/15 identical parsed actions**, and the raw model text is identical too.
- The old binary returned `[]` for commands 1–2 on the first pass because its cold first request (about 600 tokens at 10 tok/s) exceeded `parse_command`'s 20 s timeout. Rerun with a warmed server, both commands came out identical.
- **Risk:** the new build's cold first parse took 18.9 s, 14.2 s and 13.4 s across three runs, all under the 20 s limit but close. On a hot phone the robot could return `[]` for its first command, as the old binary already did.

| # | command | old | new |
|---|---|---|---|
| 1 | Find Chiara and tell her the pizza is here | find_person Chiara "the pizza is here" (after warm-up) | same |
| 2 | Go to the bathroom | navigate_to bathroom (warm) | same |
| 3 | Patrol the whole apartment | patrol [] | same |
| 4 | Find my phone in the living room | find_object phone, living_room | same |
| 5 | Come back | come_back | same |
| 6 | Tell Abdel the meeting starts in ten minutes | find_person Abdel "the meeting starts in ten minutes" | same |
| 7 | Go to the kitchen and say lunch is ready | navigate_to kitchen + say "lunch is ready" | same |
| 8 | Check the bedroom and the hallway | patrol [bedroom, hallway] | same |
| 9 | Where are my glasses | find_object glasses | same |
| 10 | Say good night | say "good night" | same |
| 11 | Find someone and tell them the door is open | find_person someone "the door is open" | same |
| 12 | Go to the living room then come back | navigate_to living_room + come_back | same |
| 13 | Look for the cat in the bedroom | find_object cat, bedroom | same |
| 14 | Tell Chiara I love her | find_person Chiara "I love her" | same |
| 15 | Blue banana seven | say "Blue banana seven" | same |

Typical warm parse time: about 5–6 s old, 2–3.6 s new.

**c. Generation, 200 tokens** (rounds alternated old, new, old, new; no thermal gate is possible from proot): median **6.46 → 8.52 tok/s** (+32%); prompt eval 10.0 → 31.6 tok/s.

## 4. Diff
The only change in the robot repo:
```diff
-LLAMA_SERVER = HOME + "/llama.cpp/build/bin/llama-server"
+LLAMA_SERVER = HOME + "/llama.cpp-b1609-dotprod/build/bin/llama-server"
```
Smoke test through the edited `server_manager`: `setup_q4` started the new binary with the robot's exact flags on port 8080, `parse_command("Go to the kitchen")` returned navigate_to kitchen in 13.4 s, and `stop` left no server running.

The comment next to it still says "11-12 tok/s" and names no `--cache-ram 0` flag, which the server command doesn't pass. I left both alone because you limited the change to the path.

## Review (AGY `gemini-3.1-pro-high`, `--sandbox`, Ponytail off, all material inlined), verbatim
The frozen files were re-verified afterwards.

---
Here is the review of the robot's llama.cpp `b1609` update and `server_manager.py` change.

### 1. Build Verification
**Confirmed.** The build targets the exact same commit (`e1a1abb7`, b1609) as the original robot binary. The CMake flags and environment are identical except for the expected `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16` change. Building in a completely new directory (`llama.cpp-b1609-dotprod`) safely preserves the existing working build. 

### 2. Dotprod Evidence
**Sufficient and conclusive.** 
- The `system_info` output explicitly shows `DOTPROD` and `FP16_VA` are active.
- `llvm-objdump` proves that the `sdot` (1045) and `fmla vN.8h` (456) instructions were successfully compiled into the new `libggml-cpu.so`, whereas they were absent in the old build.
- Verification using `/proc/<pid>/maps` confirms that the runtime linker resolves the new library correctly without relying on `LD_LIBRARY_PATH`.

### 3. Behaviour checks ("speed only, not behaviour")
The evidence robustly supports that behaviour remains identical, with discrepancies fully explained by execution time and CPU instruction limits:
*   **s1o flips (2/120):** This is expected and acceptable. Hardware dot-product instructions alter the order of floating-point/integer accumulation compared to naive NEON execution, resulting in tiny numerical drift. The author provided excellent diagnostic proof that this drift stems strictly from the dotprod kernels (a dotprod-only, no-fp16 build had the exact same flips). 
*   **Voice parsing mismatches / timeouts:** The old binary failed the first two cold voice commands because its prompt evaluation speed (~10-13 tok/s) caused it to exceed the hardcoded 20-second `requests.post` timeout in `main.py`'s `parse_command()`. The new binary evaluates prompts much faster (31-37 tok/s), successfully returning the JSON array in 18.9s. 
*   **Generative exactness:** The follow-up tests with warm caches guarantee that when both binaries actually complete the generation, they output the exact same parsed text.

### 4. Risks in Path Change
*   **Runtime/RUNPATH:** No risk. The RUNPATH baked into the new binary uses the absolute `/data/data/com.termux/files/usr/lib` path and its local `build/bin` path. It will cleanly find its own updated `libggml-cpu.so` when executed natively by the robot on Termux.
*   **Timeout Margin (Risk):** While the new binary succeeds where the old one failed, **18.9s is dangerously close to the 20s hard timeout** in `main.py`. Any thermal throttling, background OS tasks, or slightly longer prompts will cause the robot to ignore the user's first command (returning `[]`). While this PR mitigates an existing problem, the 20-second timeout in `main.py` should be increased, or the server should run a dummy warm-up prompt on startup.

### 5. Check Script Defects
The evaluation scripts are logically sound. 
- Using `ast.parse` and `exec` to extract `parse_command` from `main.py` without loading the heavy robot modules (camera, etc.) is clever. 
- The mock of `requests.post` effectively intercepts the data for raw text comparison. 
- The piggybacking in `voice_warm.py` / `smoke.py` (`sys.argv` and `exec` substring slicing) is brittle and hacky, but functionally correct for a one-off evidence gathering script.

***

**APPROVE WITH NOTES** 

**Notes for integration:** 
The build and integration are safe to merge. However, it is highly recommended to submit a follow-up patch to `main.py` increasing the `parse_command` timeout (e.g., to 30 or 40 seconds) or implementing a background cache warm-up during `setup_q4`, as 18.9s leaves almost no thermal headroom for the first cold interaction.

---

**Decisions for you:**
1. **"Speed only, not behaviour" is met for voice but not strictly for s1o.** 2/120 decisions change (both toward the correct answer), and no dotprod build avoids that. Accept or reject on that basis.
2. **The 20 s parse timeout / warm-up** the reviewer recommends is a separate change to `main.py`, not made here.
3. **The diagnostic `llama.cpp-b1609-dotprod-nofp16` directory** can be deleted if you don't need it; say so and I'll remove it.

The check scripts and outputs are in the scratchpad under `dp/`, and the review's stdout and stderr are in separate files there.
