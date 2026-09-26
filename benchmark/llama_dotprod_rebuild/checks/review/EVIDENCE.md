# Evidence: robot llama.cpp b1609 rebuilt with dotprod (+fp16)

## 1. Build
Commit confirmed from the binary: old `llama-server --version` -> `version: 1609 (e1a1abb7)`; old source tree
/data/data/com.termux/files/home/llama.cpp at e1a1abb78746c025f5e9039f590e37ccdb758ae7, `git status` clean,
rev-list count 1609, tag b10194. Old build dir: /data/data/com.termux/files/home/llama.cpp/build (untouched).
New dir: /data/data/com.termux/files/home/llama.cpp-b1609-dotprod (git clone of the old tree, checkout e1a1abb7).
Toolchain: Termux clang 21.1.8, cmake 4.3.2, GNU make 4.4.1 (run from proot with PATH/LD_LIBRARY_PATH/PREFIX
pointing at /data/data/com.termux/files/usr; the build dir uses the native path so RUNPATH works natively).
```
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/data/data/com.termux/files/usr/bin/cc -DCMAKE_CXX_COMPILER=/data/data/com.termux/files/usr/bin/c++ \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
make -C build -j6 llama-server        # 7m01s
```
CMakeCache GGML_*/LLAMA_*/BUILD_SHARED_LIBS/CMAKE_BUILD_TYPE/compilers/CMAKE_C(XX)_FLAGS* vs the old build: the
only difference is `GGML_CPU_ARM_ARCH:STRING=` -> `armv8.2-a+dotprod+fp16`. ggml-cpu compile flag added:
`-march=armv8.2-a+dotprod+fp16`. Configure feature tests: HAVE_DOTPROD Success, HAVE_FP16_VECTOR_ARITHMETIC
Success, HAVE_SVE/MATMUL_INT8/SME Failed. Build notes: web-UI bundle for b1609 not downloadable (fell back to
"latest" UI bundle; npm/node step failed under proot) - browser UI only, the robot uses /completion.
New RUNPATH: /data/data/com.termux/files/usr/bin/../../usr/lib:/data/data/com.termux/files/home/llama.cpp-b1609-dotprod/build/bin
CPU: Pixel 7 Tensor G2 (4x A55, 2x A78, 2x X1), /proc/cpuinfo has asimddp and asimdhp.

## 2. dotprod evidence
system_info (--verbose, same flags as server_manager):
- old: `CPU : NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |`
- new: `CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | DOTPROD = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |`
llvm-objdump -d | grep -c sdot (and fp16 `fmla vN.8h`):
| file | old sdot | new sdot | old fp16 fmla | new fp16 fmla |
|---|---|---|---|---|
| llama-server (6 KB launcher) | 0 | 0 | 0 | 0 |
| libggml-cpu.so.0.18.0 | 0 | 1045 | 0 | 456 |
| libggml-base / libllama / libllama-server-impl | 0 | 0 | 0 | 0 |
/proc/<pid>/maps confirms each server loads its own build's libggml-cpu.so, both started without LD_LIBRARY_PATH.

## 3a. s1o, full ladder (60 cases x 2 orders, 4 threads, cores 4-7, same S1O adapter & flags; weights-cold)
Old reference: s1o_b1609 main block of /termux-home/ladder/s1o_speed_20260926T045648Z (same cases file).
| | old b1609 | new b1609+dotprod+fp16 |
|---|---|---|
| correct / acceptable | 92/120 (77%) / 97 (81%) | 94/120 (78%) / 99 (82%) |
| order flips | 6/60 | 8/60 |
| median / P95 decision ms | 6650 / 13665 | 2478 / 4605 |
| prompt eval | 13 tok/s | 37 tok/s |
Choice agreement: 118/120. Differing: (L3_01, reversed) old 'go to the kitchen' p=0.562 (wrong) -> new 'go to the
living room' p=0.584 (correct); (L3_12, written) old 'go to the bedroom' p=0.758 (wrong) -> new 'go to the living
room' p=0.642 (correct). Max |probability difference| 0.4005 (L3_12 written, option 'go to the bedroom').
Diagnostics: (i) new build is deterministic: L3 rerun 24/24 same, max prob diff 0.0; old was deterministic in its
own run (main vs t4 0/60, 0.0000). (ii) dotprod-ONLY build (same commit, -march=armv8.2-a+dotprod, dir
llama.cpp-b1609-dotprod-nofp16): same 2 flips vs old; vs dotprod+fp16 0/120 choice differences, max prob diff
0.0001 -> the flips come from the dotprod kernels, not fp16. (iii) context: b2351 (other upstream build, FA off)
differs from old b1609 in 4/120 decisions, max prob diff 0.88.

## 3b. Voice parsing: main.py parse_command() + PARSE_SYS (exec'd from main.py source), server via
server_manager.start_setup('setup_q4') (exact robot flags, port 8080), only LLAMA_SERVER swapped.
Result: 15/15 identical parsed actions and identical raw model text once the old binary's timeouts are excluded.
Old binary: commands 1-2 of the first pass returned [] because parse_command's 20 s timeout expired on the cold
first request (~600-token prompt at ~10 tok/s); later requests reused the server's prompt cache. Follow-ups:
cmd 2 with warm cache -> identical; cmd 1 after the old server finished its warm-up -> identical (raw text
identical). New binary: first cold parse 18.9 s (first pass), 14.2 s (follow-up warm-up), 13.4 s (smoke) - under
the 20 s timeout but close. See VOICE table below.

## 3c. Generation: 200 tokens, temperature 0, ignore_eos, cache_prompt false; rounds old,new,old,new x3
old: [6.69, 6.42, 6.5, 6.71, 6.11, 6.27] tok/s, median 6.46; prompt eval median 10.0 tok/s
new: [8.36, 8.41, 8.06, 9.07, 8.68, 8.63] tok/s, median 8.52 (+32%); prompt eval median 31.6 tok/s
No thermal gate (the root thermal logger cannot run from proot); rounds were interleaved to spread drift.

## 4. Code change: server_manager.py LLAMA_SERVER only (diff below). Smoke: start_setup('setup_q4') launched the
new binary with the robot's flags, parse_command('Go to the kitchen') -> [navigate_to kitchen] in 13.4 s,
start_setup('stop') left no server. No other file in the robot repo changed.
## VOICE (first pass, old vs new)
 1. 'Find Chiara and tell her the pizza is here'  DIFFERENT  (20068 ms old / 18948 ms new)
    old: []
    new: [{"type": "find_person", "name": "Chiara", "message": "the pizza is here"}]
 2. 'Go to the bathroom'  DIFFERENT  (20025 ms old / 2382 ms new)
    old: []
    new: [{"type": "navigate_to", "room": "bathroom"}]
 3. 'Patrol the whole apartment'  SAME  (5926 ms old / 2194 ms new)
    old: [{"type": "patrol", "rooms": []}]
    new: [{"type": "patrol", "rooms": []}]
 4. 'Find my phone in the living room'  SAME  (5584 ms old / 3236 ms new)
    old: [{"type": "find_object", "object": "phone", "room": "living_room"}]
    new: [{"type": "find_object", "object": "phone", "room": "living_room"}]
 5. 'Come back'  SAME  (3279 ms old / 1778 ms new)
    old: [{"type": "come_back"}]
    new: [{"type": "come_back"}]
 6. 'Tell Abdel the meeting starts in ten minutes'  SAME  (5988 ms old / 3638 ms new)
    old: [{"type": "find_person", "name": "Abdel", "message": "the meeting starts in ten minutes"}]
    new: [{"type": "find_person", "name": "Abdel", "message": "the meeting starts in ten minutes"}]
 7. 'Go to the kitchen and say lunch is ready'  SAME  (6215 ms old / 3560 ms new)
    old: [{"type": "navigate_to", "room": "kitchen"}, {"type": "say", "message": "lunch is ready"}]
    new: [{"type": "navigate_to", "room": "kitchen"}, {"type": "say", "message": "lunch is ready"}]
 8. 'Check the bedroom and the hallway'  SAME  (4816 ms old / 2571 ms new)
    old: [{"type": "patrol", "rooms": ["bedroom", "hallway"]}]
    new: [{"type": "patrol", "rooms": ["bedroom", "hallway"]}]
 9. 'Where are my glasses'  SAME  (4308 ms old / 2431 ms new)
    old: [{"type": "find_object", "object": "glasses"}]
    new: [{"type": "find_object", "object": "glasses"}]
10. 'Say good night'  SAME  (4081 ms old / 2249 ms new)
    old: [{"type": "say", "message": "good night"}]
    new: [{"type": "say", "message": "good night"}]
11. 'Find someone and tell them the door is open'  SAME  (6003 ms old / 3451 ms new)
    old: [{"type": "find_person", "name": "someone", "message": "the door is open"}]
    new: [{"type": "find_person", "name": "someone", "message": "the door is open"}]
12. 'Go to the living room then come back'  SAME  (6286 ms old / 3504 ms new)
    old: [{"type": "navigate_to", "room": "living_room"}, {"type": "come_back"}]
    new: [{"type": "navigate_to", "room": "living_room"}, {"type": "come_back"}]
13. 'Look for the cat in the bedroom'  SAME  (5362 ms old / 3005 ms new)
    old: [{"type": "find_object", "object": "cat", "room": "bedroom"}]
    new: [{"type": "find_object", "object": "cat", "room": "bedroom"}]
14. 'Tell Chiara I love her'  SAME  (5725 ms old / 3287 ms new)
    old: [{"type": "find_person", "name": "Chiara", "message": "I love her"}]
    new: [{"type": "find_person", "name": "Chiara", "message": "I love her"}]
15. 'Blue banana seven'  SAME  (4401 ms old / 2279 ms new)
    old: [{"type": "say", "message": "Blue banana seven"}]
    new: [{"type": "say", "message": "Blue banana seven"}]
old: generation [6.69, 6.42, 6.5, 6.71, 6.11, 6.27] tok/s (n=[200, 200, 200, 200, 200, 200]), median 6.46; prompt eval median 10.0 tok/s
new: generation [8.36, 8.41, 8.06, 9.07, 8.68, 8.63] tok/s (n=[200, 200, 200, 200, 200, 200]), median 8.52; prompt eval median 31.6 tok/s

## VOICE follow-up (warm)
cmd 2 warm: old [navigate_to bathroom] 8260 ms / new same 2634 ms -> SAME
cmd 1 old after warm-up finished (4122 ms): [find_person Chiara 'the pizza is here'], raw identical to new
