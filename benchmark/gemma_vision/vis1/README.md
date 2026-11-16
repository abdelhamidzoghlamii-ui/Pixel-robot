# VIS-1 — owner-timed Gemma text and vision, motors off

Preparation only. No measured results exist yet. All new repository files are in this directory.

Run from native Termux after closing all Claude/AGY/Codex sessions and stopping existing servers
on 8080/8090/8091, robot/chat processes and RobotCam. Motors stay physically disconnected.
Use 75–80% battery initially, unplugged; 50–80% Discharging is enforced throughout (#92).
Keep the screen on and Termux foreground; do not use other apps during the run.

```bash
bash ~/robot/benchmark/gemma_vision/vis1/oneshot.sh
```

Owner-only `oneshot.sh --preflight` checks native and Debian prerequisites, then returns.
It does not start the idle, monitor, bridge or benchmark. Native prerequisite checks still
require root and inspect battery/ports/affinity; this is not an offline check.

Five-minute idle, then T1,T2,V1,V2,V3,V4. Each starts cold, waits for health, discards one warm-up,
measures a 180 s request window and stops its owned server. A request unfinished at 180 s is excluded.
Server lifetime is independently capped by timeout at 900 s, with 3 s kill grace. No other server PID
is stopped. Help capability detection for T2 happens only during the owner run; unsupported MTP flags
are omitted and recorded. The current Gemma builder supplies text prompts (thinking off).

Expected duration: roughly 30 minutes plus cooling gates; allow 60–120 minutes. A skin gate waits
until VIRTUAL-SKIN <= the original post-idle baseline +1.5 C, and stops after 15 minutes if missed.
A second gate after the cache drop ensures launch is still cool. Load and warm-up each allow 180 s.
Battery/thermal/affinity/agent failures stop the session and retain its evidence. This is a short-block
heat comparison: 180 s does not establish sustained thermal equilibrium (#93).

Native mode of the same runner supplies root telemetry through a per-launch bridge directory.
It runs on LITTLE cores. Power is sampled at nominal 0.37 s, with 5 s HAL dumps and agent checks;
CPU caps and memory are recorded with every power sample. Actual timestamps/coverage are retained.
Campaign `power.weighted` integrates receipt-time current_now × voltage_now without extrapolating gaps;
power_map's later-sample cap accounting is a marked copy. Peak VmHWM/PSS includes load and warm-up.
LMK counts use the coresidency patterns; unrelated logcat contents are never written to files.
The root CPU sensor readings are diagnostic readings, not the skin-governor heat state (#126).

Run records and all replies go in `runs/<UTC>/`; bridge telemetry and console go in
`runs/bridge_<UTC>/`. No JPEG/base64 image data is written by this benchmark. The saved request manifest
contains payload hashes, code hashes, commands, model hashes, binary hashes and base commit.
Vision request identities include frame name, budget and mmproj. The 180 s limit may leave some of the
ten frames unvisited; Phase 2 must show those missing replies explicitly, without filling them in.

Resume with the same `--only` selection (default all six) and the unchanged code:

```bash
bash ~/robot/benchmark/gemma_vision/vis1/oneshot.sh --resume /termux-home/robot/benchmark/gemma_vision/vis1/runs/<UTC>
```

Only VALID completed blocks are skipped. Interrupted block attempts remain separate. The original skin
baseline is retained; a new five-minute idle and new wrapper/bridge are required. `--only T1,V2` selects
from the canonical order. `python3 -B vis1_run.py --dry-run` is **NOT VALID TIMING** and starts no servers.

Resume checks the full manifest, including the wrapper hash. The preflight-only wrapper fix
changes that hash, so it cannot resume runs made with the original wrapper. Preserve those
records and use the exact original code at the original paths for an owner-approved resume;
do not edit `run.json`, bypass the mismatch, or select only the unfinished blocks. A missed
cooling gate does not invalidate an earlier VALID block. Close agents and unnecessary apps,
let the phone cool naturally, and retain the original baseline and all battery/thermal gates.
If it still cannot cool within 15 minutes, stop and retain the new failed attempt.

Phase 2 is performed only after the owner reports completion: read the saved folder, create results.md
and a PIL replies PDF, and copy both into `/storage/emulated/0/Download/vis1/`. No repeat run is implied.
Compare T1 with #137's 19.18 gen tok/s /39% acceptance as historical context: #137 preceded TPL-FIX,
so prompt tokenization differs. Q8_0 projector provenance is in `mmproj.json`.
