# CAMPAIGN_P1_FIX3C — Coder report

Role: Coder, Claude Code CLI, claude-opus-5-5, Ponytail lite. Reviewer: Codex CLI gpt-6.1-sol, effort medium, Ponytail off.

## HEAD / status
- HEAD at start: `95bb0d858a1a7a9fa32a8575ea1bdba63fb71bb5` (matches).
- `git status --short` at start: empty.
- Nothing committed or pushed.

## Ponytail check
`~/.config/ponytail/config.json` contains `"defaultMode": "lite"`; the session start hook reported "PONYTAIL MODE ACTIVE — level: lite". Reviewer was run with `PONYTAIL_DEFAULT_MODE=off` and the request starts with "Ponytail mode: off".

## Cause (from owner bytes)
`owner_rehearsal_p1_fix3_warmup_L0.json` → `monitor_errors[0]` (verbatim, Python repr):

```
'RuntimeError: agents/robot/other runners resident: 1312 /debug_ramdisk/su -c { taskset -p cf $$ >/dev/null; pin_rc=$?; actual=$(taskset -p $$); read_rc=$?; printf \'CAMPAIGN_ROOT_MASK pid=%s actual=%s\\n\' "$$" "$actual"; [ "$pin_rc" = 0 ] || exit 97; [ "$read_rc" = 0 ] || exit 98; case "$actual" in *": cf") ;; *) exit 98;; esac; read_node() {   if [ -r "$1" ]; then     value=$(cat "$1" 2>/dev/null) || value=UNREADABLE     printf \'%s\\t%s\\n\' "$1" "$value"   else printf \'%s\\tUNREADABLE\\n\' "$1"; fi } for f in /sys/class/power_supply/battery/capacity /sys/class/power_supply/battery/temp  /sys/class/thermal/thermal_zone*/type /sys/class/thermal/thermal_zone*/temp  /sys/class/thermal/cooling_device*/type /sys/class/thermal/cooling_device*/cur_state  /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq  /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; do read_node "$f"; done paths=$(find /sys/devices/platform /sys/devices/system/cpu /sys/kernel /sys/power /dev/cpuctl -maxdepth 5 -type f 2>/dev/null |  grep -Ei \'(boost.*(min|max)|(min|max).*boost|power.?hint|/(cpu|gpu|mif|int).*_(min|max)_freq)\') printf \'optional_discovery\\tbest effort: readable boost min/max and power-hint nodes\\n\' for f in $paths; do read_node "$f"; done; } </dev/null >/data/local/tmp/coresidency_campaign_diagnostics.txt 2>&1; rc=$?; cat /data/local/tmp/coresidency_campaign_diagnostics.txt; exit $rc'
```

- Guard: `benchmark/campaign/runtime.py:48-58` (base) `clear_processes`, `pgrep -fa` with the alternatives at `runtime.py:49-51`; base allowed only `{os.getpid(), server_pid}`.
- Each alternative tested on the line: **only `node` matches**, 4 times. The matched text is `read_node() {` (diagnostics.py:40), `read_node "$f"` (diagnostics.py:50 and :54) and `power-hint nodes` (diagnostics.py:53).
- pid 1312 is the su client spawned by `coresidency.root` (`coresidency.py:125`, `subprocess.run(['su','-c',...])`) for `diag.snapshot` (`diagnostics.py:57`). Termux's `$PREFIX/bin/su` wrapper does `exec /debug_ramdisk/su "$@"`, so the process is a direct child of the runner.
- The guard ran in the warm-up `shared_check` worker (`phase1.py:335-336`) while that diagnostics su was alive. The failure then propagated to `phase1.py:751`.
- Only this one line was listed, although the Magisk root-side shell carrying the same text must have been alive. This indicates uid-0 processes are invisible to the Termux-uid pgrep (hidepid).

## Fix (file:line)
- `benchmark/campaign/runtime.py:48-51` `parent_pid(pid)`: reads ppid from `/proc/PID/stat`, parsed after the last `)`.
- `benchmark/campaign/runtime.py:54-76` `own_process(pid)`: walks ppid at check time. It returns True when the chain reaches `os.getpid()` and False when it reaches pid ≤ 1. It returns None when the matched pid itself has exited (no longer resident). Any other read/parse error, a vanished or hidden ancestor, or a chain longer than 4096 raises.
- `benchmark/campaign/runtime.py:79-96` `clear_processes`: the pattern is unchanged. A matched line is skipped only for the server pid or own ancestry. An ancestry error makes the line resident with `[process ancestry unreadable: ...]` appended (fail closed).
- `benchmark/campaign/self_check.py`: the old mocks used fake pid 99999, which is nonexistent and now skipped as exited. They now patch `parent_pid → 1` (foreign). 2 lines changed.
- No shared file changed. The guard lives only in `benchmark/campaign/runtime.py`, used by `phase1.py` and `lag_probe.py`. The separate `desk2/speed_block.py` pgrep guard is not called by campaign paths.

```diff
diff --git a/benchmark/campaign/runtime.py b/benchmark/campaign/runtime.py
index d54c142..f31d003 100644
--- a/benchmark/campaign/runtime.py
+++ b/benchmark/campaign/runtime.py
@@ -45,6 +45,37 @@ def imports():
     return v, v.sb, v.sb.pm, v.sb.cr, speed_session
 
 
+def parent_pid(pid):
+    """ppid from /proc/PID/stat (comm may hold spaces or parentheses, so split after the last ')')."""
+    with open(f'/proc/{pid}/stat') as f:
+        return int(f.read().rsplit(')', 1)[1].split()[1])
+
+
+def own_process(pid):
+    """True if pid is this runner or its descendant, by ppid ancestry read now; never by command text.
+
+    Our su clients, root shells, pumps and diagnostics are our children, and their command lines can
+    match the guard pattern (diagnostics' read_node matched 'node'). None: the matched pid itself
+    exited before its ancestry was read, so it is no longer resident. Any other unreadable link
+    (including an ancestor hidden by hidepid) raises: fail closed.
+    """
+    me, chain = os.getpid(), pid
+    for _ in range(4096):
+        if chain == me:
+            return True
+        if chain <= 1:
+            return False
+        try:
+            chain = parent_pid(chain)
+        except (FileNotFoundError, ProcessLookupError) as e:
+            if chain == pid:
+                return None
+            raise RuntimeError(f'process ancestry unreadable: pid {pid} at {chain}: {e}') from e
+        except (OSError, ValueError, IndexError) as e:
+            raise RuntimeError(f'process ancestry unreadable: pid {pid} at {chain}: {e}') from e
+    raise RuntimeError(f'process ancestry unreadable: pid {pid}: chain too long')
+
+
 def clear_processes(server_pid=None):
     pattern = (r'claude|agy|node|codex|llama-server|chat\.py|main\.py|run_mission\.py|'
                r'benchmark/.*\.py|phase1\.py|lag_probe\.py|power_map\.py|coresidency\.py|thermal_char\.py|'
@@ -52,8 +83,15 @@ def clear_processes(server_pid=None):
     result = subprocess.run(['pgrep', '-fa', pattern], capture_output=True, text=True)
     if result.returncode not in (0, 1):
         raise RuntimeError('process enumeration failed: '+result.stderr)
-    allowed = {os.getpid(), server_pid}
-    others = [l for l in result.stdout.splitlines() if int(l.split()[0]) not in allowed]
+    others = []
+    for line in result.stdout.splitlines():
+        pid = int(line.split()[0])
+        try:
+            if pid == server_pid or own_process(pid) is not False:
+                continue
+        except RuntimeError as e:  # fail closed: unproven ancestry counts as resident
+            line += f' [{e}]'
+        others.append(line)
     if others:
         raise RuntimeError('agents/robot/other runners resident: '+'\n'.join(others))
 
diff --git a/benchmark/campaign/self_check.py b/benchmark/campaign/self_check.py
index 7300a25..5004c18 100644
--- a/benchmark/campaign/self_check.py
+++ b/benchmark/campaign/self_check.py
@@ -141,7 +141,7 @@ def check_runtime():
     for rc in (0,1):
         with patch.object(rt.subprocess,'run',return_value=NS(returncode=rc,stdout='',stderr='')):rt.clear_processes()
     for output in ('99999 codex','99999 main.py','99999 llama-server','99999 phase1.py'):
-        with patch.object(rt.subprocess,'run',return_value=NS(returncode=0,stdout=output,stderr='')):
+        with patch.object(rt.subprocess,'run',return_value=NS(returncode=0,stdout=output,stderr='')),patch.object(rt,'parent_pid',return_value=1):
             try:rt.clear_processes()
             except RuntimeError:pass
             else:raise AssertionError('resident accepted')
@@ -417,7 +417,7 @@ def check_root_adapters():
     assert all(abs((b['t']-a['t'])-.37)<1e-6 for a,b in zip(rows[:3],rows[1:4]))
     # Expanded bare-filename process refusal coverage.
     for name in ('power_map.py','coresidency.py','thermal_char.py','duty_cycle.py','camera_power.py','speed_block.py','session.py'):
-        with patch.object(rt.subprocess,'run',return_value=NS(returncode=0,stdout='99999 python '+name,stderr='')) as command:
+        with patch.object(rt.subprocess,'run',return_value=NS(returncode=0,stdout='99999 python '+name,stderr='')) as command,patch.object(rt,'parent_pid',return_value=1):
             try:rt.clear_processes()
             except RuntimeError:pass
             else:raise AssertionError('bare runner accepted')
```

## Audit (generated command lines vs guard pattern)
From `self_check_fix3c.check_audit`: real functions are called, subprocess is patched to record argv, and each argv is joined as pgrep -f sees it.

| Source | Spawn | Matches |
|---|---|---|
| diag.battery | su -c (cat capacity) | none |
| **diag.snapshot** | su -c diagnostics script | **node** (read_node / "nodes"): own child, now excluded by ancestry |
| rt.camera_version | su -c dumpsys package | none |
| rt.memory_sample | su -c root_sample (dumpsys meminfo, pidof, ps) + su -c pidof | none |
| cr.read_dump | su -c dumpsys thermalservice | none |
| cr.lmk_lines | su -c logcat/grep | none |
| cr.camera_end_check | su -c am force-stop; su -c pidof | none |
| sb.battery_sample | su -c cat battery | none |
| sb.Screen start/restore | termux-wake-lock/unlock, su -c settings | none |
| pm.camera_start | am start (Termux wrapper execs app_process … com.termux.termuxam.Am) | none |
| cr.camera_stop | am broadcast | none |
| cr.RootShell (pumps, persistent monitor/fast shells) | `su` (scripts via stdin, not argv) | none |
| cr.server_cmd | llama-server Popen | llama-server: own child and allowed server pid |

Not in argv or not visible: RootShell stdin scripts (fast_sample, power sampler, root_mask, taskset). Root-side shells and their children run as uid 0 from magiskd and are hidden per the owner evidence above. Pumps are threads. `lag_probe.py` uses the same `rt.clear_processes`.

**Residual (not live-validated):** a Magisk root-side shell is not our descendant. If it were visible to pgrep, a diagnostics shell containing `read_node` would still be refused (fail closed). The owner evidence indicates it is hidden.

## Regressions (`benchmark/campaign/self_check_fix3c.py`)
- Exact failure: the archived owner line is fed to the **base** guard (loaded from `git show 95bb0d8:…`), which refuses with a message equal to the owner `monitor_errors[0]`. The fixed guard accepts it with ancestry `1312→runner` and a deeper own descendant.
- Identical text, foreign (`1312→1`, `1312→500→1`, runner's parent side): refused with the owner's exact message. Foreign codex/claude/run_mission/phase1 are refused. The server pid is allowed.
- Fail closed: PermissionError, garbage stat, vanished ancestor, hidden ancestor and cycle are all refused with the ancestry error attached. A pgrep failure raises.
- Real `/proc` (no root): an own child with the owner text is accepted. A reparented orphan with identical text is refused. An exited pid is skipped.

```
PASS owner monitor_errors reproduced on base (own su diagnostics client, pattern node <- read_node); fixed: own descendants accepted, identical-text foreign refused, foreign agents refused
PASS fail closed: unreadable/garbage/vanished-ancestor/hidden/cyclic ancestry refuse; pgrep failure raises
PASS real /proc: own child 6193 accepted, reparented orphan 6195 with identical text refused, exited pid skipped
WARNING: saved screen timeout is already 2147483647; restore preserves it
screen timeout saved 2147483647 ms; set/read back 2147483647
screen timeout restored/read back 2147483647 ms
{"source": "diag.battery", "spawned": 1, "matches": [], "example": "su -c { taskset -p f $$ >/dev/null; pin_rc=$?; actual=$(taskset -p $$); read_rc=$?; printf 'CAMPAIGN_ROOT_MASK pid=%s actual=%s\\n' \"$$\" \"$actual\"; [ \"$pin_rc\" ="}
{"source": "diag.snapshot", "spawned": 1, "matches": ["node"], "example": "su -c { taskset -p f $$ >/dev/null; pin_rc=$?; actual=$(taskset -p $$); read_rc=$?; printf 'CAMPAIGN_ROOT_MASK pid=%s actual=%s\\n' \"$$\" \"$actual\"; [ \"$pin_rc\" ="}
{"source": "rt.camera_version", "spawned": 1, "matches": [], "example": "su -c { taskset -p f $$ >/dev/null; pin_rc=$?; actual=$(taskset -p $$); read_rc=$?; printf 'CAMPAIGN_ROOT_MASK pid=%s actual=%s\\n' \"$$\" \"$actual\"; [ \"$pin_rc\" ="}
{"source": "rt.memory_sample (root_sample + pidof)", "spawned": 1, "matches": [], "example": "su -c { taskset -p f $$ >/dev/null; pin_rc=$?; actual=$(taskset -p $$); read_rc=$?; printf 'CAMPAIGN_ROOT_MASK pid=%s actual=%s\\n' \"$$\" \"$actual\"; [ \"$pin_rc\" ="}
{"source": "cr.read_dump", "spawned": 1, "matches": [], "example": "su -c { taskset -p f $$ >/dev/null; pin_rc=$?; actual=$(taskset -p $$); read_rc=$?; printf 'CAMPAIGN_ROOT_MASK pid=%s actual=%s\\n' \"$$\" \"$actual\"; [ \"$pin_rc\" ="}
{"source": "cr.lmk_lines", "spawned": 1, "matches": [], "example": "su -c { taskset -p f $$ >/dev/null; pin_rc=$?; actual=$(taskset -p $$); read_rc=$?; printf 'CAMPAIGN_ROOT_MASK pid=%s actual=%s\\n' \"$$\" \"$actual\"; [ \"$pin_rc\" ="}
{"source": "cr.camera_end_check", "spawned": 2, "matches": [], "example": "su -c { taskset -p f $$ >/dev/null; pin_rc=$?; actual=$(taskset -p $$); read_rc=$?; printf 'CAMPAIGN_ROOT_MASK pid=%s actual=%s\\n' \"$$\" \"$actual\"; [ \"$pin_rc\" ="}
{"source": "sb.battery_sample", "spawned": 1, "matches": [], "example": "su -c { taskset -p f $$ >/dev/null; pin_rc=$?; actual=$(taskset -p $$); read_rc=$?; printf 'CAMPAIGN_ROOT_MASK pid=%s actual=%s\\n' \"$$\" \"$actual\"; [ \"$pin_rc\" ="}
{"source": "sb.Screen start/restore", "spawned": 7, "matches": [], "example": "termux-wake-lock"}
{"source": "pm.camera_start (am)", "spawned": 1, "matches": [], "example": "am start -n com.pixelrobot.robotcam/.StartActivity --es mode B --ei rate 1"}
{"source": "cr.camera_stop (am)", "spawned": 1, "matches": [], "example": "am broadcast -n com.pixelrobot.robotcam/.ControlReceiver -a com.pixelrobot.robotcam.STOP"}
{"source": "cr.RootShell (pumps, persistent monitor shell)", "spawned": 1, "matches": [], "example": "su"}
{"source": "cr.server_cmd (llama-server Popen argv)", "spawned": 1, "matches": ["llama-server"], "example": "/data/data/com.termux/files/home/llama.cpp-b1609-dotprod/build/bin/llama-server -m /data/data/com.termux/files/home/models/gemma-4-E2B-it-Q4_0.gguf --port 8080 "}
PASS audit: of 13 spawn sources only diagnostics (node) and the owned llama-server match; both are direct children of the runner (subprocess.run/Popen), excluded by ancestry/server pid
PASS CAMPAIGN_P1_FIX3C self-check (offline; agents resident; proot; NOT VALID for timing)
```

## Archive row
Copied unchanged from `/termux-home/storage/downloads/campaign/`, SHA-256 verified. Inventory: `benchmark/campaign/runs/archive_sha256_p1_fix3c.json`. No block files existed.

| Path | Bytes | SHA-256 |
|---|---|---|
| `benchmark/campaign/runs/owner_rehearsal_p1_fix3.json` | 313129 | `dea0bebb2eff230d329fa7e1b30e2660b16c51dad02a8cde3b4e5dc29a8c4d0b` |
| `benchmark/campaign/runs/owner_rehearsal_p1_fix3.stdout` | 2322 | `6fa55674026ba6fa338c76b6426b0097bb568cdb6f2358b89bb11f6a973577e0` |
| `benchmark/campaign/runs/owner_rehearsal_p1_fix3.stderr` | 767 | `2be49d9adb050bb4dc7f2f1d4e5388c00e3bc7b1b6c3833957061523712254e5` |
| `benchmark/campaign/runs/owner_rehearsal_p1_fix3_llama-server.log` | 3028 | `75742a8c5041529dee4e0ec809ac864013ba55c2606063ddf8c40a403247cac8` |
| `benchmark/campaign/runs/owner_rehearsal_p1_fix3_warmup_L0.json` | 67973 | `9861bfe4a07da182e254d2cec301220332379a7f7ffec9f223f1205278096c41` |

RUN_INDEX row (fixed label): **owner_rehearsal_p1_fix3: NOT VALID — REHEARSAL INCOMPLETE.** Setup, 3 idle sub-phases and the warm-up ran; the warm-up process guard flagged the runner's own root diagnostics shell as "agents/robot/other runners resident" (failure_kind shared), so no pause or block ran. Diagnostic only: warm-up YOLO 320 20/20, 640 4/4, camera 20/20, selector 1 call. These figures were verified against the warm-up summary.

## Checks
Termux python, repo root, proot, agents resident, offline mocks only. **NOT VALID for timing; no live/root validation.** Outputs are in `benchmark/campaign/checks/p1_fix3c/`.

| Check | Command | Exit |
|---|---|---|
| self_check | `benchmark/campaign/self_check.py` | 0 |
| self_check_fix2 | `benchmark/campaign/self_check_fix2.py` | 0 |
| self_check_fix3 | `benchmark/campaign/self_check_fix3.py` | 0 |
| self_check_fix3b | `benchmark/campaign/self_check_fix3b.py` | 0 |
| self_check_fix3c | `benchmark/campaign/self_check_fix3c.py` | 0 |
| coresidency_units | `benchmark/coresidency/test_coresidency.py` | 0 |
| power_map_units | `benchmark/power_map/test_power_map.py` | 0 |
| duty_cycle_units | `benchmark/duty_cycle/test_duty_cycle.py` | 0 |
| camera_power_units | `benchmark/camera_power/test_camera_power.py` | 0 |
| thermal_char_units | `benchmark/thermal_char/test_thermal_char.py` | 0 |
| speed1_self_check | `benchmark/relate_anything/speed1/self_check.py` | 0 |
| desk2_self_check | `benchmark/relate_anything/desk2/self_check.py` | 0 |
| detector_units | `test_detector_robotcam.py` | 0 |
| mission_units | `test_run_mission.py` | 0 |
| cycle_units | `run_cycle_safety_test.py` | 0 |
| dry_run_mock | `benchmark/campaign/phase1.py --dry-run --mock --output /termux-home/robot/benchmark/campaign/checks/p1_fix3c/coder_dry_mock_p1_fix3c.json` | 0 |

The mock dry run's stdout is identical to FIX3B's (stem aside): label NOT VALID — DRY RUN, `rehearsal_pass` false by design.

## RUN.md
The header is now CAMPAIGN_P1_FIX3C, with a Fix3c paragraph and a verification-limits entry.
1. Rehearsal: `owner_rehearsal_p1_fix3c`. Files: `.json`, `.stdout`, `.stderr`, `_llama-server.log`, `_warmup_L0.json`, `_block_01_L0.json` … `_block_06_L0.json`.
2. Full session (only after the rehearsal passes): `owner_session_p1_fix3c`. Same file set.

Frozen manifest: `benchmark/campaign/frozen_manifest_campaign_p1_fix3c.json`, 52 entries, sha256 `0bfdf757756ee0c39cbe81e4bc0a1fe61eb768296542d6954e6b24f0d6b83168`. After the review the live tree was re-verified: 0 mismatches.

## Reviewer
Isolated copy: `/termux-home/review_campaign_p1_fix3c` (git archive of 95bb0d8 plus the candidate files, no .git, no models). Request: `/termux-home/storage/downloads/campaign/review_request_campaign_p1_fix3c.txt` (52 KB, sha256 `a0d097aa379005fc3fb41dd364d1606655393a52181a9b52b188242afb590e0e`). Command:

```sh
cd /termux-home/review_campaign_p1_fix3c && PONYTAIL_DEFAULT_MODE=off \
  codex --no-daemon exec --skip-git-repo-check -m gpt-6.1-sol -c model_reasoning_effort="medium" \
    -s read-only - < /termux-home/storage/downloads/campaign/review_request_campaign_p1_fix3c.txt \
    > review_campaign_p1_fix3c_r1.txt 2> review_campaign_p1_fix3c_r1.stderr
```
Round 1, exit 0. Diagnostics: `review_campaign_p1_fix3c_r1.stderr` (590,699 B, kept separately).

### Review R1 (verbatim)

VERDICT: APPROVED

No blocking findings.

**Informational — benchmark/campaign/runtime.py:57, :268:** Magisk may create root-side shells from its daemon, outside the runner’s ancestry. Those shells are not automatically accepted by this fix. The owner evidence supports their being hidden from Termux’s process enumeration; if visible, a diagnostics shell containing `read_node` could still be refused. This remains a live-validation limitation.

**Q1.** Yes. Ownership is determined from `/proc/PID/stat` ancestry read during each check, never from command text. The existing explicit owned-server-PID exception remains.

**Q2.** Yes, for matching processes visible to enumeration. Foreign agents, robot scripts and runners remain refused, including foreign processes with identical command text. I independently verified the exact owner text against own-child, deeper-descendant and foreign ancestry.

**Q3.** Yes. Permission failures, malformed data, missing ancestors and cyclic ancestry fail closed. Skipping the matched PID when it has itself exited is acceptable: it is no longer resident. PID reuse between enumeration and inspection remains the acknowledged race.

**Q4.** The audit covers the campaign’s direct spawn sources: battery and diagnostics, camera version, memory/PSS and absence confirmation, thermal reads, LMK reads, camera cleanup, screen/wake-lock control, camera start/stop, persistent root shells and server startup. Only diagnostics (`node`) and the server (`llama-server`) match among those argv values. Pumps are threads; persistent-shell scripts travel through stdin. `lag_probe` uses the same guard. The Magisk root-side shell caveat above prevents an unconditional claim about every daemon-created process.

**Q5.** No additional reproducible self-detection or state-dependent guard failure was found for the documented native-Termux invocation. Campaign callers use the corrected guard; the separate desk2 guard is not called by these campaign paths.

**Q6.** Yes. The regressions cover the exact archived failure before/after the fix, identical-text foreign refusal and unreadable ancestry. The recorded base reproduction requires Git history, unavailable in this isolated copy. I independently passed the post-fix checks, fail-closed checks and real `/proc` child/orphan/exited-PID regression.

All **52 manifest entries** and **five archived owner files** match their recorded sizes and SHA-256 hashes. Original Downloads provenance relies on the Coder’s recorded source verification. The RUN_INDEX label, fresh `fix3c` stems and expected filenames are correct. All ten campaign Python files parse successfully; the mock evidence correctly remains **NOT VALID**, with `rehearsal_pass=false`.

No files were changed and no hardware commands were run. Live root/rehearsal/session validation remains outstanding.

## Proposed commit (NOT done; awaiting "yes commit", then a separate "yes push")
Message: `benchmark: campaign phase-1 own-process guard fix, rehearsal archive`

Files (`runs/owner_rehearsal_p1_fix3_llama-server.log` is Git-ignored and needs `git add -f`):
- `benchmark/campaign/RUN.md`
- `benchmark/campaign/RUN_INDEX.md`
- `benchmark/campaign/checks/p1_fix3c/camera_power_units.stderr`
- `benchmark/campaign/checks/p1_fix3c/camera_power_units.stdout`
- `benchmark/campaign/checks/p1_fix3c/coder_dry_mock_p1_fix3c.json`
- `benchmark/campaign/checks/p1_fix3c/coder_dry_mock_p1_fix3c_block_01_L0.json`
- `benchmark/campaign/checks/p1_fix3c/coder_dry_mock_p1_fix3c_block_02_L1.json`
- `benchmark/campaign/checks/p1_fix3c/coder_dry_mock_p1_fix3c_block_03_L2.json`
- `benchmark/campaign/checks/p1_fix3c/coder_dry_mock_p1_fix3c_block_04_L3.json`
- `benchmark/campaign/checks/p1_fix3c/coder_dry_mock_p1_fix3c_block_05_L4.json`
- `benchmark/campaign/checks/p1_fix3c/coder_dry_mock_p1_fix3c_block_06_L0.json`
- `benchmark/campaign/checks/p1_fix3c/coder_dry_mock_p1_fix3c_warmup_L0.json`
- `benchmark/campaign/checks/p1_fix3c/coresidency_units.stderr`
- `benchmark/campaign/checks/p1_fix3c/coresidency_units.stdout`
- `benchmark/campaign/checks/p1_fix3c/cycle_units.stderr`
- `benchmark/campaign/checks/p1_fix3c/cycle_units.stdout`
- `benchmark/campaign/checks/p1_fix3c/desk2_self_check.stderr`
- `benchmark/campaign/checks/p1_fix3c/desk2_self_check.stdout`
- `benchmark/campaign/checks/p1_fix3c/detector_units.stderr`
- `benchmark/campaign/checks/p1_fix3c/detector_units.stdout`
- `benchmark/campaign/checks/p1_fix3c/dry_run_mock.stderr`
- `benchmark/campaign/checks/p1_fix3c/dry_run_mock.stdout`
- `benchmark/campaign/checks/p1_fix3c/duty_cycle_units.stderr`
- `benchmark/campaign/checks/p1_fix3c/duty_cycle_units.stdout`
- `benchmark/campaign/checks/p1_fix3c/mission_units.stderr`
- `benchmark/campaign/checks/p1_fix3c/mission_units.stdout`
- `benchmark/campaign/checks/p1_fix3c/power_map_units.stderr`
- `benchmark/campaign/checks/p1_fix3c/power_map_units.stdout`
- `benchmark/campaign/checks/p1_fix3c/results.json`
- `benchmark/campaign/checks/p1_fix3c/self_check.stderr`
- `benchmark/campaign/checks/p1_fix3c/self_check.stdout`
- `benchmark/campaign/checks/p1_fix3c/self_check_fix2.stderr`
- `benchmark/campaign/checks/p1_fix3c/self_check_fix2.stdout`
- `benchmark/campaign/checks/p1_fix3c/self_check_fix3.stderr`
- `benchmark/campaign/checks/p1_fix3c/self_check_fix3.stdout`
- `benchmark/campaign/checks/p1_fix3c/self_check_fix3b.stderr`
- `benchmark/campaign/checks/p1_fix3c/self_check_fix3b.stdout`
- `benchmark/campaign/checks/p1_fix3c/self_check_fix3c.stderr`
- `benchmark/campaign/checks/p1_fix3c/self_check_fix3c.stdout`
- `benchmark/campaign/checks/p1_fix3c/speed1_self_check.stderr`
- `benchmark/campaign/checks/p1_fix3c/speed1_self_check.stdout`
- `benchmark/campaign/checks/p1_fix3c/thermal_char_units.stderr`
- `benchmark/campaign/checks/p1_fix3c/thermal_char_units.stdout`
- `benchmark/campaign/runs/archive_sha256_p1_fix3c.json`
- `benchmark/campaign/runs/owner_rehearsal_p1_fix3.json`
- `benchmark/campaign/runs/owner_rehearsal_p1_fix3.stderr`
- `benchmark/campaign/runs/owner_rehearsal_p1_fix3.stdout`
- `benchmark/campaign/runs/owner_rehearsal_p1_fix3_llama-server.log`
- `benchmark/campaign/runs/owner_rehearsal_p1_fix3_warmup_L0.json`
- `benchmark/campaign/runtime.py`
- `benchmark/campaign/self_check.py`
- `benchmark/campaign/self_check_fix3c.py`
- `benchmark/campaign/frozen_manifest_campaign_p1_fix3c.json`
