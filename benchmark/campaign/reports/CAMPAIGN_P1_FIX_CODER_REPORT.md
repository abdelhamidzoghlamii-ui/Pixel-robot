# CAMPAIGN_P1_FIX coder report

HEAD at entry and completion: `c47fc41999d7190b4e8171b8e7806c6b77b5e7dd` (requested base matched). Entry `git status --short`: empty. No staging, commit or push performed.

EXECUTION PROFILE: Coder | Codex CLI | gpt-6.1-sol | medium | Ponytail lite.
FALLBACK: AGY CLI | gemini-3.1-pro-high | effort N/A | Ponytail lite; not used.
REVIEWER PROFILE: Reviewer | Claude Code CLI headless (request on stdin) | claude-opus-5-5 | effort N/A | Ponytail off via PONYTAIL_DEFAULT_MODE=off.
Reviewer fallback: Codex CLI `codex exec -s read-only`, gpt-6.1-sol, medium, Ponytail off, owner approval required; not used.

# CAMPAIGN_P1_FIX root-cause evidence

Base: c47fc41999d7190b4e8171b8e7806c6b77b5e7dd. Initial HEAD matched; initial `git status --short` was empty.
Coder: Codex CLI, gpt-6.1-sol, medium, Ponytail lite. No fallback used.

## Proven failure mechanism; probable upstream cause

Owner `owner_session_p1a2.stderr` names base `phase1.py:545 -> :483 -> runtime.py:53 -> desk2/speed_block.py:203 -> :176`.
Base desk2 `:201` reads `previous = os.sched_getaffinity(0)`; `:202` computes `monitor_cpus = previous - cpus`; `:203` checks it, before RootShell creation at `:209` or any root readback at `:218-219`.
Subtraction cannot include measured CPUs. Thus this specific traceback proves an **empty derived safe mask**, not an unsafe root-shell readback. There is no root/su/pump PID at that failing call.

Both failed JSONs record only successful L0 in `layout_checks`. Base phase1 `:480-492` appends after all checks, so the next failing iteration is L1, not L0. L1 measured set is `{0,1,2,3,4,5}` (`LAYOUTS`, `variants.CLUSTERS` and `phase1:482`). Therefore the main thread's nonempty kernel affinity was a subset of those CPUs. The exact subset was not logged and cannot be recovered. `p1a2` L0 safe set was only `{0,1,2,3}`, whereas other runs often had `{0,1,2,3,6,7}`: CPU availability already varied across owner invocations. The owner's phrase "at L0" means the last recorded layout; the traceback is the following L1 prepare, with no idle or timed block.

The initial `cr.check_cores('preflight')` passed in both failed invocations, so cores 6–7 were lost later during setup. `p1a` retained 6–7 at L0 prepare; `p1a2` had already lost them by L0 prepare. This sharpens the transition window without proving Android's state.

Ranked upstream causes:

1. Android moved Termux out of top-app during setup, restricting it to 0–5; this is consistent with the missing 6–7 and the project's `benchmark/strategic_selector/ladder/measure.py:36-43`, which explicitly documents top-app 0–7 versus foreground 0–5. **Probable, not proven**: no cpuset transition record exists in owner evidence.
2. Android temporarily restricted the caller, then restored the cpuset but left its requested affinity narrowed. A stale thread mask also explains failure after a transient move. This cannot be distinguished from (1) with old evidence.
3. An undocumented native-library/thread affinity side effect. There is no explicit main-thread pin leak in the inspected Python: `pm.build_detector:109-145` pins only newly created workers and its PinnedDetector thread; `sb.build:153-164` does likewise. `prepare_monitor:205-214` restores the caller in `finally`, but writes its captured effective `previous` as an explicit requested mask. If captured during a temporary restriction, this restoration can persist the narrower request after Android expands the cpuset. This is a concrete code mechanism for candidate 2, not an explicit worker-pin leak. Refreshing the caller also safely covers this candidate. No library internals or model files were changed.

Root/su readback, cached shells, malformed root output, persistent wrappers and leftover processes cannot explain this traceback: it stops before those operations on L1. There is no cached RootShell in campaign setup, and the previous L0 shell closes at base phase1 `:497`. A race after refresh remains possible; checks still refuse it. Continuing Android cpuset restrictions require the owner to restore Termux foreground and use a new stem; code does not override kernel cpusets or manufacture a passing preflight.

## Session/preflight comparison, line by line before the shared call

Base `main()` (`phase1.py:516-545`):

| Lines | Operation | Session versus --preflight |
|---|---|---|
| 516–523 | Build parser, mutually exclusive flags, parse output and blocks | Only a.preflight boolean differs; no native side effect |
| 524–525 | Parse the same layout list | Identical for the same --blocks |
| 526–527 | Derive block evidence paths; refuse existing JSON/server/block files | Identical; a.preflight does not affect paths |
| 528 | Create output directory | Identical |
| 529–532 | Compute estimates and print plan/warning | Identical |
| 533–537 | Create result, including SESSION STARTING label | Identical; only a.dry_run affects label |
| 538 | Empty resources dictionary | Identical; nothing started |
| 539–542 | Same native lock path, mkdir/open, acquire flock | Identical; only a.dry_run affects lock path |
| 543–544 | Write initial JSON; enter non-dry branch | Identical |
| 545 | preflight(names, output, result, resources) | Identical arguments; no mode passed |
| 546–548 | First mode-dependent branch, AFTER setup returned | Preflight cleans up/returns; session next starts idle |

The shared setup (`phase1.py:468-513` at base) checks native/processes/cores/battery/hashes/version/skin, then starts screen and wake lock. Per layout: compute the same measured set, start one new monitor, verify sensors/root/pumps, enter the transient root scope for dump/battery then restore it, build YOLO and optional M2 workers/callers, append checks, release engines, close monitor. No sampler loop runs here. After all layouts, start one server, health/warmup/cases, start camera, verify frame, stop camera, then process/battery/cores/dump checks. Both modes own identical resources in identical order. Failure occurred before server/camera startup. There is **no mode-specific setup difference in committed code**; a mode-causal explanation would be unsupported.

## Fix and diagnostics

Campaign `runtime.setup_affinity` requests configured device CPUs for the calling thread only, reads back, and repeats the existing core gate immediately before every monitor preparation, including lag and future live blocks. Kernel cpuset filtering is retained; continuing restriction fails before RootShell creation. No retry happens during any measured interval. Successful per-layout and live-block primary/fast-monitor evidence records before/requested/actual masks. The refreshed core-gate refusal also prints measured and allowed CPUs; assigning monitor metadata is protected by cleanup. The fix is proven by mocks for a stale affinity mask; the upstream Android explanation and native success remain probable/unvalidated.

`desk2/speed_block.py` changes are justified because the actual failure and misleading error originate in its initial `prepare_monitor` check. Only diagnostic parameters/messages changed there; shared desk2 affinity behavior remains unchanged. It identifies the caller before shell creation, persistent shell PID and su PID, with actual mask, measured CPUs, allowed set and read method. Campaign verification adds the same details for persistent shell/su/pump and configured masks, including safe-but-unexpected drift. Transient root wrappers print PID and taskset readback before refusing; diagnostic lines are stripped on successful service calls to preserve legacy parsers.

Both modes still call the same setup function, with an explicit `setup_complete` marker at the idle boundary. Expanded self-check compares ordered mocked setup/cleanup traces in both modes for success and eight refusals, stopping the session on its first mocked sleep. Another test reproduces the exact original L1 empty-mask refusal using the real desk2 prepare function, then passes through real campaign prepare after refresh, and verifies continuing cpuset refusal without starting root. An ordinary shell executes the transient wrapper with a fake taskset; no root is used.

Fresh command stems plus `set -C` in RUN.md protect redirect files. A Python nonempty-log check cannot stop the shell's prior truncation, and stdout is normally nonempty by setup time, so no such check was added.

All owner files were read, copied byte-for-byte and SHA-256 compared; archive inventory includes every available .json/.stdout/.stderr and preflight server log. Committed evidence remains unchanged. No model files, motors, native root preflight, real timed block or real session were run. Agents resident and proot cannot validate native timing/root behavior.

## Independent review round 1 and correction

Claude opus-5-5 returned CHANGES REQUESTED solely because three Git-ignored owner preflight server logs were omitted from discovery, the freeze, isolated snapshot and commit list. The archive copies and source SHA matches existed, but were insufficient without explicit candidate membership. Round 2 explicitly includes all three logs (including ignored paths) in manifest, snapshot and proposed commit list; they will require `git add -f` only after human commit authorization. No .gitignore or committed evidence was modified. Small optional findings also clarified the stale-request mechanism, configured-versus-online wording, added measured/allowed to the affinity-refresh refusal, placed metadata assignment inside cleanup, and retained block monitor refresh metadata. No bounded wait was added: continuing CPU restrictions still refuse setup. The reviewer accepted the bounded fix and causal confidence otherwise.

## Diff summary

- `runtime.py`: campaign-only caller affinity refresh immediately before every monitor setup; continuing kernel restrictions refuse. Record before/requested/readback. Detailed configured, persistent-shell, su, pump and transient-shell mask diagnostics. Preserve legacy service output by stripping successful diagnostic markers.
- `phase1.py`: record per-layout setup affinity and explicit setup_complete at the shared setup/idle boundary. Both modes already shared the path; retain it and assert parity rather than invent a mode-specific difference.
- `self_check.py`: original L1 failure-site reproduction, refresh success, continuing cpuset refusal, executed transient wrapper with mock taskset, noclobber preservation, and ordered real main/preflight setup trace parity for both modes and refusal cases.
- `desk2/speed_block.py`: diagnostic-only optional context parameters and messages at the actual failure site and root/su readbacks. Existing mask selection and acceptance behavior unchanged. This exception is justified by the traced root-cause check, with campaign-only remediation outside shared legacy behavior.
- `RUN.md`: fresh preflight and split-session commands, exact filenames, noclobber protection, affinity-refresh behavior and native-validation limits. `RUN_INDEX.md`, 21 unchanged owner files plus SHA inventory, fresh mock/check evidence and frozen manifest added.

## Archive table

# Campaign owner evidence

Archived unchanged from owner Downloads, SHA-256 verified against each source. Includes the three Git-ignored preflight `_llama-server.log` files, explicitly listed in the frozen manifest and commit proposal (force-add only after the human gate). Full file inventory: [archive_sha256.json](runs/archive_sha256.json). No timed benchmark completed.

| Stem | Fixed label | Evidence |
|---|---|---|
| [owner_lag_p1](runs/owner_lag_p1.json) | POWER LAG PROBE — method check, not a benchmark | Complete, 11.5 Hz. current_now rise 50 % 0.25 s, 90 % 0.55 s, fall 50 % 0.29 s; current_avg slow (rise 50 % 3.0 s, 90 % 6.5 s): current_now is the valid power source. |
| [owner_preflight_p1](runs/owner_preflight_p1.json) | PREFLIGHT ONLY — NO TIMING | Passed; all layouts. |
| [owner_preflight_p1b](runs/owner_preflight_p1b.json) | PREFLIGHT ONLY — NO TIMING | Passed; all layouts. |
| [owner_preflight_p1c](runs/owner_preflight_p1c.json) | PREFLIGHT ONLY — NO TIMING | Passed; --blocks L0,L1,L2. |
| [owner_session_p1a](runs/owner_session_p1a.json) | NOT VALID — SESSION INCOMPLETE | Setup refusal "root shell/monitor mask empty or includes inference cores" after L0 check; only L0 recorded. No idle, no block. .stdout/.stderr were overwritten by an immediate rerun that refused with "evidence exists". |
| [owner_session_p1a2](runs/owner_session_p1a2.json) | NOT VALID — SESSION INCOMPLETE | Same refusal reproduced with full traceback; only L0 recorded. No idle, no block. |

Archive inventory: `benchmark/campaign/runs/archive_sha256.json` includes path, source path, size and SHA-256 for every file. All 21 sources and destinations were rehashed before freeze. Existing committed evidence was not changed.

## Checks

| Check | First attempt | Final result |
|---|---|---|
| self_check | exit 0 | exit 0 |
| speed1_self_check | exit 0 | exit 0 |
| desk2_self_check | exit 0 | exit 0 |
| power_map_units | exit 1 | exit 0 |
| coresidency_units | exit 1 | exit 0 |
| detector_units | exit 0 | exit 0 |
| mission_units | exit 0 | exit 0 |
| cycle_units | exit 0 | exit 0 |

Expanded final campaign self-check (`self_check_r3`, after round-1 corrections) including noclobber: exit 0. Compile changed Python without bytecode writes, RUN.md shell syntax without execution, SHA archive verification and complete dry-run invariants: passed. `git diff --check`: passed.

The first power_map attempt exited 1 due to missing frames in its fake child; the first coresidency attempt exited 1 because its 2.1-second cleanup test observed 8 unreadable reads, below its hardcoded minimum 10. These are wall-clock mock paths, unrelated to changed campaign imports. Both passed sequential retries with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. All first failure and retry outputs are preserved; no production/test logic outside the requested scope was altered to obtain passes.

Portable `python3 phase1.py --dry-run`: exit 0, six 6-second blocks L0,L1,L2,L3,L4,L0, all hardware mocked, all mock_only, no unrun blocks, null watts. Label: NOT VALID — DRY RUN (all hardware mocked). Exact copied files: `benchmark/campaign/checks/p1_fix/coder_dry_p1_fix.{json,stdout,stderr}` and six matching block JSONs. This is neither a native timed block nor a real session.

Check commands, environment and scope: `benchmark/campaign/checks/p1_fix/results.json`. Raw check outputs: same directory. No real timed block/session, native root preflight, model inference, camera service or motor command was run. Proot cannot reach native root and agents were resident. Native root/cpuset/service/cleanup success remains owner validation.

## RUN.md owner path

Preflight: `owner_preflight_p1_fix.{json,stdout,stderr}` and `owner_preflight_p1_fix_llama-server.log`, default all layouts.
Session A: `owner_session_p1a_fix.{json,stdout,stderr}`, `owner_session_p1a_fix_llama-server.log`, block_01_L0, block_02_L1, block_03_L2 JSONs.
Session B: `owner_session_p1b_fix.{json,stdout,stderr}`, `owner_session_p1b_fix_llama-server.log`, block_01_L3, block_02_L4, block_03_L0 JSONs.
The exact native Termux commands are in `benchmark/campaign/RUN.md`; each subshell sets noclobber. Never rerun the same stem, including a refused attempt. Restore Termux foreground before trying a new stem. Preflight pass covers that invocation's shared start path, not future CPU availability. Exit agents before owner runs.

## Freeze and review

Frozen manifest: `benchmark/campaign/frozen_manifest_campaign_p1_fix.json`, 64 payload files plus manifest (self-excluded from its own hashes). Source and isolated-copy hashes verified. Snapshot: `/termux-home/review_campaign_p1_fix/`. Request: `/termux-home/storage/downloads/campaign/review_request_campaign_p1_fix.txt` (95420 bytes, below 100 KB).
Review rounds retained: 2; latest verdict: **APPROVED**. Latest JSON: `/termux-home/review_campaign_p1_fix/review_campaign_p1_fix_r2.json`. JSON/stderr also copied to Downloads/campaign; result-only text saved as `/termux-home/storage/downloads/campaign/review_campaign_p1_fix_result.txt`. Candidate remained frozen while reviewing. No silent fallback or additional reviewer was launched.

## Reviewer results verbatim

### Round 1

**Review: CAMPAIGN_P1_FIX** (reviewer: Claude Code CLI, claude-opus-5-5, Ponytail off). I only read files. I ran no commands, edited nothing and started no other agents.

## Causal confidence

- **Proven:** the immediate failure. The traceback in `runs/owner_session_p1a2.stderr:10-19` stops at base `speed_block.py:203`. There `previous - cpus` came out empty during the L1 prepare, before any root shell existed. That rules out root/su readback, cached shells and wrappers as the cause. I agree with this.
- **Probable only:** the upstream cause, a cpuset or stale-mask restriction from Android. The owner evidence can't prove it. There is extra in-run evidence that the coder didn't use (optional item 1).
- **No mode difference in code:** confirmed. `phase1.py:517-547` does the same thing for session and preflight until after `preflight()` returns. The only branch is at `:549`.
- **The bounded fix is acceptable without live validation.** The core gate still refuses a cpuset restriction that persists. It does not claim to override the kernel. Nothing is retried inside a measured interval: both `rt.prepare` calls at `phase1.py:337,354` and `lag_probe.py:42-43` run before `t0`. The other main-thread work is unaffected, since PinnedDetector and the workers pin themselves. Whether a real session now succeeds still depends on Android state and has not been validated.

## Blocking

**B1. The freeze and archive leave out three owner server logs.**
- `runs/archive_sha256.json:38-43,62-67,86-91` lists `runs/owner_preflight_p1{,b,c}_llama-server.log` as archived and SHA-verified.
- `final_verification.stdout` says "all 21 owner files rehashed". 21 = 18 + 3 logs.
- None of the three logs appear in `frozen_manifest_campaign_p1_fix.json`, in the inventory I was given, or in the snapshot (no `benchmark/campaign/**/*.log` exists).
- The task requires all available owner files archived, verified and frozen. As it stands, either the frozen candidate leaves out required evidence, or the archive inventory points at files that won't be committed (for example if `*.log` is ignored).
- **Fix:**
  - Add the three logs to the manifest with their sizes and hashes.
  - Confirm they are present in the candidate tree and will be part of the commit list, even if `.gitignore` excludes `*.log`.
  - Optionally mention them in the RUN_INDEX preflight rows.

## Optional, non-blocking

1. **Root-cause note (`checks/p1_fix/root_cause.md`).** It lists the `finally` in `prepare_monitor` only as clearing it of a pin leak. That `finally` (`speed_block.py:215-216`) actually rewrites `previous` as an explicit thread mask, so a narrowed mask captured during a temporary restriction stays in place. That is a concrete mechanism for candidate 2. Two more facts from the evidence:
   - `check_cores('preflight')` at `phase1.py:472` passed in both sessions, so CPUs 6–7 were lost during setup.
   - p1a still had 6–7 at the L0 prepare (`owner_session_p1a.json:36-42`). p1a2 had already lost them (`owner_session_p1a2.json:36-41`).

   This sharpens the "probable" ranking but doesn't change the fix.
2. **`runtime.py:75-77`.** When `setup_affinity` refuses, the error shows before/requested/actual, process and read method, but not the measured CPUs. This is now the first refusal path for the observed failure. Pass `measured` in for completeness.
3. **`runtime.py:85`.** `campaign_setup_affinity` is assigned outside the `try` at `:86`. If that assignment ever raised, the shell would leak. It's only theoretical; moving the line inside the `try` would close it.
4. **Wording.** `os.cpu_count()` (`runtime.py:69`) counts configured CPUs, not online ones as RUN.md says. It's harmless because the kernel intersects the mask with the cpuset.
5. **Live-block evidence.** Block records don't store `setup_affinity` the way `layout_checks` do (`phase1.py:493-494`). Storing it would make future refusals or drift diagnosable.
6. **A possible safe alternative:** a bounded `cr.wait_cores` (`coresidency.py:301`) during setup only, before monitor preparation. This could ride out a short top-app loss instead of refusing. It is not required.

## Verified as correct

- **Shared setup:** `phase1.py:546-551`. The preflight and session traces are compared over success and eight refusals, with only the excluded `sleep` call at index 18 differing (`self_check.py:171-209`).
- **Reproduction:** the old L1 empty-mask refusal is reproduced through the real `sb.prepare_monitor`. It then passes after the refresh, and a continuing restriction is refused without starting root (`self_check.py:555-583`).
- **Diagnostics:** desk2 changes are diagnostic only. The existing positional callers still work, and `threading` is imported (`speed_block.py:14`). Campaign checks report actual/measured/allowed/process/read method for the persistent shell, su and the pump (`runtime.py:46-60`).
- **Transient wrapper:** it prints the readback line before refusing, strips it on success, and was exercised in a plain shell with a fake taskset.
- **Same-stem reruns:** `set -C` in a subshell plus fresh `_p1_fix` stems is a sound choice, and the reason given for not adding a Python log-size check is correct. The noclobber self-check covers both redirect files.
- **RUN.md:** it gives fresh preflight, session part A, session part B and dry-run commands, and lists the exact files for each.
- **RUN_INDEX:**
  - It uses the three required labels.
  - The lag values match `owner_lag_p1.json:6235-6254`: 11.48 Hz, current_now 0.252/0.553/0.288 s, current_avg 3.00/6.53 s.
  - The p1a overwrite and the p1a2 full traceback are documented.
- **Check runs:** the power_map and coresidency failures on the first attempt are preserved, and both passed on the sequential retry. Neither imports the changed production logic.

## Final verdict: CHANGES REQUESTED

The only reason is B1: the freeze doesn't cover the archived server logs. The code fix, diagnostics, self-checks and RUN.md are otherwise acceptable. No commit or push is authorized.

### Round 2

I'm approving round 2. I only read files: I ran no commands, edited nothing, wrote no plan file and started no other agents.

## Round-1 blocking item B1: resolved

- **The three server logs are now frozen.** All three are in `frozen_manifest_campaign_p1_fix.json`:
  - `owner_preflight_p1_llama-server.log` (`:238-240`)
  - `owner_preflight_p1b_llama-server.log` (`:258-260`)
  - `owner_preflight_p1c_llama-server.log` (`:278-280`)

  Each entry has size 2270 and the same SHA-256 as the inventory and `runs/archive_sha256.json:38-43,62-67,86-91`.
- **They are in the snapshot.** The `runs/` listing shows all three, and the 21 files there match the "all 21 owner files rehashed" line in `final_verification.stdout`.
- **The Git-ignore issue is documented.** `RUN_INDEX.md:3` says the logs are ignored by Git and will need force-adding only after the human commit gate. `root_cause.md` (round-1 section) says the same, and `.gitignore` was not changed.
- **The commit list must include them.** It should contain every manifest path, with these three logs added by `git add -f`.

## Round-1 optional items

| # | Item | Status |
|---|---|---|
| 1 | Stale-mask mechanism in `prepare_monitor` | Added to `root_cause.md` as the code path for candidate 2. Also records that `check_cores('preflight')` passed and that p1a still had CPUs 6–7 at L0 while p1a2 had lost them. |
| 2 | Measured CPUs missing from the refusal | Fixed: `runtime.py:75-78` now prints before, requested, actual, measured, allowed, process and read method. |
| 3 | Metadata assignment outside `try` | Fixed: `runtime.py:86-87`, so `close()` covers it. |
| 4 | `cpu_count` wording | Clarified ("configured device CPUs", RUN.md). |
| 5 | Live blocks didn't record setup affinity | Fixed: `phase1.py:339,356` now records it for both the primary and fast monitors. |
| 6 | Bounded wait | Deliberately not added. That's acceptable: it was optional, and a restriction that persists still refuses safely. |

## Regression check on the round-2 changes

- **Lag probe:** `lag_probe.py:42-43` goes through `rt.prepare`, so it runs the refresh too. Both calls happen before any measured interval.
- **No retry inside a measured window:** in `live_block`, both `rt.prepare` calls (`phase1.py:337,355`) come before `t0` (`:368`).
- **No new exposure from the refresh:**
  - The refresh only widens the main thread's mask.
  - The M2 workers and caller built at `:351` pin themselves and are not affected.
  - The main thread's mask already overlapped the measured cores in the base code (`previous` minus `cpus` was non-empty).
- **Self-check:** the round-3 run passes, including the reproduction of the old failure and the shared-path trace comparison (`self_check.py:171-209,555-607`).

## Causal confidence

- **Proven:** the immediate failure. Base `speed_block.py:203` computed an empty safe mask for the calling thread on the L1 prepare, before any root shell existed. That rules out root/su readback, cached shells and wrappers.
- **No session-versus-preflight difference in the code:** proven. Both run the same `preflight()` call with no mode argument; the only branch comes after it returns (`phase1.py:548-553`).
- **Probable only:** the upstream cause. Either Android moved Termux out of top-app (losing CPUs 6–7), or a temporary restriction left a narrowed mask in place through `prepare_monitor`'s `finally`. The owner evidence can't tell these apart.
- **The bounded fix is acceptable without live validation.** It resets the calling thread's mask to all configured CPUs, reads it back and re-runs the core check. A restriction that persists still refuses before any root shell starts, and nothing is retried inside a measured interval. Whether a real session now succeeds still depends on Android state and has not been validated; nothing in the evidence claims it has.

## Findings

- **Blocking:** none.
- **Optional:** none new.

No commit or push is authorized. The coder should propose the exact file list under the title "benchmark: campaign phase-1 session start fix, lag probe and preflight archive", including the three force-added logs, and wait for "yes commit". Pushing needs a separate "yes push".

## Final verdict: APPROVED

## Proposed commit

Independent review APPROVED; human commit permission is still required. Proposed single commit:

`benchmark: campaign phase-1 session start fix, lag probe and preflight archive`

Exact file list:

- `benchmark/campaign/RUN.md`
- `benchmark/campaign/RUN_INDEX.md`
- `benchmark/campaign/checks/p1_fix/coder_dry_p1_fix.json`
- `benchmark/campaign/checks/p1_fix/coder_dry_p1_fix.stderr`
- `benchmark/campaign/checks/p1_fix/coder_dry_p1_fix.stdout`
- `benchmark/campaign/checks/p1_fix/coder_dry_p1_fix_block_01_L0.json`
- `benchmark/campaign/checks/p1_fix/coder_dry_p1_fix_block_02_L1.json`
- `benchmark/campaign/checks/p1_fix/coder_dry_p1_fix_block_03_L2.json`
- `benchmark/campaign/checks/p1_fix/coder_dry_p1_fix_block_04_L3.json`
- `benchmark/campaign/checks/p1_fix/coder_dry_p1_fix_block_05_L4.json`
- `benchmark/campaign/checks/p1_fix/coder_dry_p1_fix_block_06_L0.json`
- `benchmark/campaign/checks/p1_fix/coresidency_units.stderr`
- `benchmark/campaign/checks/p1_fix/coresidency_units.stdout`
- `benchmark/campaign/checks/p1_fix/coresidency_units_r2.stderr`
- `benchmark/campaign/checks/p1_fix/coresidency_units_r2.stdout`
- `benchmark/campaign/checks/p1_fix/cycle_units.stderr`
- `benchmark/campaign/checks/p1_fix/cycle_units.stdout`
- `benchmark/campaign/checks/p1_fix/desk2_self_check.stderr`
- `benchmark/campaign/checks/p1_fix/desk2_self_check.stdout`
- `benchmark/campaign/checks/p1_fix/detector_units.stderr`
- `benchmark/campaign/checks/p1_fix/detector_units.stdout`
- `benchmark/campaign/checks/p1_fix/final_verification.stdout`
- `benchmark/campaign/checks/p1_fix/mission_units.stderr`
- `benchmark/campaign/checks/p1_fix/mission_units.stdout`
- `benchmark/campaign/checks/p1_fix/power_map_units.stderr`
- `benchmark/campaign/checks/p1_fix/power_map_units.stdout`
- `benchmark/campaign/checks/p1_fix/power_map_units_r2.stderr`
- `benchmark/campaign/checks/p1_fix/power_map_units_r2.stdout`
- `benchmark/campaign/checks/p1_fix/results.json`
- `benchmark/campaign/checks/p1_fix/root_cause.md`
- `benchmark/campaign/checks/p1_fix/self_check.stderr`
- `benchmark/campaign/checks/p1_fix/self_check.stdout`
- `benchmark/campaign/checks/p1_fix/self_check_r2.stderr`
- `benchmark/campaign/checks/p1_fix/self_check_r2.stdout`
- `benchmark/campaign/checks/p1_fix/self_check_r3.stderr`
- `benchmark/campaign/checks/p1_fix/self_check_r3.stdout`
- `benchmark/campaign/checks/p1_fix/speed1_self_check.stderr`
- `benchmark/campaign/checks/p1_fix/speed1_self_check.stdout`
- `benchmark/campaign/frozen_manifest_campaign_p1_fix.json`
- `benchmark/campaign/phase1.py`
- `benchmark/campaign/runs/archive_sha256.json`
- `benchmark/campaign/runs/owner_lag_p1.json`
- `benchmark/campaign/runs/owner_lag_p1.stderr`
- `benchmark/campaign/runs/owner_lag_p1.stdout`
- `benchmark/campaign/runs/owner_preflight_p1.json`
- `benchmark/campaign/runs/owner_preflight_p1.stderr`
- `benchmark/campaign/runs/owner_preflight_p1.stdout`
- `benchmark/campaign/runs/owner_preflight_p1_llama-server.log`
- `benchmark/campaign/runs/owner_preflight_p1b.json`
- `benchmark/campaign/runs/owner_preflight_p1b.stderr`
- `benchmark/campaign/runs/owner_preflight_p1b.stdout`
- `benchmark/campaign/runs/owner_preflight_p1b_llama-server.log`
- `benchmark/campaign/runs/owner_preflight_p1c.json`
- `benchmark/campaign/runs/owner_preflight_p1c.stderr`
- `benchmark/campaign/runs/owner_preflight_p1c.stdout`
- `benchmark/campaign/runs/owner_preflight_p1c_llama-server.log`
- `benchmark/campaign/runs/owner_session_p1a.json`
- `benchmark/campaign/runs/owner_session_p1a.stderr`
- `benchmark/campaign/runs/owner_session_p1a.stdout`
- `benchmark/campaign/runs/owner_session_p1a2.json`
- `benchmark/campaign/runs/owner_session_p1a2.stderr`
- `benchmark/campaign/runs/owner_session_p1a2.stdout`
- `benchmark/campaign/runtime.py`
- `benchmark/campaign/self_check.py`
- `benchmark/relate_anything/desk2/speed_block.py`

The three owner `_llama-server.log` files are ignored by Git and must be force-added explicitly only after the human commit gate; their hashes are in the freeze and archive inventory. No other files proposed. STOP at the human gate: commit only after "yes commit"; push only after a separate "yes push".

## Completion working-tree status

```text
 M benchmark/campaign/RUN.md
 M benchmark/campaign/phase1.py
 M benchmark/campaign/runtime.py
 M benchmark/campaign/self_check.py
 M benchmark/relate_anything/desk2/speed_block.py
?? benchmark/campaign/RUN_INDEX.md
?? benchmark/campaign/checks/p1_fix/
?? benchmark/campaign/frozen_manifest_campaign_p1_fix.json
?? benchmark/campaign/runs/
```

Doc Keeper handoff: campaign evidence is archived under `benchmark/campaign/runs/` and labelled in `benchmark/campaign/RUN_INDEX.md`. No deployed robot behavior or canonical status/decision file changed.

## Authorized commit and push completion

The earlier sections record the reviewed pre-commit candidate. The owner subsequently instructed "commit and push", explicitly authorizing both actions.

Committed `10be01e296815e2173855afebd49cb0e52ee66fd` with the exact approved 65-file list (including three force-added owner server logs). Parent/base unchanged: `c47fc41999d7190b4e8171b8e7806c6b77b5e7dd`. Commit title: `benchmark: campaign phase-1 session start fix, lag probe and preflight archive`.

The initial staged whitespace check flagged an existing trailing blank line in the frozen raw `power_map_units.stderr`; that approved evidence was preserved byte-for-byte. The staged source/Markdown/JSON whitespace check passed. No source or evidence bytes were changed after approval.

Author: root <root@localhost.localdomain>
Committer: root <root@localhost.localdomain>

Push `git push origin main` exited 0. Exact output:

```text
To https://github.com/abdelhamidzoghlamii-ui/Pixel-robot.git
   c47fc41..10be01e  main -> main
```

Independent `git ls-remote origin refs/heads/main` returned `10be01e296815e2173855afebd49cb0e52ee66fd`, matching local HEAD. Final `git status --short` is empty. All frozen payload SHA-256 hashes and the exact committed file list were reverified. No native validation or benchmark was run.
