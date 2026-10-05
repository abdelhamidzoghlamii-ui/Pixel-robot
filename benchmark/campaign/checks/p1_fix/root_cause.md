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
