# RELATE_SPEED1 owner commands

M2 fp32 is the benchmark reference, not deployed. Native Termux Python only,
motors off, charger unplugged for the timed session, Termux in front and screen
on. Exit Codex, Claude, AGY, node, robot runners and model servers before step 2.
The Downloads directory below is created during Coder preparation. Existing
evidence is refused; use a fresh output stem and matching log filenames on reruns.

No changed variant qualified: int8 build/load failed, XNNPACK is available but
failed inference/parity, img384/img336 failed the unchanged quality criterion.
One reference photo contains only two triplets, so the literal >=9 overlap rule
also fails there. The script reads the frozen quality evidence and refuses failed
or changed artifacts. The plan is:

1. REF fp32 MID — CPUs 4–5, two threads.
2. fp32 BIG — CPUs 6–7, two threads.
3. fp32 LITTLE — CPUs 0–3, four threads, matching power_map R3_little.
4. REF fp32 MID — CPUs 4–5, two threads, for session drift.

Step 1 — session dry-run (two calls per block, no idle/gates/sensors, NOT VALID):

```sh
python -u ~/robot/benchmark/relate_anything/speed1/session.py --session --dry-run --output ~/storage/downloads/relate_speed1/owner_dry_speed1.json > ~/storage/downloads/relate_speed1/owner_dry_speed1.stdout 2> ~/storage/downloads/relate_speed1/owner_dry_speed1.stderr
```

Expected about 40 seconds including loads/cadence (informal estimate only).
On success, stdout contains the full planned block list and owner-session time
estimate, each block summary with `NOT VALID — DRY RUN`, median/P95, verified
caller/worker CPU masks, graph/bank/sidecar hashes, `ORT_ENABLE_ALL`, and null
block/inside-call/outside-call mean watts with zero sample counts. It ends with
`INFORMAL — agents resident, NOT VALID TIMING — SESSION DRY RUN` and `Evidence:`.
No valid speed or power evidence is produced. Send these exact files:

- `owner_dry_speed1.stdout`, `owner_dry_speed1.stderr`, `owner_dry_speed1.json`
- `owner_dry_speed1_block_01_fp32_MID.json`
- `owner_dry_speed1_block_02_fp32_BIG.json`
- `owner_dry_speed1_block_03_fp32_LITTLE.json`
- `owner_dry_speed1_block_04_fp32_MID.json`

Step 2 — timed session, owner only, after exiting all agents:

```sh
python -u ~/robot/benchmark/relate_anything/speed1/session.py --session --output ~/storage/downloads/relate_speed1/owner_session_speed1.json > ~/storage/downloads/relate_speed1/owner_session_speed1.stdout 2> ~/storage/downloads/relate_speed1/owner_session_speed1.stderr
```

Expected 17 minutes plus setup/load time with immediately satisfied gates; up to
49 minutes plus setup/load time with four maximum eight-minute gates. All
preflight refusals (including root/su masks, policies, saved input, output access,
engines and worker pinning) precede the one 300-second idle. Each block loads a
fresh session, uses the original skin gate relative to the initial idle baseline,
and schedules 36 calls at 0,5,...,175 seconds in 180 seconds. This is a warm-cache
comparison, not a cold-load benchmark. References bracket the session; inspect
their drift before comparing clusters. VmHWM is cumulative for the process.

Success stdout prints the plan and estimate first, screen timeout saved/set,
`Idling 300 s once`, four block records, restored screen timeout, then
`SESSION COMPLETE; inspect every block validity and warm-start flags` and
`Evidence:`. Every full block has a `validity` field: `VALID`,
`NOT VALID — WARM START` (gate timed out), or `NOT VALID — INCOMPLETE`.
Per-block power records use one-second current_now/voltage_now receipt timestamps;
summary means/counts are sample-weighted, with read-boundary counts. Five-second
status/memory and skin monitoring remain. The fast sampler, root/su processes and
pump avoid the measured cluster and retain a nonempty allowed mask.

Only cadence failures with successful cleanup and no monitor errors can continue
to a later block. Warm-start blocks remain recorded and the next block has its
own gate. Root/sensors, thermal limits, charger, processes, cores/affinity,
inference and cleanup failures stop the session; refused blocks and the unrun plan
are recorded. A session-complete label is not blanket validity for its blocks.
Send all of these exact files, including failures:

- `owner_session_speed1.stdout`, `owner_session_speed1.stderr`, `owner_session_speed1.json`
- `owner_session_speed1_block_01_fp32_MID.json`
- `owner_session_speed1_block_02_fp32_BIG.json`
- `owner_session_speed1_block_03_fp32_LITTLE.json`
- `owner_session_speed1_block_04_fp32_MID.json`

The same runner also accepts `--variant {fp32,int8,xnnpack,img384,img336}` and
`--cluster {MID,BIG,LITTLE}` for one-block owner runs; failed variants refuse even
in dry-run. Omit `--variant` or use `--session` for the fixed ordered session.
No timed block/session was run by the Coder. Live root affinity, power sampling,
skin gates and timed session transitions remain owner validation limits.
