# RELATE_FIX1 owner commands

The archived owner runs are listed in [RUN_INDEX.md](../RUN_INDEX.md). They used
frozen R1. The updated runner samples `current_now`/`voltage_now` each second and
reports block/inside-call/outside-call mean watts and sample counts. Status/memory
and skin remain at five seconds. Empty/null power fields in dry runs are expected.
These commands use fresh FIX1 filenames and preserve R1 evidence.

Native Termux only, from any directory. Motors off. Both models passed the fixed
horse + blocked_1 real-pair parity criterion. Review/owner decision precedes full
runs. Do not launch a timed block while Codex, Claude, AGY or node is resident.
Unplug the charger, keep Termux in front. Each full invocation keeps the screen
awake, restores its validated timeout, idles 300 s, then applies the skin gate.
No valid timing was taken by the Coder. Existing evidence is never overwritten;
choose a new --output filename if a command reports that its output exists.

1. Dry-run M1 (two calls; no sensors or idle, NOT VALID):

```sh
python -u ~/robot/benchmark/relate_anything/desk2/speed_block.py --model relsgg-vits16plus --dry-run --output ~/storage/downloads/relate_fix1/owner_dry_m1_fix1.json
```

Success prints `INFORMAL — agents resident, NOT VALID TIMING — dry-run`, JSON
with that label ending in `DRY RUN`, median/P95 ORT ms, VmHWM, and `Evidence:`.
Send the complete stdout, stderr and owner_dry_m1.json. This proves inference and
MID caller/worker pinning only. It supplies no valid speed/power measurement.

2. Dry-run M2 (eligible because its new two-photo parity passed):

```sh
python -u ~/robot/benchmark/relate_anything/desk2/speed_block.py --model relsgg-vits16 --dry-run --output ~/storage/downloads/relate_fix1/owner_dry_m2_fix1.json
```

Success prints the same NOT VALID label and summary for M2. Send complete stdout,
stderr and owner_dry_m2.json. Missing/failed parity_valid_relsgg-vits16.json refuses
before loading, locking or inference; old horse_parity.json never authorizes M2.

3. Timed block M1, after exiting all agents:

```sh
python -u ~/robot/benchmark/relate_anything/desk2/speed_block.py --model relsgg-vits16plus --output ~/storage/downloads/relate_fix1/owner_speed_m1_fix1.json
```

Success prints the saved/set screen timeout, `Idling 300 s`, the restored timeout,
then JSON labelled `TIMING COMPLETE; warm starts flagged; no cold-cache load claim`
and `Evidence:`. Send complete stdout, stderr and owner_speed_m1.json, including
raw sensor/affinity/call rows, warm-start gate and CPU-cap summary. A refusal or
`NOT VALID — INCOMPLETE` is a failed block; report it and restore the stated
conditions before a fresh invocation with a new output filename.

4. Timed block M2, in a separate process after M1:

```sh
python -u ~/robot/benchmark/relate_anything/desk2/speed_block.py --model relsgg-vits16 --output ~/storage/downloads/relate_fix1/owner_speed_m2_fix1.json
```

Success prints the same screen/idle/restoration messages and TIMING COMPLETE
summary for M2. Send complete stdout, stderr and owner_speed_m2.json. Each block
plans 36 calls at 0,5,...,175 s across 180 s on MID cores 4–5, intra-op threads=2.
Do not compare speed until both block JSONs have been checked for validity.
