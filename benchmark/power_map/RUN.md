# YOLO power map — human runs from native Termux

Motors off. No USB/serial or motor imports. Do not run inside Debian/proot.
Before **each** run: quit every agent and exit Debian; stop robot-chat/chat.py
and any llama-server. In native Termux run:

```bash
pkill -f security_reminder_hook
```

Unplug the charger (launcher requires Discharging). Keep the screen on and
Termux in front throughout. Leave the phone in its robot mount. Agent residency
invalidates timings; run these commands only after exiting the agents.

Smoke R1 (three 20 s blocks, no gate waiting, about 6 min + load/checks):

```bash
bash ~/robot/benchmark/power_map/oneshot.sh --set R1 --smoke
```

Full R1 and R2 (each three 180 s blocks; 14–46 min + load/checks):

```bash
bash ~/robot/benchmark/power_map/oneshot.sh --set R1
bash ~/robot/benchmark/power_map/oneshot.sh --set R2
```

These are separate runs. R1 uses rate 2, R2 rate 1; cadence 5 s, 10 s, off.
Every successful new frame uses exactly one detector size: 320 normally,
640 instead when the drive-mode SizePolicy interval elapses. No person search.

R3 (three 180 s blocks; 14–46 min + load/checks). Replace placeholders with
rate 1 or 2 and cadence 5, 10 or off:

```bash
bash ~/robot/benchmark/power_map/oneshot.sh --set R3 --rate <1-or-2> --cadence <5-or-10-or-off>
```

CONFIRM (one 1200 s block; 25–41 min + load/checks). Replace all placeholders:

```bash
bash ~/robot/benchmark/power_map/oneshot.sh --set CONFIRM --rate <1-or-2> --cadence <5-or-10-or-off> --threads <default-or-mid-or-little>
```

Add `--smoke` to R3 for three 20 s blocks (6 min + load/checks), or to CONFIRM
for one 60 s block (6 min + load/checks). All estimates include the 5 min idle,
up to 8 min for each block gate and the initial load/warm-up gate. Gemma stays
resident throughout an invocation; one untimed selector warm-up; no cache drop,
cold-load experiment or MTP snapshot.

Resume with the **same set, rate, cadence, threads and smoke flags**. Examples:

```bash
bash ~/robot/benchmark/power_map/oneshot.sh --set R1 --resume ~/power_map/run_<UTC>_R1
bash ~/robot/benchmark/power_map/oneshot.sh --set R1 --smoke --resume ~/power_map/run_<UTC>_R1_smoke
bash ~/robot/benchmark/power_map/oneshot.sh --set R3 --rate <1-or-2> --cadence <5-or-10-or-off> --resume ~/power_map/run_<UTC>_R3
```

Completed block JSONs are kept; an interrupted block is redone. Resume loads a
new resident Gemma and warms it once. Original idle skin baseline is retained.
Outputs: `~/power_map/run_<UTC>_<set>/` (or `_smoke`), including `report.txt`,
`run.json`, block JSONs and server logs. Report is also copied to
`~/storage/downloads/power_map_<run>_report.txt`. Console/thermal logs are in
`~/power_map/`. zone9/10/11 in the logs/JSON are **CPU-core reading, not a heat state**.

The skin-only gate requires VIRTUAL-SKIN <= original idle + 1.5 C; if unmet
in 8 min the block starts marked warm. Block limits and INCOMPLETE rules are
inherited from coresidency: CRITICAL status, battery >=45 C, CPU fault >=110 C
in three samples; stale sensors fail closed. Failed camera reads, missing
required size/drift, failed samples/end checks, missing selector slots,
server death, non-Discharging battery, unreadable cpufreq or failed LMK query
produce INCOMPLETE and nonzero exit. Cadence off requires only 320. ORT
verification failures and absent skin slope also produce INCOMPLETE. Limit
stops remain results (unless no first frame or fail-closed). To redo a kept
block, move its `block_<name>.json` out of the run folder, then resume.

ORT default uses `detect_person.Detector()` unchanged. Mid/little use explicit
intra/inter settings and pin only session workers plus a dedicated detector
calling thread (ORT includes the caller in intra-op work). Main runner affinity
is unchanged. Per-second `/proc` samples record all runner threads, ORT worker
and caller CPUs/allowed lists. Stat processor is the last scheduled CPU, not an
exhaustive execution trace. CPU lists count samples only after utime+stime ticks
advance relative to the previous sample; the first sample is a baseline. Idle
workers retain a stale last CPU and show no sampled CPU tick advance. Sub-tick
work and the initial sampling interval may be missed. Real pinning and rate-1 pacing remain untested until
the phone smoke run.

## If something is left running

```bash
am broadcast -n com.pixelrobot.robotcam/.ControlReceiver -a com.pixelrobot.robotcam.STOP
su -c "ps -A | grep -E 'python|llama'"     # then: su -c "kill <PID>"  (never pkill -f, COMMANDS §8)
su -c "settings get system screen_off_timeout </dev/null >/data/local/tmp/st.txt 2>&1"; su -c "cat /data/local/tmp/st.txt"
rm -r ~/power_map/.lock    # only if no launcher is running (a killed launcher leaves it; the next start refuses)
```

Offline checks (no root, camera or real server):

```bash
python ~/robot/benchmark/power_map/test_power_map.py
python ~/robot/benchmark/coresidency/test_coresidency.py
bash ~/robot/benchmark/coresidency/test_oneshot.sh
```
