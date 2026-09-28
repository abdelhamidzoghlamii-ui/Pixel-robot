ROLE: Reviewer (independent, fresh session). PONYTAIL: off — do a correctness/safety review, not an over-engineering review.
PLATFORM: Codex CLI codex exec, read-only sandbox. MODEL: gpt-6-sol. EFFORT: medium.
Rules: do NOT edit, stage, commit or push anything; do not run motors, main.py, llama-server or any model; do not launch another reviewer. Read files only (you may run read-only commands such as cat, grep, git diff, python3 -m py_compile).

Repository: /termux-home/robot (Pixel Robot). Base commit: main 825152b75fff (working tree otherwise clean). Read AGENTS.md, docs/WORKFLOW.md, docs/REVIEWER.md for your role.

TASK given to the Coder (verbatim summary):
Prepare a SHORT unattended speed run testing MTP speculative decoding (--spec-type draft-mtp) and Qwen3.5-4B Q4_0 on the robot build ~/llama.cpp-b1609-dotprod (upstream tag b10194).
1. Download (SHA-256 checked vs HF LFS): unsloth/Qwen3.5-4B-MTP-GGUF Qwen3.5-4B-Q4_0.gguf -> ~/models/qwen35/mtp/Qwen3.5-4B-Q4_0-MTP.gguf; unsloth/gemma-4-E2B-it-GGUF MTP/mtp-gemma-4-E2B-it-Q8_0.gguf -> ~/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf.
2. Smoke tests: server_manager.py flags + --cache-ram 0, (i) Qwen MTP --spec-type draft-mtp --spec-draft-n-max 3, (ii) Gemma E2B Q4_0 --model-draft <drafter> --spec-type draft-mtp --spec-draft-n-max 3; show MTP/draft loaded and draft acceptance. If (ii) fails on the robot build, fall back to ~/llama.cpp-upstream for both Gemma rows.
3. Write conv_speed_mtp.py next to conversation/conv_speed.py (do NOT edit archived conv_speed.py or bench.py). Reuse their prompt wrappers, cold handshake, thermal gate, cpuset guard, cores 4-7, ctx 2048, --cache-ram 0. Changes: temperature 0 for every prompt; prompts restricted to A5 A6 B1 B2 B6 B7 C1 C6 C7 C8 C9 C10 C11; cold block only; Qwen think OFF. Configs in order: 1 gemma_e2b_q40 (MTP off), 2 gemma_e2b_q40_mtp (drafter, n-max 3), 3 qwen35_4b_q4km (MTP off, existing file), 4 qwen35_4b_q40mtp (MTP off, new file), 5 qwen35_4b_q40mtp_on (MTP on, n-max 3). Report per config: ttft, prompt tok/s, gen tok/s (token-weighted, >=10 tokens), peak MiB, draft acceptance, thermal start/end, and for rows 2 and 5 how many of 13 replies are byte-identical to rows 1 and 4. Score C replies with conversation/score_c.py (report only). Write run_conv_mtp.sh for ~/ladder/oneshot.sh.
4. Toy run on 2 prompts per config (no timing claims; agent resident).
RULES: no motors; do not edit main.py, server_manager.py or robot code.

CANDIDATE (new, untracked files; full diff below):
- benchmark/llm_objective_setting/conversation/conv_speed_mtp.py
- benchmark/llm_objective_setting/conversation/run_conv_mtp.sh (a byte-identical copy, SHA-256 4db32d39…d5817, is at /termux-home/ladder/run_conv_mtp.sh because oneshot.sh requires $L/$RUN_SCRIPT)

SURROUNDING CONTEXT to read yourself:
- benchmark/llm_objective_setting/conversation/conv_speed.py (archived runner imported as cs; identical to /termux-home/ladder/conv_speed.py, SHA 9b721d86…)
- benchmark/llm_objective_setting/bench.py (prompts, format_prompt, parse_and_grade_c, strip_think)
- benchmark/llm_objective_setting/conversation/score_c.py and benchmark/llm_objective_setting/aggregate.py
- /termux-home/ladder/ladder.py (live module imported at runtime; make_cold, thermal_gate, check_cores, wait_cores)
- /termux-home/ladder/oneshot.sh (SHA-256 db1f3156…8dbdc8e8, the launcher) and run_conversation.sh (previous run script)
- server_manager.py (robot flags; it now itself passes --cache-ram 0; conv_speed.server_cmd builds its own argv from sm constants, adding --cache-ram 0 once)

CODER DECISIONS to check:
- Smoke (ii) succeeded on the robot build (no assert), so the upstream fallback was not used; all configs use server_manager.LLAMA_SERVER. The only warning was "Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)" / "[spec] failed to measure draft model memory", then the drafter loaded and acceptance was reported.
- Draft acceptance = sum(draft_n_accepted)/sum(draft_n) from the final streamed event's timings (fields verified present in smoke/toy output; absent -> None for MTP-off configs).
- Byte-identical compares the full streamed reply text per prompt_id.
- c_results.json writes C rows in bench.py result format (run_index 1, graded by bench.parse_and_grade_c, source_sha256 = bench.SOURCE_SHA256) so score_c.py can read it.
- A5 uses conv_speed's behavior (final user turn only, no history), unchanged from the archived run.
- --ids option restricts to a subset of the 13 for the toy run only.

ROUND 2. This is a fresh session; a previous independent round returned this verdict verbatim:
---
**Verdict: REQUEST CHANGES**

1. **MAJOR — Paired replies receive different prompts.** [conv_speed_mtp.py](/termux-home/robot/benchmark/llm_objective_setting/conversation/conv_speed_mtp.py:62) regenerates prompts for each config, while `bench.format_prompt()` inserts a random nonce. At temperature 0, an MTP-on reply can differ from its MTP-off twin because the prompt bytes differ. The reported byte-identical count therefore cannot isolate MTP’s effect. Reuse each prompt’s exact bytes across its pair.

2. **MAJOR — Resume can silently accept an incomplete prompt set.** [conv_speed_mtp.py](/termux-home/robot/benchmark/llm_objective_setting/conversation/conv_speed_mtp.py:166) treats any completed block in the output directory as done without checking its prompt IDs against the current `--ids`. Resuming a two-prompt toy directory with the default 13 IDs skips those blocks and produces a report from two replies per config.

3. **MAJOR — An interrupted write can double-count a block on resume.** [conv_speed_mtp.py](/termux-home/robot/benchmark/llm_objective_setting/conversation/conv_speed_mtp.py:194) flushes turns before writing the completion record. If the unattended run stops between those writes, resume reruns the block, and `cs.load_done()` combines both sets of turns under one completion record. Rates, acceptance, scoring, and the identical-reply denominator can then be wrong.

The supplied smoke logs show both drafters loading and reporting acceptance; the two-prompt toy run completed. Neither check covers these three cases.
---
Coder's changes in response (all three accepted): (1) prompts() seeds `random` (random.seed(0)) before cs.prompts(), so each family's nonces are deterministic and a pair gets byte-identical prompts; nonces still differ between prompts, and each block is a fresh server with --cache-ram 0 and cache_prompt false. (2) On --resume, a completed block whose prompt ids differ from --ids is refused (SystemExit). (3) On --resume, turns.jsonl is renamed to turns.jsonl.before_resume_<stamp> and rewritten with only the turns of blocks that have a blocks.jsonl record, so an interrupted block's orphan turns are not merged with its redo. Verify these fixes and review the whole candidate again, not only the delta.

CHECK OUTPUT:
Downloads: Qwen3.5-4B-Q4_0-MTP.gguf 2669209920 bytes sha256 14e6ef39302330c63c2c1a1ab548c7f6f1b7e36b3150ca8b42cab7193b0c3669 (= HF LFS oid); mtp-gemma-4-E2B-it-Q8_0.gguf 97817664 bytes sha256 9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919 (= HF LFS oid).
Smoke evidence: /termux-home/ladder/conv_mtp_prep/smoke.py (run against an earlier draft), /termux-home/ladder/conv_mtp_prep/smoke_qwen_mtp.log, /termux-home/ladder/conv_mtp_prep/smoke_gemma_mtp_robot.log.
Model-free checks on the fixed code: prompts() twice per family is equal; 13 prompts in IDS order; 13 distinct prompt prefixes; temperature 0 everywhere; a 2-id subset equals the matching rows of the full set -> "prompt checks ok". Resume of the 2-prompt toy dir with default ids -> refused: "--resume: gemma_e2b_q40's completed block has prompts ['A6', 'C1'], not [...13 ids...]". Orphan test: copy of toy dir with qwen35_4b_q40mtp_on's block record removed (its 2 turns orphaned), --resume --models gemma_e2b_q40 -> turns.jsonl 10 -> 8 lines, backup turns.jsonl.before_resume_* 10 lines, 0 orphan rows left.
Toy run on the fixed code: /termux-home/ladder/conv_mtp_prep/toy2/ (stdout /termux-home/ladder/conv_mtp_prep/toy2.stdout.txt), --ids A6,C1, exit 0:
[gemma_e2b_q40] cold: 2 turns
[gemma_e2b_q40_mtp] cold: 2 turns
[qwen35_4b_q4km] cold: 2 turns
[qwen35_4b_q40mtp] cold: 2 turns
[qwen35_4b_q40mtp_on] cold: 2 turns
MTP SPEED REPORT  (server_manager flags + --cache-ram 0, cores 4-7, cold block, temperature 0)
ttft = client time to first streamed token, median; prompt tok/s median; gen tok/s token-weighted (>= 10 tokens);
accept = accepted / drafted tokens; same = replies byte-identical to the MTP-off twin

model                 load s  ttft s prompt t/s  gen t/s peak MiB trunc  accept    same   thermal start -> end (z9 z10 z11 degC); load state
gemma_e2b_q40            5.5    2.34       70.1    13.68     3394     0     n/a       -   n/a -> n/a; weights-cold
gemma_e2b_q40_mtp        7.6    2.36       71.3    23.06     4533     0   56.4%     2/2   n/a -> n/a; weights-cold
qwen35_4b_q4km           6.9    4.92       31.4     6.56     5281     0     n/a       -   n/a -> n/a; weights-cold
qwen35_4b_q40mtp         6.2    4.45       34.6     7.57     5028     0     n/a       -   n/a -> n/a; weights-cold
qwen35_4b_q40mtp_on      7.0    4.66       33.0     9.52     4978     0   66.7%     2/2   n/a -> n/a; weights-cold

score_c.py on toy2/c_results.json: C1 exact 1/1 for all 5 configs, all regimes.
(Round-1 toy run on the earlier code: /termux-home/ladder/conv_mtp_prep/toy/.)

DELTA SINCE ROUND 1 (diff of the two candidate diffs):
3c3
< index 0000000..df24106
---
> index 0000000..533dcfb
6c6
< @@ -0,0 +1,206 @@
---
> @@ -0,0 +1,218 @@
22a23
> +import random
69c70,73
< +    """conv_speed's prompts (bench.py wrappers, stop tokens, nonce), restricted to ids, at temperature 0."""
---
> +    """conv_speed's prompts (bench.py wrappers, stop tokens, nonce), restricted to ids, at temperature 0. The nonce
> +    generator is seeded so an MTP-on config and its MTP-off twin get byte-identical prompts (the nonces still differ
> +    between prompts; every block starts a fresh server with --cache-ram 0 and cache_prompt false)."""
> +    random.seed(0)
172a177
> +    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
173a179,185
> +        for (m, _), (_, rows) in done.items():
> +            if sorted(r["prompt_id"] for r in rows) != sorted(ids):
> +                raise SystemExit(f"--resume: {m}'s completed block has prompts {[r['prompt_id'] for r in rows]}, not {ids}")
> +        turns = a.out / "turns.jsonl"
> +        if turns.exists():  # drop turns of a block interrupted before its blocks.jsonl record; keep the old file
> +            turns.rename(a.out / f"turns.jsonl.before_resume_{stamp}")
> +            turns.write_text("".join(json.dumps(r) + "\n" for _, rows in done.values() for r in rows))
176c188
< +    (a.out / f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json").write_text(json.dumps({
---
> +    (a.out / f"run_{stamp}.json").write_text(json.dumps({


FULL CANDIDATE DIFF:
diff --git a/benchmark/llm_objective_setting/conversation/conv_speed_mtp.py b/benchmark/llm_objective_setting/conversation/conv_speed_mtp.py
new file mode 100644
index 0000000..533dcfb
--- /dev/null
+++ b/benchmark/llm_objective_setting/conversation/conv_speed_mtp.py
@@ -0,0 +1,218 @@
+#!/usr/bin/env python3
+"""MTP speculative-decoding speed runner. Research only: offline, no motors, no main.py.
+
+Usage: conv_speed_mtp.py --out DIR [--models a,b] [--ids A6,C1] [--resume] [--thermal-log PATH]
+
+conv_speed.py's cold block (thermal gate, page-cache drop through the handshake, cores 4-7, cpuset guard, redo on
+lost cores, --resume) with these changes: temperature 0 for every prompt, only the 13 prompts in IDS, no cached
+block, and per config extra server flags (MTP drafting). Qwen think is OFF as before.
+Draft acceptance = sum(draft_n_accepted) / sum(draft_n) from the server's final timings. For each MTP-on config the
+report counts replies byte-identical to its MTP-off twin (PAIRS). C-bucket replies are also written in bench.py's
+result format to DIR/c_results.json for score_c.py.
+Writes DIR/blocks.jsonl, DIR/turns.jsonl, DIR/report.txt, DIR/c_results.json.
+"""
+import argparse
+import hashlib
+import json
+import random
+import statistics
+import subprocess
+import time
+from datetime import datetime, timezone
+from pathlib import Path
+
+import conv_speed as cs  # the archived runner: its prompts, server flags, streaming turn, stop, VmHWM, resume
+from conv_speed import bench, ladder, sm
+
+M = cs.M
+GEMMA_DRAFT = M + "mtp/mtp-gemma-4-E2B-it-Q8_0.gguf"
+MTP = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
+CONFIGS = {  # config -> (gguf, prompt family, think, extra flags)
+    "gemma_e2b_q40": (M + "gemma-4-E2B-it-Q4_0.gguf", "gemma", None, []),
+    "gemma_e2b_q40_mtp": (M + "gemma-4-E2B-it-Q4_0.gguf", "gemma", None, ["--model-draft", GEMMA_DRAFT, *MTP]),
+    "qwen35_4b_q4km": (M + "qwen35/Qwen3.5-4B-Q4_K_M.gguf", "chatml", "OFF", []),
+    "qwen35_4b_q40mtp": (M + "qwen35/mtp/Qwen3.5-4B-Q4_0-MTP.gguf", "chatml", "OFF", []),
+    "qwen35_4b_q40mtp_on": (M + "qwen35/mtp/Qwen3.5-4B-Q4_0-MTP.gguf", "chatml", "OFF", MTP),
+}
+PAIRS = {"gemma_e2b_q40_mtp": "gemma_e2b_q40", "qwen35_4b_q40mtp_on": "qwen35_4b_q40mtp"}  # MTP on -> MTP off
+IDS = "A5 A6 B1 B2 B6 B7 C1 C6 C7 C8 C9 C10 C11".split()
+BY_ID = {p["id"]: p for p in bench.PROMPTS_A + bench.PROMPTS_B + bench.PROMPTS_C}
+
+
+def server_cmd(name):
+    """conv_speed.server_cmd (server_manager.py's binary and flags plus --cache-ram 0) with this config's flags."""
+    gguf, _, _, extra = CONFIGS[name]
+    return cs.server_cmd(gguf) + extra
+
+
+def start(name, log, timeout=180):
+    """conv_speed.start with server_cmd(name): cores 4-7; returns (process, seconds from spawn to /health ok)."""
+    if cs.healthy(cs.PORT):
+        raise SystemExit(f"port {cs.PORT} is already serving; stop that server first")
+    began = time.perf_counter()
+    proc = subprocess.Popen(["taskset", "-c", ladder.CORES, *server_cmd(name)], stdout=log, stderr=subprocess.STDOUT,
+                            start_new_session=True)
+    while not cs.healthy(cs.PORT):
+        if proc.poll() is not None or time.perf_counter() - began > timeout:
+            cs.stop(proc)
+            raise RuntimeError(f"llama-server did not become healthy (exit {proc.poll()}); see the server log")
+        time.sleep(0.1)
+    return proc, time.perf_counter() - began
+
+
+def prompts(family, think, ids):
+    """conv_speed's prompts (bench.py wrappers, stop tokens, nonce), restricted to ids, at temperature 0. The nonce
+    generator is seeded so an MTP-on config and its MTP-off twin get byte-identical prompts (the nonces still differ
+    between prompts; every block starts a fresh server with --cache-ram 0 and cache_prompt false)."""
+    random.seed(0)
+    return [(pid, dict(body, temperature=0)) for pid, body in cs.prompts(family, think) if pid in ids]
+
+
+def run_block(name, ids, out, thermal):
+    """conv_speed.run_block's cold block for one config."""
+    gguf, family, think, extra = CONFIGS[name]
+    files = [gguf] + [GEMMA_DRAFT] * ("--model-draft" in extra)
+    ladder.check_cores("before cold")
+    therm = {}
+    if thermal:
+        therm["start"] = ladder.thermal_gate(thermal[0], thermal[1], f"{name} cold")
+        print(f"  [{name}] thermal start: {ladder.fmt_thermal(therm['start'])} (waited {therm['start']['waited_s']} s)",
+              flush=True)
+    load_state = ladder.make_cold(name, files)
+    ladder.check_cores("before starting cold")
+    with open(out / "logs" / f"{name}.cold.server.log", "w") as log:
+        proc, load_s = start(name, log)
+        rows = []
+        try:
+            for pid_, body in prompts(family, think, ids):
+                ttft, tm, stop_type, text = cs.turn(body)
+                rows.append({"model": name, "block": "cold", "prompt_id": pid_, "ttft_s": ttft, "stop_type": stop_type,
+                             "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
+                             "prompt_tok_s": tm.get("prompt_per_second"), "predicted_n": tm.get("predicted_n"),
+                             "predicted_ms": tm.get("predicted_ms"), "gen_tok_s": tm.get("predicted_per_second"),
+                             "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
+                             "timings": tm, "reply": text})
+                ladder.check_cores("during cold")
+            peak = cs.vm_hwm_mib(proc.pid)
+        finally:
+            cs.stop(proc)
+    if thermal:
+        therm["end"] = ladder.read_thermal(thermal[0])
+        print(f"  [{name}] thermal end:   {ladder.fmt_thermal(therm['end'])}", flush=True)
+    ladder.check_cores("after cold")
+    return {"model": name, "block": "cold", "load_s": load_s, "load_state": load_state, "peak_rss_mib": peak,
+            "thermal": therm or None}, rows
+
+
+def acceptance(rows):
+    drafted = sum(r["draft_n"] or 0 for r in rows)
+    return sum(r["draft_n_accepted"] or 0 for r in rows) / drafted if drafted else None
+
+
+def report(done, models):
+    f = lambda v, spec: format(v, spec) if v is not None else "n/a"
+    t = lambda b, k: ladder.fmt_thermal(b["thermal"][k]).split(" degC")[0] if b.get("thermal") else "n/a"
+    L = ["MTP SPEED REPORT  (server_manager flags + --cache-ram 0, cores 4-7, cold block, temperature 0)",
+         "ttft = client time to first streamed token, median; prompt tok/s median; gen tok/s token-weighted (>= 10 tokens);",
+         "accept = accepted / drafted tokens; same = replies byte-identical to the MTP-off twin", ""]
+    L.append(f"{'model':<21}{'load s':>7}{'ttft s':>8}{'prompt t/s':>11}{'gen t/s':>9}{'peak MiB':>9}{'trunc':>6}{'accept':>8}"
+             f"{'same':>8}   thermal start -> end (z9 z10 z11 degC); load state")
+    for name in models:
+        if (name, "cold") not in done:
+            L.append(f"{name:<21}  not run")
+            continue
+        meta, rows = done[(name, "cold")]
+        s = cs.summary(meta, rows)
+        same = "-"
+        if name in PAIRS and (PAIRS[name], "cold") in done:
+            off = {r["prompt_id"]: r["reply"] for r in done[(PAIRS[name], "cold")][1]}
+            same = f"{sum(off.get(r['prompt_id']) == r['reply'] for r in rows)}/{len(rows)}"
+        L.append(f"{name:<21}{f(s['load_s'], '.1f'):>7}{f(s['ttft_s'], '.2f'):>8}{f(s['prompt_tok_s'], '.1f'):>11}"
+                 f"{f(s['gen_tok_s'], '.2f'):>9}{s['peak_rss_mib']:>9}{s['truncated']:>6}{f(acceptance(rows), '.1%'):>8}"
+                 f"{same:>8}   {t(meta, 'start')} -> {t(meta, 'end')}; {meta['load_state']['mode'].split(' (')[0]}")
+    return "\n".join(L)
+
+
+def c_results(done):
+    """C-bucket rows in bench.py's result format (run_index 1), graded by bench.parse_and_grade_c, for score_c.py."""
+    out = []
+    for (name, _), (_, rows) in done.items():
+        for r in rows:
+            p = BY_ID[r["prompt_id"]]
+            if r["prompt_id"][0] != "C":
+                continue
+            clean = bench.strip_think(r["reply"])
+            parse_ok, exact, crit, viol = bench.parse_and_grade_c(clean, p["expected"])
+            out.append({"config": name, "bucket": "C", "prompt_id": p["id"], "lang": p["lang"], "run_index": 1,
+                        "source_sha256": bench.SOURCE_SHA256, "reply_raw": r["reply"], "reply_clean": clean,
+                        "truncated": r["stop_type"] == "limit", "C_parse_ok": parse_ok, "C_exact": exact,
+                        "C_critical_failure": crit, "C_reject_violation": viol})
+    return out
+
+
+def main():
+    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
+    ap.add_argument("--out", type=Path, required=True)
+    ap.add_argument("--models", default=",".join(CONFIGS))
+    ap.add_argument("--ids", default=",".join(IDS), help="prompt ids (a subset of IDS, for toy runs)")
+    ap.add_argument("--resume", action="store_true")
+    ap.add_argument("--thermal-log", default=None)
+    a = ap.parse_args()
+    models, ids = a.models.split(","), a.ids.split(",")
+    if set(models) - set(CONFIGS):
+        raise SystemExit(f"unknown models {sorted(set(models) - set(CONFIGS))}")
+    if set(ids) - set(IDS):
+        raise SystemExit(f"ids outside the 13 {sorted(set(ids) - set(IDS))}")
+    if a.resume and not a.out.is_dir():
+        raise SystemExit(f"--resume: {a.out} does not exist")
+    a.out.mkdir(parents=True, exist_ok=a.resume)
+    (a.out / "logs").mkdir(exist_ok=True)
+    done = cs.load_done(a.out) if a.resume else {}
+    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
+    if a.resume:
+        for (m, _), (_, rows) in done.items():
+            if sorted(r["prompt_id"] for r in rows) != sorted(ids):
+                raise SystemExit(f"--resume: {m}'s completed block has prompts {[r['prompt_id'] for r in rows]}, not {ids}")
+        turns = a.out / "turns.jsonl"
+        if turns.exists():  # drop turns of a block interrupted before its blocks.jsonl record; keep the old file
+            turns.rename(a.out / f"turns.jsonl.before_resume_{stamp}")
+            turns.write_text("".join(json.dumps(r) + "\n" for _, rows in done.values() for r in rows))
+        print("resume: keeping " + (", ".join(m for m, _ in done) or "nothing"), flush=True)
+    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
+    (a.out / f"run_{stamp}.json").write_text(json.dumps({
+        "conv_speed_mtp_sha256": sha(__file__), "conv_speed_sha256": sha(cs.__file__), "bench_sha256": bench.SOURCE_SHA256,
+        "ladder_sha256": sha(ladder.__file__), "ids": ids,
+        "llama_server": sm.LLAMA_SERVER, "libggml_cpu_sha256": sha(Path(sm.LLAMA_SERVER).parent / "libggml-cpu.so"),
+        "server_cmds": {m: server_cmd(m) for m in models}}, indent=1) + "\n")
+    ladder.wait_cores("start")
+    thermal = None
+    if a.thermal_log:
+        idle = ladder.read_thermal(a.thermal_log)
+        print(f"thermal idle reading (gate z9 <= idle + {ladder.GATE_MC / 1000:.0f} degC): {ladder.fmt_thermal(idle)}", flush=True)
+        thermal = (a.thermal_log, idle)
+    with open(a.out / "turns.jsonl", "a") as tlog, open(a.out / "blocks.jsonl", "a") as blog:
+        for name in models:
+            if (name, "cold") in done:
+                print(f"[{name}] cold: already completed, skipped (--resume)", flush=True)
+                continue
+            print(f"[{name}] cold: {len(ids)} turns", flush=True)
+            while True:
+                ladder.wait_cores(f"{name} cold")
+                try:
+                    meta, rows = run_block(name, ids, a.out, thermal)
+                    break
+                except ladder.CoresLost as e:
+                    print(f"  [{name} cold] {e}: discarding this block's data and redoing it", flush=True)
+            tlog.writelines(json.dumps(r) + "\n" for r in rows)
+            tlog.flush()
+            blog.write(json.dumps(meta) + "\n")
+            blog.flush()
+            done[(name, "cold")] = (meta, rows)
+    (a.out / "c_results.json").write_text(json.dumps(c_results(done), indent=1) + "\n")
+    text = report(done, models)
+    (a.out / "report.txt").write_text(text + "\n")
+    print(text)
+
+
+if __name__ == "__main__":
+    main()
diff --git a/benchmark/llm_objective_setting/conversation/run_conv_mtp.sh b/benchmark/llm_objective_setting/conversation/run_conv_mtp.sh
new file mode 100755
index 0000000..44d03e4
--- /dev/null
+++ b/benchmark/llm_objective_setting/conversation/run_conv_mtp.sh
@@ -0,0 +1,26 @@
+#!/bin/bash
+# MTP speculative decoding and Qwen3.5-4B Q4_0 speed on the robot's llama.cpp (server_manager.py's binary and flags
+# plus --cache-ram 0). Research only: offline, no motors, no main.py. Five configs, one cold block each over 13 of the
+# DECISIONS #110 prompts at temperature 0: gemma_e2b_q40, gemma_e2b_q40_mtp (drafter), qwen35_4b_q4km,
+# qwen35_4b_q40mtp (MTP off), qwen35_4b_q40mtp_on. Every block waits until z9 <= idle + 4 degC; losing cores 4-7
+# pauses the run and redoes the block; cold loads drop the page cache through the handshake (LADDER_COLD_HANDSHAKE=1).
+# Afterwards score_c.py grades the C replies (report only) into OUT_DIR/c_scores.txt.
+#
+# Usage: run_conv_mtp.sh [OUT_DIR]          new run
+#        run_conv_mtp.sh --resume OUT_DIR   keep OUT_DIR's completed blocks, run the rest
+# Unattended, from native Termux:  RUN_SCRIPT=run_conv_mtp.sh bash ~/ladder/oneshot.sh [--resume OUT_DIR]
+set -euo pipefail
+C=/termux-home/robot/benchmark/llm_objective_setting/conversation
+cd /termux-home/ladder
+RESUME=()
+if [ "${1:-}" = --resume ]; then
+  OUT=${2:?usage: run_conv_mtp.sh --resume OUT_DIR}
+  RESUME=(--resume)
+else
+  OUT=${1:-/termux-home/ladder/conv_mtp_$(date -u +%Y%m%dT%H%M%SZ)}
+fi
+python3 "$C/conv_speed_mtp.py" --out "$OUT" "${RESUME[@]}" \
+  --thermal-log /termux-home/ladder/thermal.log \
+  > >(tee -a "$OUT.stdout.txt") 2> >(tee -a "$OUT.stderr.txt" >&2)
+python3 "$C/score_c.py" "$OUT/c_results.json" > "$OUT/c_scores.txt"
+cat "$OUT/c_scores.txt"


Please return a final review with: verdict (APPROVE / APPROVE WITH NOTES / REQUEST CHANGES), numbered findings with severity (BLOCKER/MAJOR/MINOR/NOTE), file:line, and a concrete failure scenario for each.
