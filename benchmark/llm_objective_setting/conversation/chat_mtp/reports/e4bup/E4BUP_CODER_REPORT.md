# E4B-UP Coder report — Stage 1

Coder: Codex CLI, gpt-6.1-sol, effort medium, Ponytail lite. Reviewer: Claude Code CLI, claude-opus-5-5, effort medium, Ponytail off. Stage 2 has not started. No commit or push performed.

## Base check

- Fetch succeeded. HEAD = origin/main = live refs/heads/main = `0fc03dc4f31dd7c4c2a7feee50607f8c8819f9de`. Parent `2ce2b3b7d2cb9ef09c2613de85e106d0192e74c5`. Clean status; empty stash.
- HEAD adds only the three PICKPROMPT-1 files:
  - `README.md` SHA-256 `30752d54df8e8b68cc63be605cae3e1afc8fb86588777939faa38d9022d9ed1e`.
  - `harness.py` SHA-256 `44e30a1bb12dfad3720f2577e1b7ecd6abdae95d04808d3aec16c32a3e2e11cb`.
  - `test_harness.py` SHA-256 `821a5f8df4290a9eed2af58c59bdcd66866f46300d475506ac91357e47e5f084`.

## Storage and downloads

Initial disk gate failed at 2414141440 bytes free; resume recheck 2466160640. Owner authorized deletion of old review working copies. Deleted four `/tmp/clean2-review-*` directories and the 21 `copy_r*` / `r*/copy` directories under `/termux-home/review/`; reports/logs/diffs/manifests remained. The first temporary source had already been verified and removed during the briefly requested archive operation; its new archive and the second incomplete archive were subsequently deleted when the owner changed the instruction to deletion. No existing models, experiment folders, benchmark results or caches were deleted.

Shared storage and Termux use the same 118015217664-byte filesystem. Moving uncompressed folders would not free space. Cleanup left 11344572416 bytes free, passing the 9680000000-byte pre-download gate. Largest observed holdings before new downloads: models 39.04 GB, yolo_bench 7.12 GB, sel-candidates 5.55 GB, laya-test 3.46 GB, von-test 3.17 GB, laya-onnx-test 2.00 GB; Debian /root 8.10 GB. Counts were collected during cleanup, not a full consistent phone inventory. No historical snapshot establishes when 50 GB was free.

Hub metadata confirms publisher unsloth, license apache-2.0, revision `bfc15c382204943c3a8fff0c750b94ae2364d7a3`. Primary source: https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/bfc15c382204943c3a8fff0c750b94ae2364d7a3 . API snapshots: `hub_model.json`, `hub_tree.json` in the work folder.

Current published E2B Q4_0 is 3041378400 bytes; retained local E2B is 2841481184 bytes. Both local E2B base/drafter hashes rechecked unchanged before/after downloads. Existing E4B Q4_K_M retained. Downloads used `curl -f -L -C - --retry 3 --connect-timeout 30 -o <destination> <pinned URL>`. Expected hashes are pinned LFS oid values; post-download bytes and SHA-256 matched:

- URL: https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/bfc15c382204943c3a8fff0c750b94ae2364d7a3/gemma-4-E4B-it-Q4_0.gguf
  - Destination `/termux-home/models/gemma-4-E4B-it-Q4_0.gguf`; bytes `4836002944`; SHA-256 `4a403d2e4d80281063e4f517b1c061ded8476b4011a4fc2ba7dbff707075547e`.
- URL: https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/bfc15c382204943c3a8fff0c750b94ae2364d7a3/MTP/mtp-gemma-4-E4B-it-Q8_0.gguf
  - Destination `/termux-home/models/mtp/mtp-gemma-4-E4B-it-Q8_0.gguf`; bytes `98653248`; SHA-256 `b6a723115efa510d3b3215db1e26790dae84cd08c2134a764f3d194f1f0c3376`.

E2B retained hashes:
```
8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52  /termux-home/models/gemma-4-E2B-it-Q4_0.gguf
9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919  /termux-home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf
```

## Changes and mirrored E2B lines

Six candidate files only (manifest below). Added E4B Q4_0 as the last menu entry, preserving all existing entries and menu numbers; old E4B remains labeled Q4_K_M. Drafter on by default for this chat entry, missing/wrong size warns yellow and loads without drafter, MTP off shown. Opt-in robot setups refuse invalid drafter before killing or starting any server. Robot default unchanged.

Mirrored E2B chat tuple verbatim:
```python
('Gemma 4 E2B Q4_0', 'gemma-4-E2B-it-Q4_0.gguf', 'gemma', 4096),
```
Mirrored E2B robot setup lines verbatim:
```python
return start_server(
    HOME + "/models/gemma-4-E2B-it-Q4_0.gguf",
    port=8080, ctx=2048
)
```
Mirrored E2B drafter flags verbatim:
```python
MTP_ARGS = ["--model-draft", MTP_DRAFT, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
```

Both E4B paths use the existing b1609 dotprod binary, threads4/batch4/parallel1, swa-full/cache-ram0. Chat preserves taskset rule and ctx4096; robot ctx2048. E4B drafter size 98653248. Runner --model e4b selects only the E4B command/hashes and labels its output E4B; e2b remains default, existing E2B command/payload/report/hash behavior and resume format preserved. Existing runner shell wrappers already forward arguments unchanged.

/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
PASS: old setup branches/helper functions AST-identical; old menu tuples/indices, commands, payloads unchanged; E2B runner commands, hashes, prompts, payload and report byte-identical; scope exactly six files.

## Checks: base versus candidate

| Suite | Base | Candidate | Result |
|---|---:|---:|---|
| run_cycle_safety_test | 22 | 22 | PASS |
| test_detector_robotcam | 17 | 17 | PASS |
| test_heat_control | 29 | 29 | PASS |
| test_run_mission | 10 | 10 | PASS |
| test_picker | 29 | 29 | PASS |
| test_chat_mtp | 10 | 10 | PASS |
| test_e4b_mtp | 0 | 6 | PASS |
| runner | 32 | 35 | PASS |
| harness | 4 | 4 | PASS (one intentional skip) |

Base first six suites passed together: 117 tests; runner32 and harness4 (one skip). Candidate first six117 plus six new E4B tests, runner35 and harness4 (one skip). Exact candidate/native reviewer test commands are in `test_commands.json` and the review request. Native Termux Python 3.13.13 was invoked from proot, using installed numpy/Pillow/requests/ONNX Runtime. Initial attempts with Debian Python failed at import because those dependencies are absent; logs retained; no package downloads/installations were performed.

Base and candidate navigation output are byte-identical, MD5 `fbbfe5b3248f1c930ac7c0d5fc67d4c6`. This is the output MD5, not the Python source MD5. `git diff --check` passed. Old robot setup branch bodies/helper ASTs, existing chat tuples/indices/commands/payloads, and default E2B runner commands/hashes/prompts/payload/report compared against git base and matched. No forbidden file changed.

## Non-timed smoke: functionality only

Proot, agent resident, no root, no camera/motors, one server at a time. `setup_e4b_q4_mtp` invoked through start_setup; os.system intercepted to capture identical argv/log stderr and skip only broad pkill after asserting no server existed. Existing wait_for_server used unchanged. Chat command generated by chat.server_cmd with the same taskset rule. Server processes supervised and stopped after replies. Load seconds informational (setup includes its existing 2 s prelaunch sleep); NOT performance measurements.

| Command | Load s | MemAvailable before kB | After kB | Drafted | Accepted | Servers left |
|---|---:|---:|---:|---:|---:|---|
| setup_e4b_q4_mtp | 13.30 | 3728888 | 1984064 | 93 | 17 | NONE |
| chat_e4b_q4_mtp | 10.76 | 4671796 | 1839092 | 90 | 24 | NONE |

Full commands, payloads, raw replies/timings, process exits and memory snapshots: `smoke.json`; full stdout/stderr and both server logs in the work folder. Relevant log lines:
```
setup_e4b_q4_mtp
0.08.894.827 I common_speculative_init_result: loading draft model '/data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E4B-it-Q8_0.gguf'
0.10.452.777 I srv  llama_server: listening on http://127.0.0.1:8080
0.20.937.497 I slot print_timing: id  0 | task 0 | draft acceptance = 0.18280 (   17 accepted /    93 generated), mean len =  1.55
chat_e4b_q4_mtp
0.08.508.623 I common_speculative_init_result: loading draft model '/data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E4B-it-Q8_0.gguf'
0.10.271.258 I srv  llama_server: listening on http://127.0.0.1:8080
0.19.403.938 I slot print_timing: id  0 | task 0 | draft acceptance = 0.26667 (   24 accepted /    90 generated), mean len =  1.80
```

Both logs include the known speculative memory-fitting warning (`Gemma4Assistant requires ctx_other` / failed to measure draft model memory), followed by successful loading and accepted tokens. This warning is also present in the archived E2B MTP logs. No server remains.

## Independent reviews

PICKPROMPT-1 flags retained: `--model claude-opus-5-5 --effort medium --safe-mode --restricted --disable-slash-commands --tools Read,Grep,Glob,Bash --permission-mode dontAsk --permission-prompts none --strict-mcp-config --mcp-config {"mcpServers":{}} --no-session-persistence`. `--allowedTools` contains Read/Grep/Glob, precisely the three unit-test command patterns in the request and `Bash(wc -l *)`, `Bash(sha256sum *)`. Added `--output-format stream-json --verbose` to audit every actual reviewer shell command. Full exact command/request/diagnostics retained separately in the work folder.

### Round 1

{
  "valid": true,
  "errors": [],
  "model": "claude-opus-5-5",
  "shell_commands": [
    "wc -l docs/HANDOFF.md",
    "wc -l docs/STATUS.md",
    "wc -l docs/DECISION_INDEX.md",
    "wc -l docs/CLAUDE.md",
    "/data/data/com.termux/files/usr/bin/python3 -B -m unittest run_cycle_safety_test test_detector_robotcam test_heat_control test_run_mission test_picker test_chat_mtp test_e4b_mtp",
    "/data/data/com.termux/files/usr/bin/python3 -B -m unittest discover -s benchmark/llm_objective_setting/conversation/chat_mtp -p test_chat_mtp_runner.py",
    "/data/data/com.termux/files/usr/bin/python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py",
    "sha256sum chat.py",
    "sha256sum server_manager.py",
    "sha256sum test_chat_mtp.py",
    "sha256sum test_e4b_mtp.py",
    "sha256sum benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py",
    "sha256sum benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py",
    "sha256sum candidate.diff",
    "sha256sum e4bup_evidence/candidate_r1.diff",
    "wc -l chat.py",
    "wc -l server_manager.py",
    "wc -l test_chat_mtp.py",
    "wc -l test_e4b_mtp.py",
    "wc -l benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py",
    "wc -l benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py",
    "wc -l docs/WORKFLOW.md",
    "sha256sum e4bup_evidence/base_nav_stdout.txt",
    "sha256sum e4bup_evidence/candidate_nav_stdout.txt",
    "sha256sum frozen_manifest.json",
    "sha256sum e4bup_evidence/frozen_manifest_r1.json",
    "wc -l e4bup_evidence/download.log",
    "wc -l verified_line_counts.txt"
  ],
  "read_files": [
    "CLAUDE.md",
    "TASK.txt",
    "benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py",
    "benchmark/llm_objective_setting/conversation/chat_mtp/run_chat_mtp.sh",
    "benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py",
    "candidate.diff",
    "chat.py",
    "docs/CLAUDE.md",
    "docs/DECISIONS.md",
    "docs/DECISION_INDEX.md",
    "docs/HANDOFF.md",
    "docs/REVIEWER.md",
    "docs/STATUS.md",
    "docs/WORKFLOW.md",
    "e4bup_evidence/base_main_tests.txt",
    "e4bup_evidence/base_runner_native_tests.txt",
    "e4bup_evidence/base_runner_tests.txt",
    "e4bup_evidence/candidate_runner_tests.txt",
    "e4bup_evidence/candidate_test_chat_mtp_tests.txt",
    "e4bup_evidence/candidate_test_e4b_mtp_tests.txt",
    "e4bup_evidence/chat_e4b_q4_mtp.server.log",
    "e4bup_evidence/download.log",
    "e4bup_evidence/downloads.json",
    "e4bup_evidence/e2b_before.sha256",
    "e4bup_evidence/hub_e2b_current.json",
    "e4bup_evidence/hub_model.json",
    "e4bup_evidence/hub_tree.json",
    "e4bup_evidence/invariance.txt",
    "e4bup_evidence/setup_e4b_q4_mtp.server.log",
    "e4bup_evidence/smoke.json",
    "e4bup_evidence/smoke.stdout.txt",
    "e4bup_evidence/test_commands.json",
    "frozen_manifest.json",
    "server_manager.py",
    "test_chat_mtp.py",
    "test_e4b_mtp.py",
    "verified_line_counts.txt"
  ],
  "line_count_entries": 37,
  "citations_verified": 19,
  "searched": [
    {
      "pattern": "MTP|E4B|chat\\.py|server_manager",
      "path": "/data/data/com.termux/files/home/review/e4bup/copy_r1/docs/STATUS.md",
      "output_mode": "content"
    },
    {
      "pattern": "^#+ .*(#?137|#?120)\\b",
      "path": "/data/data/com.termux/files/home/review/e4bup/copy_r1/docs/DECISIONS.md",
      "output_mode": "content"
    },
    {
      "pattern": "^(#+ )?\\*{0,2}(#|No\\. ?)?13[67]\\b",
      "path": "/data/data/com.termux/files/home/review/e4bup/copy_r1/docs/DECISIONS.md",
      "output_mode": "content"
    },
    {
      "pattern": "^(e4bup_evidence/|candidate\\.diff|frozen_manifest|TASK\\.txt|chat\\.py|server_manager\\.py|test_chat_mtp\\.py|test_e4b_mtp\\.py|benchmark/llm_objective_setting/conversation/chat_mtp/|docs/(HANDOFF|WORKFLOW|REVIEWER|STATUS|DECISION_INDEX|DECISIONS|CLAUDE)\\.md|CLAUDE\\.md)",
      "path": "/data/data/com.termux/files/home/review/e4bup/copy_r1/verified_line_counts.txt",
      "output_mode": "content",
      "-n": false
    },
    {
      "pattern": "gemma-4-E4B-it-Q4_0\\.gguf|mtp-gemma-4-E4B-it-Q8_0\\.gguf",
      "path": "/data/data/com.termux/files/home/review/e4bup/copy_r1/e4bup_evidence/hub_tree.json",
      "output_mode": "content",
      "-B": 10,
      "-A": 2
    },
    {
      "pattern": "ctx_other|failed to measure draft|draft acceptance|loading draft model|n_ctx_slot",
      "path": "/data/data/com.termux/files/home/review/e4bup/copy_r1/benchmark/llm_objective_setting/conversation/chat_mtp/runs/chat_mtp_20261014T224541Z/server_block01_ON.log",
      "output_mode": "content",
      "head_limit": 8
    },
    {
      "pattern": "^(Ran |OK|FAILED)",
      "path": "/data/data/com.termux/files/home/review/e4bup/copy_r1/e4bup_evidence",
      "output_mode": "content"
    },
    {
      "pattern": "server_manager|start_setup|model_key\\(|check_hashes\\(",
      "glob": "*.{py,sh}",
      "output_mode": "files_with_matches"
    },
    {
      "pattern": "MODEL_FILES|chat\\.MODELS|MTP_MODEL",
      "glob": "*.py",
      "output_mode": "content"
    },
    {
      "pattern": "run_chat_mtp|chat_mtp\\.py|\\$@|\\$\\*|SCRIPT|RUN=",
      "path": "/data/data/com.termux/files/home/review/e4bup/copy_r1/benchmark/llm_objective_setting/conversation/chat_mtp/oneshot.sh",
      "output_mode": "content"
    },
    {
      "pattern": "E2B-it-Q4_0|after",
      "path": "/data/data/com.termux/files/home/review/e4bup/copy_r1/e4bup_evidence",
      "glob": "!hub_*.json",
      "output_mode": "files_with_matches"
    }
  ],
  "verdict": "VERDICT: APPROVED"
}

Verbatim reviewer final response:

# Independent review: E4B-UP Stage 1, round 1

**Reviewer:** Claude Code CLI, claude-opus-5-5, effort medium, Ponytail off. Isolated copy with no `.git`. I edited nothing and ran only the allowed commands.

## Scope and identity

I hashed all six candidate files with `sha256sum` and counted them with `wc -l`. Every result matches `frozen_manifest.json`. `frozen_manifest.json` and `e4bup_evidence/frozen_manifest_r1.json` have the same hash (`9b5d10af…`), as do `candidate.diff` and `e4bup_evidence/candidate_r1.diff` (`45d896a8…`). The diff touches exactly the six manifest files.

## Mirrored E2B lines and flags

**Chat.** The new menu entry is appended last, so existing menu numbers are kept. It uses context 4096, the same as the E2B Q4_0 entry:

`chat.py:17`
```python
    ('Gemma 4 E4B Q4_0', 'gemma-4-E4B-it-Q4_0.gguf', 'gemma', 4096),
```

The E4B entry goes through the same `server_cmd()` base list as E2B: b1609 dotprod binary, 4 threads and batch threads, parallel 1, `--swa-full`, `--cache-ram 0`. Only the drafter list differs:

`chat.py:82`
```python
    draft_args = E4B_MTP_ARGS if m['file'] == E4B_MTP_MODEL else MTP_ARGS
```

The taskset rule is unchanged and shared by both entries. A missing or wrong-size drafter gives a yellow warning, the server starts without the drafter, and the header shows `MTP off`:

`chat.py:108`
```python
    if draft and not (e4b_mtp_draft_ok() if e4b else mtp_draft_ok()):
```

**Robot setups.** `setup_e4b_q4` and `setup_e4b_q4_mtp` mirror `setup_q4` with context 2048. The drafter check runs before `start_server()`, so before `kill_servers()`. A bad drafter therefore stops nothing and starts nothing, and the setup returns `False`:

`server_manager.py:116`
```python
        if not e4b_mtp_draft_ok():
```

The opt-in comment is present:

`server_manager.py:108`
```python
    # E4B Q4_0: opt-in, not the robot default.
```

The drafter size matches the Hub LFS size:

`server_manager.py:27`
```python
E4B_MTP_DRAFT_SIZE = 98653248
```

The usage line lists all setups:

`server_manager.py:139`
```python
        print("Usage: python3 server_manager.py [setup_q4|setup_q4_mtp|setup_e4b|setup_e4b_q4|setup_e4b_q4_mtp|stop]")
```

`setup_q4`, `setup_q4_mtp` and `setup_e4b` are untouched in the diff. The only change near them is an added "Opt-in:" line in the unknown-setup branch.

## Runner (`chat_mtp.py`)

`--model` defaults to `e2b`:

`benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py:396`
```python
    ap.add_argument('--model', choices=('e2b', 'e4b'), default='e2b', help='Q4_0 chat model (default: e2b)')
```

The default path is the same as before:
- `model_key('e2b')` gives the same error text as the old `MODEL` path.
- `command()`, `check_hashes(SHA256)`, the report header and the `chat_mtp_` folder prefix are unchanged.
- `--resume` compares the same manifest, so old E2B runs still resume.
- Resuming across models is refused, because the recorded commands differ.

E4B hashes are checked at run start, after the refusal checks and before any core wait or server start:

`benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py:422`
```python
        hashes = check_hashes(a.model)
```

The pinned values match the Hub LFS metadata:

`benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py:65`
```python
              '4a403d2e4d80281063e4f517b1c061ded8476b4011a4fc2ba7dbff707075547e',
```

`benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py:66`
```python
              sm.E4B_MTP_DRAFT: 'b6a723115efa510d3b3215db1e26790dae84cd08c2134a764f3d194f1f0c3376'}
```

`oneshot.sh` forwards its arguments through `run_chat_mtp.sh` to `chat_mtp.py`, so the owner command is `oneshot.sh --model e4b` (plus `--smoke` if wanted).

## Download metadata

- `hub_model.json` has `"sha": "bfc15c382204943c3a8fff0c750b94ae2364d7a3"`.
- In `hub_tree.json`, `gemma-4-E4B-it-Q4_0.gguf` has LFS size 4836002944 and SHA `4a403d2e…`. `MTP/mtp-gemma-4-E4B-it-Q8_0.gguf` has LFS size 98653248 and SHA `b6a72311…`.
- `downloads.json` and the `VERIFIED` lines in `download.log` show the same URLs (resolved at the pinned commit), byte counts and SHA-256 values.
- `hub_e2b_current.json` shows the publisher's E2B file is now 3041378400 bytes. The task's do-not-replace rule is consistent with that.
- `e2b_before.sha256` matches the runner's E2B constants.

**I did not hash the external weights.** Everything above is consistency of the Coder's records, not an independent hash.

## Smoke evidence (not a measurement)

- **Setup run, `setup_e4b_q4_mtp`.** The recorded command has no taskset, context 2048, and the E4B drafter flags. The server log shows `n_ctx_slot = 2048`, the draft model loaded, and a reply was generated. Draft acceptance 17/93. The process exited 0 and `servers_left` is empty.
- **Chat run.** The command is `taskset -c 4-7` plus context 4096 plus the drafter. The log shows `n_ctx_slot = 4096` and draft acceptance 24/90. The process stopped and `servers_left` is empty.
- **Memory.** MemAvailable went 3728888 → 1984064 kB in the setup run and 4671796 → 1839092 kB in the chat run, with the agent resident.
- **Memory-fitting warning.** This E4B log line:

  `e4bup_evidence/setup_e4b_q4_mtp.server.log:8`
  ```python
  0.00.968.779 E llama_init_from_model: failed to initialize the context: Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)
  ```

  is identical in kind to the archived E2B baseline:

  `benchmark/llm_objective_setting/conversation/chat_mtp/runs/chat_mtp_20261014T224541Z/server_block01_ON.log:8`
  ```python
  0.00.940.007 E llama_init_from_model: failed to initialize the context: Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)
  ```

  In both logs it is followed by a successful draft load and accepted tokens, so it is harmless.

## Do the tests exercise behaviour?

Mostly, yes:
- `test_e4b_mtp.py` runs the real `start_server` and `start_setup` with faked filesystem, `os.system`, `Popen` and HTTP.
- The base menu entries and the setup command strings are pinned as literals, not derived from the implementation.
- The bad-drafter refusal is checked as `calls == []`, which also covers no pkill:

  `test_e4b_mtp.py:86`
  ```python
              self.assertEqual(calls, [])
  ```

- The chat fallback test checks the E4B drafter path, the yellow warning and `mtp_on == False` for sizes None, 0, ±1.
- The taskset fallback is tested.
- The runner tests check the pinned hashes, refusal on mismatch or missing file, and that the CLI calls `check_hashes('e4b')` and records the E4B commands.

## Findings

No Critical, High or Medium findings.

1. **Low: runner docstring is stale.** The docstring is also argparse's `description`. It does not mention `--model e4b`, the `chat_mtp_e4b_` folder, or that the E4B hashes are pinned Hub LFS values rather than `conv_mtp/ARTIFACTS.md` values:

   `benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py:26`
   ```python
   Usage: chat_mtp.py [--smoke] [--resume RUN_DIR] [--thermal-log PATH]
   ```

   `benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py:24`
   ```python
   model or drafter SHA-256 not the conv_mtp/ARTIFACTS.md values.
   ```

   `README.md` and `RUN.md` in that folder are also unchanged. The CLI `--help` still lists `--model`. This is cosmetic and can go as a DOC DIFF or a later fix.

2. **Low: some E2B-default checks are self-referential.** This line compares the default against itself:

   `benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py:562`
   ```python
               self.assertEqual(cm.command(config, 'e2b'), cm.command(config))
   ```

   No test asserts that a default `main()` calls `check_hashes('e2b')` or uses the `chat_mtp_` prefix. The usage test checks the unknown-setup message, not the `__main__` usage line:

   `test_e4b_mtp.py:92`
   ```python
           self.assertIn('setup_e4b_q4, setup_e4b_q4_mtp', self.launch('unknown')[2])
   ```

   Mitigation: the existing E2B command, G1 flags, hash and report-header tests pin the default behaviour, and I confirmed the default branches by inspection. The script behind `invariance.txt` is not in the copy, so I could not check its PASS claim.

3. **Low: evidence gaps.**
   - There is no E2B "after" hash record in `e4bup_evidence/`, only `e2b_before.sha256`.
   - The free-space figures (11344572416 bytes against the 9680000000-byte gate) and the owner's deletion authorisation appear only in your request, not in the evidence.
   - The smoke driver's source is not in the copy. Its claimed behaviour (intercepting `os.system`, skipping only the broad pkill, no `get_temp`/root calls) is consistent with the logs but not verifiable from source. For example, the server log has stderr content even though the setup command redirects `2>/dev/null`.

   The Coder report should include all three.

4. **Info: memory headroom.** MemAvailable after load was about 1.84–1.98 GB with the agent resident. The archived E2B co-residency minimum (#125) was 2523 MiB. Robot co-residency with E4B is not established, so `setup_e4b_q4*` must stay described as opt-in and unvalidated.

5. **Info: runner timing under heat.** Block timing is tuned for E2B:

   `benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py:69`
   ```python
   BLOCK_S, LAST_START_S, SMOKE_GATE_S = 180, 150, 20
   ```

   At E4B speeds (about 5.7–7.5 tok/s in the smoke runs, which are not measurements), a 512-token reply under throttling could miss the 175 s deadline. The run then stops with a RuntimeError. That fails safe and is not a defect, but the owner should know.

## Checks I ran (all exit 0)

| Command | Result |
|---|---|
| `python3 -B -m unittest run_cycle_safety_test test_detector_robotcam test_heat_control test_run_mission test_picker test_chat_mtp test_e4b_mtp` | Ran 123, OK |
| `… discover -s benchmark/llm_objective_setting/conversation/chat_mtp -p test_chat_mtp_runner.py` | Ran 35, OK |
| `… discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py` | Ran 4, OK (skipped=1) |

These match the Coder's results:
- **Main suites:** base native 117 tests. Candidate 22+17+29+10+29+10 = 117, plus 6 new = 123.
- **Runner:** 32 → 35.
- **Harness:** 4 skip 1 at both base and candidate.
- **Proot-only base runs:** `base_main_tests.txt` and `base_runner_tests.txt` show import errors in proot. That is an environment limit, and the native reruns supersede them.
- **Nav output:** `sha256sum` of `base_nav_stdout.txt` and `candidate_nav_stdout.txt` match (`86a2bf53…`). I cannot compute MD5 within the allowed commands, so I did not check the `fbbfe5b3…` value directly.

## Files read

| File | Lines |
|---|---|
| TASK.txt | 37 |
| docs/HANDOFF.md | 68 |
| docs/WORKFLOW.md | 303 |
| docs/REVIEWER.md | 15 |
| docs/STATUS.md (partial 115–139 + grep) | 449 |
| docs/DECISION_INDEX.md | 54 |
| docs/DECISIONS.md (partial 1650–1669, #137) | 1669 |
| docs/CLAUDE.md | 127 |
| CLAUDE.md | 4 |
| candidate.diff | 468 |
| frozen_manifest.json | 38 |
| verified_line_counts.txt (partial 1–591 + grep) | 3015 |
| chat.py | 395 |
| server_manager.py | 141 |
| test_chat_mtp.py | 210 |
| test_e4b_mtp.py | 96 |
| benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py | 478 |
| benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py | 611 |
| benchmark/llm_objective_setting/conversation/chat_mtp/run_chat_mtp.sh | 9 |
| e4bup_evidence/downloads.json | 16 |
| e4bup_evidence/download.log | 12 |
| e4bup_evidence/e2b_before.sha256 | 2 |
| e4bup_evidence/hub_e2b_current.json | 12 |
| e4bup_evidence/hub_model.json | 200 |
| e4bup_evidence/hub_tree.json (partial 1–12 + grep context) | 1459 |
| e4bup_evidence/invariance.txt | 3 |
| e4bup_evidence/smoke.json | 285 |
| e4bup_evidence/smoke.stdout.txt | 14 |
| e4bup_evidence/setup_e4b_q4_mtp.server.log | 36 |
| e4bup_evidence/chat_e4b_q4_mtp.server.log | 36 |
| e4bup_evidence/test_commands.json | 109 |
| e4bup_evidence/candidate_test_e4b_mtp_tests.txt | 5 |
| e4bup_evidence/candidate_test_chat_mtp_tests.txt | 5 |
| e4bup_evidence/candidate_runner_tests.txt | 7 |
| e4bup_evidence/base_runner_native_tests.txt | 7 |
| e4bup_evidence/base_runner_tests.txt | 26 |
| e4bup_evidence/base_main_tests.txt | 87 |

**Line-count crosscheck:** all counts above agree with `verified_line_counts.txt` except two:
- `download.log`: `wc` gives 12, the list gives 171, because curl progress output uses `\r` and the list apparently splits on it.
- `verified_line_counts.txt`: not listed in itself; `wc` gives 3015.

**Searched only (grep/glob, not read in full):**
- In `e4bup_evidence/`: base_main_native_tests.txt, candidate_run_cycle_safety_test_tests.txt, candidate_test_detector_robotcam_tests.txt, candidate_test_heat_control_tests.txt, candidate_test_run_mission_tests.txt, candidate_test_picker_tests.txt, candidate_harness_tests.txt, base_harness_tests.txt, base_harness_native_tests.txt.
- In `benchmark/llm_objective_setting/conversation/chat_mtp/`: oneshot.sh, and runs/chat_mtp_20261014T224541Z/server_block01_ON.log.
- Repo-wide grep for `server_manager`, `start_setup`, `MODEL_FILES` and `MTP_MODEL` callers. No caller depends on changed behaviour.

## Limitations

- No `.git`, so I could not diff the prohibited paths against base myself. I rely on the six-file diff and the Coder's invariance claim; the script behind it is not in the copy.
- I did not access or hash the external weights.
- The smoke driver source is not in the copy.
- No hardware, server or network use.
- Timings and memory figures are not measurements.

My recommendation is approval, with findings 1–3 as follow-ups for the Coder report or Doc Keeper. This is a review result only, not permission to commit.

VERDICT: APPROVED

## Frozen manifest

```json
[
  {
    "path": "benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py",
    "bytes": 25869,
    "sha256": "b2c370e23e5f7fa93996a28a9aaed2f3f5ee546f2ef70f94eaebe4b34a29cd24",
    "lines": 478
  },
  {
    "path": "benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py",
    "bytes": 34420,
    "sha256": "d8f6e5f2ae98a6bb965041c00452825c152e442345c231541e3e45774688392c",
    "lines": 611
  },
  {
    "path": "chat.py",
    "bytes": 15167,
    "sha256": "b1ef174ef38d6efcef14faff5908e5fc3116f866ec0ea698dddaa5c9ef4d2ced",
    "lines": 395
  },
  {
    "path": "server_manager.py",
    "bytes": 5548,
    "sha256": "4f63062a4dbc842538b72d1c28a8b2a721a0664a1a2bed8d2936e712984d84aa",
    "lines": 141
  },
  {
    "path": "test_chat_mtp.py",
    "bytes": 9945,
    "sha256": "346875c9387368678725b9b190e45d1aa3139e56a5b717b9d6f5a8b00048b961",
    "lines": 210
  },
  {
    "path": "test_e4b_mtp.py",
    "bytes": 4661,
    "sha256": "29d8d367afb339b1f13416733c75ee4847a276026c4394d1fe3638b8f4c4834d",
    "lines": 96
  }
]
```

## UNKNOWN / limitations

- Timed E4B OFF/ON performance, reply quality, sustained heat, memory cost and robot co-residency are unmeasured. These short functional smokes do not establish them.
- No robot/hardware execution was requested or performed. Production robot default stays E2B.
- Stage 2 DEV/TEST scoring has not begun; waiting for Stage 1 review and separate human commit/push gates.

## Owner command for timed E4B OFF/ON

From native Termux after exiting all agents; keep Termux in front, charger unplugged, battery >=80%. This is the existing launcher with the new model option; it performs the thermal idle/root sampling:
```bash
bash ~/robot/benchmark/llm_objective_setting/conversation/chat_mtp/oneshot.sh --model e4b
```
Smoke timing-run option (one prompt per config): add `--smoke`. Resume an E4B run with `--model e4b --resume <run-directory>`. Do not run these from proot.

## Human gate

No staging, commit or push. After valid approval, wait for explicit "yes commit"; stage only manifest paths and use:
```text
llm: Gemma 4 E4B Q4_0 + MTP drafter in robot-chat, opt-in setup_e4b_q4(_mtp) (E4B-UP)
```
Push requires a separate "yes push". Stage 2 starts only after that push.

## Review follow-ups and supplementary evidence

Round 1 is VALID, APPROVED: before/after live/copy identities identical, only permitted shell commands, all 37 file-count entries checked (curl progress log counted by newline/wc, with CR distinction disclosed), all 19 cited lines checked after removing Markdown fence indentation. No code changed after review.

The reviewer noted three Low items. The runner's original E2B-only docstring/help description and README/RUN text are intentionally retained; the new --model option is listed by argparse. No documentation files changed. Existing literal command/hash/payload tests and the frozen-base invariance check cover the E2B default; the reviewer could not inspect that helper because it was absent from its packet. Supporting files below are now supplied outside the unchanged frozen copy so the original review evidence remains intact. The reviewer did not inspect these supplementary attachments; its limitations remain verbatim above.

- The E2B before and after SHA-256 records are byte-identical.
- Disk gate values and deletion authority are from the session transcript: initial free 2414141440, resume 2466160640; owner first asked to archive, then explicitly said “no just delete them”; cleanup free 11344572416 before the two model downloads. The download driver's assert enforces free >=9680000000 immediately before curl.
- The actual smoke driver, invariance check and download driver are attached below with hashes.
- E4B timed runs retain the existing runner's 175 s watchdog; a long reply under throttling may exceed it and cause the run to stop safely. No E4B duration parameters were changed.
- The Low/Info findings do not alter the APPROVED verdict. E4B robot co-residency, heat and timed OFF/ON performance remain unvalidated.

Supplementary attachments (not reviewed, no candidate changes):
- `e4bup_smoke_driver.py` — SHA-256 `408b3557bb70c660f69c5b768fa8a487c4031baf27a8e9fdd758def4962aa8de`.
- `e4bup_invariance_check.py` — SHA-256 `13b9f6a171b19f8563baa96b91c0ed86ef5e4bdaf00d62ec549bf7201db236c0`.
- `e4bup_download_driver.py` — SHA-256 `2a51fbf5e0989b32d46b46f8779d890f75060e9ba1d889e5d24cb9fedd85f505`.
- `e4bup_e2b_before.sha256` — SHA-256 `5ab119d02956413459aadc245f0be7af2cbf2ed891a82230bc7426702306ed99`.
- `e4bup_e2b_after.sha256` — SHA-256 `5ab119d02956413459aadc245f0be7af2cbf2ed891a82230bc7426702306ed99`.
- `e4bup_validity_r1.json` — SHA-256 `c9a231a3886bd1b3dcbb3dd3dbf79b3fa6256c0d6e49534354c28988d31eba8c`.
- `e4bup_command_r1.txt` — SHA-256 `b1f22cc2f9c42a28be61cc1014b7c1670c4a042382f00307ee82bbf54b78b6c1`.

E2B after-hash record:
```text
8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52  /termux-home/models/gemma-4-E2B-it-Q4_0.gguf
9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919  /termux-home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf
```

## Owner-approved cleanup before commit — 2026-10-17 decision

Owner explicitly authorized deleting the five proposed unreferenced E2B projectors/adapters plus the two older E2B quants, despite their optional menu/benchmark references. Exactly these seven files were deleted; no code was edited:

- `~/models/mmproj-gemma-4-E2B-it-Q8_0.gguf`
- `~/models/mmproj-gemma-4-e2b-it-q4_0.gguf`
- `~/models/mmproj-google_gemma-4-E2B-it-bf16.gguf`
- `~/models/gemma4-nav-lora.gguf`
- `~/models/gemma4-voice-lora.gguf`
- `~/models/gemma-4-e2b-it-q4_k_m.gguf`
- `~/models/gemma-4-e2b-it-q8_0.gguf`

Both older llama.cpp trees and the deployed dotprod build were retained, as were the protected E2B Q4_0/drafter, new E4B base/drafter, old E4B Q4_K_M and everything outside the seven-file list. Protected model/build identities checked unchanged.

DF before: 6124175360 bytes available, 111756824576 used.
DF immediately after: 16496918528 bytes available, 101384081408 used.
Total filesystem space reclaimed: 10372743168 bytes (10.37 GB).

Required tests rerun with native Termux Python from proot: main123 OK; runner35 OK; harness4 OK (skipped=1). Raw outputs: post_cleanup_main_tests.txt, post_cleanup_runner_tests.txt, post_cleanup_harness_tests.txt in /termux-home/review/e4bup/. git diff --check passed.

All six frozen candidate hashes verified unchanged after cleanup and tests:
```text
b2c370e23e5f7fa93996a28a9aaed2f3f5ee546f2ef70f94eaebe4b34a29cd24  benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py
d8f6e5f2ae98a6bb965041c00452825c152e442345c231541e3e45774688392c  benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
b1ef174ef38d6efcef14faff5908e5fc3116f866ec0ea698dddaa5c9ef4d2ced  chat.py
4f63062a4dbc842538b72d1c28a8b2a721a0664a1a2bed8d2936e712984d84aa  server_manager.py
346875c9387368678725b9b190e45d1aa3139e56a5b717b9d6f5a8b00048b961  test_chat_mtp.py
29d8d367afb339b1f13416733c75ee4847a276026c4394d1fe3638b8f4c4834d  test_e4b_mtp.py
```

No staging, commit or push. Waiting for the owner’s separate "yes commit". Benchmarks that require deleted model files cannot run unchanged; no model-dependent archived benchmark was rerun.

## Owner-authorized local commit

Commit: `8a840407b0fade86c90ed981080aa27224da3c64`. Exactly the six frozen manifest paths committed after rechecking working-tree and staged SHA-256 values. Message: `llm: Gemma 4 E4B Q4_0 + MTP drafter in robot-chat, opt-in setup_e4b_q4(_mtp) (E4B-UP)`. Git used its automatically configured identity `root <root@localhost.localdomain>`. Working tree clean; main ahead of origin/main by one. No push performed; waiting for separate owner authorization. Stage 2 has not started.

## Owner-authorized push

Pushed main to origin: `0fc03dc..8a84040`. HEAD, origin/main and live remote refs/heads/main all equal `8a840407b0fade86c90ed981080aa27224da3c64`. Working tree clean after push. Stage 2 base check against this commit passed; frozen PICKPROMPT-1 set hash `19773ea00d376d67e78641844241b6a8a310c0c8a7b1a048113f9c435a76ae44`; no TEST files.
