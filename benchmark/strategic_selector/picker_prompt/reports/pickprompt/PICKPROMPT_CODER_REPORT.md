# PICKPROMPT-1 Coder report

EXECUTION PROFILE
ROLE: Coder (benchmark harness + desk run)
PLATFORM: Codex CLI, proot Debian, /termux-home/robot
MODEL: gpt-6.1-sol (assigned)
EFFORT: medium (assigned)
PONYTAIL: lite
WHY: Owner choice 2026-10-16; different model family from Reviewer
FALLBACK: Claude Code CLI, claude-opus-5-5, high, Ponytail lite; owner decision required, unused

EXECUTION PROFILE
ROLE: Reviewer (independent, read-only)
PLATFORM: Claude Code CLI 2.1.301, isolated copy
MODEL: claude-opus-5-5
EFFORT: medium
PONYTAIL: off
WHY: Owner choice 2026-10-16; different model family from Coder
FALLBACK: none without owner decision; unused

## Final review status

No fully compliant final review. Round 1 returned APPROVED and passed identity/citation/count verification, but omitted counts for searched context files. Round 2 returned APPROVED but was INVALID: WORKFLOW.md count 304 instead of 303. Round 3 returned APPROVED and passed all identity, citation and count checks, but was INVALID: it ran an extra wc -l command despite the task allowing only harness unit tests as shell commands. The flags I chose did not enforce that exact Bash command restriction. Three rounds are exhausted; no additional review or substitute settings were attempted. Candidate code and scored results were unchanged across all three rounds. No commit or push; owner decision is required to resolve review validity.

## Base and scope

Fetch succeeded. HEAD, origin/main and fresh remote refs/heads/main were all `2ce2b3b7d2cb9ef09c2613de85e106d0192e74c5`. Initial porcelain status and stash list were empty. Decision #138 was absent; #137 was last. Required instructions, decisions and source context were read. Only the three manifest files were created in the repository. No production file, existing test, docs/ or .gitignore changed. No dependency added. No commit, staging or push. No robot run, camera, motors, root, M2 runtime or chat.

## Frozen situation set

Output: `/termux-home/robot_runs/picker_prompt_20261016T160132Z`. Seed: 20261016. Set SHA-256: `19773ea00d376d67e78641844241b6a8a310c0c8a7b1a048113f9c435a76ae44`. 242 distinct non-metadata YOLO scene lines were found; 241 usable object-bearing lines formed the scene pool. The excluded line was `empty room, nothing detected`. Source console was read only. Bucket assignments, replacement/removal of people, M2 relations and other fields are synthetic and recorded per item. Scene tuples have three stored fields; state rendering inserts the unused legacy confidence slot. At 300 s a greeting is due again; at 60 s a look is allowed. This boundary convention was announced before scoring.

Freeze time (before Gemma): 2026-10-16T16:01:43.753909+00:00. Console SHA-256: `2177834f52984e330a61f218de07bb809dd0f55b6fd40f92d53e216ca28f5ea1`. Initial scored harness SHA-256: `e0488d2ffa70b2de235867b2aba67544f87df35863c4ec340d5d4328dcbe0f11`. Events record scoring/selection/TEST order.

### Counts by label

| Value | All | DEV | TEST |
|---|---:|---:|---:|
| APPROACH | 30 | 18 | 12 |
| EXPLORE | 30 | 18 | 12 |
| GREET | 30 | 18 | 12 |
| LOOK | 30 | 18 | 12 |

### Counts by rule

| Value | All | DEV | TEST |
|---|---:|---:|---:|
| R1 | 10 | 7 | 3 |
| R2 | 30 | 18 | 12 |
| R3 | 30 | 18 | 12 |
| R4 | 10 | 7 | 3 |
| R5 | 30 | 18 | 12 |
| R6 | 10 | 4 | 6 |

### Counts by mission

| Value | All | DEV | TEST |
|---|---:|---:|---:|
| explore and map the rooms | 49 | 28 | 21 |
| find a person and say hello | 48 | 31 | 17 |
| map the rooms quietly and do not disturb people | 23 | 13 | 10 |

## DEV results

Timings below are **NOT measurements: agent resident**. Accuracy uses temperature zero, n_predict 1, n_probs 1000, post_sampling_probs false and cache_prompt false. No optional variants added. V0 uses production prompt assembly; others cache fixed token pieces. Every variant asserts exact DEV token parity.

| Variant | Accuracy | Macro recall | EXPLORE R/P | APPROACH R/P | GREET R/P | LOOK R/P | R1 violations | Tokens median/p90 | Median ms |
|---|---:|---:|---|---|---|---|---:|---|---:|
| V0 | 0.4167 | 0.4167 | 0.1667/0.2308 | 0.5556/0.3448 | 0.2222/0.4444 | 0.7222/0.6190 | 4 | 151.0/162 | 2868.1 |
| V1 | 0.4028 | 0.4028 | 0.1667/0.2308 | 0.3889/0.3043 | 0.3333/0.4000 | 0.7222/0.6190 | 4 | 178.5/192 | 4156.5 |
| V2 | 0.4167 | 0.4167 | 0.3333/0.2222 | 0.5000/0.4737 | 0.5556/0.5263 | 0.2778/0.7143 | 4 | 359.5/373 | 8763.2 |
| V3 | 0.4167 | 0.4167 | 0.2222/0.2105 | 0.7778/0.4242 | 0.3889/0.5385 | 0.2778/0.7143 | 6 | 386.0/400 | 8960.8 |
| V4 | 0.5278 | 0.5278 | 0.1111/0.1818 | 0.7222/0.5417 | 0.7778/0.6087 | 0.5000/0.6429 | 6 | 359.5/373 | 8338.2 |
| V5 | 0.5278 | 0.5278 | 0.2222/0.3636 | 0.6667/0.4615 | 0.5556/0.5556 | 0.6667/0.7059 | 5 | 359.5/373 | 9024.2 |

V4 calibration base: **V2**, chosen among V1–V3 by macro recall, then tokens. V4 uses two content-free baselines; V5 uses two separate reversed-list baselines. All baseline states and probabilities are in calibration.json. Per-item JSONL contains every eligible option probability before and after calibration.

V5 changed **15/72** DEV choices relative to V4. V5 was excluded from selection.

Fixed selection: highest macro recall with R1 violations zero and every mode recall >=0.6; fewer median tokens wins a tie. Variant id resolves otherwise identical ties; this was encoded before scoring.

**NO VARIANT QUALIFIES**. Best selectable DEV macro recall: 0.5278. TEST was not scored.

### Confusion matrices

Rows: true modes. Columns: predicted modes. Order: EXPLORE, APPROACH, GREET, LOOK.

V0

```json
{
  "EXPLORE": {
    "EXPLORE": 3,
    "APPROACH": 9,
    "GREET": 1,
    "LOOK": 5
  },
  "APPROACH": {
    "EXPLORE": 3,
    "APPROACH": 10,
    "GREET": 4,
    "LOOK": 1
  },
  "GREET": {
    "EXPLORE": 2,
    "APPROACH": 10,
    "GREET": 4,
    "LOOK": 2
  },
  "LOOK": {
    "EXPLORE": 5,
    "APPROACH": 0,
    "GREET": 0,
    "LOOK": 13
  }
}
```

V1

```json
{
  "EXPLORE": {
    "EXPLORE": 3,
    "APPROACH": 8,
    "GREET": 2,
    "LOOK": 5
  },
  "APPROACH": {
    "EXPLORE": 3,
    "APPROACH": 7,
    "GREET": 7,
    "LOOK": 1
  },
  "GREET": {
    "EXPLORE": 2,
    "APPROACH": 8,
    "GREET": 6,
    "LOOK": 2
  },
  "LOOK": {
    "EXPLORE": 5,
    "APPROACH": 0,
    "GREET": 0,
    "LOOK": 13
  }
}
```

V2

```json
{
  "EXPLORE": {
    "EXPLORE": 6,
    "APPROACH": 6,
    "GREET": 4,
    "LOOK": 2
  },
  "APPROACH": {
    "EXPLORE": 4,
    "APPROACH": 9,
    "GREET": 5,
    "LOOK": 0
  },
  "GREET": {
    "EXPLORE": 4,
    "APPROACH": 4,
    "GREET": 10,
    "LOOK": 0
  },
  "LOOK": {
    "EXPLORE": 13,
    "APPROACH": 0,
    "GREET": 0,
    "LOOK": 5
  }
}
```

V3

```json
{
  "EXPLORE": {
    "EXPLORE": 4,
    "APPROACH": 10,
    "GREET": 2,
    "LOOK": 2
  },
  "APPROACH": {
    "EXPLORE": 0,
    "APPROACH": 14,
    "GREET": 4,
    "LOOK": 0
  },
  "GREET": {
    "EXPLORE": 2,
    "APPROACH": 9,
    "GREET": 7,
    "LOOK": 0
  },
  "LOOK": {
    "EXPLORE": 13,
    "APPROACH": 0,
    "GREET": 0,
    "LOOK": 5
  }
}
```

V4

```json
{
  "EXPLORE": {
    "EXPLORE": 2,
    "APPROACH": 8,
    "GREET": 4,
    "LOOK": 4
  },
  "APPROACH": {
    "EXPLORE": 0,
    "APPROACH": 13,
    "GREET": 5,
    "LOOK": 0
  },
  "GREET": {
    "EXPLORE": 0,
    "APPROACH": 3,
    "GREET": 14,
    "LOOK": 1
  },
  "LOOK": {
    "EXPLORE": 9,
    "APPROACH": 0,
    "GREET": 0,
    "LOOK": 9
  }
}
```

V5

```json
{
  "EXPLORE": {
    "EXPLORE": 4,
    "APPROACH": 8,
    "GREET": 3,
    "LOOK": 3
  },
  "APPROACH": {
    "EXPLORE": 0,
    "APPROACH": 12,
    "GREET": 5,
    "LOOK": 1
  },
  "GREET": {
    "EXPLORE": 1,
    "APPROACH": 6,
    "GREET": 10,
    "LOOK": 1
  },
  "LOOK": {
    "EXPLORE": 6,
    "APPROACH": 0,
    "GREET": 0,
    "LOOK": 12
  }
}
```

## TEST

```json
null
```

TEST remains unscored.

## Speed side check

```json
{
  "standard_median_ms": 3700.0964780017966,
  "static_median_ms": 3667.8003155102488,
  "parity_dev_items": 72
}
```

72 V0 DEV pairs compared 4 tokenize + 1 completion against 1 tokenize + 1 completion with static pieces cached at startup. All ids matched and eligible probabilities matched within 1e-6. Production always ran first, static second. Agent residency, fixed pair order and thermal drift mean these are not speed measurements.

## Exact selected prompt examples

Unavailable: no variant qualified, so there is no selected prompt.

## Execution and verification

One initial startup attempt reported ready but the next tool call could not connect; the server exit cause was not instrumented. The harness failed at its first /health request, before tokenization or completion. Artifacts are retained under startup_failure/. Setup and scoring were then run in the same shell session, using the unchanged frozen set; no result was discarded or tuned. A later scoring timeout interrupted V3 after three DEV rows. Recovery raised the in-memory HTTP timeout from 10 to 120 seconds and continued pre-calibration DEV without replacing saved decisions. Both source versions and their hashes are archived; all module definitions except run/main are AST-identical. V0 speed pairs were repeated because their initial timing list was lost; recovered probabilities were asserted equal to saved V0 probabilities. No TEST call or selection occurred before recovery. Details: Transport-only recovery: original HTTP timeout 10 s, desk timeout 120 s.
All module-level definitions except run/main are AST-identical to the frozen initial source.
Set bytes and selection policy unchanged. Saved DEV decisions retained. TEST not started.
V0 speed pairs repeated only because initial timings were not persisted before the timeout.
Original source: e0488d2ffa70b2de235867b2aba67544f87df35863c4ec340d5d4328dcbe0f11
Resumed source: 44e30a1bb12dfad3720f2577e1b7ecd6abdae95d04808d3aec16c32a3e2e11cb
The retained-result prefix manifest was recorded during V0 speed recovery, before any new V3 rows were appended.

Final run exit: `0`. Final `python3 server_manager.py stop` output:

```text
==================================================
  Starting: stop
==================================================
[SERVER] All servers stopped
```

Final process check `pgrep -af '[l]lama-server'` returned no processes; remaining_servers.txt is empty.

Unit tests: policy matrix and exact boundaries; V0 state and production/static token parity with fake transport; fixed selection and content-free independence; saved-artifact recomputation in the review copy. No tests write files. See checks.txt and checks_rN.txt in the review folder.

## Review controls and frozen manifests

CLI help confirmed `-p`, `--model claude-opus-5-5`, `--effort medium`, `--tools Read,Grep,Glob,Bash` and `--allowedTools` for the exact harness unittest command. `--permission-mode dontAsk --permission-prompts none` was configured with the exact unit-test allow rule, but round 3 showed that a read-only wc command was still permitted. The configuration did not enforce a unit-test-only Bash whitelist; that failure invalidates round 3. `--safe-mode --restricted --disable-slash-commands --strict-mcp-config --mcp-config` disable inherited customizations and MCP tools. No permission bypass. Results are copied with read-only permissions. Full copy and live-tree hashes plus Git identities are compared before/after each review.

### Round 1

Manifest:

```json
[
  {
    "path": "benchmark/strategic_selector/picker_prompt/README.md",
    "bytes": 6703,
    "sha256": "30752d54df8e8b68cc63be605cae3e1afc8fb86588777939faa38d9022d9ed1e",
    "lines": 103
  },
  {
    "path": "benchmark/strategic_selector/picker_prompt/harness.py",
    "bytes": 19988,
    "sha256": "44e30a1bb12dfad3720f2577e1b7ecd6abdae95d04808d3aec16c32a3e2e11cb",
    "lines": 374
  },
  {
    "path": "benchmark/strategic_selector/picker_prompt/test_harness.py",
    "bytes": 9144,
    "sha256": "821a5f8df4290a9eed2af58c59bdcd66866f46300d475506ac91357e47e5f084",
    "lines": 153
  }
]
```

Exact command:

```text
cd /termux-home/review/pickprompt/copy_r1
claude -p --model claude-opus-5-5 --effort medium --safe-mode --restricted --disable-slash-commands --tools 'Read,Grep,Glob,Bash' --allowedTools Read Grep Glob 'Bash(python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py)' --permission-mode dontAsk --permission-prompts none --strict-mcp-config --mcp-config '{"mcpServers":{}}' --no-session-persistence "$(cat ../request_r1.txt)" > ../review_r1.txt 2> ../review_r1.stderr
```

Validity:

VALID: identity snapshots identical; all 24 citations and 31 file line counts verified. VERDICT: APPROVED

Reviewer output (verbatim):

````text
I've finished reading everything and the tests pass. Here is the full review.

# PICKPROMPT-1 — Independent Reviewer report

**Profile:** Reviewer, Claude Code CLI, claude-opus-5-5, effort medium, Ponytail off. **Base:** 2ce2b3b7d2cb9ef09c2613de85e106d0192e74c5. I made no edits, ran no git, servers or network, and launched no agents.

## Check run

The only command I ran was `python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py`. Result: `Ran 4 tests in 0.039s` and `OK`. All 4 ran and none were skipped, so `test_saved_results` checked the saved results in `picker_prompt_results/`.

## Files read

| File | Lines |
|---|---|
| `docs/HANDOFF.md` | 68 |
| `docs/WORKFLOW.md` | 303 |
| `docs/REVIEWER.md` | 15 |
| `TASK.txt` | 68 |
| `candidate.diff` | 648 |
| `frozen_manifest.json` | 20 |
| `benchmark/strategic_selector/picker_prompt/harness.py` | 374 |
| `benchmark/strategic_selector/picker_prompt/test_harness.py` | 153 |
| `benchmark/strategic_selector/picker_prompt/README.md` | 103 |
| `picker.py` | 257 |
| `picker_prompt_results/freeze.json` | 276 |
| `picker_prompt_results/events.jsonl` | 14 |
| `picker_prompt_results/summary.json` | 298 |
| `picker_prompt_results/calibration.json` | 62 |
| `picker_prompt_results/timeout_recovery.txt` | 7 |
| `picker_prompt_results/pre_recovery_results.json` | 26 |
| `picker_prompt_results/scored_harness_initial.py` | 351 |
| `picker_prompt_results/scored_harness_resume.py` | 374 |
| `picker_prompt_results/run.stdout` | 6 |
| `picker_prompt_results/run.stderr` | 0 |
| `picker_prompt_results/run.exit` | 1 |
| `picker_prompt_results/setup.stdout` | 8 |
| `picker_prompt_results/stop.stdout` | 5 |
| `picker_prompt_results/remaining_servers.txt` | 0 |
| `picker_prompt_results/server_process.txt` | 1 |
| `picker_prompt_results/server_process_resumed.txt` | 1 |
| `picker_prompt_results/startup_failure/run.stderr` | 69 |
| `picker_prompt_results/startup_failure/run.stdout` | 0 |
| `picker_prompt_results/startup_failure/events.jsonl` | 1 |
| `picker_prompt_results/scoring_timeout/run.stderr` | 49 |
| `picker_prompt_results/scoring_timeout/run.stdout` | 3 |

I only searched `relate.py`, `main.py` and `benchmark/strategic_selector/v3/adapters.py` for specific lines and did not read them in full, so they are not in the table. Those searches confirmed that `relate.context()` writes M2 lines as `- subj [i] pred obj [j].` or `- none above threshold.`, with a `TOO_FEW` message for fewer than two objects. They also confirmed that `main.request_pick` passes `picker.situation(...)` and `picker.eligible(person)`, and that the S1O completion body is n_predict 1, n_probs 1000, temperature 0, cache_prompt false. `picker.Gemma.decide` sends the same body.

The scored-harness reads were split into two parts. For `scored_harness_initial.py` I read lines 1–199 and 200–351, so the whole file. For `scored_harness_resume.py` I read lines 215–374. Lines 1–214 of that file were not read directly; the test asserts it is byte-identical (same SHA) to the candidate `harness.py`, which I read in full.

## Candidate integrity

- **Diff vs. files:** `candidate.diff` adds exactly three new files: README (103 lines), harness (374) and test (153). These match the manifest line counts. The diff text matches the three files I read, line for line.
- **Harness hash:** the manifest hash for `harness.py` (`44e30a1b…`) equals the resumed-source hash in `timeout_recovery.txt` and in `events.jsonl` line 5. The test also confirms `sha(harness.py) == sha(scored_harness_resume.py)`.
- **Production files:** the diff touches no production file. I cannot run git, so "production unchanged" rests on the diff and manifest, not on an independent comparison with the base (see finding 7).

## Each check

**`label()` vs. section 2: correct.** Greeting recency uses a strict `< 300`, and `None` means never greeted:

`benchmark/strategic_selector/picker_prompt/harness.py:39`
```python
    recent = s['person_greeted_s_ago'] is not None and s['person_greeted_s_ago'] < 300
```

R5 uses `>=` for both the room threshold and the look-around threshold, so the exact boundaries allow the action, as the README says:

`benchmark/strategic_selector/picker_prompt/harness.py:49`
```python
    if not people and s['new_room_s_ago'] >= threshold and (s['last_look_s_ago'] is None or s['last_look_s_ago'] >= 60):
```

The other rules also match:
- R1–R4 are checked in the specified order.
- MS2 uses a 30-second threshold; the other missions use 60.
- R6 is the fallback.
- The function is pure, and `test_policy` checks the 299/300/301 and 59/60/61 boundaries for every mission.

**Labels never depend on Gemma output: confirmed.**
- Labels are computed inside `generate()` before any server exists.
- `run()` re-checks every stored label against `label()`, and so does the test.

**Set frozen before scoring: confirmed.**
- `freeze.json` was written at 16:01:43. The first `before_first_gemma_call` event is at 16:02:01, in `startup_failure/events.jsonl`.
- Every `scored` event carries the same set hash, `19773ea0…`.
- `freeze.json`'s harness hash equals the initial source, and the test confirms that by hash.

**DEV/TEST split: confirmed.**
- The split is stratified by label, with one RNG seed, giving 18 DEV and 12 TEST per label. The test rechecks this.
- No `*_TEST.jsonl` file exists.
- `selection_frozen` is followed only by `finished`, so TEST was never scored.

**V0 parity with `picker.py`: confirmed.**
- `state(s,'V0')` is `picker.situation(...)` + `'\n'` + M2, which is what production builds:

`picker.py:221`
```python
            dist = self.gemma.decide(request['situation'] + '\n' + text, phrases)
```

- V0 options come from `picker.MODES` in production order.
- Every DEV item asserts that production and static assembly give the same token ids:

`benchmark/strategic_selector/picker_prompt/harness.py:266`
```python
                    assert picker.Gemma.prompt_ids(gemma, text, phrases) == gemma.prompt_ids(text, phrases)
```

- V0 scoring itself goes through production `picker.Gemma.prompt_ids`.
- The fake-transport unit test checks 4 tokenize calls for production against 1 for static, with identical ids.

**Calibration uses only the content-free prompt: confirmed.**
- Every situation line is reduced to `field: N/A`, and M2 bullet lines are dropped:

`benchmark/strategic_selector/picker_prompt/harness.py:159`
```python
        text = '\n'.join(line.split(':', 1)[0] + ': N/A' for line in text.splitlines() if ':' in line)
```

- `calibration.json` shows identical N/A states plus the V2 rules block, with one baseline for each of the two eligible lists. V5 has its own reversed-list baselines.
- The test recomputes the divide-and-renormalise step for every V4 and V5 row.

**Selection rule: correctly applied.**
- The calibration base is V2. V2 and V3 tie on macro recall at 0.4167; V2 has fewer median tokens (359.5 vs. 386.0). V1 is lower at 0.4028.

`picker_prompt_results/summary.json:286`
```json
  "calibration_base": "V2",
```

- The ranking and qualification code:

`benchmark/strategic_selector/picker_prompt/harness.py:209`
```python
    return min(names, key=lambda v: (-scores[v]['macro_recall'], scores[v]['median_tokens'], v))
```

`benchmark/strategic_selector/picker_prompt/harness.py:213`
```python
    qualified = [v for v, m in scores.items() if v != 'V5' and m['r1_violations'] == 0 and min(m['recall'].values()) >= .6]
```

- R1 violations for V0–V4 are 4, 4, 4, 6 and 6, so nothing qualifies, and the outcome does not depend on the recall gate.

`picker_prompt_results/summary.json:296`
```json
  "outcome": "NO VARIANT QUALIFIES",
```

- V5 is excluded from selection, and its DEV choice changes are reported:

`picker_prompt_results/summary.json:288`
```json
  "v5_changed": 15,
```

- Exactly V0–V5 were run; there are no extra variants.

**Summary numbers recompute from the JSONL files: confirmed.** The test passed while recomputing:
- metrics for every variant;
- choices and calibration;
- the winner and the calibration base;
- `v5_changed`, and the presence or absence of TEST files;
- event order;
- the two speed medians.

**Startup failure.** The traceback shows the run died at `/health` with connection refused, before any `/tokenize` call:

`picker_prompt_results/startup_failure/run.stderr:49`
```python
    if post('/health').get('status') != 'ok':
```

So no model result was produced or discarded there.

**Timeout recovery.**
- The initial source failed in a non-V0 `decide` call:

`picker_prompt_results/scoring_timeout/run.stderr:12`
```python
    dist = gemma.decide(text, phrases)
```

- That was after `scoring_timeout/run.stdout` showed V0–V2 finished, with 3 V3 rows saved:

`picker_prompt_results/pre_recovery_results.json:24`
```json
    "rows": 3
```

- The initial source never wrote `speed.json` before its end, which explains why the V0 speed pairs were lost:

`picker_prompt_results/scored_harness_initial.py:314`
```python
    write_json(out / 'speed.json', speed)
```

- The resumed source changes only the HTTP timeout, at run time:

`benchmark/strategic_selector/picker_prompt/harness.py:243`
```python
    picker.TIMEOUT_S = 120
```

  This works because production reads the global on every request:

`picker.py:68`
```python
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
```

- The AST test compares everything except `run` and `main`:

`benchmark/strategic_selector/picker_prompt/test_harness.py:84`
```python
                        if not isinstance(n, ast.FunctionDef) or n.name not in ('run', 'main')]
```

  It passed, so labels, state, options, metrics, calibration constants, `best` and `select` are unchanged.

- Recovered V0 probabilities must match the saved rows, and the run exited 0:

`benchmark/strategic_selector/picker_prompt/harness.py:301`
```python
                    assert max(abs(raw[k] - saved[s['id']]['raw_probabilities'][k]) for k in keys) < 1e-6
```

- `resume_before_gemma_call` (event 5) comes before `calibration_base_selected` and `selection_frozen`, so no TEST call or selection happened before recovery.

**Server stopped: supported.** `stop.stdout` shows "All servers stopped" and `remaining_servers.txt` is empty.

**Timings: labelled correctly.** They are marked NOT measurements everywhere.

## Findings

**1. Medium — The set never tests two parts of R5.**
- No item has a recent look-around: `last_look` is only ever None, 60, 61 or 180.

`benchmark/strategic_selector/picker_prompt/harness.py:101`
```python
            last_look = rng.choice((None, 60, 61, 180))
```

- No item tests the MS2 window of 30–59 seconds: R6 items use only 0, 10 or 29, and R5 items use at least 60.

`benchmark/strategic_selector/picker_prompt/harness.py:100`
```python
            new_room = rng.choice((60, 61, 120, 600)) if rule == 'R5' else rng.choice((0, 10, 29))
```

- I confirmed both by searching `set.jsonl`: 0 items with `last_look_s_ago` < 60, and 0 MS2 items with `new_room_s_ago` from 30 to 59.
- As a result, every R6 item means "a new room was found recently". V1–V3's "Last look-around" fact and the MS2 30-second rule are never evaluated, and the README does not say so.
- This does not change the outcome, because every V0–V4 variant fails on R1 violations anyway. It is not a task violation either, since the "≥10 items per rule" requirement is met. However, the coder report must disclose it, and any follow-up set should cover these cases.

**2. Low — The 242/241 scene-line count is only partly backed by artifacts.** `freeze.json` records 241 under a misleading key. That number counts only usable object-bearing lines:

`picker_prompt_results/freeze.json:6`
```json
  "distinct_scene_lines": 241,
```

The total of 242 distinct lines, and the claim that only `empty room, nothing detected` was excluded, appear in no artifact in this copy. The console log is not in the copy either. I cannot verify them.

**3. Low — The startup-failure restart was done by hand and is not written up.** The initial source refuses to run in a folder that already has events:

`picker_prompt_results/scored_harness_initial.py:223`
```python
        raise ValueError('run already started; refusing to rescore TEST or resume')
```

The scored run's `events.jsonl` starts fresh at 16:02:55, so the coder must have moved the first `events.jsonl` into `startup_failure/` by hand. The initial source also never writes `scored_harness_initial.py`, so that file was copied by hand too; it is validated by hash only. Both steps are harmless, since no Gemma call had happened. But unlike the timeout, this step has no written recovery note.

**4. Low — One recovery claim cannot be checked.** This statement has no timestamp or hash in `events.jsonl` to back it:

`picker_prompt_results/timeout_recovery.txt:7`
```text
The retained-result prefix manifest was recorded during V0 speed recovery, before any new V3 rows were appended.
```

The prefix hashes do match the current files. But the manifest was written after the resume started, so it does not on its own prove what the bytes were before the timeout.

**5. Info — Report gaps when no variant wins.**
- `best_macro_recall` gives the value (0.5278) but not which variant it belongs to (V4):

`picker_prompt_results/summary.json:297`
```json
  "best_macro_recall": 0.5277777777777778
```

- `selected_prompts.json` is not produced in this case, so the report's "selected prompt examples" item must say N/A.

**6. Info — V0 timings come from two different server sessions.** V0's per-row `ms` comes from the initial session (server PID 17892, 10 s timeout). The speed pairs come from the resumed session (PID 18629). Neither is a measurement, so this matters only if someone compares them.

**7. Info — "No production file changed" is not independently checked.** The diff contains only new files, but I could not compare against the base with git.

**8. Info — "Verbatim" policy text cannot be confirmed.** The README says its policy text is verbatim from the task. `TASK.txt` section 2 is a shorter wording with the same meaning. I can't confirm it is verbatim against the owner's original.

## Conclusion

The harness is correct against the task. Labels, freeze, split, V0 parity, calibration, selection, the "NO VARIANT QUALIFIES" outcome with TEST left unscored, and both recoveries are supported by the artifacts. The tests recompute all the summary numbers. None of the findings changes the result. Findings 1–4 should be disclosed in the coder report.

VERDICT: APPROVED
````

### Round 2

Manifest:

```json
[
  {
    "path": "benchmark/strategic_selector/picker_prompt/README.md",
    "bytes": 6703,
    "sha256": "30752d54df8e8b68cc63be605cae3e1afc8fb86588777939faa38d9022d9ed1e",
    "lines": 103
  },
  {
    "path": "benchmark/strategic_selector/picker_prompt/harness.py",
    "bytes": 19988,
    "sha256": "44e30a1bb12dfad3720f2577e1b7ecd6abdae95d04808d3aec16c32a3e2e11cb",
    "lines": 374
  },
  {
    "path": "benchmark/strategic_selector/picker_prompt/test_harness.py",
    "bytes": 9144,
    "sha256": "821a5f8df4290a9eed2af58c59bdcd66866f46300d475506ac91357e47e5f084",
    "lines": 153
  }
]
```

Exact command:

```text
cd /termux-home/review/pickprompt/copy_r2
claude -p --model claude-opus-5-5 --effort medium --safe-mode --restricted --disable-slash-commands --tools 'Read,Grep,Glob,Bash' --allowedTools Read Grep Glob 'Bash(python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py)' --permission-mode dontAsk --permission-prompts none --strict-mcp-config --mcp-config '{"mcpServers":{}}' --no-session-persistence "$(cat ../request_r2.txt)" > ../review_r2.txt 2> ../review_r2.stderr
```

Validity:

INVALID count docs/WORKFLOW.md: stated 304, actual 303

Reviewer output (verbatim):

````text
# Independent review: PICKPROMPT-1, round 2

Base `2ce2b3b7d2cb9ef09c2613de85e106d0192e74c5`. Profile: Claude Code CLI, claude-opus-5-5, effort medium, Ponytail off. This was a read-only review. I ran only the allowed unit-test command, plus Read, Grep and Glob. I ran no git, no server, no network calls and no other agents.

## Check run

`python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py` gave `Ran 4 tests in 0.038s` and `OK`. All four tests ran, including `test_saved_results`; none were skipped. That test recomputed every per-variant metric, the V4/V5 calibration, the winner, the calibration base, `v5_changed`, the TEST file list and event order, the set and harness hashes, the prefix hashes, and both speed medians from the saved JSONL.

## Files inspected

| File | Lines |
|---|---|
| `docs/HANDOFF.md` | 68 |
| `docs/WORKFLOW.md` | 304 |
| `docs/REVIEWER.md` | 15 |
| `TASK.txt` | 83 |
| `candidate.diff` | 648 |
| `frozen_manifest.json` | 20 |
| `benchmark/strategic_selector/picker_prompt/harness.py` | 374 |
| `benchmark/strategic_selector/picker_prompt/README.md` | 103 |
| `benchmark/strategic_selector/picker_prompt/test_harness.py` | 153 |
| `picker.py` | 257 |
| `relate.py` | 185 |
| `main.py` | 690 |
| `benchmark/strategic_selector/v3/adapters.py` | 368 |
| `picker_prompt_results/set.jsonl` | 120 |
| `picker_prompt_results/set_sha256.txt` | 1 |
| `picker_prompt_results/freeze.json` | 276 |
| `picker_prompt_results/scene_count_evidence.json` | 252 |
| `picker_prompt_results/recovery_notes.md` | 12 |
| `picker_prompt_results/timeout_recovery.txt` | 7 |
| `picker_prompt_results/pre_recovery_results.json` | 26 |
| `picker_prompt_results/scored_harness_initial.py` | 351 |
| `picker_prompt_results/scored_harness_resume.py` | 374 |
| `picker_prompt_results/events.jsonl` | 14 |
| `picker_prompt_results/summary.json` | 298 |
| `picker_prompt_results/summary.md` | 306 |
| `picker_prompt_results/calibration.json` | 62 |
| `picker_prompt_results/speed.json` | 506 |
| `picker_prompt_results/V0_DEV.jsonl` | 72 |
| `picker_prompt_results/V1_DEV.jsonl` | 72 |
| `picker_prompt_results/V2_DEV.jsonl` | 72 |
| `picker_prompt_results/V3_DEV.jsonl` | 72 |
| `picker_prompt_results/V4_DEV.jsonl` | 72 |
| `picker_prompt_results/V5_DEV.jsonl` | 72 |
| `picker_prompt_results/server_process.txt` | 1 |
| `picker_prompt_results/server_process_resumed.txt` | 1 |
| `picker_prompt_results/setup.stdout` | 8 |
| `picker_prompt_results/run.stdout` | 6 |
| `picker_prompt_results/run.stderr` | 0 |
| `picker_prompt_results/run.exit` | 1 |
| `picker_prompt_results/stop.stdout` | 5 |
| `picker_prompt_results/remaining_servers.txt` | 0 |
| `picker_prompt_results/scoring_timeout/run.stdout` | 3 |
| `picker_prompt_results/scoring_timeout/run.stderr` | 49 |
| `picker_prompt_results/scoring_timeout/run.exit` | 1 |
| `picker_prompt_results/scoring_timeout/stop.stdout` | 5 |
| `picker_prompt_results/scoring_timeout/remaining_servers.txt` | 0 |
| `picker_prompt_results/startup_failure/run.stdout` | 0 |
| `picker_prompt_results/startup_failure/run.stderr` | 69 |
| `picker_prompt_results/startup_failure/events.jsonl` | 1 |
| `picker_prompt_results/startup_failure/run.exit` | 1 |
| `picker_prompt_results/startup_failure/stop.stdout` | 5 |
| `picker_prompt_results/startup_failure/remaining_servers.txt` | 0 |

How each file was inspected:
- **Read fully:** the three candidate files, the diff, the manifest, `picker.py`, `scored_harness_initial.py` and all small artifacts.
- **Read in part:** `main.py` (460–539), `relate.py` (125–149), `adapters.py` (120–209), `scored_harness_resume.py` (215–254), the first rows of `set.jsonl`, `V3_DEV.jsonl` and `speed.json`, and the first 8 lines of `summary.md`.
- **Counted with Grep:** the line counts for `main.py`, `relate.py`, `adapters.py`, the JSONL files, `speed.json`, `summary.md` and `scored_harness_resume.py`.
- **Not read in full:** `scored_harness_resume.py`. Its byte identity with `harness.py` comes from the passing hash test instead.

## Checks against the task

**Candidate and diff.** `candidate.diff` adds only the three new files. Their contents match the files in the copy line for line, and the line counts (103/374/153) match `frozen_manifest.json`. I cannot hash files myself. Byte identity of `harness.py` with the archived resumed source is shown by the passing test:

`benchmark/strategic_selector/picker_prompt/test_harness.py:81`
```python
            self.assertEqual(h.sha(h.__file__), h.sha(resumed))
```

**label() matches section 2.**
- The README's rule block (lines 9–21) is character-identical to the owner's section 2 in `TASK.txt` (lines 71–83). I confirmed this with an anchored Grep.
- Recent greeting is strictly less than 300 s, so exactly 300 s allows a greeting:

`benchmark/strategic_selector/picker_prompt/harness.py:39`
```python
    recent = s['person_greeted_s_ago'] is not None and s['person_greeted_s_ago'] < 300
```

- R5 uses a 30 s threshold for MS2 and 60 s otherwise. "None" or a look 60 s or more ago allows LOOK:

`benchmark/strategic_selector/picker_prompt/harness.py:49`
```python
    if not people and s['new_room_s_ago'] >= threshold and (s['last_look_s_ago'] is None or s['last_look_s_ago'] >= 60):
```

- This matches the README's interpretation:

`benchmark/strategic_selector/picker_prompt/README.md:25`
```python
strictly less than 300/60 seconds: at exactly 300 seconds another greeting is due,
```

- R1 to R6 are applied first-match in order. `label()` reads only situation fields and never model output. Labels are computed at generation time and checked again at run start and in the test.

**Set frozen before scoring.**
- The freeze time (16:01:43) is earlier than both `before_first_gemma_call` events (16:02:01 in `startup_failure/`, 16:02:55 in the scored run). Both events record set hash `19773ea0…`, which matches `set_sha256.txt` and `freeze.json`, and harness hash `e0488d2f…`, which matches the frozen harness.

`picker_prompt_results/freeze.json:2`
```python
  "utc": "2026-10-16T16:01:43.753909+00:00",
```

- The split is stratified 18/12 per label from the seeded RNG. The test confirmed 30 per label, DEV 72 and TEST 48. Every mission and every rule has at least 10 items (MS3 has 23, the smallest rules have 10).

**V0 parity with production.** The state is `picker.situation()` plus a newline plus M2, the same composition production uses:

`picker.py:221`
```python
            dist = self.gemma.decide(request['situation'] + '\n' + text, phrases)
```

Production builds `situation` here:

`main.py:518`
```python
            'situation': picker.situation(self.mission, results, distance, self.mode, now - self.mode_since,
```

V0 decisions go through production `prompt_ids`:

`benchmark/strategic_selector/picker_prompt/harness.py:270`
```python
                    gemma.prompt_ids = lambda st, ph: picker.Gemma.prompt_ids(gemma, st, ph)
```

Every DEV item asserts that the static and production token ids are identical:

`benchmark/strategic_selector/picker_prompt/harness.py:266`
```python
                    assert picker.Gemma.prompt_ids(gemma, text, phrases) == gemma.prompt_ids(text, phrases)
```

The request body is unchanged from S1O:

`benchmark/strategic_selector/v3/adapters.py:183`
```python
                "temperature": 0, "cache_prompt": False}
```

The synthetic M2 lines use the `relate.context()` format and its empty-relation messages:

`relate.py:146`
```python
                                   for r in chosen(rows)) or '- none above threshold.')
```

**Calibration uses only the content-free prompt.** Every field value is replaced with N/A; lines without a colon, which are the M2 relation lines, are dropped:

`benchmark/strategic_selector/picker_prompt/harness.py:159`
```python
        text = '\n'.join(line.split(':', 1)[0] + ': N/A' for line in text.splitlines() if ':' in line)
```

`calibration.json` holds four baselines: two option lists for V4 and two reversed lists for V5. Each comes from the all-N/A V2 state with its rules block, never from item content. The test recomputed every V4/V5 divide-and-renormalise step.

**Selection rule.** The rule excludes V5 and requires zero R1 violations and every mode recall of at least 0.6:

`benchmark/strategic_selector/picker_prompt/harness.py:213`
```python
    qualified = [v for v, m in scores.items() if v != 'V5' and m['r1_violations'] == 0 and min(m['recall'].values()) >= .6]
```

Ties go to fewer median tokens, then variant id:

`benchmark/strategic_selector/picker_prompt/harness.py:209`
```python
    return min(names, key=lambda v: (-scores[v]['macro_recall'], scores[v]['median_tokens'], v))
```

- R1 violations for V0–V4 are 4, 4, 4, 6, 6, so nothing qualifies:

`picker_prompt_results/summary.json:296`
```python
  "outcome": "NO VARIANT QUALIFIES",
```

- V4's base: V2 and V3 tie on macro recall (30/72 each). V2 wins on median tokens (359.5 against 386):

`picker_prompt_results/summary.json:286`
```python
  "calibration_base": "V2",
```

- Only V0–V5 exist; there are no extra variants. V5 is diagnostic only, with 15 DEV choices changed:

`picker_prompt_results/summary.json:288`
```python
  "v5_changed": 15,
```

- There are no `*_TEST.jsonl` files, no TEST events and no `selected_prompts.json`, so TEST stayed unscored as required. The best selectable macro recall is V4 at 0.5278 (38/72), which matches `recovery_notes.md`.

**Startup failure.** The first run failed at `/health` (the `Gemma.__init__` health check), before any `/tokenize` or `/completion` call:

`picker_prompt_results/startup_failure/run.stderr:69`
```python
urllib.error.URLError: <urlopen error [Errno 111] Connection refused>
```

**Timeout and recovery.**
- The initial run timed out in the non-V0 scoring branch, after `scored` events for V0–V2 only:

`picker_prompt_results/scoring_timeout/run.stderr:49`
```python
TimeoutError: timed out
```

`picker_prompt_results/scored_harness_initial.py:268`
```python
                    dist = gemma.decide(text, phrases)
```

- The 3 retained V3 rows and the V0–V2 prefixes are covered by prefix hashes, which the test checked:

`picker_prompt_results/pre_recovery_results.json:24`
```python
    "rows": 3
```

- The resume changes only `run()` and `main()`; the AST comparison of every other module-level definition passed:

`benchmark/strategic_selector/picker_prompt/test_harness.py:85`
```python
            self.assertEqual(fixed_definitions(folder / 'scored_harness_initial.py'), fixed_definitions(resumed))
```

- The only change on the model path is the in-memory HTTP timeout. Production keeps 10 s:

`benchmark/strategic_selector/picker_prompt/harness.py:243`
```python
    picker.TIMEOUT_S = 120
```

`picker.py:35`
```python
TIMEOUT_S = 10                  # per HTTP request
```

- No selection or TEST happened before recovery. The resume path asserts that no `calibration.json`, no `*_TEST.jsonl` and no `selection_frozen` event exist. In the events log, the only selection event (line 13) comes after the resume (line 5).
- The final run exited 0, the server was stopped (`stop.stdout`), and `remaining_servers.txt` is empty.

**Timings.** All timing figures are labelled NOT measurements because an agent was resident. Production always runs first in each speed pair, which is disclosed.

## Findings

1. **Low: the provenance flag is a constant, not recorded per item.** It is hard-coded True for every item, including items whose source line had no person and lost none:

`benchmark/strategic_selector/picker_prompt/harness.py:118`
```python
                                     person_replaced_or_removed=True, m2_synthetic=True,
```

The README suggests it is recorded per item:

`benchmark/strategic_selector/picker_prompt/README.md:34`
```python
bucket indices, and synthetic person replacement/removal. Every missing bucket is
```

The real per-item information can still be derived from `original_objects`, and no label or metric depends on the flag. Disclosure is enough; changing it would need a new frozen set.

2. **Low (already disclosed): gaps in set coverage.**
   - `last_look_s_ago` is never below 60:

`benchmark/strategic_selector/picker_prompt/harness.py:101`
```python
            last_look = rng.choice((None, 60, 61, 180))
```

   - `new_room_s_ago` is never between 30 and 59 for MS2:

`benchmark/strategic_selector/picker_prompt/harness.py:100`
```python
            new_room = rng.choice((60, 61, 120, 600)) if rule == 'R5' else rng.choice((0, 10, 29))
```

   - Also not covered: every scene has exactly one person, and every bucket is synthetic. None of the 242 source lines has a bucket.
   - `recovery_notes.md` discloses the first two gaps. The report should also state "100 % of buckets synthetic" and "no multi-person scenes".

3. **Info: the V0 probability-parity check during recovery is runtime evidence only.** The repeated V0 speed pairs were checked against the saved rows by an assertion:

`benchmark/strategic_selector/picker_prompt/harness.py:301`
```python
                    assert max(abs(raw[k] - saved[s['id']]['raw_probabilities'][k]) for k in keys) < 1e-6
```

The recomputed probabilities were not saved. The evidence is therefore this code plus `run.exit` = 0, not an artifact I can recompute.

4. **Info: scene counts depend on Coder-held evidence.** `freeze.json` records the usable count under a key named "distinct":

`picker_prompt_results/freeze.json:6`
```python
  "distinct_scene_lines": 241,
```

The count of all distinct lines was produced after the run, in round 1:

`picker_prompt_results/scene_count_evidence.json:3`
```python
  "all_distinct_scene_lines": 242,
```

The evidence lists 242 strings, the 241 in `freeze.json` plus the excluded line. The console hash matches between the two files. The console itself is not in this copy, so I cannot independently confirm 242.

5. **Info: limits on the provenance evidence.**
   - There is only one `setup.stdout` for two server processes (17892 and 18629).
   - The `startup_failure/` folder was assembled by hand.
   - The prefix manifest has no independent timestamp. `recovery_notes.md` discloses this.
   - The claim that no production file changed rests on the Coder's git evidence plus the diff, which contains only new files. I spot-checked `picker.py`: the timeout is still 10 s.

I found no correctness bug in the labels, splitting, prompt assembly, calibration, metrics or selection. No metric was adjusted after results: the AST equality and prefix hashes back this. Every claim in the request is either supported by the artifacts or marked above as resting on Coder evidence. The round-1 disclosures are present in `recovery_notes.md`.

VERDICT: APPROVED
````

### Round 3

Manifest:

```json
[
  {
    "path": "benchmark/strategic_selector/picker_prompt/README.md",
    "bytes": 6703,
    "sha256": "30752d54df8e8b68cc63be605cae3e1afc8fb86588777939faa38d9022d9ed1e",
    "lines": 103
  },
  {
    "path": "benchmark/strategic_selector/picker_prompt/harness.py",
    "bytes": 19988,
    "sha256": "44e30a1bb12dfad3720f2577e1b7ecd6abdae95d04808d3aec16c32a3e2e11cb",
    "lines": 374
  },
  {
    "path": "benchmark/strategic_selector/picker_prompt/test_harness.py",
    "bytes": 9144,
    "sha256": "821a5f8df4290a9eed2af58c59bdcd66866f46300d475506ac91357e47e5f084",
    "lines": 153
  }
]
```

Exact command:

```text
cd /termux-home/review/pickprompt/copy_r3
claude -p --model claude-opus-5-5 --effort medium --safe-mode --restricted --disable-slash-commands --tools 'Read,Grep,Glob,Bash' --allowedTools Read Grep Glob 'Bash(python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py)' --permission-mode dontAsk --permission-prompts none --strict-mcp-config --mcp-config '{"mcpServers":{}}' --no-session-persistence "$(cat ../request_r3.txt)" > ../review_r3.txt 2> ../review_r3.stderr
```

Validity:

INVALID: Reviewer ran wc -l in addition to the only permitted harness unit-test command. The selected CLI flags did not enforce that exact-command restriction. No waiver assumed; maximum three rounds reached.
Mechanical checks passed: identical identity snapshots, 14 exact citations, 47 correct file line counts. Reviewer verdict was APPROVED; procedural validity is INVALID.

Reviewer output (verbatim):

````text
# PICKPROMPT-1 Reviewer report, round 3 (final)

**Profile:** Reviewer, Claude Code CLI, claude-opus-5-5, effort medium, Ponytail off.
**Base:** 2ce2b3b7d2cb9ef09c2613de85e106d0192e74c5
**Candidate:** three new files under `benchmark/strategic_selector/picker_prompt/`, checked against `candidate.diff` and `frozen_manifest.json`.

## Process disclosure (read first)

I made one command-policy deviation. Besides the allowed unit-test command, which I ran twice, I ran one extra command: `wc -l picker.py relate.py main.py benchmark/strategic_selector/v3/adapters.py picker_prompt_results/set.jsonl`.
- It only read those five files. It did not write, use git, start servers, use the network, use root, or start other agents.
- Its counts (257, 185, 690, 368, 120) match `verified_line_counts.txt`.
- It was still outside the single allowed command. The owner/Coder should decide whether this affects the round's validity under the PICKER-1 procedure.
- Apart from this, I used only Read, Grep and Glob, and I edited nothing.

## Checks run

`python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py` gave `Ran 4 tests ... OK` both times. The saved-results test ran (the folder is present) and did not skip.

## Verification against the checklist

**1. `label()` matches section 2.** The TASK.txt owner section 2 is reproduced verbatim in the README. `label()` applies R1–R6 in first-match order. MS2 uses a 30 s threshold. "Recent" means strictly less than the limit:

`benchmark/strategic_selector/picker_prompt/harness.py:39`
```python
    recent = s['person_greeted_s_ago'] is not None and s['person_greeted_s_ago'] < 300
```
`benchmark/strategic_selector/picker_prompt/harness.py:49`
```python
    if not people and s['new_room_s_ago'] >= threshold and (s['last_look_s_ago'] is None or s['last_look_s_ago'] >= 60):
```
This matches the README's stated reading: greeting recent below 300 s, look recent below 60 s, and exact boundaries allow the action. A visible person always reaches R1–R4, because every bucket is one of the four values. `test_policy` covers both boundaries and checks that the input is not mutated.

**2. Labels never depend on Gemma output.** `label()` reads only situation fields. `generate()` asserts the intended rule, and `run()` re-asserts every label before any call (harness line 235). The saved-results test recomputes all 120 labels.

**3. Set frozen before scoring.**
- `freeze.json` set_sha256 `19773ea0…` matches `set_sha256.txt`.
- It also matches the first event of both the startup-failure log and the final `events.jsonl`, and the hash in every `scored` event.
- The frozen harness hash `e0488d2f…` equals `scored_harness_initial.py`, which the test asserts.
- The freeze time is 16:01:43Z; the first model event is 16:02:55Z.

**4. DEV/TEST split and TEST use.** The split is by seed and stratified: 18 DEV and 12 TEST per mode (72/48). No variant qualified, so there is no `*_TEST.jsonl`, no TEST event, and `summary.test` is null.

**5. V0 matches `picker.py` token for token.**
- V0 state is `picker.situation(...) + '\n' + m2`. This is the same assembly as production:

`picker.py:221`
```python
            dist = self.gemma.decide(request['situation'] + '\n' + text, phrases)
```
- `main.request_pick` passes `picker.situation(...)` and `picker.eligible(person)`.
- The M2 strings match the `relate.context()` format and `TOO_FEW`:

`relate.py:146`
```python
                                   for r in chosen(rows)) or '- none above threshold.')
```
- V0 scoring swaps in `picker.Gemma.prompt_ids`.
- Every DEV prompt asserts static-vs-production id parity, for all variants.
- The fake-transport test also asserts 1 tokenize call (static path) against 4 (production path).
- Request bodies are unchanged, matching adapters `S1O.completion_probs`: n_predict 1, n_probs 1000, temperature 0, cache_prompt false.

**6. Calibration uses only the content-free prompt.**
- All four `calibration.json` states are field-name `N/A` lines plus the fixed V2 rule block. The M2 item lines are removed.
- There is one baseline per eligible list for V4, plus separate reversed-list baselines for V5.
- The test recomputes every V4/V5 divide-and-renormalise result.
- The base is V2: V2 and V3 tie on macro recall at 0.41667, and V2 has fewer median tokens (359.5 vs 386.0).

**7. Selection rule (section 5).**
- `select()` excludes V5 and requires zero R1 violations and every per-mode recall of at least 0.6. Ties go to median tokens, then variant id:

`benchmark/strategic_selector/picker_prompt/harness.py:213`
```python
    qualified = [v for v, m in scores.items() if v != 'V5' and m['r1_violations'] == 0 and min(m['recall'].values()) >= .6]
```
- Every variant from V0 to V4 has R1 violations (4, 4, 4, 6, 6) and a minimum recall below 0.6, so none qualifies.
- `best_macro_recall` is 0.52778 (V4, 38/72). The outcome is NO VARIANT QUALIFIES and TEST stays unscored:

`picker_prompt_results/summary.json:287`
```python
  "selected": null,
```

**8. Summary numbers recompute from the JSONL.** The test recomputes:
- per-variant metrics from all six DEV files;
- every row's choice;
- the calibration transformations;
- the winner and the calibration base;
- `v5_changed` = 15;
- the TEST file set and event order;
- the 72 speed pairs and both medians.

`run.stdout` matches `summary.json`.

**9. Production unchanged.** The diff and manifest contain only the three new files. The manifest's harness hash `44e30a1b…` equals the resumed hash in `events.jsonl` and `timeout_recovery.txt`. The test proves `harness.py` is byte-identical to `scored_harness_resume.py`. I cannot run git, so the claim that no production file changed rests on the Coder's git evidence (stated in recovery_notes) plus the diff and manifest.

**10. Recovery evidence.**
- **Startup failure:** the traceback ends at `/health` with Connection refused, at initial harness lines 234 and 173. No V files and no `/tokenize` or `/completion` call happened.
- **Timeout failure:** the traceback (initial harness line 268, non-V0 branch) is consistent with three saved V3 rows of about 9.3–9.8 s each under the 10 s production limit:

`picker.py:68`
```python
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
```
- **Recovery change:** recovery changed only the module global:

`benchmark/strategic_selector/picker_prompt/harness.py:243`
```python
    picker.TIMEOUT_S = 120
```
`http()` reads the global at call time, so the change is effective, and it affects only the desk process.
- **Why V0 timings were lost:** the initial harness held speed pairs only in memory and wrote them at the end:

`picker_prompt_results/scored_harness_initial.py:314`
```python
    write_json(out / 'speed.json', speed)
```
- **V0 re-runs:** these assert agreement with saved decisions within 1e-6 and append nothing:

`benchmark/strategic_selector/picker_prompt/harness.py:301`
```python
                    assert max(abs(raw[k] - saved[s['id']]['raw_probabilities'][k]) for k in keys) < 1e-6
```
- **Resume guards:** resume refuses if `calibration.json`, any TEST file, or a `selection_frozen` event exists. Events 1–4 contain no selection or TEST event. The AST test proves every module-level definition except `run` and `main` is unchanged.
- **Prefix hashes:** `pre_recovery_results.json` prefix hashes match the final files (V0–V2 are 72 rows each, V3 is 3 rows). As recovery_notes discloses, they are not contemporaneous before-time proof.
- **Server stopped:** `stop.stdout` and the empty `remaining_servers.txt` support that the server was stopped.

**11. Scene count.** `scene_count_evidence.json` lists 242 distinct strings with the same console SHA as `freeze.json`. The only exclusion is `empty room, nothing detected`. `freeze.json` stores the 241 usable lines under a misleading key name, which is disclosed:

`picker_prompt_results/freeze.json:6`
```python
  "distinct_scene_lines": 241,
```

**12. Round-2 prose correction confirmed by Grep on `set.jsonl`:**
- 80 scenes contain a scene-level person tuple.
- No line contains two person tuples, so there are no multi-person scenes.
- 40 scenes have no person.
- No item has an empty `buckets_synthetic`; every source line in the evidence lacks a bucket, so buckets are 100% synthetic.
- `person_replaced_or_removed` is never false; it is constant, as disclosed.

## Findings

1. **Low, reviewer process, not a candidate defect.** The extra read-only `wc -l` command described in the process disclosure. The owner should rule on round validity.
2. **Low, coverage, disclosed.** Two rule clauses are never decisive in the scored set:
   - No item has `last_look_s_ago` below 60, so the look-around clause of R5 is never exercised:

     `benchmark/strategic_selector/picker_prompt/harness.py:101`
     ```python
                 last_look = rng.choice((None, 60, 61, 180))
     ```
   - No MS2 item has a new-room age from 30 to 59 s, so the MS2-specific threshold never changes a label:

     `benchmark/strategic_selector/picker_prompt/harness.py:100`
     ```python
                 new_room = rng.choice((60, 61, 120, 600)) if rule == 'R5' else rng.choice((0, 10, 29))
     ```

   Unit tests cover both clauses; the model was not evaluated on them. This is disclosed in recovery_notes and does not change the outcome.
3. **Low, provenance, disclosed.** There are three gaps:
   - The flag is constant:

     `benchmark/strategic_selector/picker_prompt/harness.py:118`
     ```python
                                          person_replaced_or_removed=True, m2_synthetic=True,
     ```
   - The recovered V0 distributions were not saved, so the parity evidence rests only on the runtime assertion.
   - Only the final `setup.stdout` was kept.
4. **Informational.** V0 decision-row `ms` (server 17892) and the recovered speed pairs (server 18629) come from different server processes, and all timings were taken with an agent resident. None of the timings are measurements. This is disclosed.

**Unsupported claims:** none found. Every claim in the request is either supported by the artifacts or explicitly disclosed as resting only on Coder evidence (git status, the server staying alive between runs, when the prefix manifest was created).

## Files inspected

| File | Lines |
|---|---|
| `docs/HANDOFF.md` | 68 |
| `docs/WORKFLOW.md` | 303 |
| `docs/REVIEWER.md` | 15 |
| `TASK.txt` | 83 |
| `candidate.diff` | 648 |
| `frozen_manifest.json` | 20 |
| `verified_line_counts.txt` | 55 |
| `benchmark/strategic_selector/picker_prompt/harness.py` | 374 |
| `benchmark/strategic_selector/picker_prompt/README.md` | 103 |
| `benchmark/strategic_selector/picker_prompt/test_harness.py` | 153 |
| `picker.py` | 257 |
| `relate.py` | 185 |
| `main.py` | 690 |
| `benchmark/strategic_selector/v3/adapters.py` | 368 |
| `picker_prompt_results/set.jsonl` | 120 |
| `picker_prompt_results/set_sha256.txt` | 1 |
| `picker_prompt_results/freeze.json` | 276 |
| `picker_prompt_results/scene_count_evidence.json` | 252 |
| `picker_prompt_results/recovery_notes.md` | 15 |
| `picker_prompt_results/timeout_recovery.txt` | 7 |
| `picker_prompt_results/pre_recovery_results.json` | 26 |
| `picker_prompt_results/scored_harness_initial.py` | 351 |
| `picker_prompt_results/scored_harness_resume.py` | 374 |
| `picker_prompt_results/events.jsonl` | 14 |
| `picker_prompt_results/summary.json` | 298 |
| `picker_prompt_results/summary.md` | 306 |
| `picker_prompt_results/calibration.json` | 62 |
| `picker_prompt_results/speed.json` | 506 |
| `picker_prompt_results/V3_DEV.jsonl` | 72 |
| `picker_prompt_results/V4_DEV.jsonl` | 72 |
| `picker_prompt_results/V5_DEV.jsonl` | 72 |
| `picker_prompt_results/server_process.txt` | 1 |
| `picker_prompt_results/server_process_resumed.txt` | 1 |
| `picker_prompt_results/setup.stdout` | 8 |
| `picker_prompt_results/run.stdout` | 6 |
| `picker_prompt_results/run.stderr` | 0 |
| `picker_prompt_results/run.exit` | 1 |
| `picker_prompt_results/stop.stdout` | 5 |
| `picker_prompt_results/remaining_servers.txt` | 0 |
| `picker_prompt_results/scoring_timeout/run.stdout` | 3 |
| `picker_prompt_results/scoring_timeout/run.stderr` | 49 |
| `picker_prompt_results/scoring_timeout/run.exit` | 1 |
| `picker_prompt_results/scoring_timeout/stop.stdout` | 5 |
| `picker_prompt_results/startup_failure/run.stderr` | 69 |
| `picker_prompt_results/startup_failure/events.jsonl` | 1 |
| `picker_prompt_results/startup_failure/run.exit` | 1 |
| `picker_prompt_results/startup_failure/stop.stdout` | 5 |

I read `scored_harness_resume.py` only from line 215 to 244; the test's SHA equality proves the whole file is byte-identical to `harness.py`, which I read in full. I read `main.py`, `relate.py` and `adapters.py` in the relevant ranges after searching them. I only searched `set.jsonl`, `V4_DEV.jsonl` and `V5_DEV.jsonl`.

## Conclusion

The candidate harness, frozen set, labels, calibration, selection and recovery are correct and match the task. The no-qualifier outcome is valid and TEST correctly stayed unscored. The remaining findings are already-disclosed limitations, plus my own read-only command deviation, which needs an owner ruling.

VERDICT: APPROVED
````

## Reviewer disclosures

# Recovery and coverage disclosure

Recorded after review round 1; these notes do not change the frozen set, prompts, decisions, probabilities or metrics.

- Coverage: no set item has last_look_s_ago below 60 seconds, and no MS2 item has new_room_s_ago from 30 to 59 seconds. R6 therefore covers only recent-room cases. Unit tests cover these omitted rule boundaries, but the model was not evaluated on them. A follow-up needs a new frozen set; this set was not changed after results.
- Scene counts: scene_count_evidence.json contains all 242 extracted, distinct, non-metadata YOLO scene strings, linked to the unchanged original console SHA-256. freeze.json contains the 241 usable strings. The only excluded string is "empty room, nothing detected". The original complete console was not copied; it remains read-only on the phone. The earlier key distinct_scene_lines in freeze.json means usable lines.
- First startup recovery: the original run failed at its first /health call, before tokenize/completion. The Coder manually created startup_failure/ and moved events.jsonl, run.stdout, run.stderr, run.exit, stop.stdout and remaining_servers.txt into it. The subsequent invocation used the same set and harness bytes, started setup_q4 and scored within one shell session. The failed server exit cause was not instrumented.
- Before changing the timeout/recovery logic, the Coder manually copied the initial harness to scored_harness_initial.py. Its SHA-256 matches the hash frozen before any model call. The frozen set and all saved DEV decision rows were retained.
- The pre_recovery_results.json prefix manifest was created during recovery, while repeating V0 speed pairs and before adding more V3 rows. There is no independent, contemporaneous event/hash proving that creation time or proving those prefixes existed before the timeout. Its hashes prove only that the recorded prefixes match the final files. This is a provenance limitation, not independent before-time evidence.
- Original V0 decision-row timings and the recovered speed pairs came from different server processes (17892 and 18629). Each speed pair ran both paths in the resumed session, production first. All timings are NOT measurements: agent resident.
- No variant qualified. V4 has the best selectable DEV macro recall, 38/72 = 0.5277777777777778. No selected prompt exists, so selected prompt examples are N/A, and TEST stayed unscored.
- Production unchanged was checked by the Coder with git diff --exit-code (empty) and porcelain status showing exactly the three new candidate files. The Reviewer is prohibited from running git; this assertion therefore depends on Coder evidence and the frozen manifest/diff, rather than a Reviewer Git comparison.

- Additional round-2 disclosures: 100% of distance buckets are synthetic. There are 80 person-present scenes with exactly one person each and 40 scenes with no person; no multi-person case is evaluated. The person_replaced_or_removed provenance flag is constant True even when no actual person change was needed; original_objects plus scene provide the actual per-item information. These limitations do not change the no-qualifier outcome.
- Recovered V0 probabilities were checked at runtime but not separately saved. That parity evidence consists of the source assertion and successful run exit, rather than independently recomputable recovered distributions. Only the final setup.stdout was preserved for both scoring server processes.


## UNKNOWN / limitations

Coder model/effort are the owner-assigned profile; the tool interface supplies no independent runtime attestation. Reviewer model/effort are explicitly selected in the recorded CLI command. No claim of real-world prompt accuracy, hardware acceptance, person identity memory, controlled latency, power/heat improvement or deployment suitability. The set is balanced and mostly synthetic; the source scene pool comes from one dry run. No extra prompt variants, model alternatives, dependency, or production wiring. Timing and position-bias evidence are limited to this desk session. Any incomplete or invalid review is recorded above and does not authorize a commit.

Waiting for the human gate. On explicit “yes commit”, stage manifest paths only with message `benchmark: picker prompt desk test harness (PICKPROMPT-1)`. Results remain on the phone; push requires separate “yes push”.
