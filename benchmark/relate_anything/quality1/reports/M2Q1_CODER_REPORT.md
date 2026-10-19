# M2-Q1 Coder Report — RelateAnything M2 Quality & Gemma A/B Desk Test

## Phone-Sized Summary (10 lines max)
- **Status:** COMPLETE (motors off; no tracked files modified; all timings `NOT A MEASUREMENT (agent resident)`).
- **Base commit:** `33c2c78d2f05bac27820d7511588ecb66ccf1f8e` | **Coder model:** `gemini-4-argon-medium` (`Gemini 4 Argon (Medium)`).
- **Unit tests:** 7/7 offline fake tests passed (`test_quality1.py`).
- **Independent review:** Codex `gpt-6.1-sol` (effort `medium`, Ponytail `off`): Round 1 `CHANGES REQUESTED` -> fixed; Round 2 `APPROVED`.
- **Reviewed harness hash:** `bda996b683706bbaffefdfb6ddbf19f51c60a732aa62aecd5e8f6cc8afc6eb7d` (matches full-run `manifest.json`).
- **Photos processed:** 70 total (`38` `bench_photos/`, `32` `test_photos/`): **60** with `>= 2` boxes, **10** with `< 2` boxes, **0** with `> 32` boxes, **0** failed.
- **Determinism (2x runs):** M2 rows identical twice: **60 / 60**; Gemma replies identical twice: **60 / 60** (both variants).
- **PDF outputs (< 20 MB):** `sheet_m2.pdf` (6.27 MB, 61 pages) and `sheet_ab.pdf` (4.34 MB, 60 pages), copied to `/storage/emulated/0/Download/m2q1/`.
- **Blindness:** `key.json` (`random.Random(20261017)`) written only to run folder; absent from PDFs, `results.json`, and this report.
- **Git state:** `git status --porcelain=v1 -uall` shows only new untracked files under `benchmark/relate_anything/quality1/`.

---

## 1. Execution & Reviewer Profiles

```text
EXECUTION PROFILE
ROLE: Coder (harness author and desk-test runner)
PLATFORM: AGY/Antigravity CLI, interactive session in proot Debian, /termux-home/robot
MODEL: gemini-4-argon-medium (Gemini 4 Argon (Medium))
EFFORT: N/A (the "Medium" is part of the AGY model label)
PONYTAIL: lite
WHY: Owner's choice for this task; new benchmark-only files, no robot code touched
FALLBACK: Claude Code CLI, claude-opus-5-5, effort medium, Ponytail lite (Reviewer stays Codex)

REVIEWER PROFILE
ROLE: Reviewer (independent, read-only)
PLATFORM: Codex CLI (`codex exec`), run inside an isolated copy
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: off
WHY: Different model family from the Coder; owner's choice
FALLBACK: none without an owner decision
```

### `agy models` Output
```text
gemini-4-argon-medium     Gemini 4 Argon (Medium)
gemini-4-argon-high       Gemini 4 Argon (High)
gemini-4-argon-low        Gemini 4 Argon (Low)
gemini-3.8-flash-high     Gemini 3.8 Flash (High)
gemini-3.8-flash-medium   Gemini 3.8 Flash (Medium)
gemini-3.8-flash-low      Gemini 3.8 Flash (Low)
claude-opus-5-5-low       Claude Opus 5.5 (Low)
claude-opus-5-5-medium    Claude Opus 5.5 (Medium)
claude-opus-5-5-high      Claude Opus 5.5 (High)
claude-sonnet-5-5-low     Claude Sonnet 5.5 (Low)
claude-sonnet-5-5-medium  Claude Sonnet 5.5 (Medium)
claude-sonnet-5-5-high    Claude Sonnet 5.5 (High)
gpt-oss-120b-medium       GPT-OSS 120B (Medium)
```

- **Base commit (`git rev-parse HEAD`):** `33c2c78d2f05bac27820d7511588ecb66ccf1f8e`
- **Accuracy & timing label:** `Accuracy VALID (temperature 0); timings NOT measurements (agent resident); grading pending (Local AI, blind).`
- **Ponytail lite note:** Reused `relate.Head`, `relate.chosen`, `relate.context`, `detect_person.get_detector`, `detect_person.scene_from_detections`, and `chat.build_prompt` / `chat.get_stop_tokens` directly by import instead of duplicating model or template logic.

---

## 2. Interpreter, Models, and Server Verification

- **Interpreter used:** `/data/data/com.termux/files/usr/bin/python3`
  - Python: `3.13.13` (`sys.platform == "android"`)
  - `onnxruntime`: `1.25.1`
  - `numpy`: `2.4.4`
  - `PIL` (Pillow): `12.2.0`
- **YOLO (`detect_person.get_detector(pinned=False)` at size 640, `CONF=0.35`, `IOU=0.45`):**
  - `/data/data/com.termux/files/home/robot/yolo11s_320.onnx`: `9eab5ebc83d1254e0049abf2a939ef12db696b7e0ef74af7ec8b70cec993f863`
  - `/data/data/com.termux/files/home/robot/yolo11s_640.onnx`: `dc9fe7c50e66ef272db1318f282d6e6dc4ba6adaa0103a2197a87300fbb9dcb5`
- **RelateAnything M2 (`relate.Head`, directory `/data/data/com.termux/files/home/models/relate_anything/relsgg-vits16`):**
  - `relateanything.onnx`: `24a2e2a2ef8ea7d1b7cffef9fe5e35ab9c8eeb413e73eb4dee5afa9b24dd98d5`
  - `relateanything.json`: `e70152faa7875bf9ae865d14b3539b7aac358f33bfb10db265365441db26bfaa`
  - `predicate_bank.npz`: `6043ff8fb765db34d89019617979d566b79967b7740c72bb4007786179839f48`
- **Gemma 4 E2B Q4_0 Server (`127.0.0.1:8080`, `setup_q4`):**
  - `/health`: `{"status": "ok"}`
  - `model_path`: `/data/data/com.termux/files/home/models/gemma-4-E2B-it-Q4_0.gguf`
  - GGUF SHA-256: `8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52`
  - `/props` summary: `model_ftype="Q4_0"`, `n_ctx=2048`, `total_slots=1`, `build_info="b1609-e1a1abb7"`, `bos_token="<bos>"`, `eos_token="<eos>"`, `modalities={"vision": false, "video": false, "audio": false}`.

---

## 3. Code Mirroring (`relate.py` vs `quality1.py`) & Fixed Instruction Constant

### Reproduction of `Relate.relations()` without thread pinning

`relate.py:174-185`:
```python
    def relations(self, image, detections):
        """(picker text, boxes used, relations in the text, inference ms or None). Fewer than 2 boxes: no M2 run.
        More than 32: the 32 highest-confidence boxes, in detector order (copied from phase1.run_cycle's m2())."""
        if len(detections) < 2:
            return TOO_FEW, len(detections), 0, None
        if len(detections) > MAX_BOXES:
            keep = sorted(sorted(range(len(detections)), key=lambda i: -detections[i]['conf'])[:MAX_BOXES])
            detections = [detections[i] for i in keep]
        started = time.perf_counter()
        rows = self.head.infer(image, detections)
        ms = (time.perf_counter() - started) * 1000
        return context(rows), len(detections), len(chosen(rows)), ms
```

`benchmark/relate_anything/quality1/quality1.py:137-171`:
```python
def select_m2_detections(detections):
    """Reproduce Relate.relations() >32 rule: keep 32 highest-confidence boxes in detector order."""
    if len(detections) > relate.MAX_BOXES:
        keep = sorted(sorted(range(len(detections)), key=lambda i: -detections[i]['conf'])[:relate.MAX_BOXES])
        detections = [detections[i] for i in keep]
    return detections


def run_m2_once(head, image, detections):
    """Reproduce Relate.relations() without thread pinning."""
    if len(detections) < 2:
        return {
            'ran': False,
            'boxes_used': len(detections),
            'used_detections': list(detections),
            'rows': [],
            'shown_rows': [],
            'shown_text': relate.TOO_FEW,
            'infer_ms_not_a_measurement': None,
            'timing_label': TIMING_LABEL,
        }
    used = select_m2_detections(detections)
    started = time.perf_counter()
    rows = head.infer(image, used)
    ms = (time.perf_counter() - started) * 1000.0
    return {
        'ran': True,
        'boxes_used': len(used),
        'used_detections': used,
        'rows': rows,
        'shown_rows': relate.chosen(rows),
        'shown_text': relate.context(rows),
        'infer_ms_not_a_measurement': ms,
        'timing_label': TIMING_LABEL,
    }
```

### Fixed Instruction Constant (`quality1.py:61-64`)
```python
INSTRUCTION = (
    "You are the robot's eyes. Using only the information below, describe in one or two short sentences "
    "what is in front of the robot. Do not mention anything that is not listed."
)
```

---

## 4. Files Created & SHA-256 Digests

### Harness, Tests, Documentation, and Run Artifacts
| File | SHA-256 | Bytes |
|---|---|---:|
| `benchmark/relate_anything/quality1/quality1.py` | `bda996b683706bbaffefdfb6ddbf19f51c60a732aa62aecd5e8f6cc8afc6eb7d` | 27,765 |
| `benchmark/relate_anything/quality1/test_quality1.py` | `43ea5c7d7c4665ff3e3e415dae007f7261623a40d123d8fca2097d67c40addfa` | 13,009 |
| `benchmark/relate_anything/quality1/README.md` | `8682306b929e863a462710ce3dc3ed7ba9bf953f420d54f33751d3d0a5eae69d` | 2,622 |
| `benchmark/relate_anything/quality1/runs/20261017T212311Z/manifest.json` (smoke) | `77fc619a4c6a0e84d2f828b099c92907db74aff9674e76366995a2cd12fb0bd9` | 16,188 |
| `benchmark/relate_anything/quality1/runs/20261017T212311Z/results.json` (smoke) | `90cdcfac4e954df932588c123ccc728d2db86f6170e7e54455316199dce873bb` | 43,186 |
| `benchmark/relate_anything/quality1/runs/20261017T212311Z/key.json` (smoke) | `e3b5d52902912df873718749aafe24f0f8329c41a2774db8b7a2b52257d7bd4f` | 225 |
| `benchmark/relate_anything/quality1/runs/20261017T212311Z/sheet_m2.pdf` (smoke) | `d930207de9e75cb0a8e8187ebc2197b3e115f88413e564b9a42bfa1d797fc7e3` | 389,710 |
| `benchmark/relate_anything/quality1/runs/20261017T212311Z/sheet_ab.pdf` (smoke) | `17d934ff7dd777cc53645d412699de9e475571806bc649f4e878e79d77a31bb7` | 238,754 |
| `benchmark/relate_anything/quality1/runs/20261017T212311Z/run.log` (smoke) | `358133507fa19048e6cfe5be33f71f3107bf5c310e3d4db73f994cac0f2df9c1` | 1,140 |
| `benchmark/relate_anything/quality1/runs/20261017T213604Z/manifest.json` (full) | `235e19a4a213c1db0d238c6f3029c4ff5a676774f279c7707166dd16258e2346` | 44,521 |
| `benchmark/relate_anything/quality1/runs/20261017T213604Z/results.json` (full) | `924fcc10ad7c6249c2baac1c8fd4f63bd5749bbf081dba30274be0101808be9f` | 558,579 |
| `benchmark/relate_anything/quality1/runs/20261017T213604Z/key.json` (full) | `4ee960baccd0eedc86e32a73412c2064fcfb04be89af4bb091617b0421dc8eea` | 4,777 |
| `benchmark/relate_anything/quality1/runs/20261017T213604Z/sheet_m2.pdf` (full) | `3445940770c59a7cf899727326b592b201fcc69eeb157658f193f42cc8229585` | 6,267,115 |
| `benchmark/relate_anything/quality1/runs/20261017T213604Z/sheet_ab.pdf` (full) | `bfb0fcede7f2c508f06191754da2310730dc4fb7bd6886e7fe31288da91399ed` | 4,339,568 |
| `benchmark/relate_anything/quality1/runs/20261017T213604Z/run.log` (full) | `5558f8eac0d6d80cf02014996d348c370ef189d1b8007c20ff15a290b0e0a19a` | 10,509 |

### Tracked Imported Files (Unchanged)
| File | SHA-256 |
|---|---|
| `relate.py` | `0f4790bd9f8822f2131adcafa6813983a2b1d5b8a76abaaea074a44ea258ad20` |
| `detect_person.py` | `2420d70b11996f51a78e589dcc112e1cc25a33f3dcf6f1881ad3c791ce098b21` |
| `picker.py` | `b152947be98f8bde83ad0d6e34118c999f90b59025308cef2f3ebc3cf745df66` |
| `chat.py` | `b1ef174ef38d6efcef14faff5908e5fc3116f866ec0ea698dddaa5c9ef4d2ced` |
| `server_manager.py` | `4f63062a4dbc842538b72d1c28a8b2a721a0664a1a2bed8d2936e712984d84aa` |

---

## 5. Offline Unit Tests Run & Output

Command:
```bash
/data/data/com.termux/files/usr/bin/python3 -B benchmark/relate_anything/quality1/test_quality1.py -v
```

Output:
```text
test_2_to_32_and_over_32_rule_matches_relate (__main__.Quality1Test.test_2_to_32_and_over_32_rule_matches_relate) ... ok
test_module_origins_and_provenance (__main__.Quality1Test.test_module_origins_and_provenance) ... ok
test_pdf_builders_and_key_never_leaked_into_sheets (__main__.Quality1Test.test_pdf_builders_and_key_never_leaked_into_sheets) ... ok
test_prepare_frame_and_server_check_fakes (__main__.Quality1Test.test_prepare_frame_and_server_check_fakes) ... ok
test_prompts_call_gemma_and_strip_tokens (__main__.Quality1Test.test_prompts_call_gemma_and_strip_tokens) ... ok
test_too_few_handling (__main__.Quality1Test.test_too_few_handling) ... ok
test_xy_assignment_reproducible_and_order_independent (__main__.Quality1Test.test_xy_assignment_reproducible_and_order_independent) ... ok

----------------------------------------------------------------------
Ran 7 tests in 3.107s

OK
```

---

## 6. Full Run Results Summary (`runs/20261017T213604Z`)

- **Start UTC:** `2026-10-17T21:36:04Z`
- **End UTC:** `2026-10-17T22:11:44Z`
- **Elapsed (`NOT A MEASUREMENT (agent resident)`):** `2139.63 s`
- **Photo counts:**
  - **Total photos discovered & processed:** `70` (`38` in `bench_photos/`, `32` in `test_photos/`)
  - **Photos with `>= 2` boxes (M2 + Gemma A/B run):** `60`
  - **Photos with `< 2` boxes (skipped M2 + Gemma, listed on summary page of `sheet_m2.pdf`):** `10`
    - `bench_photos/empty_1.jpg` (0 boxes)
    - `bench_photos/empty_3.jpg` (1 box: `chair`)
    - `bench_photos/empty_5.jpg` (1 box: `potted plant`)
    - `bench_photos/roomsig_bed_1.jpg` (1 box: `bed`)
    - `bench_photos/roomsig_toilet_3.jpg` (1 box: `toilet`)
    - `bench_photos/roomsig_tv_1.jpg` (1 box: `tv`)
    - `bench_photos/roomsig_tv_2.jpg` (1 box: `tv`)
    - `test_photos/PXL_20260405_213536608.jpg` (1 box: `refrigerator`)
    - `test_photos/PXL_20260405_213539353.MP.jpg` (1 box: `toilet`)
    - `test_photos/empty_5.jpg` (1 box: `potted plant`)
  - **Photos with `> 32` boxes:** `0` (max detections on a photo was 17 on `bench_photos/blocked_1.jpg` and `test_photos/table_top.jpg`)
- **M2 identical-twice count:** `60 / 60` (`100%` identical indices, predicates, scores, thresholds, and pass flags across both runs)
- **Gemma identical-twice count:**
  - Reply `X` identical twice: `60 / 60`
  - Reply `Y` identical twice: `60 / 60`
  - Both variants identical twice: `60 / 60` (`100%`)
- **Failed photos:** `0` (none failed)

### PDF Paths
- Repo full-run paths:
  - `/termux-home/robot/benchmark/relate_anything/quality1/runs/20261017T213604Z/sheet_m2.pdf` (`6,267,115` bytes, SHA-256 `3445940770c59a7cf899727326b592b201fcc69eeb157658f193f42cc8229585`)
  - `/termux-home/robot/benchmark/relate_anything/quality1/runs/20261017T213604Z/sheet_ab.pdf` (`4,339,568` bytes, SHA-256 `bfb0fcede7f2c508f06191754da2310730dc4fb7bd6886e7fe31288da91399ed`)
- Phone Download copies (for attaching to Local AI chat):
  - `/storage/emulated/0/Download/m2q1/sheet_m2.pdf` (`6,267,115` bytes, SHA-256 `3445940770c59a7cf899727326b592b201fcc69eeb157658f193f42cc8229585`)
  - `/storage/emulated/0/Download/m2q1/sheet_ab.pdf` (`4,339,568` bytes, SHA-256 `bfb0fcede7f2c508f06191754da2310730dc4fb7bd6886e7fe31288da91399ed`)

---

## 7. Independent Review Record (`codex exec`)

### `codex exec --help` Available Flags Recorded
`codex-cli 0.163.0`:
- `-c, --config <key=value>`
- `--enable <FEATURE>` / `--disable <FEATURE>`
- `--strict-config`
- `-i, --image <FILE>...`
- `-m, --model <MODEL>`
- `--oss` / `--local-provider <OSS_PROVIDER>`
- `-p, --profile <CONFIG_PROFILE_V2>`
- `-s, --sandbox <SANDBOX_MODE>` (`[possible values: read-only, workspace-write, danger-full-access]`)
- `-C, --cd <DIR>`
- `--worktree`
- `--add-dir <DIR>`
- `--skip-git-repo-check`
- `--ephemeral`
- `--ignore-user-config` / `--ignore-rules`
- `--output-schema <FILE>`
- `--color <COLOR>`
- `--json`
- `-o, --output-last-message <FILE>`

Invocation used inside each isolated copy (`round1_copy`, `round2_copy`):
```bash
codex exec -m gpt-6.1-sol -c model_reasoning_effort="medium" --sandbox read-only --skip-git-repo-check --ephemeral "<request>"
```

### Review Rounds Summary & Pre/Post Integrity Audits
- **Round 1:**
  - Pre/Post audit (`/termux-home/review/m2q1/round1_audit.json`): `{"head_changed": false, "git_status_changed": false, "copy_hashes_changed": false, "live_hashes_changed": false, "returncode": 0}`
  - Citations verified: 100% exact against `round1_copy`.
  - Verdict: `VERDICT: CHANGES REQUESTED` (1. `chat.py` prepended `/data/data/com.termux/files/home/robot` to `sys.path` before other imports; 2. PDF blindness test checked raw bytes rather than rendered text strings; 3. trailing `<` fragment on truncated turn tag).
  - Resolution: Fixed import order and `sys.path` pinning in `quality1.py`, added `verify_and_hash_imports()` + `test_module_origins_and_provenance`, captured `ImageDraw.ImageDraw.text` strings in `test_pdf_builders_and_key_never_leaked_into_sheets`, and extended `TEMPLATE_TOKEN_RE` for trailing `<` fragments.
- **Round 2:**
  - Pre/Post audit (`/termux-home/review/m2q1/round2_audit.json`): `{"head_changed": false, "git_status_changed": false, "copy_hashes_changed": false, "live_hashes_changed": false, "returncode": 0}`
  - Citations verified: 100% exact against `round2_copy` (`quality1.py` 695 lines, `test_quality1.py` 302 lines, `README.md` 48 lines; all 5 code block citations verified line-for-line).
  - Verdict: `VERDICT: APPROVED`
  - Separate logs preserved at:
    - `/termux-home/review/m2q1/round1_stdout.txt` & `/termux-home/review/m2q1/round1_stderr.txt`
    - `/termux-home/review/m2q1/round2_stdout.txt` & `/termux-home/review/m2q1/round2_stderr.txt`

### Verbatim Round 1 Review (`round1_stdout.txt`)
```text
Review completed read-only. Candidate line counts:

| File | Lines |
|---|---:|
| `benchmark/relate_anything/quality1/quality1.py` | 668 |
| `benchmark/relate_anything/quality1/test_quality1.py` | 247 |
| `benchmark/relate_anything/quality1/README.md` | 48 |

**1. Must fix: imports can escape the reviewed checkout and invalidate provenance.**

`benchmark/relate_anything/quality1/quality1.py:30-33`:
```python
from chat import build_prompt, get_stop_tokens
import detect_person
from picker import URL
import relate
```

Importing `chat` executes `chat.py:3`:
```python
sys.path.insert(0, '/data/data/com.termux/files/home/robot')
```

Consequently, when that live directory is accessible, the subsequent imports can load `detect_person`, `picker`, and `relate` from the live repository instead of the isolated copy. The transitive `server_manager` import can escape too.

Meanwhile, `benchmark/relate_anything/quality1/quality1.py:614` hashes files relative to the harness checkout:
```python
        'code_sha256': {rel_p: relate.sha256(ROBOT / rel_p) for rel_p in imported_files},
```

This can record hashes for different code than actually executed. It also prevents the isolated tests from reliably checking the frozen reference implementation.

Fix this within the new candidate files: ensure the imported project modules resolve to the intended checkout despite `chat`’s path mutation, and verify/hash their actual source paths. Add a regression check for module origins without editing tracked files.

**2. Test weakness: PDF byte searches do not verify visible blindness.**

`benchmark/relate_anything/quality1/test_quality1.py:204-206`:
```python
            for forbidden in (b'Y-ONLY', b'Y+M2', b'key.json'):
                self.assertNotIn(forbidden, m2_bytes)
                self.assertNotIn(forbidden, ab_bytes)
```

PIL renders the page text into pixels. A visibly printed variant label need not appear as an ASCII string in the PDF bytes, so these assertions can pass despite a leak. Capture drawing calls and inspect their text, or otherwise test the rendered content.

The requested checks otherwise gave these results:

- **M2 correctness:** The inspected source constructs imported `relate.Head`, never `relate.Relate`. The reference Head checks all three SHA-256 hashes before loading. The candidate matches the reference’s fewer-than-two handling, highest-confidence 32-box selection with detector order preserved, and use of `relate.context`/`relate.chosen`. Qualifying photos receive two M2 calls; comparisons include indices, predicates, scores, thresholds, and pass flags. These conclusions are subject to finding 1’s import-resolution problem.
- **Blindness:** Assignment uses `random.Random(20261017)`, one choice per qualifying photo in sorted order. The explicit X/Y mapping is written only to `key.json`. The inspected PDF builders do not draw the mapping or variant names. The A/B builder draws the annotated frame and X/Y replies without adding an object list or M2 block. Supplied results’ replies contain no variant labels. Both prompts are retained separately, as requested.
- **Tracked files:** The supplied patch contains only the three new candidate files. Six current source hashes match the smoke manifest. This copy has no `.git`, so I cannot independently establish that every protected tracked file is byte-identical to the stated base.
- **Hardware/server safety:** No candidate motor, mission, teleop, USB, server-launch, or server-kill path was found. Importing the inspected chat/server modules does not launch their command-line entry points. Initial server checks stop with owner instructions when unavailable or the model basename differs.
- **Privacy:** No secrets, emails, or account names were found in the candidate files or supplied smoke text/JSON.
- **Timing:** Harness durations carry the exact required label, `NOT A MEASUREMENT (agent resident)`. No candidate heat or power claims were found.
- **Evidence:** Supplied output reports six passing tests and three qualifying smoke photos, all with identical repeated M2 rows and Gemma replies. One saved reply retains a trailing `<` fragment; template cleanup deserves a regression case.

I did not rerun the file-writing tests or harness, inspect the absent PDFs, or establish base-file identity from Git. No files were created, modified, or deleted; the import/provenance issue remains unresolved.

VERDICT: CHANGES REQUESTED
```

### Verbatim Round 2 Final Review (`round2_stdout.txt`)
```text
No blocking correctness or safety findings. The three Round 1 fixes are present.

Candidate files reviewed in full:

| File | Line count |
|---|---:|
| `benchmark/relate_anything/quality1/quality1.py` | 695 |
| `benchmark/relate_anything/quality1/test_quality1.py` | 302 |
| `benchmark/relate_anything/quality1/README.md` | 48 |

1. **M2 correctness:** The harness uses the imported `relate.Head`, preserving its three-file SHA-256 check, and never constructs `relate.Relate` or calls its pinning methods. The fewer-than-two branch skips inference and returns `relate.TOO_FEW`. Selection above 32 boxes matches the reference exactly.

   `benchmark/relate_anything/quality1/quality1.py:139-141`:
   ```python
       if len(detections) > relate.MAX_BOXES:
           keep = sorted(sorted(range(len(detections)), key=lambda i: -detections[i]['conf'])[:relate.MAX_BOXES])
           detections = [detections[i] for i in keep]
   ```

   `relate.py:179-181`:
   ```python
           if len(detections) > MAX_BOXES:
               keep = sorted(sorted(range(len(detections)), key=lambda i: -detections[i]['conf'])[:MAX_BOXES])
               detections = [detections[i] for i in keep]
   ```

   Output uses `relate.chosen()` and `relate.context()`. Both M2 executions are compared, including indices, predicates, scores, thresholds and pass flags; differing second-run rows are retained.

   `benchmark/relate_anything/quality1/quality1.py:532-534`:
   ```python
           m2_1 = run_m2_once(head, decoded, detections)
           m2_2 = run_m2_once(head, decoded, detections)
           m2_same = rows_identical(m2_1['rows'], m2_2['rows'])
   ```

2. **Blindness:** The mapping uses seed `20261017`, one coin for each qualifying photo in sorted order. Results contain the required variant prompts and X/Y replies, without serializing their mapping. Neither PDF builder receives the key. The strengthened test captures rendered text and checks the complete A/B text sequence.

   `benchmark/relate_anything/quality1/quality1.py:270-275`:
   ```python
       rng = random.Random(seed)
       key = {}
       for photo in sorted(qualifying_photos):
           first = rng.choice(('Y-ONLY', 'Y+M2'))
           second = 'Y+M2' if first == 'Y-ONLY' else 'Y-ONLY'
           key[photo] = {'X': first, 'Y': second}
   ```

3. **Tracked files:** The supplied diff contains only the three new candidate files. The supplied Git proof reports an empty tracked diff. I independently hashed all seven reference files; every hash matches that proof. The candidate harness hash also matches the smoke manifest. Live Git state could not be independently rechecked here.

4. **Motor/server safety:** No candidate code opens USB, imports motor control, starts or kills servers, or invokes the imported server-control functions. Network operations check the existing localhost server and request completions.

5. **Privacy:** No secrets, email addresses or account identifiers were found in the candidate text or supplied smoke outputs. This was a textual review; generated PDF/image artifacts were not visually inspected.

6. **Timing and claims:** Recorded inference, completion and elapsed timings carry the required label. No candidate heat or power claims were found.

   `benchmark/relate_anything/quality1/quality1.py:54`:
   ```python
   TIMING_LABEL = 'NOT A MEASUREMENT (agent resident)'
   ```

The supplied evidence reports seven passing tests in both checkouts and three successful smoke photos, with identical repeated M2 rows and Gemma replies. The truncated-token regression is present. I did not rerun tests: they create temporary files, and interpreter execution was rejected here. No files were changed. Full-dataset execution and blind grading remain pending.

VERDICT: APPROVED
```

---

## 8. Git Status Confirmation

`git -C /termux-home/robot diff HEAD` is empty (0 bytes).
`git -C /termux-home/robot status --porcelain=v1 -uall` shows only new untracked files under `benchmark/relate_anything/quality1/`:

```text
?? benchmark/relate_anything/quality1/README.md
?? benchmark/relate_anything/quality1/quality1.py
?? benchmark/relate_anything/quality1/runs/20261017T212311Z/frames/bench_photos__blocked_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T212311Z/frames/bench_photos__blocked_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T212311Z/frames/bench_photos__blocked_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T212311Z/key.json
?? benchmark/relate_anything/quality1/runs/20261017T212311Z/manifest.json
?? benchmark/relate_anything/quality1/runs/20261017T212311Z/results.json
?? benchmark/relate_anything/quality1/runs/20261017T212311Z/sheet_ab.pdf
?? benchmark/relate_anything/quality1/runs/20261017T212311Z/sheet_m2.pdf
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/ (70 JPEGs)
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/key.json
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/manifest.json
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/results.json
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/sheet_ab.pdf
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/sheet_m2.pdf
?? benchmark/relate_anything/quality1/test_quality1.py
```
