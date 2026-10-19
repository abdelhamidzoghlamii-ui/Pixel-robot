1. TPL-FIX v2 implemented; live chat regressed.
2. Gemma builders/stops corrected; Qwen/Mistral unchanged.
3. Used three live b1609 captures, including native thinking prelude.
4. Voice parsed OLD 15/15, NEW 15/15; same actions 14/15; no leaks.
5. "Blue banana seven": OLD say "[]"; NEW find banana.
6. Chat leak items: OLD 0/5, NEW 5/5.
7. Broken answers: OLD 1/5, NEW 4/5; NEW four hit the 128-token cap.
8. Tests 178 -> 183 (root 123 -> 128), two skips; nav same; safety PASS.
9. AGY round 3 APPROVED; rounds 1–2 INVALID citation checks.
10. Server unchanged, settings restored; no USB/motors/commit/push.

## Scope, provenance and authorization

Assigned Coder: Codex CLI, gpt-6.1-sol, effort medium, Ponytail lite. Assigned and log-verified Reviewer: AGY headless, gemini-3.8-flash-high, effort N/A, Ponytail off. No fallback used.

Base commit: `33c2c78d2f05bac27820d7511588ecb66ccf1f8e`. Initial check found no modified tracked files and neither TPLFIX directory existed; unrelated untracked quality1 files were retained untouched. Owner later confirmed readiness and approved both the single stale archive-test assertion update and canonical omission of an empty `permissions.ask` key in the temporary overlay.

Only tracked changes are chat.py, main.py and the approved test assertion. Qwen/Mistral branches, server flags, PARSE_SYS, generation parameters/timeouts, JSON handling, warm_up, navigation, picker, detector, heat control, builds and model files are unchanged. The AST contract check and root regressions verify this.

Runtime scope search found no callers/imports of configs/setup_b.py, setup_c.py or setup_d.py. Their old builders remain untouched. Other root old-marker senders are historical diagnostics: full_benchmark.py, nav_logic_test.py, thermal_real.py, swa_benchmark.py, benchmark_nav.py, system_benchmark.py, quality_benchmark.py, benchmark_lora.py, benchmark_compare.py, json_benchmark.py, thermal_benchmark3.py, quality_benchmark2.py, mission_test.py, thermal_benchmark2.py and thermal_benchmark.py. The new comparison script mentions old tokens only in leak detection/self-check and intentionally loads OLD base code. Other benchmark archives remain unchanged.

## Native template captures

The robot b1609 endpoint is available; the VIS0 fallback was not needed. POST /apply-template with add_generation_prompt=true and the three requested message lists produced the exact small fixture below. Each prompt includes native system plus `<|think|>`; there is no synthetic Understood turn and no empty `<|channel>thought` generation prefill. The template Jinja fallback default(false) differs from the live server-supplied thinking option. This difference from VIS0 explains why token-correct native prompts can still expose thought channels.

```json
{
  "source": "b1609-e1a1abb7 POST /apply-template, default template options",
  "model": "gemma-4-E2B-it-Q4_0.gguf",
  "cases": [
    {
      "request": {
        "messages": [
          {
            "role": "user",
            "content": "Hi"
          }
        ],
        "add_generation_prompt": true
      },
      "prompt": "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
    },
    {
      "request": {
        "messages": [
          {
            "role": "system",
            "content": "S"
          },
          {
            "role": "user",
            "content": "Hi"
          }
        ],
        "add_generation_prompt": true
      },
      "prompt": "<|turn>system\n<|think|>\nS<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
    },
    {
      "request": {
        "messages": [
          {
            "role": "user",
            "content": "Hi"
          },
          {
            "role": "assistant",
            "content": "Yo"
          },
          {
            "role": "user",
            "content": "Again"
          }
        ],
        "add_generation_prompt": true
      },
      "prompt": "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nYo<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n"
    }
  ]
}
```

## Live comparison and limitations

Run: `benchmark/tplfix/runs/20261018T182009Z`. The reviewed, hash-frozen compare.py imported OLD main.py/chat.py from an external temporary copy of the base commit and NEW from the working tree. Voice used the exact 15 #114 commands and unchanged parser temperature/timeout/prefill. Chat used five fixed English prompts, temperature 0, n_predict 128. Raw replies, actions, syntactic JSON-array parse success, token markers, hashes and server props are in results.json/manifest.json. Every request returned a raw reply; no timeout occurred. Parse success is syntactic, not a semantic correctness grade.

# TPL-FIX OLD vs NEW

NOT A MEASUREMENT (agent resident)

| Voice command | OLD parsed | NEW parsed | Same actions | OLD leaks | NEW leaks |
|---|---|---|---|---|---|
| Find Chiara and tell her the pizza is here | True | True | True | 0 | 0 |
| Go to the bathroom | True | True | True | 0 | 0 |
| Patrol the whole apartment | True | True | True | 0 | 0 |
| Find my phone in the living room | True | True | True | 0 | 0 |
| Come back | True | True | True | 0 | 0 |
| Tell Abdel the meeting starts in ten minutes | True | True | True | 0 | 0 |
| Go to the kitchen and say lunch is ready | True | True | True | 0 | 0 |
| Check the bedroom and the hallway | True | True | True | 0 | 0 |
| Where are my glasses | True | True | True | 0 | 0 |
| Say good night | True | True | True | 0 | 0 |
| Find someone and tell them the door is open | True | True | True | 0 | 0 |
| Go to the living room then come back | True | True | True | 0 | 0 |
| Look for the cat in the bedroom | True | True | True | 0 | 0 |
| Tell Chiara I love her | True | True | True | 0 | 0 |
| Blue banana seven | True | True | False | 0 | 0 |

| Chat prompt | OLD leaks | NEW leaks | OLD repeated | NEW repeated |
|---|---|---|---|---|
| Hello! Introduce yourself in one sentence. | 0 | 1 | False | False |
| What is two plus three? Answer briefly. | 0 | 2 | False | False |
| Name three common household objects. | 0 | 1 | False | False |
| Explain why a robot should stop near an obstacle in one sentence. | 0 | 1 | False | False |
| Write a friendly good morning greeting. | 0 | 1 | False | False |

Repeated = an exact eight-word span repeated without overlap; read raw replies for quality.

### Manual broken-answer assessment

Broken means the requested answer was not produced within 128 tokens, including non-answer repetition or a thought-only response truncated at the cap. Leakage is counted separately; the NEW arithmetic reply contains the correct final answer 5 while still exposing channels. The script's exact eight-word repetition heuristic is false for all ten replies; it does not catch the OLD three-object reply repeating a four-word generic phrase. The raw replies establish that failure.

| Item | OLD broken | NEW broken | Observed result |
|---|---|---|---|

| 1 | False | True | OLD: Requested answer present. NEW: 128-token limit reached in exposed thought text; no final requested answer. |
| 2 | False | False | OLD: Requested answer present. NEW: Correct final answer 5 is present, with exposed thought channels. |
| 3 | True | True | OLD: Repeated generic phrase; does not name three objects. NEW: 128-token limit reached in exposed thought text; no final requested answer. |
| 4 | False | True | OLD: Requested answer present. NEW: 128-token limit reached in exposed thought text; no final requested answer. |
| 5 | False | True | OLD: Requested answer present. NEW: 128-token limit reached in exposed thought text; no final requested answer. |

Only differing voice item:
```json
{
  "command": "Blue banana seven",
  "OLD": {
    "raw_reply": "{\"type\":\"say\",\"message\":\"[]\"}]",
    "actions": [
      {
        "type": "say",
        "message": "[]"
      }
    ],
    "parse_succeeded": true,
    "leaks": []
  },
  "NEW": {
    "raw_reply": "{\"type\":\"find_object\",\"object\":\"banana\",\"room\":\"\"}]",
    "actions": [
      {
        "type": "find_object",
        "object": "banana",
        "room": ""
      }
    ],
    "parse_succeeded": true,
    "leaks": []
  },
  "same_actions": false
}
```

Live results expose a chat regression despite exact requested template matching and code-review approval: NEW clean chat replies are 0/5, with 4/5 lacking a final requested answer. The thought-enabled native prelude drives the additional output. The invalid/nonsense voice case also changes interpretation. No prompt/policy adjustment was made after review; this report records the requested bounded change and observed results. The independent verdict covered code and offline checks before these live results.

Robot server executable and PID were verified before/after; `/props` summaries are identical. The server remains running.

```json
{
  "build_info": "b1609-e1a1abb7",
  "model": "gemma-4-E2B-it-Q4_0.gguf",
  "total_slots": 1,
  "chat_template_sha256": "603a42db292c25278e9b23d94c7abbe77453f13e26a053c43c5c2f900b4136f7"
}
```

Context: 2048. Process evidence: server_process_before.json and server_process_after.json in this report directory.

## Tracked diff

```diff
diff --git a/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py b/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
index 13d0f0d..79aee09 100644
--- a/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
+++ b/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
@@ -46,8 +46,9 @@ class CommandPayloadTest(unittest.TestCase):
         p = cm.payload('How far can you see?')
         self.assertEqual(p, chat.chat_payload(chat.build_prompt('gemma', [{'role': 'user', 'content': 'How far can you see?'}]),
                                               'gemma'))
-        self.assertEqual(p, {'prompt': '<start_of_turn>user\nHow far can you see?<end_of_turn>\n<start_of_turn>model\n',
-                             'n_predict': 512, 'temperature': 0.7, 'stop': ['<end_of_turn>'], 'stream': False})
+        self.assertEqual(p, {'prompt': '<|turn>system\n<|think|>\n<turn|>\n'
+                                      '<|turn>user\nHow far can you see?<turn|>\n<|turn>model\n',
+                             'n_predict': 512, 'temperature': 0.7, 'stop': ['<turn|>'], 'stream': False})
         with mock.patch.object(chat, 'chat_payload', return_value='P') as f:
             self.assertEqual(cm.payload('x'), 'P')
         f.assert_called_once()
diff --git a/chat.py b/chat.py
index b712e04..ce64ff3 100644
--- a/chat.py
+++ b/chat.py
@@ -163,16 +163,19 @@ def server_running():
 def build_prompt(model_type, messages):
     if model_type == 'gemma':
         prompt = ''
+        # b1609 /apply-template defaults to a native system turn with thinking enabled.
+        if not messages or messages[0]['role'] != 'system':
+            prompt = '<|turn>system\n<|think|>\n<turn|>\n'
         for msg in messages:
             role = msg['role']
-            content = msg['content']
+            content = msg['content'].strip()
             if role == 'system':
-                prompt += f'<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n'
+                prompt += f'<|turn>system\n<|think|>\n{content}<turn|>\n'
             elif role == 'user':
-                prompt += f'<start_of_turn>user\n{content}<end_of_turn>\n'
+                prompt += f'<|turn>user\n{content}<turn|>\n'
             elif role == 'assistant':
-                prompt += f'<start_of_turn>model\n{content}<end_of_turn>\n'
-        prompt += '<start_of_turn>model\n'
+                prompt += f'<|turn>model\n{content}<turn|>\n'
+        prompt += '<|turn>model\n'
         return prompt
 
     elif model_type == 'qwen':
@@ -204,7 +207,7 @@ def build_prompt(model_type, messages):
 
 def get_stop_tokens(model_type):
     stops = {
-        'gemma':   ['<end_of_turn>'],
+        'gemma':   ['<turn|>'],
         'qwen':    ['<|im_end|>'],
         'mistral': ['</s>', '[INST]'],
     }
diff --git a/main.py b/main.py
index 5c5799f..a03e066 100644
--- a/main.py
+++ b/main.py
@@ -127,14 +127,14 @@ EXAMPLES:
 
 def parse_command(text, timeout=40):
     prompt = (
-        '<start_of_turn>user\n' + PARSE_SYS +
+        '<|turn>user\n' + PARSE_SYS +
         '\nCommand: "' + text + '"'
-        '<end_of_turn>\n<start_of_turn>model\n[' 
+        '<turn|>\n<|turn>model\n['
     )
     try:
         resp = requests.post(GEMMA_URL, json={
             'prompt': prompt, 'n_predict': 300,
-            'temperature': 0.05, 'stop': ['<end_of_turn>', '\n\n']
+            'temperature': 0.05, 'stop': ['<turn|>', '\n\n']
         }, timeout=timeout)
         raw = resp.json()['content']
         # Model starts after [ which we injected
```

## Candidate and new repository artifact hashes

The three new candidate files are test_tplfix.py, tests/gemma4_apply_template.json and benchmark/tplfix/compare.py. The three new run files are results.json, manifest.json and results.md. The comparison directory is ignored by the existing benchmark/* rule, and was explicitly included in the full review diff/copy. No ignore rule was changed.

| File | SHA-256 |
|---|---|

| `chat.py` | `e191f59db7ce650a9389f07e89d43319374d1d154a6c4e29834dce4b2334c839` |
| `main.py` | `62ef3a80789852fec4206773634db93e0ce8caaa91fa7de483539b86c780674c` |
| `test_tplfix.py` | `2f245763f6ced77e3edae6ffbc532cf9d7c7f48c0757016861c8d99f732b2026` |
| `tests/gemma4_apply_template.json` | `584791241afdd96b6e27720ad4bc450ad10f6e419a1d9b071f2e4cfed0ab6d17` |
| `benchmark/tplfix/compare.py` | `89653c841c82b66fb69a47f69839de2e8d86e719d95546d901311e809d99e588` |
| `benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py` | `cd1ed09e1de73e6d44087a02e6dd17fc40571abb4399fdf9cb757887762ac1b2` |
| `benchmark/tplfix/runs/20261018T182009Z/manifest.json` | `d2101a2c66cf92dd69671fe37d92b00745829cf16c0e4f7969cec84130f67f60` |
| `benchmark/tplfix/runs/20261018T182009Z/results.json` | `19a228b166bb54db99dd33a680e269d07858228ddbe19c1046037fc34c4212bc` |
| `benchmark/tplfix/runs/20261018T182009Z/results.md` | `371a606c621e90d4ee7e29d259290ee84316aed36da13d903af04d41b37d7d94` |

Downloaded table: `/storage/emulated/0/Download/tplfix/results.md`; its SHA-256 equals the run results.md hash.

## Search evidence

Full-repo final command (including ignored benchmark sources): `rg -l --hidden --no-ignore --glob '!.git/**' --glob '!**/__pycache__/**' --glob '!*.pyc' '<start_of_turn>' .`

```text
./thermal_benchmark2.py
./thermal_benchmark.py
./thermal_real.py
./swa_benchmark.py
./mission_test.py
./system_benchmark.py
./full_benchmark.py
./nav_logic_test.py
./benchmark_nav.py
./benchmark_lora.py
./benchmark_compare.py
./json_benchmark.py
./quality_benchmark.py
./quality_benchmark2.py
./thermal_benchmark3.py
./benchmark/tplfix/compare.py
./docs/FILES.md
./docs/COMMANDS.md
./benchmark/gemma_vision/vis0/runs/20261018T010701Z/results.json
./benchmark/qwen_vision/qwen_vision_bench_test.py
./benchmark/qwen_vision/qwen35_vision_probe.py
./benchmark/gemma_vision/vis0/runs/20261018T004817Z/results.json
./benchmark/llama_dotprod_rebuild/checks/voice_gen.py
./benchmark/gemma_vision/vis0/README.md
./benchmark/llama_dotprod_rebuild/checks/review/prompt.txt
./benchmark/gemma_vision/vis0/test_vis0.py
./benchmark/gemma_vision/vis0/vis0.py
./benchmark/llm_objective_setting/memory_check/memcheck_pilot.py
./benchmark/llm_objective_setting/memory_check/memcheck.py
./benchmark/llm_objective_setting/conversation/chat_mtp/runs/chat_mtp_e4b_20261017T115533Z/run.json
./benchmark/llm_objective_setting/runs/bench.py
./benchmark/llm_objective_setting/bench.py
./benchmark/llm_objective_setting/conversation/prep/review/prompt.txt
./benchmark/strategic_selector/picker_prompt/runs/picker_prompt2_20261017T002036Z/B3-E4B_DEV.jsonl
./benchmark/strategic_selector/picker_prompt/runs/picker_prompt2_20261017T002036Z/B3-E2B_DEV.jsonl
./benchmark/strategic_selector/picker_prompt/runs/picker_prompt2_20261017T002036Z/B2-E4B_DEV.jsonl
./benchmark/strategic_selector/picker_prompt/runs/picker_prompt2_20261017T002036Z/B2-E2B_DEV.jsonl
./benchmark/strategic_selector/picker_prompt/runs/picker_prompt2_20261017T002036Z/B1-E4B_DEV.jsonl
./benchmark/strategic_selector/picker_prompt/runs/picker_prompt2_20261017T002036Z/B1-E2B_DEV.jsonl
./benchmark/strategic_selector/picker_prompt/runs/picker_prompt2_20261017T000147Z_interrupted/attempt_source/test_harness2.py
./benchmark/strategic_selector/picker_prompt/runs/picker_prompt2_20261017T000147Z_interrupted/attempt_source/harness2.py
./benchmark/strategic_selector/picker_prompt/reports/pickprompt2/PICKPROMPT2_CODER_REPORT.md
./benchmark/strategic_selector/picker_prompt/test_harness2.py
./benchmark/strategic_selector/picker_prompt/harness2.py
./benchmark/llm_objective_setting/conversation/conversation_20260926T185957Z/turns.jsonl
./benchmark/run_benchmark.py
./configs/setup_d.py
./configs/setup_c.py
./configs/setup_b.py
./benchmark/llm_objective_setting/conversation/conv_mtp/reviews/review5/stderr.txt
./benchmark/llm_objective_setting/conversation/conv_mtp/reviews/review4/stderr.txt
./benchmark/llm_objective_setting/conversation/conv_mtp/reviews/review3/stderr.txt
./benchmark/llm_objective_setting/conversation/conv_mtp/reviews/review2/stderr.txt
./benchmark/llm_objective_setting/conversation/chat_mtp/runs/e4bup_evidence/smoke.stdout.txt
./benchmark/llm_objective_setting/conversation/conv_mtp/reviews/review1/stderr.txt
./benchmark/llm_objective_setting/conversation/conv_mtp/conv_mtp_20260928T081906Z/turns.jsonl
./benchmark/llm_objective_setting/conversation/chat_mtp/runs/chat_mtp_20261014T224541Z/run.json
./benchmark/llm_objective_setting/conversation/chat_mtp/runs/e4bup_evidence/smoke.json
./benchmark/llm_objective_setting/conversation/chat_mtp/runs/chat_mtp_20261014T221203Z_smoke/run.json
./benchmark/relate_anything/quality1/runs/20261017T213604Z/results.json
./benchmark/yolo_speed/reviews/review2_stderr.txt
./benchmark/relate_anything/quality1/runs/20261017T212311Z/results.json
./benchmark/relate_anything/quality1/test_quality1.py
```

Runtime config/caller search:
```text
thermal_real.py:79:    print('  ⚠️  All configs ran hot — needs heatsink or longer rest')
thermal_benchmark2.py:136:configs = [
thermal_benchmark2.py:145:for rest, cycles, label in configs:
configs/setup_b.py:4:NAME = "setup_b_gemma4_e4b"
configs/setup_d.py:4:NAME = "setup_d_hybrid"
configs/setup_d.py:33:def build_vision_prompt(user_input, image_path):
thermal_benchmark3.py:171:configs = [
thermal_benchmark3.py:177:for rest, cycles, label in configs:
configs/setup_c.py:4:NAME = "setup_c_gemma4_e2b"
```

## Verification output

All recorded durations are NOT A MEASUREMENT (agent resident). The root suite includes run_cycle_safety_test. The 55 additional unit tests ran against base chat code for baseline and candidate code after the approved assertion update. Archive-publication tests were not run because they perform forbidden Git writes, even in temporary repositories. Interactive/hardware diagnostics and historical duplicate archived tests were not run. The two optional benchmark-test skips are unchanged. This report does not claim every test in every historical archive ran.


### contract_checks.txt

```text
PASS: parse_command AST differs only in the two marker strings (prefill, payload, JSON handling unchanged).
PASS: PARSE_SYS and warm_up identical.
PASS: Qwen/Mistral branches and chat payload, HTTP, server management functions identical.
```


### tests_before.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
.............................................................................................................  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
PING: ALIVE
distance: 400 cm

mission: explore and map the rooms
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MAP] Found: bedroom
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] GREET -> EXPLORE: x
  [MODE] EXPLORE -> GREET: x

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MODE] LOOK -> EXPLORE: LOOK done after 4 cycles
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 5 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [PERSON] close person — FWD_L held
  [PERSON] close person — FWD_R held
  [PERSON] close person — FORWARD held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_L held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_R held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FORWARD held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_L held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_R held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FORWARD held
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_L held
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_R held
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
..............
----------------------------------------------------------------------
Ran 123 tests in 3.068s

OK
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] Photo failed

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=n/a
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] missing frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] bad frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] other_session frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [VISION] inference failure — stop

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test
  [MODE] EXPLORE -> FORWARD: pick (start)

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
```


### tests_after.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
.............................................................................................................  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
PING: ALIVE
distance: 400 cm

mission: explore and map the rooms
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MAP] Found: bedroom
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] GREET -> EXPLORE: x
  [MODE] EXPLORE -> GREET: x

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MODE] LOOK -> EXPLORE: LOOK done after 4 cycles
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 5 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [PERSON] close person — FWD_L held
  [PERSON] close person — FWD_R held
  [PERSON] close person — FORWARD held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_L held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_R held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FORWARD held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_L held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_R held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FORWARD held
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_L held
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_R held
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
...................
----------------------------------------------------------------------
Ran 128 tests in 3.743s

OK
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] Photo failed

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=n/a
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] missing frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] bad frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] other_session frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [VISION] inference failure — stop

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test
  [MODE] EXPLORE -> FORWARD: pick (start)

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
```


### benchmark_unit_tests_before.txt

```text
benchmark/llm_objective_setting EXIT 0
..
----------------------------------------------------------------------
Ran 2 tests in 0.015s

OK
ok

benchmark/llm_objective_setting/conversation/chat_mtp EXIT 0
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
...................................
----------------------------------------------------------------------
Ran 35 tests in 0.150s

OK

benchmark/strategic_selector/picker_prompt EXIT 0
.s......s.........
----------------------------------------------------------------------
Ran 18 tests in 0.291s

OK (skipped=2)

```


### benchmark_unit_tests.txt

```text
benchmark/llm_objective_setting EXIT 0
..
----------------------------------------------------------------------
Ran 2 tests in 0.008s

OK
ok

benchmark/llm_objective_setting/conversation/chat_mtp EXIT 0
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
...................................
----------------------------------------------------------------------
Ran 35 tests in 0.172s

OK

benchmark/strategic_selector/picker_prompt EXIT 0
.s......s.........
----------------------------------------------------------------------
Ran 18 tests in 0.333s

OK (skipped=2)

```


### nav_before.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
OBSTACLE_DIST = 25

=== single-shot: distance -> move (no history) ===
  dist    5  ->  LEFT          stuck=False
  dist   14  ->  LEFT          stuck=False
  dist   15  ->  STRAFE_LEFT   stuck=False
  dist   20  ->  STRAFE_LEFT   stuck=False
  dist   24  ->  STRAFE_LEFT   stuck=False
  dist   25  ->  FORWARD       stuck=False
  dist   26  ->  FORWARD       stuck=False
  dist  100  ->  FORWARD       stuck=False
  dist  400  ->  FORWARD       stuck=False
  dist  999  ->  FORWARD       stuck=False

=== persistent blockage: 16 cycles at 20cm ===
   (feeding each returned move back into last_moves, as move() does)
  cycle  1  ->  STRAFE_LEFT   side=LEFT  stuck=False
  cycle  2  ->  STRAFE_LEFT   side=LEFT  stuck=False
  cycle  3  ->  LEFT          side=LEFT  stuck=False
  cycle  4  ->  LEFT          side=LEFT  stuck=False
  cycle  5  ->  LEFT          side=LEFT  stuck=False
  cycle  6  ->  LEFT          side=LEFT  stuck=False
  cycle  7  ->  RIGHT         side=RIGHT stuck=True
  cycle  8  ->  RIGHT         side=RIGHT stuck=False
  cycle  9  ->  RIGHT         side=RIGHT stuck=False
  cycle 10  ->  RIGHT         side=RIGHT stuck=False
  cycle 11  ->  RIGHT         side=RIGHT stuck=False
  cycle 12  ->  RIGHT         side=RIGHT stuck=False
  cycle 13  ->  RIGHT         side=RIGHT stuck=False
  cycle 14  ->  STRAFE_RIGHT  side=RIGHT stuck=False
  cycle 15  ->  STRAFE_RIGHT  side=RIGHT stuck=False
  cycle 16  ->  STRAFE_RIGHT  side=RIGHT stuck=False

=== blocked then cleared ===
  cycle 1  dist  20  ->  STRAFE_LEFT   stuck=False
  cycle 2  dist  20  ->  STRAFE_LEFT   stuck=False
  cycle 3  dist  20  ->  LEFT          stuck=False
  cycle 4  dist  20  ->  LEFT          stuck=False
  cycle 5  dist  20  ->  LEFT          stuck=False
  cycle 6  dist 300  ->  FORWARD       stuck=False
  cycle 7  dist 300  ->  FORWARD       stuck=False
```


### nav_after.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
OBSTACLE_DIST = 25

=== single-shot: distance -> move (no history) ===
  dist    5  ->  LEFT          stuck=False
  dist   14  ->  LEFT          stuck=False
  dist   15  ->  STRAFE_LEFT   stuck=False
  dist   20  ->  STRAFE_LEFT   stuck=False
  dist   24  ->  STRAFE_LEFT   stuck=False
  dist   25  ->  FORWARD       stuck=False
  dist   26  ->  FORWARD       stuck=False
  dist  100  ->  FORWARD       stuck=False
  dist  400  ->  FORWARD       stuck=False
  dist  999  ->  FORWARD       stuck=False

=== persistent blockage: 16 cycles at 20cm ===
   (feeding each returned move back into last_moves, as move() does)
  cycle  1  ->  STRAFE_LEFT   side=LEFT  stuck=False
  cycle  2  ->  STRAFE_LEFT   side=LEFT  stuck=False
  cycle  3  ->  LEFT          side=LEFT  stuck=False
  cycle  4  ->  LEFT          side=LEFT  stuck=False
  cycle  5  ->  LEFT          side=LEFT  stuck=False
  cycle  6  ->  LEFT          side=LEFT  stuck=False
  cycle  7  ->  RIGHT         side=RIGHT stuck=True
  cycle  8  ->  RIGHT         side=RIGHT stuck=False
  cycle  9  ->  RIGHT         side=RIGHT stuck=False
  cycle 10  ->  RIGHT         side=RIGHT stuck=False
  cycle 11  ->  RIGHT         side=RIGHT stuck=False
  cycle 12  ->  RIGHT         side=RIGHT stuck=False
  cycle 13  ->  RIGHT         side=RIGHT stuck=False
  cycle 14  ->  STRAFE_RIGHT  side=RIGHT stuck=False
  cycle 15  ->  STRAFE_RIGHT  side=RIGHT stuck=False
  cycle 16  ->  STRAFE_RIGHT  side=RIGHT stuck=False

=== blocked then cleared ===
  cycle 1  dist  20  ->  STRAFE_LEFT   stuck=False
  cycle 2  dist  20  ->  STRAFE_LEFT   stuck=False
  cycle 3  dist  20  ->  LEFT          stuck=False
  cycle 4  dist  20  ->  LEFT          stuck=False
  cycle 5  dist  20  ->  LEFT          stuck=False
  cycle 6  dist 300  ->  FORWARD       stuck=False
  cycle 7  dist 300  ->  FORWARD       stuck=False
```


### copy_checks.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
.............................................................................................................  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
PING: ALIVE
distance: 400 cm

mission: explore and map the rooms
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MAP] Found: bedroom
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] GREET -> EXPLORE: x
  [MODE] EXPLORE -> GREET: x

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MODE] LOOK -> EXPLORE: LOOK done after 4 cycles
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 5 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [PERSON] close person — FWD_L held
  [PERSON] close person — FWD_R held
  [PERSON] close person — FORWARD held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_L held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_R held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FORWARD held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_L held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_R held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FORWARD held
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_L held
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_R held
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
...................
----------------------------------------------------------------------
Ran 128 tests in 3.568s

OK
..
----------------------------------------------------------------------
Ran 2 tests in 0.008s

OK
...................................
----------------------------------------------------------------------
Ran 35 tests in 0.171s

OK
.s......s.........
----------------------------------------------------------------------
Ran 18 tests in 0.310s

OK (skipped=2)
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] Photo failed

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=n/a
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] missing frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] bad frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] other_session frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [VISION] inference failure — stop

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test
  [MODE] EXPLORE -> FORWARD: pick (start)

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
ok
TPL-FIX runner self-check: PASS
Copy imports verified; root tests and runner self-check passed.
```


### onphone.stdout.txt

```text
TPL-FIX runner self-check: PASS
voice 1 saved
voice 2 saved
voice 3 saved
voice 4 saved
voice 5 saved
voice 6 saved
voice 7 saved
voice 8 saved
voice 9 saved
voice 10 saved
voice 11 saved
voice 12 saved
voice 13 saved
voice 14 saved
voice 15 saved
chat 1 saved
chat 2 saved
chat 3 saved
chat 4 saved
chat 5 saved
Saved /data/data/com.termux/files/home/robot/benchmark/tplfix/runs/20261018T182009Z
```


### onphone.stderr.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
```

## Independent review audit and verbatim reviews

Two initial smoke calls returned the required response on the correct model but were INVALID due to settings serialization changes; no code review ran in those calls. After owner authorization, the temporary overlay preselected the exact requested model and canonically omitted empty ask; allow was exactly the sole review-check command and deny exactly the requested list. Later smoke and review snapshots show no changes to live tree, isolated copy, Git refs/status/stashes/remote, or temporary settings. Each call restored and verified original settings SHA-256 `6e5273bf3864677e135b104758ba5724c8c45f8a17d3e55d9af19e66ee4b7421`. Existing settings.json.bak_m2q1 was observed and left untouched. No blanket permission bypass was used. CLI diagnostic copies retain model-verification lines only, omitting authentication identities; stdout and stderr remain separate.

Three code-review rounds were used, all on the same frozen code and exact model label. Round 1 was INVALID because every reported full length exceeded newline/splitlines counts by one; quoted source positions matched. Round 2 was INVALID because a regex quotation lost genuine escaping under Markdown table pipe decoding. Round 3 was APPROVED; all quoted complete lines, all six lengths, model label, frozen source hashes and before/after state audits were independently verified by the Coder. No code changed after any review round or the live run.


### Round 1 audit

```json
{
  "returncode": 0,
  "equal": true,
  "differences": [],
  "original_settings_sha256": "6e5273bf3864677e135b104758ba5724c8c45f8a17d3e55d9af19e66ee4b7421"
}
```

### Round 1 review — verbatim

# Review TPL-FIX v2

## Execution Profile

```text
EXECUTION PROFILE
ROLE: Reviewer (independent, read-only)
PLATFORM: AGY/Antigravity CLI headless, isolated copy
MODEL: Gemini 3.8 Flash (High) (identifier: gemini-3.8-flash-high)
EFFORT: N/A (part of model label)
PONYTAIL: off
WHY: Owner-selected different model family from Coder
FALLBACK: gemini-3.8-flash-medium only if high absent; no silent substitution
```

- **Base commit**: `33c2c78d2f05bac27820d7511588ecb66ccf1f8e`
- **Candidate diff**: [TPLFIX_EVIDENCE/full_candidate.diff](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/TPLFIX_EVIDENCE/full_candidate.diff)
- **Review verdict**: **APPROVED**

---

## 1. Independent Verification Checks

The single authorized terminal check command was executed in the isolated candidate copy:
```bash
/data/data/com.termux/files/usr/bin/python3 -B TPLFIX_EVIDENCE/REVIEW_CHECKS.py
```

### Verification Results
1. **Root test suite**: 128 unit tests passed in 3.903s (`OK`), cleanly up from 123 tests prior to candidate changes due to the 5 new unit tests added in [test_tplfix.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/test_tplfix.py).
2. **Additional benchmark unit test suites**:
   - `benchmark/llm_objective_setting` (`test_aggregate`, `test_bench`): 2 tests passed (`OK`).
   - `benchmark/llm_objective_setting/conversation/chat_mtp` (`test_chat_mtp_runner`): 35 tests passed (`OK`).
   - `benchmark/strategic_selector/picker_prompt` (`test_harness`, `test_harness2`): 18 tests passed (`OK`, 2 skipped).
3. **Module copy import isolation**: Dynamic path assertion verified that `chat`, `server_manager`, and `run_cycle_safety_test` resolved exclusively to the candidate review copy, redirecting legacy paths.
4. **Comparison runner offline self-check**: `benchmark/tplfix/compare.py --self-check` passed (`TPL-FIX runner self-check: PASS`).
5. **No hardware / server side effects**: The suite invoked zero motor, USB, or server start/stop actions.
6. **No repository mutations**: Bytecode generation was suppressed (`-B`, `sys.dont_write_bytecode = True`). All files remain completely frozen.

---

## 2. Mandatory Analysis & Findings

### A. Template Matching & Native System / Thinking Prelude
In [chat.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/chat.py) (full line count: 399), line 163 defines `build_prompt`:
- Line 163:
  ```python
  def build_prompt(model_type, messages):
  ```
- Lines 167–168:
  ```python
          if not messages or messages[0]['role'] != 'system':
              prompt = '<|turn>system\n<|think|>\n<turn|>\n'
  ```
- Line 173:
  ```python
                  prompt += f'<|turn>system\n<|think|>\n{content}<turn|>\n'
  ```
- Line 175:
  ```python
                  prompt += f'<|turn>user\n{content}<turn|>\n'
  ```
- Line 177:
  ```python
                  prompt += f'<|turn>model\n{content}<turn|>\n'
  ```
- Line 178:
  ```python
          prompt += '<|turn>model\n'
  ```

In [tests/gemma4_apply_template.json](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/tests/gemma4_apply_template.json) (full line count: 55), the captures from the live robot b1609 server (`b1609-e1a1abb7` default options) show:
- Line 15 (Case 1: User `Hi`):
  ```json
        "prompt": "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
  ```
- Line 31 (Case 2: System `S` + User `Hi`):
  ```json
        "prompt": "<|turn>system\n<|think|>\nS<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
  ```
- Line 51 (Case 3: User `Hi` + Assistant `Yo` + User `Again`):
  ```json
        "prompt": "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nYo<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n"
  ```

**Distinction from VIS0**: The template in [TPLFIX_EVIDENCE/server_props_capture.json](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/TPLFIX_EVIDENCE/server_props_capture.json) establishes that b1609 defaults `enable_thinking` to true. Under `enable_thinking`, the template condition `{%- if enable_thinking or tools or (messages and messages[0]['role'] in ['system', 'developer']) -%}` unconditionally renders a native system turn containing `<|think|>\n`, even when no user system prompt is provided. VIS0 renders without this prelude because it evaluated with thinking disabled or outside the b1609 default template options. `chat.py:build_prompt` faithfully reflects the live b1609 renderer default captures byte-for-byte.

In [chat.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/chat.py) (full line count: 399), lines 208–212 configure stop tokens:
- Line 208:
  ```python
  def get_stop_tokens(model_type):
  ```
- Line 210:
  ```python
          'gemma':   ['<turn|>'],
  ```
- Line 211:
  ```python
          'qwen':    ['<|im_end|>'],
  ```
- Line 212:
  ```python
          'mistral': ['</s>', '[INST]'],
  ```
Gemma stop token is correctly updated to `<turn|>`.

### B. Voice Command Parser (`main.py:parse_command`) Contract Preservation
In [main.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/main.py) (full line count: 691), `parse_command` retains its exact runtime contract:
- Line 128:
  ```python
  def parse_command(text, timeout=40):
  ```
- Line 130:
  ```python
          '<|turn>user\n' + PARSE_SYS +
  ```
- Line 132:
  ```python
          '<turn|>\n<|turn>model\n['
  ```
- Line 137:
  ```python
              'temperature': 0.05, 'stop': ['<turn|>', '\n\n']
  ```
- Line 141:
  ```python
          raw = '[' + raw
  ```

1. **Turn markers**: Updated to Gemma 4 `<|turn>user\n` and `<turn|>\n<|turn>model\n[`.
2. **Injected `[` prefill**: Intact in both prompt construction (line 132) and recovery logic (line 141: `raw = '[' + raw`).
3. **Parameters**: `n_predict: 300`, `temperature: 0.05`, and default `timeout=40` remain identical.
4. **Stop tokens**: Updated to `['<turn|>', '\n\n']`.
5. **System text & filtering**: `PARSE_SYS` text is untouched. JSON array slicing (`raw.find('[')` to `raw.rfind(']') + 1`) and validation of dicts requiring `'type'` remain intact.
6. **Warm-up**: `main.warm_up()` remains untouched.
7. **Rationale for distinction between `chat.py` and `main.py`**: In `main.py`, `PARSE_SYS` is embedded into the single user turn and primed with `[` to force direct JSON output without thinking channels or multi-turn conversational preludes. Injecting `<|think|>` into `main.parse_command` would cause reasoning token generation and break the prefilled JSON array boundary. Maintaining the prefilled user-turn prompt for voice parsing while aligning conversational turns in `chat.py` with b1609 `/apply-template` is the correct design.

### C. Family Isolation (Qwen / Mistral)
In [chat.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/chat.py) (full line count: 399), lines 181–205 and lines 211–212 for Qwen (`<|im_start|>`, `<|im_end|>`) and Mistral (`[INST]`, `</s>`) are completely untouched. Unit tests in [test_tplfix.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/test_tplfix.py) explicitly verify that prompt construction and stop tokens for Qwen and Mistral remain bit-for-bit identical to pre-candidate behavior.

### D. Full-Repo Old-Marker Search & Justification for Unchanged Legacy Senders
Audit of [TPLFIX_EVIDENCE/full_repo_old_marker_files.txt](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/TPLFIX_EVIDENCE/full_repo_old_marker_files.txt) and [TPLFIX_EVIDENCE/old_marker_grep.txt](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/TPLFIX_EVIDENCE/old_marker_grep.txt) confirms that only runtime files (`chat.py`, `main.py`) and the single owner-approved test file (`benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py`) were modified.
All other matches are justified as unchanged:
1. **Legacy configurations** ([configs/setup_b.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/configs/setup_b.py), `setup_c.py`, `setup_d.py`): These are early prototype standalone scripts. Neither `main.py`, `run_mission.py`, `chat.py`, nor `server_manager.py` imports or executes anything from `configs/`. Modifying them would be churn on dormant legacy files.
2. **Archived executed benchmarks**: `benchmark/llm_objective_setting/`, `benchmark/strategic_selector/picker_prompt/`, and `benchmark/relate_anything/` are completed, historical runs. Touching them violates benchmark immutability rules.
3. **Historical benchmark scripts and harnesses**: `thermal_benchmark*.py`, `benchmark_nav.py`, `swa_benchmark.py`, `quality_benchmark*.py`, `nav_logic_test.py`, and `mission_test.py` are superseded artifacts documented in `docs/FILES.md`.
4. **Documentation**: Mentions in `docs/FILES.md` and `docs/COMMANDS.md` describe historical behavior and require no runtime mutation.

### E. Test Suite Coverage & Comparison Runner (`benchmark/tplfix/compare.py`)
1. **Unit tests** ([test_tplfix.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/test_tplfix.py), full line count: 54):
   - Line 11:
     ```python
     class GemmaTemplateTest(unittest.TestCase):
     ```
   - Line 12:
     ```python
         def test_matches_robot_apply_template_captures(self):
     ```
   - Line 19:
     ```python
         def test_stop_tokens(self):
     ```
   - Line 24:
     ```python
         def test_other_families_unchanged(self):
     ```
   - Line 34:
     ```python
         def test_voice_prefill_payload_and_json_filter(self):
     ```
   - Line 45:
     ```python
         def test_voice_invalid_json_still_fails_closed(self):
     ```
   All 5 tests run cleanly and provide direct verification of captured fixtures, payload invariants, filtering, fail-closed handling, and model-family isolation.
2. **Chat MTP runner assertion update** ([benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py), full line count: 613):
   - Lines 49–51:
     ```python
             self.assertEqual(p, {'prompt': '<|turn>system\n<|think|>\n<turn|>\n'
                                           '<|turn>user\nHow far can you see?<turn|>\n<|turn>model\n',
                                  'n_predict': 512, 'temperature': 0.7, 'stop': ['<turn|>'], 'stream': False})
     ```
   This single assertion update was explicitly approved by the owner to reconcile the test with `chat.build_prompt('gemma', ...)`. The test passes.
3. **Comparison runner** ([benchmark/tplfix/compare.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/benchmark/tplfix/compare.py), full line count: 185):
   - Line 28:
     ```python
     TIMING_LABEL = 'NOT A MEASUREMENT (agent resident)'
     ```
   - Line 78:
     ```python
         payload = module.chat_payload(module.build_prompt('gemma', [{'role': 'user', 'content': prompt}]), 'gemma')
     ```
   - Line 79:
     ```python
         payload.update(temperature=0, n_predict=128)
     ```
   - Line 136:
     ```python
         with tempfile.TemporaryDirectory(prefix='tplfix_old_') as directory:
     ```
   - Line 139:
     ```python
                 (old / filename).write_bytes(subprocess.check_output(['git', 'show', f'{base}:{filename}'], cwd=ROOT))
     ```
   - **Isolation**: OLD base code is extracted into a temporary directory via `git show` and loaded dynamically.
   - **Safety**: Runner does not manage, spawn, or kill llama-server processes, and instantiates no `Robot` or hardware drivers.
   - **Verification scope**: Evaluates the 15 voice commands from `voice_gen.py` and 5 fixed English chat prompts, logging raw replies, actions, parse validity, token leaks, and repetitions.
   - **Privacy and provenance**: SHA-256 digests of all candidate files and base files are hashed into `manifest.json`. No secrets or personal account names are embedded.
   - **Timing discipline**: Labeled `NOT A MEASUREMENT (agent resident)`.

---

## 3. Approved Exceptions, Limitations, and Uncertainties

1. **Owner-approved test assertion update**: Updating the single stale assertion in `benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py` (lines 49–51) was owner-approved; all other benchmark archives remain strictly frozen.
2. **AGY overlay permissions representation**: The temporary permissions overlay canonically omitted the empty `"permissions.ask"` key per AGY's internal structure; runtime `Ask:[]` remained empty as approved.
3. **Archive-publication test limitation**: Tests verifying Git publication actions were not run because they invoke prohibited Git write operations in a frozen review session. This limitation is noted and expected.
4. **Pending on-phone comparison**: The live on-phone comparison (`benchmark/tplfix/compare.py BASE_SHA`) is deliberately pending this independent review and has not yet run against a live server. It is validated offline via its self-check.

---

## 4. Verdict

**APPROVED**

The candidate changes in TPL-FIX v2 are minimal, correct, fully tested, and strictly adhere to all architectural, safety, and formatting constraints.

*(Notice: Reviewer approval is a technical assessment and does not constitute authorization to commit or push under `docs/WORKFLOW.md`.)*


### Round 2 audit

```json
{
  "returncode": 0,
  "equal": true,
  "differences": [],
  "original_settings_sha256": "6e5273bf3864677e135b104758ba5724c8c45f8a17d3e55d9af19e66ee4b7421"
}
```

### Round 2 review — verbatim

### Verdict: APPROVED

### 1. Executive Summary & Verification Findings
Fresh independent Round 2 review of frozen candidate **TPL-FIX v2** (base `33c2c78d2f05bac27820d7511588ecb66ccf1f8e`). The candidate satisfies all template correction requirements, preserves safety and inference contracts, isolates non-Gemma families, and passes all verification suites without side effects.

- **Exact Template Rendering**: In [chat.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/chat.py), `build_prompt('gemma', ...)` matches all three live captures from [tests/gemma4_apply_template.json](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/tests/gemma4_apply_template.json) byte-for-byte (`[user Hi]`, `[system S, user Hi]`, and `[user Hi, assistant Yo, user Again]`). Stop tokens correctly map to `['<turn|>']`.
- **Observed Live b1609 vs Jinja Default / VIS0**: In the raw Jinja template (`server_props_capture.json`), `enable_thinking` has `default(false)`, which in standalone offline contexts (such as VIS0) omits the system turn for user-only prompts. On the live robot server (`b1609-e1a1abb7`), the server enables thinking by default in `/apply-template`, emitting the native `<|turn>system\n<|think|>\n<turn|>\n` prelude even for a user-only turn. [chat.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/chat.py) strictly matches the live server capture.
- **Voice Parser Contract Preserved**: In [main.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/main.py):`parse_command`, turn markers are updated to `<|turn>user\n` and `<turn|>\n<|turn>model\n[`. The injected `[` prefill, `PARSE_SYS` prompt, `n_predict=300`, `temperature=0.05`, `timeout=40`, fail-closed JSON extraction/filtering, and `warm_up()` remain identical.
- **Model Isolation**: Qwen and Mistral prompt builders and stop tokens in [chat.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/chat.py) remain untouched. Navigation, motor safety, thermal governor, RelateAnything, and YOLO detector pipelines are completely unaltered.
- **Legacy Senders Justified**: Full-repo grep confirms old markers remain only in historical/archived benchmarks and inactive configs ([configs/setup_b.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/configs/setup_b.py), `setup_c.py`, `setup_d.py`). These configs have no import paths or call sites in runtime entry points ([main.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/main.py), [run_mission.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/run_mission.py), [chat.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/chat.py)), justifying leaving them untouched. The single approved archive update in [test_chat_mtp_runner.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py) updates only the stale template assertion.
- **Comparison Runner & Safety**: [benchmark/tplfix/compare.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/benchmark/tplfix/compare.py) compares OLD (isolated via `git show BASE:file`) vs NEW across 15 voice commands and 5 fixed English chat prompts. It captures raw replies, actions, parse success, and regex token leaks; verifies SHA-256 manifests and server props before/after; records no hardware/server mutations; contains no secrets; and explicitly tags all timings as `NOT A MEASUREMENT (agent resident)`.

### 2. Independent Execution Verification
Execution of `/data/data/com.termux/files/usr/bin/python3 -B TPLFIX_EVIDENCE/REVIEW_CHECKS.py` completed cleanly with exit code 0:
- Verified all 6 citation line lengths matching [CITATION_COUNTS.json](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/TPLFIX_EVIDENCE/CITATION_COUNTS.json).
- Ran 128 root unit tests (123 baseline + 5 in [test_tplfix.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/test_tplfix.py)): OK.
- Ran 55 additional benchmark tests (`test_aggregate`, `test_bench`, `test_chat_mtp_runner`, `test_harness`, `test_harness2`): OK (2 skipped).
- Verified copy-redirected imports for `chat`, `server_manager`, and `run_cycle_safety_test`.
- Passed [benchmark/tplfix/compare.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/benchmark/tplfix/compare.py) offline self-check.
- Evidence confirms [nav_test.py](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/nav_test.py) output before and after candidate changes is byte-identical.

### 3. Verbatim Source Citations
All stated file lengths reflect standard newline counts verified by [CITATION_COUNTS.json](file:///data/data/com.termux/files/home/review/tplfix/round1/candidate/TPLFIX_EVIDENCE/CITATION_COUNTS.json).

| File | Full Line Count | Line | Verbatim Quoted Source Line |
|---|---|---|---|
| `chat.py` | 398 | 167 | `        if not messages or messages[0]['role'] != 'system':` |
| `chat.py` | 398 | 168 | `            prompt = '<\|turn>system\n<\|think|>\n<turn\|>\n'` |
| `chat.py` | 398 | 210 | `        'gemma':   ['<turn\|>'],` |
| `main.py` | 690 | 130 | `        '<\|turn>user\n' + PARSE_SYS +` |
| `main.py` | 690 | 132 | `        '<turn\|>\n<\|turn>model\n['` |
| `main.py` | 690 | 137 | `            'temperature': 0.05, 'stop': ['<turn\|>', '\n\n']` |
| `tests/gemma4_apply_template.json` | 54 | 15 | `      "prompt": "<\|turn>system\n<\|think|>\n<turn\|>\n<\|turn>user\nHi<turn\|>\n<\|turn>model\n"` |
| `test_tplfix.py` | 53 | 17 | `                self.assertEqual(chat.build_prompt('gemma', case['request']['messages']), case['prompt'])` |
| `test_tplfix.py` | 53 | 20 | `        self.assertEqual(chat.get_stop_tokens('gemma'), ['<turn\|>'])` |
| `benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py` | 612 | 51 | `                             'n_predict': 512, 'temperature': 0.7, 'stop': ['<turn\|>'], 'stream': False})` |
| `benchmark/tplfix/compare.py` | 184 | 28 | `TIMING_LABEL = 'NOT A MEASUREMENT (agent resident)'` |
| `benchmark/tplfix/compare.py` | 184 | 29 | `TOKEN = re.compile(r'<(?:\|[^<>\n]*[>\|]\|[^<>\n]*\|>\|start_of_turn>\|end_of_turn>)')` |
| `benchmark/tplfix/compare.py` | 184 | 109 | `    print('TPL-FIX runner self-check: PASS')` |

### 4. Limitations
1. **On-Phone Live Comparison**: The live server run (`compare.py BASE_SHA`) is pending this review and must run only on frozen reviewed code.
2. **Archive-Publication Tests**: Skipped as they invoke prohibited Git write operations.
3. **AGY Configuration Overlay**: Omits empty `permissions.ask` key per canonical AGY serialization; verified effective runtime `Ask: []`.
4. **Approval Scope**: This verdict constitutes technical review approval only; it is not authorization to commit or push.


### Round 3 audit

```json
{
  "returncode": 0,
  "equal": true,
  "differences": [],
  "original_settings_sha256": "6e5273bf3864677e135b104758ba5724c8c45f8a17d3e55d9af19e66ee4b7421"
}
```

### Round 3 review — verbatim

## Verdict
**APPROVED**
*(Note: Reviewer approval is a technical verification finding only and does not constitute authorization to commit or push.)*

### Verified Candidate Source Lengths
| File | Verified Line Count |
|---|---|
| `chat.py` | 398 |
| `main.py` | 690 |
| `tests/gemma4_apply_template.json` | 54 |
| `test_tplfix.py` | 53 |
| `benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py` | 612 |
| `benchmark/tplfix/compare.py` | 184 |

### Mandatory Review Findings

1. **Template Matching & Server Prelude**:
In `chat.py` (full length 398 lines), the prompt builder for family `gemma` matches all three robot b1609 `POST /apply-template` captures recorded in `tests/gemma4_apply_template.json` (full length 54 lines). The live robot renderer enables thinking externally on the server, producing `<|turn>system\n<|think|>\n<turn|>\n` prelude even for user-only requests, unlike offline VIS0 Jinja rendering which falls back to disabled thinking. The stop token is correctly updated in `chat.py` line 210:
```python
        'gemma':   ['<turn|>'],
```

2. **Voice Contract & Safety Invariants**:
In `main.py` (full length 690 lines), `parse_command` updates turn markers to Gemma 4 format while preserving the injected `[` prefill in `main.py` line 132:
```python
        '<turn|>\n<|turn>model\n['
```
Stop tokens are `['<turn|>', '\n\n']`, `n_predict=300`, `temperature=0.05`, and default `timeout=40`. `PARSE_SYS` text, `warm_up()`, and fail-closed JSON array extraction and validation remain untouched.

3. **Family Isolation & Legacy Senders**:
Qwen and Mistral prompt builders and stop tokens remain completely untouched in `chat.py`. Navigation, heat control, detector, and picker logic are unmodified. Repository search confirmed remaining occurrences of `<start_of_turn>` and `<end_of_turn>` reside exclusively in archived benchmarks, obsolete configs, or documentation, none of which are active runtime senders. The only active senders are `chat.py` and `main.py`. The owner-approved test assertion update in `benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py` (full length 612 lines) aligns with the new builder output and passes.

4. **Test Suite & Verification**:
In `test_tplfix.py` (full length 53 lines), five unit tests verify builder fixture parity, stop tokens, family isolation, voice prefill payload, and fail-closed JSON filtering. Line 17 verifies exact capture matching:
```python
                self.assertEqual(chat.build_prompt('gemma', case['request']['messages']), case['prompt'])
```
The designated review check command ran independently: 128 root tests (123 baseline + 5 new), 55 benchmark suite tests (2 + 35 + 18, 2 skipped), and offline runner self-check passed without error. Navigation regression output is byte-identical, and safety tests passed. Archive-publication tests invoke prohibited Git writes and were not run.

5. **Comparison Runner & Environment**:
`benchmark/tplfix/compare.py` (full length 184 lines) verifies offline self-check, base commit isolation via temporary directory, manifest SHA-256 recording, and raw response/leak tracking without modifying server state or executing hardware. Chat comparison parameters are set in `benchmark/tplfix/compare.py` line 79:
```python
    payload.update(temperature=0, n_predict=128)
```
All outputs are tagged `NOT A MEASUREMENT (agent resident)`. The canonical omission of empty `permissions.ask` in the temporary overlay preserves empty runtime ask lists. Live on-phone comparison is properly deferred until after this frozen review.


## DOC DIFF for Doc Keeper

No canonical documentation was edited. Proposed evidence-backed update after owner discussion: runtime Gemma builders now use Gemma 4 turn markers; voice 15/15 parsed with 14/15 action equality vs base and one changed nonsense case; native default chat thinking prelude leaks channels 5/5 and reaches the 128-token cap before a final answer 4/5. Do not record a clean-chat success. Evidence: this report and benchmark/tplfix/runs/20261018T182009Z.

## Final Git status

No staging, commit, push, stash, reset, checkout or clean was performed. Unrelated pre-existing quality1 files are retained.

```text
 M benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
 M chat.py
 M main.py
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
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__blocked_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__blocked_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__blocked_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__blocked_4.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__empty_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__empty_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__empty_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__empty_4.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__empty_5.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_040cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_060cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_080cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_100cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_150cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_200cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_300cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_040cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_060cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_080cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_100cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_150cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_200cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_300cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_bed_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_bed_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_bed_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_couch_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_couch_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_couch_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_tv_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_tv_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_tv_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213419918.MP.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213509269.MP.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213514191.MP.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213532927.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213536608.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213539353.MP.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_close_back.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_close_front.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_close_left.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_close_right.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_far_back.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_far_front.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_far_left.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_far_right.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_close_back.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_close_front.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_close_left.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_close_right.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_far_back.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_far_front.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_far_left.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_far_right.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__empty_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__empty_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__empty_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__empty_4.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__empty_5.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__scene_test.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__stereo_a.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__stereo_b.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__table_stereo_a.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__table_top.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/key.json
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/manifest.json
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/results.json
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/sheet_ab.pdf
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/sheet_m2.pdf
?? benchmark/relate_anything/quality1/test_quality1.py
?? test_tplfix.py
?? tests/gemma4_apply_template.json
```


# TPL-FIX r2 — thinking OFF
1. PASS: NEW answers 10/10 chat cases with zero detected token leaks.
2. Without system: 5/5 answers; zero leaks; one greeting reaches the cap.
3. With system: 5/5 answers; zero leaks; zero cap hits.
4. OLD control: 9/10 answers; zero leaks; zero cap hits.
5. Voice: both parse 15/15, no leaks; NEW actions match v2 NEW 15/15.
6. main.py is byte-identical to v2; v2 run is untouched.
7. Captured native thinking-OFF prompts; retained ON captures as unused.
8. Tests 183 -> 184 (root 128 -> 129), two skips; nav same; safety PASS.
9. AGY round 4 APPROVED; citations, exact model and state audits verified.
10. Server unchanged; settings restored; no USB/motors/stage/commit/push.
## r2 scope and provenance

Assigned Coder: Codex CLI, gpt-6.1-sol, effort medium, Ponytail lite. Independent Reviewer: AGY headless, gemini-3.8-flash-high, effort N/A, Ponytail off. Exact model label verified in the AGY log; no fallback used. Owner authorized follow-up rounds 4–6; only round 4 was needed. Base remains `33c2c78d2f05bac27820d7511588ecb66ccf1f8e`.

Incremental r2 edits are limited to the Gemma branch of chat.py, its fixture/tests, the matching chat-MTP assertion, and the comparison extension required for system-message cases/cap reporting. No automatic system turn is emitted; explicit system turns contain only supplied content, with no `<|think|>`. User/assistant/generation-prefix strings and `<turn|>` stops are unchanged from v2. `.strip()` is retained because the native template trims string content (system content line 199, message text line 327 in the captured template); a new padded-content unit test covers all three roles. No dependencies or server/model/build settings changed.

`main.py` SHA-256 remains `62ef3a80789852fec4206773634db93e0ce8caaa91fa7de483539b86c780674c`, matching v2 byte-for-byte. Qwen/Mistral branches, stops, chat payload, server lifecycle functions and comparison voice/parser/base-import functions are unchanged, verified by AST/byte checks. The historical v2 run files have their original SHA-256. The old report prefix is unchanged; this r2 section is appended.

## Thinking-OFF captures and preserved ON evidence

POST /apply-template on the running robot b1609 accepts `chat_template_kwargs: {"enable_thinking": false}`. It did not ignore the option, and no VIS0 fallback was needed. The three captured request/response pairs are the active `cases` in tests/gemma4_apply_template.json. The old ON cases remain under `thinking_on_not_used_by_robot`, labelled `not used by the robot`, and were verified equal to the v2 cases by the independent review checks.

```json
[
  {
    "request": {
      "messages": [
        {
          "role": "user",
          "content": "Hi"
        }
      ],
      "add_generation_prompt": true,
      "chat_template_kwargs": {
        "enable_thinking": false
      }
    },
    "prompt": "<|turn>user\nHi<turn|>\n<|turn>model\n"
  },
  {
    "request": {
      "messages": [
        {
          "role": "system",
          "content": "S"
        },
        {
          "role": "user",
          "content": "Hi"
        }
      ],
      "add_generation_prompt": true,
      "chat_template_kwargs": {
        "enable_thinking": false
      }
    },
    "prompt": "<|turn>system\nS<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
  },
  {
    "request": {
      "messages": [
        {
          "role": "user",
          "content": "Hi"
        },
        {
          "role": "assistant",
          "content": "Yo"
        },
        {
          "role": "user",
          "content": "Again"
        }
      ],
      "add_generation_prompt": true,
      "chat_template_kwargs": {
        "enable_thinking": false
      }
    },
    "prompt": "<|turn>user\nHi<turn|>\n<|turn>model\nYo<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n"
  }
]
```

## Live results

Run: `benchmark/tplfix/runs/20261019T062853Z`. OLD is base-commit code imported from a temporary directory outside the repository; NEW is the hash-frozen reviewed candidate. Same 15 voice commands, same five chat prompts, temperature 0 and n_predict 128 for chat. The additional five cases use exactly `You are a small home robot. Answer in English, briefly.` Both OLD and NEW were tested in both chat modes. The parser keeps its existing temperature/timeout and JSON handling. All requests returned raw replies.

Answer presence was manually assessed from each raw reply, with reasons recorded in chat_assessment.json alongside the raw results.json hash. The raw results.json and manifest.json were not edited. The generated results.md is preserved as a prefix; the manual per-item table was appended, and the complete table was copied to `/storage/emulated/0/Download/tplfix/results_r2.md` with a matching hash. All timings are **NOT A MEASUREMENT (agent resident)**.

# TPL-FIX OLD vs NEW

NOT A MEASUREMENT (agent resident)

| Voice command | OLD parsed | NEW parsed | Same actions | OLD leaks | NEW leaks |
|---|---|---|---|---|---|
| Find Chiara and tell her the pizza is here | True | True | True | 0 | 0 |
| Go to the bathroom | True | True | True | 0 | 0 |
| Patrol the whole apartment | True | True | True | 0 | 0 |
| Find my phone in the living room | True | True | True | 0 | 0 |
| Come back | True | True | True | 0 | 0 |
| Tell Abdel the meeting starts in ten minutes | True | True | True | 0 | 0 |
| Go to the kitchen and say lunch is ready | True | True | True | 0 | 0 |
| Check the bedroom and the hallway | True | True | True | 0 | 0 |
| Where are my glasses | True | True | True | 0 | 0 |
| Say good night | True | True | True | 0 | 0 |
| Find someone and tell them the door is open | True | True | True | 0 | 0 |
| Go to the living room then come back | True | True | True | 0 | 0 |
| Look for the cat in the bedroom | True | True | True | 0 | 0 |
| Tell Chiara I love her | True | True | True | 0 | 0 |
| Blue banana seven | True | True | False | 0 | 0 |

| System | Chat prompt | OLD leaks | NEW leaks | OLD cap | NEW cap |
|---|---|---|---|---|---|
| False | Hello! Introduce yourself in one sentence. | 0 | 0 | False | False |
| False | What is two plus three? Answer briefly. | 0 | 0 | False | False |
| False | Name three common household objects. | 0 | 0 | False | False |
| False | Explain why a robot should stop near an obstacle in one sentence. | 0 | 0 | False | False |
| False | Write a friendly good morning greeting. | 0 | 0 | False | True |
| True | Hello! Introduce yourself in one sentence. | 0 | 0 | False | False |
| True | What is two plus three? Answer briefly. | 0 | 0 | False | False |
| True | Name three common household objects. | 0 | 0 | False | False |
| True | Explain why a robot should stop near an obstacle in one sentence. | 0 | 0 | False | False |
| True | Write a friendly good morning greeting. | 0 | 0 | False | False |

Token cap means stop_type=limit; answer presence is assessed from raw replies.

## Per-item answer check (manual inspection of raw replies)

| System | Prompt | OLD answer | OLD leak | OLD cap | NEW answer | NEW leak | NEW cap |
|---|---|---|---|---|---|---|---|
| False | Hello! Introduce yourself in one sentence. | True | False | False | True | False | False |
| False | What is two plus three? Answer briefly. | True | False | False | True | False | False |
| False | Name three common household objects. | False | False | False | True | False | False |
| False | Explain why a robot should stop near an obstacle in one sentence. | True | False | False | True | False | False |
| False | Write a friendly good morning greeting. | True | False | False | True | False | True |
| True | Hello! Introduce yourself in one sentence. | True | False | False | True | False | False |
| True | What is two plus three? Answer briefly. | True | False | False | True | False | False |
| True | Name three common household objects. | True | False | False | True | False | False |
| True | Explain why a robot should stop near an obstacle in one sentence. | True | False | False | True | False | False |
| True | Write a friendly good morning greeting. | True | False | False | True | False | False |

NEW no-system greeting hit the cap after providing valid greetings; it satisfies answer presence. OLD no-system object listing repeats a generic phrase without naming objects. All NEW answers are in English.

### Interpretation and limits

The no-system NEW greeting offers several valid greetings, then reaches the 128-token cap while continuing alternatives. It satisfies the stated answer-presence target; it is still verbose. All five explicit-system NEW answers are brief, complete and below the cap. OLD no-system object listing repeats a generic phrase without naming any objects; all other OLD cases contain answers. Every NEW reply is English and contains no detected thought-channel or template token. Thinking-off fixes the observed v2 thought-channel regression on these ten cases.

The NEW voice actions match v2 NEW for every command. OLD vs NEW action equality is still 14/15; the unchanged nonsense case `Blue banana seven` remains OLD say `[]` vs NEW find_object banana with an empty room. No voice logic was changed to alter that behavior.

### Server and frozen-code integrity

Server PID/executable and captured /props remain identical before/after. It is the owner-managed b1609 dotprod process on port 8080 with E2B Q4_0, ctx2048. The agent never started, stopped or killed a server. Candidate file hashes match the reviewed freeze and the completed run manifest; no code changed after review or during/after the run.

```json
{
  "build_info": "b1609-e1a1abb7",
  "model": "gemma-4-E2B-it-Q4_0.gguf",
  "total_slots": 1,
  "chat_template_sha256": "603a42db292c25278e9b23d94c7abbe77453f13e26a053c43c5c2f900b4136f7"
}
```

## r2 incremental diff against v2

```diff
--- v2/chat.py
+++ r2/chat.py
@@ -163,14 +163,11 @@
 def build_prompt(model_type, messages):
     if model_type == 'gemma':
         prompt = ''
-        # b1609 /apply-template defaults to a native system turn with thinking enabled.
-        if not messages or messages[0]['role'] != 'system':
-            prompt = '<|turn>system\n<|think|>\n<turn|>\n'
         for msg in messages:
             role = msg['role']
             content = msg['content'].strip()
             if role == 'system':
-                prompt += f'<|turn>system\n<|think|>\n{content}<turn|>\n'
+                prompt += f'<|turn>system\n{content}<turn|>\n'
             elif role == 'user':
                 prompt += f'<|turn>user\n{content}<turn|>\n'
             elif role == 'assistant':
--- v2/test_tplfix.py
+++ r2/test_tplfix.py
@@ -15,6 +15,14 @@
         for case in fixture['cases']:
             with self.subTest(messages=case['request']['messages']):
                 self.assertEqual(chat.build_prompt('gemma', case['request']['messages']), case['prompt'])
+                self.assertNotIn('<|think|>', case['prompt'])
+
+    def test_gemma_trims_content_like_the_template(self):
+        messages = [{'role': 'system', 'content': ' S\n'}, {'role': 'user', 'content': ' Hi '},
+                    {'role': 'assistant', 'content': '\nYo '}, {'role': 'user', 'content': ' Again\n'}]
+        self.assertEqual(chat.build_prompt('gemma', messages),
+                         '<|turn>system\nS<turn|>\n<|turn>user\nHi<turn|>\n'
+                         '<|turn>model\nYo<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n')
 
     def test_stop_tokens(self):
         self.assertEqual(chat.get_stop_tokens('gemma'), ['<turn|>'])
--- v2/tests/gemma4_apply_template.json
+++ r2/tests/gemma4_apply_template.json
@@ -1,5 +1,5 @@
 {
-  "source": "b1609-e1a1abb7 POST /apply-template, default template options",
+  "source": "b1609-e1a1abb7 POST /apply-template, chat_template_kwargs.enable_thinking=false",
   "model": "gemma-4-E2B-it-Q4_0.gguf",
   "cases": [
     {
@@ -10,9 +10,12 @@
             "content": "Hi"
           }
         ],
-        "add_generation_prompt": true
+        "add_generation_prompt": true,
+        "chat_template_kwargs": {
+          "enable_thinking": false
+        }
       },
-      "prompt": "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
+      "prompt": "<|turn>user\nHi<turn|>\n<|turn>model\n"
     },
     {
       "request": {
@@ -26,9 +29,12 @@
             "content": "Hi"
           }
         ],
-        "add_generation_prompt": true
+        "add_generation_prompt": true,
+        "chat_template_kwargs": {
+          "enable_thinking": false
+        }
       },
-      "prompt": "<|turn>system\n<|think|>\nS<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
+      "prompt": "<|turn>system\nS<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
     },
     {
       "request": {
@@ -46,9 +52,66 @@
             "content": "Again"
           }
         ],
-        "add_generation_prompt": true
+        "add_generation_prompt": true,
+        "chat_template_kwargs": {
+          "enable_thinking": false
+        }
       },
-      "prompt": "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nYo<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n"
+      "prompt": "<|turn>user\nHi<turn|>\n<|turn>model\nYo<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n"
     }
-  ]
+  ],
+  "thinking_on_not_used_by_robot": {
+    "label": "not used by the robot",
+    "source": "b1609-e1a1abb7 POST /apply-template, default template options",
+    "cases": [
+      {
+        "request": {
+          "messages": [
+            {
+              "role": "user",
+              "content": "Hi"
+            }
+          ],
+          "add_generation_prompt": true
+        },
+        "prompt": "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
+      },
+      {
+        "request": {
+          "messages": [
+            {
+              "role": "system",
+              "content": "S"
+            },
+            {
+              "role": "user",
+              "content": "Hi"
+            }
+          ],
+          "add_generation_prompt": true
+        },
+        "prompt": "<|turn>system\n<|think|>\nS<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
+      },
+      {
+        "request": {
+          "messages": [
+            {
+              "role": "user",
+              "content": "Hi"
+            },
+            {
+              "role": "assistant",
+              "content": "Yo"
+            },
+            {
+              "role": "user",
+              "content": "Again"
+            }
+          ],
+          "add_generation_prompt": true
+        },
+        "prompt": "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nYo<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n"
+      }
+    ]
+  }
 }
--- v2/benchmark/tplfix/compare.py
+++ r2/benchmark/tplfix/compare.py
@@ -26,6 +26,7 @@
     'Write a friendly good morning greeting.',
 )
 TIMING_LABEL = 'NOT A MEASUREMENT (agent resident)'
+CHAT_SYSTEM = 'You are a small home robot. Answer in English, briefly.'
 TOKEN = re.compile(r'<(?:\|[^<>\n]*[>|]|[^<>\n]*\|>|start_of_turn>|end_of_turn>)')
 FILES = ('chat.py', 'main.py', 'test_tplfix.py', 'tests/gemma4_apply_template.json',
          'benchmark/tplfix/compare.py',
@@ -74,15 +75,18 @@
                 leaks=TOKEN.findall(raw or ''))
 
 
-def chat_reply(module, prompt):
-    payload = module.chat_payload(module.build_prompt('gemma', [{'role': 'user', 'content': prompt}]), 'gemma')
+def chat_reply(module, prompt, system=None):
+    messages = ([{'role': 'system', 'content': system}] if system else [])
+    messages.append({'role': 'user', 'content': prompt})
+    payload = module.chat_payload(module.build_prompt('gemma', messages), 'gemma')
     payload.update(temperature=0, n_predict=128)
     response = requests.post(module.URL, json=payload, timeout=120)
     response.raise_for_status()
     data = response.json()
     reply = data['content']
     return dict(raw_reply=reply, leaks=TOKEN.findall(reply), repeated_eight_words=repeated(reply),
-                truncated=data.get('truncated'), stop_type=data.get('stop_type'))
+                truncated=data.get('truncated'), stop_type=data.get('stop_type'),
+                hit_token_cap=data.get('stop_type') == 'limit')
 
 
 def props():
@@ -96,6 +100,8 @@
 
 
 def self_check():
+    from unittest.mock import patch
+
     assert parsed_ok('{"type":"come_back"}]') and parsed_ok(']')
     assert not parsed_ok('bad') and not parsed_ok(None)
     assert TOKEN.findall('hello <turn|> <|channel>thought <start_of_turn>') == [
@@ -106,6 +112,16 @@
     fake = SimpleNamespace(parse_command=lambda text: [])
     assert voice(fake, 'ignored')['parse_succeeded'] is False
     assert len(CHAT_PROMPTS) == 5
+    builder = SimpleNamespace(URL='unused', build_prompt=lambda family, messages: messages,
+                              chat_payload=lambda prompt, family: {'prompt': prompt})
+    response = SimpleNamespace(raise_for_status=lambda: None,
+                               json=lambda: {'content': '5', 'stop_type': 'limit'})
+    with patch.object(requests, 'post', return_value=response) as post:
+        for system in (None, CHAT_SYSTEM):
+            assert chat_reply(builder, 'Hi', system)['hit_token_cap']
+            messages = ([{'role': 'system', 'content': system}] if system else [])
+            messages.append({'role': 'user', 'content': 'Hi'})
+            post.assert_called_with('unused', json={'prompt': messages, 'temperature': 0, 'n_predict': 128}, timeout=120)
     print('TPL-FIX runner self-check: PASS')
 
 
@@ -126,7 +142,7 @@
     manifest = dict(base_commit=base, timing_label=TIMING_LABEL, server_props_before=before,
                     candidate_sha256={f: sha(ROOT / f) for f in FILES},
                     voice_source=str(source.relative_to(ROOT)), voice_source_sha256=sha(source),
-                    chat_prompts=list(CHAT_PROMPTS), complete=False)
+                    chat_prompts=list(CHAT_PROMPTS), chat_system=CHAT_SYSTEM, complete=False)
 
     def save():
         (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
@@ -148,13 +164,14 @@
             results['voice'].append(row)
             save()
             print('voice', len(results['voice']), 'saved', flush=True)
-        for prompt in CHAT_PROMPTS:
-            row = dict(prompt=prompt)
-            for label, _, builder in variants:
-                row[label] = chat_reply(builder, prompt)
-            results['chat'].append(row)
-            save()
-            print('chat', len(results['chat']), 'saved', flush=True)
+        for system in (None, CHAT_SYSTEM):
+            for prompt in CHAT_PROMPTS:
+                row = dict(prompt=prompt, system=system)
+                for label, _, builder in variants:
+                    row[label] = chat_reply(builder, prompt, system)
+                results['chat'].append(row)
+                save()
+                print('chat', len(results['chat']), 'saved', flush=True)
     manifest['server_props_after'] = props()
     assert manifest['server_props_after'] == before, 'server props changed; investigate'
     assert manifest['candidate_sha256'] == {f: sha(ROOT / f) for f in FILES}, 'candidate changed'
@@ -167,11 +184,12 @@
         lines.append('| ' + ' | '.join([row['command'], str(row['OLD']['parse_succeeded']),
                      str(row['NEW']['parse_succeeded']), str(row['same_actions']),
                      str(len(row['OLD']['leaks'])), str(len(row['NEW']['leaks']))]) + ' |')
-    lines += ['', '| Chat prompt | OLD leaks | NEW leaks | OLD repeated | NEW repeated |', '|---|---|---|---|---|']
+    lines += ['', '| System | Chat prompt | OLD leaks | NEW leaks | OLD cap | NEW cap |', '|---|---|---|---|---|---|']
     for row in results['chat']:
-        lines.append('| ' + ' | '.join([row['prompt'], str(len(row['OLD']['leaks'])), str(len(row['NEW']['leaks'])),
-                     str(row['OLD']['repeated_eight_words']), str(row['NEW']['repeated_eight_words'])]) + ' |')
-    lines += ['', 'Repeated = an exact eight-word span repeated without overlap; read raw replies for quality.', '']
+        lines.append('| ' + ' | '.join([str(bool(row['system'])), row['prompt'],
+                     str(len(row['OLD']['leaks'])), str(len(row['NEW']['leaks'])),
+                     str(row['OLD']['hit_token_cap']), str(row['NEW']['hit_token_cap'])]) + ' |')
+    lines += ['', 'Token cap means stop_type=limit; answer presence is assessed from raw replies.', '']
     (out / 'results.md').write_text('\n'.join(lines))
     print('Saved', out)
 
--- v2/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
+++ r2/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
@@ -46,8 +46,7 @@
         p = cm.payload('How far can you see?')
         self.assertEqual(p, chat.chat_payload(chat.build_prompt('gemma', [{'role': 'user', 'content': 'How far can you see?'}]),
                                               'gemma'))
-        self.assertEqual(p, {'prompt': '<|turn>system\n<|think|>\n<turn|>\n'
-                                      '<|turn>user\nHow far can you see?<turn|>\n<|turn>model\n',
+        self.assertEqual(p, {'prompt': '<|turn>user\nHow far can you see?<turn|>\n<|turn>model\n',
                              'n_predict': 512, 'temperature': 0.7, 'stop': ['<turn|>'], 'stream': False})
         with mock.patch.object(chat, 'chat_payload', return_value='P') as f:
             self.assertEqual(cm.payload('x'), 'P')
```

## Full tracked diff against base commit

This cumulative worktree diff includes the earlier v2 main.py turn-token fix; r2 did not change main.py.

```diff
diff --git a/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py b/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
index 13d0f0d..dd22521 100644
--- a/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
+++ b/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
@@ -46,8 +46,8 @@ class CommandPayloadTest(unittest.TestCase):
         p = cm.payload('How far can you see?')
         self.assertEqual(p, chat.chat_payload(chat.build_prompt('gemma', [{'role': 'user', 'content': 'How far can you see?'}]),
                                               'gemma'))
-        self.assertEqual(p, {'prompt': '<start_of_turn>user\nHow far can you see?<end_of_turn>\n<start_of_turn>model\n',
-                             'n_predict': 512, 'temperature': 0.7, 'stop': ['<end_of_turn>'], 'stream': False})
+        self.assertEqual(p, {'prompt': '<|turn>user\nHow far can you see?<turn|>\n<|turn>model\n',
+                             'n_predict': 512, 'temperature': 0.7, 'stop': ['<turn|>'], 'stream': False})
         with mock.patch.object(chat, 'chat_payload', return_value='P') as f:
             self.assertEqual(cm.payload('x'), 'P')
         f.assert_called_once()
diff --git a/chat.py b/chat.py
index b712e04..8f39406 100644
--- a/chat.py
+++ b/chat.py
@@ -165,14 +165,14 @@ def build_prompt(model_type, messages):
         prompt = ''
         for msg in messages:
             role = msg['role']
-            content = msg['content']
+            content = msg['content'].strip()
             if role == 'system':
-                prompt += f'<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n'
+                prompt += f'<|turn>system\n{content}<turn|>\n'
             elif role == 'user':
-                prompt += f'<start_of_turn>user\n{content}<end_of_turn>\n'
+                prompt += f'<|turn>user\n{content}<turn|>\n'
             elif role == 'assistant':
-                prompt += f'<start_of_turn>model\n{content}<end_of_turn>\n'
-        prompt += '<start_of_turn>model\n'
+                prompt += f'<|turn>model\n{content}<turn|>\n'
+        prompt += '<|turn>model\n'
         return prompt
 
     elif model_type == 'qwen':
@@ -204,7 +204,7 @@ def build_prompt(model_type, messages):
 
 def get_stop_tokens(model_type):
     stops = {
-        'gemma':   ['<end_of_turn>'],
+        'gemma':   ['<turn|>'],
         'qwen':    ['<|im_end|>'],
         'mistral': ['</s>', '[INST]'],
     }
diff --git a/main.py b/main.py
index 5c5799f..a03e066 100644
--- a/main.py
+++ b/main.py
@@ -127,14 +127,14 @@ EXAMPLES:
 
 def parse_command(text, timeout=40):
     prompt = (
-        '<start_of_turn>user\n' + PARSE_SYS +
+        '<|turn>user\n' + PARSE_SYS +
         '\nCommand: "' + text + '"'
-        '<end_of_turn>\n<start_of_turn>model\n[' 
+        '<turn|>\n<|turn>model\n['
     )
     try:
         resp = requests.post(GEMMA_URL, json={
             'prompt': prompt, 'n_predict': 300,
-            'temperature': 0.05, 'stop': ['<end_of_turn>', '\n\n']
+            'temperature': 0.05, 'stop': ['<turn|>', '\n\n']
         }, timeout=timeout)
         raw = resp.json()['content']
         # Model starts after [ which we injected
```

## Candidate and new r2 artifact hashes

| File | SHA-256 |
|---|---|

| `chat.py` | `5f08a9d65d2a1f8fb5b77cb40050a009b8729480327def91be19ec44bf835348` |
| `main.py` | `62ef3a80789852fec4206773634db93e0ce8caaa91fa7de483539b86c780674c` |
| `test_tplfix.py` | `862673f4e1a41e699088596c91e7001d16ff1f157a6d29e012c96886002c040b` |
| `tests/gemma4_apply_template.json` | `cabc65272f082138231e1168e9835e25c0ac724e5f9e1cb1fd4ea7b86d9e2524` |
| `benchmark/tplfix/compare.py` | `3a259e917e3771ceeeb78188b360dd12dc5a20256b064508bc1e8719fa2d4089` |
| `benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py` | `35dbb1593b301f83d50ce4839b37214c5f08f4dab74453a5e5444adc832f937d` |
| `benchmark/tplfix/runs/20261019T062853Z/chat_assessment.json` | `cb1f370b72f899f26b2afc02d598a1d7ed1ae46c88551f31892114a4417a153c` |
| `benchmark/tplfix/runs/20261019T062853Z/manifest.json` | `34c6cd5b0ccadca532c99f027357bb77787aa67d2e83c0930fc0f8cbd34363be` |
| `benchmark/tplfix/runs/20261019T062853Z/results.json` | `a3d60ea439338a398a1cb80ffab8f06afde5cd0c25b98879792d076b0169e8d1` |
| `benchmark/tplfix/runs/20261019T062853Z/results.md` | `0d08aed3e0188f83d8fd41467d0c9970f07575807fd49fffebbfb2d6fb0a682d` |

## Exact ignored files for the later human commit decision

The following existing files are matched by benchmark/* and would require explicit force staging if the owner chooses to include them. **No git add -f, staging, commit or push was run.** The list includes preserved v2 evidence as well as r2 evidence.

```text
benchmark/tplfix/compare.py
benchmark/tplfix/runs/20261018T182009Z/manifest.json
benchmark/tplfix/runs/20261018T182009Z/results.json
benchmark/tplfix/runs/20261018T182009Z/results.md
benchmark/tplfix/runs/20261019T062853Z/chat_assessment.json
benchmark/tplfix/runs/20261019T062853Z/manifest.json
benchmark/tplfix/runs/20261019T062853Z/results.json
benchmark/tplfix/runs/20261019T062853Z/results.md
```

## Checks and actual output

The verified v2 baseline is 183 tests (two skips); the 128 root tests were rerun at r2 entry, before edits. After r2, 129 root plus 55 additional tests ran with no failures and the same two optional skips. Navigation output before/after is byte-identical. run_cycle_safety_test is included and passes. Physical/interactive diagnostics and archive-publication tests that invoke prohibited Git writes were not run. Historical duplicate archive tests were not re-executed. No new timing measurements are claimed.


### r2 contract_checks.txt

```text
PASS: main.py byte-identical to v2.
PASS: Qwen/Mistral branches, stop tokens, chat payload, server management unchanged.
PASS: comparison voice/parser/props/base-module loading functions unchanged.
```


### r2 tests_before.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
.............................................................................................................  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
PING: ALIVE
distance: 400 cm

mission: explore and map the rooms
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MAP] Found: bedroom
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] GREET -> EXPLORE: x
  [MODE] EXPLORE -> GREET: x

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MODE] LOOK -> EXPLORE: LOOK done after 4 cycles
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 5 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [PERSON] close person — FWD_L held
  [PERSON] close person — FWD_R held
  [PERSON] close person — FORWARD held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_L held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_R held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FORWARD held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_L held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_R held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FORWARD held
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_L held
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_R held
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
...................
----------------------------------------------------------------------
Ran 128 tests in 3.931s

OK
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] Photo failed

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=n/a
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] missing frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] bad frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] other_session frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [VISION] inference failure — stop

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test
  [MODE] EXPLORE -> FORWARD: pick (start)

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
```


### r2 tests_after.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
.............................................................................................................  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
PING: ALIVE
distance: 400 cm

mission: explore and map the rooms
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MAP] Found: bedroom
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] GREET -> EXPLORE: x
  [MODE] EXPLORE -> GREET: x

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MODE] LOOK -> EXPLORE: LOOK done after 4 cycles
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 5 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [PERSON] close person — FWD_L held
  [PERSON] close person — FWD_R held
  [PERSON] close person — FORWARD held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_L held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_R held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FORWARD held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_L held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_R held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FORWARD held
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_L held
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_R held
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
....................
----------------------------------------------------------------------
Ran 129 tests in 3.671s

OK
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] Photo failed

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=n/a
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] missing frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] bad frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] other_session frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [VISION] inference failure — stop

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test
  [MODE] EXPLORE -> FORWARD: pick (start)

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
```


### r2 benchmark_tests_after.txt

```text
benchmark/llm_objective_setting EXIT 0
..
----------------------------------------------------------------------
Ran 2 tests in 0.009s

OK
ok

benchmark/llm_objective_setting/conversation/chat_mtp EXIT 0
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
...................................
----------------------------------------------------------------------
Ran 35 tests in 0.167s

OK

benchmark/strategic_selector/picker_prompt EXIT 0
.s......s.........
----------------------------------------------------------------------
Ran 18 tests in 0.352s

OK (skipped=2)

```


### r2 copy_checks.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
............................................................................................................Verified citation file lengths: {"chat.py": {"newline_count": 395, "splitlines_count": 395}, "main.py": {"newline_count": 690, "splitlines_count": 690}, "test_tplfix.py": {"newline_count": 61, "splitlines_count": 61}, "tests/gemma4_apply_template.json": {"newline_count": 117, "splitlines_count": 117}, "benchmark/tplfix/compare.py": {"newline_count": 202, "splitlines_count": 202}, "benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py": {"newline_count": 611, "splitlines_count": 611}}
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
  [MODE] None -> EXPLORE: mission start
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
PING: ALIVE
distance: 400 cm

mission: explore and map the rooms
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE
rooms: {}
moves: []
[PICKER] disabled: server down — the mission continues in EXPLORE

mission: explore and map the rooms   [DRY — no motor commands]
ctrl-C to stop

rooms: {}
moves: []
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MAP] Found: bedroom
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] GREET -> EXPLORE: x
  [MODE] EXPLORE -> GREET: x

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 3 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 4 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [MODE] LOOK -> EXPLORE: LOOK done after 4 cycles
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 5 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=True detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [PERSON] close person — FWD_L held
  [PERSON] close person — FWD_R held
  [PERSON] close person — FORWARD held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_L held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FWD_R held
  [MODE] None -> EXPLORE: test
  [PERSON] close person — FORWARD held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_L held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FWD_R held
  [MODE] None -> APPROACH: test
  [PERSON] close person — FORWARD held
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> GREET: test
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_L held
  [MODE] None -> LOOK: test
  [PERSON] close person — FWD_R held
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [PERSON] very close person — FORWARD held
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s.....................
----------------------------------------------------------------------
Ran 129 tests in 3.759s

OK
..
----------------------------------------------------------------------
Ran 2 tests in 0.008s

OK
...................................
----------------------------------------------------------------------
Ran 35 tests in 0.171s

OK
.s......s.........
----------------------------------------------------------------------
Ran 18 tests in 0.319s

OK (skipped=2)

  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] Photo failed

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 2 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] RIGHT
  [MOTOR] RIGHT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 2 | heat n/a | dist:20cm ──
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [YOLO] size=320 person=False detect=0ms age=n/a
  [YOLO] empty room
  [NAV] FORWARD
  [MOTOR] FORWARD

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] missing frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] bad frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] other_session frame — stop

── Cycle 1 | heat n/a | dist:400cm ──
  [VISION] inference failure — stop

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] HOLD
  [MOTOR] HOLD
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:10cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] LEFT
  [MOTOR] LEFT
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT
  [MODE] None -> EXPLORE: test
  [MODE] EXPLORE -> FORWARD: pick (start)

── Cycle 1 | heat n/a | dist:20cm ──
  [YOLO] size=320 person=False detect=0ms age=0.40s
  [YOLO] empty room
  [NAV] STRAFE_LEFT
  [MOTOR] STRAFE_LEFT

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> EXPLORE: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> APPROACH: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> GREET: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
  [MODE] None -> LOOK: test

── Cycle 1 | heat n/a | dist:400cm ──
  [CAM] no new frame — stop
ok
TPL-FIX runner self-check: PASS
Copy imports verified; root tests and runner self-check passed.
```


### r2 nav_before.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
OBSTACLE_DIST = 25

=== single-shot: distance -> move (no history) ===
  dist    5  ->  LEFT          stuck=False
  dist   14  ->  LEFT          stuck=False
  dist   15  ->  STRAFE_LEFT   stuck=False
  dist   20  ->  STRAFE_LEFT   stuck=False
  dist   24  ->  STRAFE_LEFT   stuck=False
  dist   25  ->  FORWARD       stuck=False
  dist   26  ->  FORWARD       stuck=False
  dist  100  ->  FORWARD       stuck=False
  dist  400  ->  FORWARD       stuck=False
  dist  999  ->  FORWARD       stuck=False

=== persistent blockage: 16 cycles at 20cm ===
   (feeding each returned move back into last_moves, as move() does)
  cycle  1  ->  STRAFE_LEFT   side=LEFT  stuck=False
  cycle  2  ->  STRAFE_LEFT   side=LEFT  stuck=False
  cycle  3  ->  LEFT          side=LEFT  stuck=False
  cycle  4  ->  LEFT          side=LEFT  stuck=False
  cycle  5  ->  LEFT          side=LEFT  stuck=False
  cycle  6  ->  LEFT          side=LEFT  stuck=False
  cycle  7  ->  RIGHT         side=RIGHT stuck=True
  cycle  8  ->  RIGHT         side=RIGHT stuck=False
  cycle  9  ->  RIGHT         side=RIGHT stuck=False
  cycle 10  ->  RIGHT         side=RIGHT stuck=False
  cycle 11  ->  RIGHT         side=RIGHT stuck=False
  cycle 12  ->  RIGHT         side=RIGHT stuck=False
  cycle 13  ->  RIGHT         side=RIGHT stuck=False
  cycle 14  ->  STRAFE_RIGHT  side=RIGHT stuck=False
  cycle 15  ->  STRAFE_RIGHT  side=RIGHT stuck=False
  cycle 16  ->  STRAFE_RIGHT  side=RIGHT stuck=False

=== blocked then cleared ===
  cycle 1  dist  20  ->  STRAFE_LEFT   stuck=False
  cycle 2  dist  20  ->  STRAFE_LEFT   stuck=False
  cycle 3  dist  20  ->  LEFT          stuck=False
  cycle 4  dist  20  ->  LEFT          stuck=False
  cycle 5  dist  20  ->  LEFT          stuck=False
  cycle 6  dist 300  ->  FORWARD       stuck=False
  cycle 7  dist 300  ->  FORWARD       stuck=False
```


### r2 nav_after.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
OBSTACLE_DIST = 25

=== single-shot: distance -> move (no history) ===
  dist    5  ->  LEFT          stuck=False
  dist   14  ->  LEFT          stuck=False
  dist   15  ->  STRAFE_LEFT   stuck=False
  dist   20  ->  STRAFE_LEFT   stuck=False
  dist   24  ->  STRAFE_LEFT   stuck=False
  dist   25  ->  FORWARD       stuck=False
  dist   26  ->  FORWARD       stuck=False
  dist  100  ->  FORWARD       stuck=False
  dist  400  ->  FORWARD       stuck=False
  dist  999  ->  FORWARD       stuck=False

=== persistent blockage: 16 cycles at 20cm ===
   (feeding each returned move back into last_moves, as move() does)
  cycle  1  ->  STRAFE_LEFT   side=LEFT  stuck=False
  cycle  2  ->  STRAFE_LEFT   side=LEFT  stuck=False
  cycle  3  ->  LEFT          side=LEFT  stuck=False
  cycle  4  ->  LEFT          side=LEFT  stuck=False
  cycle  5  ->  LEFT          side=LEFT  stuck=False
  cycle  6  ->  LEFT          side=LEFT  stuck=False
  cycle  7  ->  RIGHT         side=RIGHT stuck=True
  cycle  8  ->  RIGHT         side=RIGHT stuck=False
  cycle  9  ->  RIGHT         side=RIGHT stuck=False
  cycle 10  ->  RIGHT         side=RIGHT stuck=False
  cycle 11  ->  RIGHT         side=RIGHT stuck=False
  cycle 12  ->  RIGHT         side=RIGHT stuck=False
  cycle 13  ->  RIGHT         side=RIGHT stuck=False
  cycle 14  ->  STRAFE_RIGHT  side=RIGHT stuck=False
  cycle 15  ->  STRAFE_RIGHT  side=RIGHT stuck=False
  cycle 16  ->  STRAFE_RIGHT  side=RIGHT stuck=False

=== blocked then cleared ===
  cycle 1  dist  20  ->  STRAFE_LEFT   stuck=False
  cycle 2  dist  20  ->  STRAFE_LEFT   stuck=False
  cycle 3  dist  20  ->  LEFT          stuck=False
  cycle 4  dist  20  ->  LEFT          stuck=False
  cycle 5  dist  20  ->  LEFT          stuck=False
  cycle 6  dist 300  ->  FORWARD       stuck=False
  cycle 7  dist 300  ->  FORWARD       stuck=False
```


### r2 onphone.stdout.txt

```text
TPL-FIX runner self-check: PASS
voice 1 saved
voice 2 saved
voice 3 saved
voice 4 saved
voice 5 saved
voice 6 saved
voice 7 saved
voice 8 saved
voice 9 saved
voice 10 saved
voice 11 saved
voice 12 saved
voice 13 saved
voice 14 saved
voice 15 saved
chat 1 saved
chat 2 saved
chat 3 saved
chat 4 saved
chat 5 saved
chat 6 saved
chat 7 saved
chat 8 saved
chat 9 saved
chat 10 saved
Saved /data/data/com.termux/files/home/robot/benchmark/tplfix/runs/20261019T062853Z
```


### r2 onphone.stderr.txt

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
```

## Independent review round 4

No other AGY process was present at review startup. The fresh isolated copy contained tracked files plus candidate/new source and prior-run evidence, excluding .git, ignored assets/photos/weights/builds. Full live-tree/copy SHA-256 inventories and Git ref/status/stash/remote output match before/after. The sole allowed check command was `/data/data/com.termux/files/usr/bin/python3 -B TPLFIX_EVIDENCE/REVIEW_CHECKS.py`; deny rules matched v2. The owner-approved canonical omission of empty permissions.ask was retained, preserving empty runtime Ask lists. Backup settings were restored immediately after each smoke/review call, SHA-256 `6e5273bf3864677e135b104758ba5724c8c45f8a17d3e55d9af19e66ee4b7421`. The existing settings.json.bak_m2q1 was left untouched. No blanket bypass flag was used.

The 2026-10-18 smoke completed successfully before interruption; after resume, model availability and smoke were refreshed before the first actual r2 review. Both smoke audits and the round 4 review audit were valid. The exact selected model label, all six file lengths and all eight quoted source lines were verified. Review stdout, stderr and model-only CLI diagnostics are separate. The live r2 run occurred only after this valid review. Rounds 5–6 remain unused.

```json
{
  "returncode": 0,
  "equal": true,
  "differences": [],
  "original_settings_sha256": "6e5273bf3864677e135b104758ba5724c8c45f8a17d3e55d9af19e66ee4b7421"
}
```

### Round 4 final review — verbatim

# Independent Review: TASK TPL-FIX r2 (Round 4)

**Status: APPROVED**

### Summary & Conformance
This candidate cleanly implements thinking-OFF prompt construction for the Gemma branch in [`chat.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/chat.py). When no system message is provided, no implicit system turn or thinking token is injected, achieving exact parity with the native `/apply-template` endpoint under `chat_template_kwargs.enable_thinking=false`. Explicit system messages render `<|turn>system\n{content}<turn|>\n` without `<|think|>`. Content stripping across all roles matches native Jinja `| trim` formatting.

All three test cases (`[user Hi]`, `[system S, user Hi]`, and `[user Hi, assistant Yo, user Again]`) in [`tests/gemma4_apply_template.json`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/tests/gemma4_apply_template.json) reflect captured thinking-OFF responses from build 1609 (`gemma-4-E2B-it-Q4_0.gguf`). The prior thinking-ON fixture captures remain preserved under `thinking_on_not_used_by_robot`.

[`main.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/main.py) is byte-identical to v2. Voice parsing prefill `[` and `<turn|>` stop tokens remain intact. Qwen and Mistral prompt builders, stop tokens, generation settings, navigation, detector, and picker logic are completely untouched. The benchmark extension in [`benchmark/tplfix/compare.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/benchmark/tplfix/compare.py) adds the five system-prompt cases, records `hit_token_cap`, and leaves voice commands and isolation logic unchanged.

### Source File Lengths
Exact newline counts verified against `CITATION_COUNTS.json`:

| File | Newline Count |
|---|---|
| [`chat.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/chat.py) | 395 |
| [`main.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/main.py) | 690 |
| [`test_tplfix.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/test_tplfix.py) | 61 |
| [`tests/gemma4_apply_template.json`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/tests/gemma4_apply_template.json) | 117 |
| [`benchmark/tplfix/compare.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/benchmark/tplfix/compare.py) | 202 |
| [`benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py) | 611 |

### Source Citations
In [`chat.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/chat.py), line 170 renders explicit system content without thinking markers:
```python
                prompt += f'<|turn>system\n{content}<turn|>\n'
```
and line 207 retains the turn stop token:
```python
        'gemma':   ['<turn|>'],
```
In [`main.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/main.py), line 132 maintains the voice parse generation prompt:
```python
        '<turn|>\n<|turn>model\n['
```
In [`test_tplfix.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/test_tplfix.py), line 18 validates the absence of thinking tokens:
```python
                self.assertNotIn('<|think|>', case['prompt'])
```
In [`tests/gemma4_apply_template.json`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/tests/gemma4_apply_template.json), line 64 retains the unused thinking-ON cases:
```json
    "label": "not used by the robot",
```
In [`benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py), line 49 updates the expected prompt format:
```python
        self.assertEqual(p, {'prompt': '<|turn>user\nHow far can you see?<turn|>\n<|turn>model\n',
```
In [`benchmark/tplfix/compare.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/benchmark/tplfix/compare.py), line 29 defines the benchmark system prompt:
```python
CHAT_SYSTEM = 'You are a small home robot. Answer in English, briefly.'
```
and line 89 records token cap limits:
```python
                hit_token_cap=data.get('stop_type') == 'limit')
```

### Verification Checks & Automated Execution
The authorized check script was executed independently:
`/data/data/com.termux/files/usr/bin/python3 -B TPLFIX_EVIDENCE/REVIEW_CHECKS.py`
- Verified newline counts across all 6 cited files.
- 129 root unit tests passed.
- 55 extra benchmark and selector tests passed (with 2 expected skips).
- Byte equality between [`main.py`](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/main.py) and v2 snapshot verified.
- Retention of unused thinking-on fixtures verified.
- Comparison runner mock self-check passed.

### Ignored Files & Limitations
- Ignored file list for later staging via `git add -f` (without executing now):
  - `benchmark/tplfix/compare.py`
- This review was strictly read-only. No live model inference, server management, or hardware operations were conducted.
- The live r2 comparison on the phone against base commit `33c2c78d2f05bac27820d7511588ecb66ccf1f8e` is pending.
- Approval does not authorize a commit or push; the human commit gate in [WORKFLOW.md](file:///data/data/com.termux/files/home/review/tplfix/round4/candidate/docs/WORKFLOW.md) remains in effect.


## DOC DIFF for Doc Keeper

No canonical docs were edited. Proposed evidence-backed addition after owner discussion: robot Gemma chat now renders native thinking-OFF prompts; main.py remains v2. Live NEW chat answers 10/10 with zero token leaks; one no-system greeting hits the 128-token cap after supplying valid greetings. All five explicit-system cases are brief and below the cap. NEW voice actions match v2 15/15. Evidence: this r2 section and benchmark/tplfix/runs/20261019T062853Z.

## r2 final Git status

No Git mutation was performed. Pre-existing unrelated quality1 untracked files are retained.

```text
 M benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py
 M chat.py
 M main.py
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
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__blocked_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__blocked_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__blocked_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__blocked_4.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__empty_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__empty_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__empty_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__empty_4.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__empty_5.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_040cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_060cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_080cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_100cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_150cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_200cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_facing_300cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_040cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_060cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_080cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_100cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_150cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_200cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__person_sideon_300cm.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_bed_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_bed_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_bed_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_couch_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_couch_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_couch_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_tv_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_tv_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/bench_photos__roomsig_tv_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213419918.MP.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213509269.MP.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213514191.MP.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213532927.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213536608.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__PXL_20260405_213539353.MP.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_close_back.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_close_front.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_close_left.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_close_right.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_far_back.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_far_front.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_far_left.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__abdel_far_right.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_close_back.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_close_front.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_close_left.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_close_right.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_far_back.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_far_front.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_far_left.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__chiara_far_right.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__empty_1.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__empty_2.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__empty_3.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__empty_4.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__empty_5.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__scene_test.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__stereo_a.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__stereo_b.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__table_stereo_a.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/test_photos__table_top.jpg
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/key.json
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/manifest.json
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/results.json
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/sheet_ab.pdf
?? benchmark/relate_anything/quality1/runs/20261017T213604Z/sheet_m2.pdf
?? benchmark/relate_anything/quality1/test_quality1.py
?? test_tplfix.py
?? tests/gemma4_apply_template.json
```
