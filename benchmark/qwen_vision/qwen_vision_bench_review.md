# Review of Frozen Qwen VL Benchmark Candidate

**Verdict:** PASS

The candidate implements a robust, standalone multimodal benchmark for Qwen 3.5 VL models on the Pixel Robot hardware. It strictly obeys all operational constraints: it utilizes the fresh `b2351` build, correctly implements the ChatML multimodal payload wrapper, forcefully manages lifecycle via HTTP and signal timeouts, safely samples memory across the process lifecycle, enforces the 50-80% discharging battery constraint, and outputs the requested atomic JSON and table summaries. 

The accompanying `qwen_vision_bench_test.py` thoroughly proves that the harness safely survives server crashes, hanging connections, partial body responses, and missing files entirely isolated from the heavy model weights.

## Ranked Findings and Limitations

1. **(Validation) Complete Total-Wall HTTP Deadline:** `qwen_vision_bench.py:108`
   The usage of `signal.setitimer(signal.ITIMER_REAL, deadline)` perfectly fulfills the requirement for a total HTTP wall deadline. It guarantees interruption and accurate timeout reporting even if the `llama.cpp` server hangs indefinitely while streaming partial HTTP response bodies back to the client.

2. **(Validation) Multimodal Payload and Context:** `qwen_vision_bench.py:73`
   The script injects exactly the required `prompt` dictionary mapping `prompt_string` and `multimodal_data` array containing the base64 encoded bytes to `llama-server`. Injecting the `LLAMA_MEDIA_MARKER` directly into the environment ensures the Qwen VL model marker correctly aligns with the `mtmd` expectations of this specific `llama.cpp` build. 

3. **(Validation) Guaranteed Child Isolation & Cleanup:** `qwen_vision_bench.py:136`
   `stop_server()` escalates gracefully from `terminate()` to a 10s wait, and finally `kill()`. By addressing the specific `subprocess.Popen` PID directly, it safely sidesteps the risks of `pkill -f` and ensures no orphaned models or background jobs leak into the next loop iteration.

4. **(Validation) Thread-Safe Memory Lifecycle Sampling:** `qwen_vision_bench.py:165`
   The background memory worker uses `threading.Event` effectively and is correctly `.join()`ed before the collected `samples` list is sliced. This prevents mutation-during-iteration race conditions while safely capturing the runtime memory footprints.

5. **(Limitation) Over-Strict Label Parser Regex:** `qwen_vision_bench.py:84`
   The regex `r'\s*(PERSON|TV|REFRIGERATOR|TOILET)\s*=\s*([LCR])\s*'` combined with `.split(',')` expects precise formatting. If a model generates a trailing comma (e.g., `PERSON=L, TV=L,`) or uses newlines instead of commas, the parser will fail and return `None` due to empty list elements failing the exact regex match. This is completely acceptable for a strict formatting benchmark, but it is a limitation to keep in mind if model scores appear artificially low.

6. **(Limitation) Fixed Thermal Cooldown:** `qwen_vision_bench.py:284`
   The script utilizes a fixed 60s cooldown (`time.sleep(COOLDOWN)`) between models. While this obeys the prompt's instruction ("fixed, not thermal equilibrium"), testing heavy 4B-parameter models consecutively might cause latent SoC heat saturation to bleed into subsequent tests, mildly skewing tail-end token generation speeds. 

**Reminder:** A reviewer pass is a finding of technical correctness and safety only. This review does not constitute authorization to commit or push. The codebase remains frozen pending separate human commit authorization per the `docs/WORKFLOW.md` gating rules.
