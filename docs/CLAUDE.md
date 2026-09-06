# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Context Discipline

**Context is a budget. Spend it on reasoning, not on re-reading.**

- Never re-read a full file already open in this session. Reference what
  you already have. If you need one function, quote that function - not
  the file.
- Summarize benchmark results as a single row, not raw logs.
  Good: `E2B Q4_K_M | 4 threads | 11-12 tok/s | 80°C peak | stable`
  Bad: pasting 15 lines of per-cycle output.
- Don't paste full stack traces. Extract only the failing line and the
  exception type. The other 20 frames are noise.
  Good: `stereo_depth.py:627 ZeroDivisionError - height_px was 0`
  Bad: the entire traceback.
- Terminal output: paste the result, not the scrollback. Server boot logs,
  model metadata dumps, and `cmake` output are almost never the signal.
- When a command produces >20 lines, pipe it: `| tail -5`, `| grep ERROR`,
  or write a summary block at the end of the script.

---

## Project-Specific Constraints

These values are **benchmarked on this hardware**. Do not change them without
re-running the corresponding benchmark and recording the result in DECISIONS.md.

**llama.cpp server flags** (`server_manager.py`):
```
--threads 4 --threads-batch 4    # optimal for Tensor G2; more threads = slower
--parallel 1                     # no slot splitting
--swa-full                       # fixes Gemma 4 SWA prompt-cache invalidation
--ctx-size 2048
```

**Accuracy targets:**
```
Voice command parsing:  ≥93%
Navigation decisions:   ≥90%
```

**Vision** (`detect_person.py`):
```
MODEL = yolo11m.onnx
CONF  = 0.35
IOU   = 0.45
Input = 640x640
```

Anything marked `UNCALIBRATED` or `PENDING` in STATUS.md is not yet trustworthy —
treat those values as placeholders, not as benchmarked constants.
