**Verdict: CHANGES REQUESTED** — one custody-guard defect. I found no scoring or metric defect affecting the supplied run.

This was a **source-only static review** of this packet. I called no tools, loaded no skills or Ponytail mode, and executed nothing. The packet specifies this method because earlier Codex sandbox execution failed under proot. All test outputs, phone artifacts, hashes and Git outputs are **supplied evidence**; static review cannot prove runtime behaviour.

1. **P1 — An external symlink can bypass the repository custody check.**

   `benchmark/mind_picker/b1_quality/b1.py:60–61` (**379 lines**):

   > `p = Path(path).absolute()`

   > `return any(q.exists() and os.path.samefile(q, REPO) for q in [p, *p.parents])  # bind mounts: same inode`

   `absolute()` does not resolve symlinks. An outside path such as `/tmp/set.json` pointing to a frozen set stored inside a repository subdirectory passes this check: neither the file nor its lexical parents is the repository root. If its contents match the pinned hash, loading proceeds despite the custody requirement.

   Resolve symlinks before checking ancestry, while retaining the inode comparison for bind-mount aliases. Add an offline test for an outside symlink targeting a set inside the repository. **The supplied run does not demonstrate this bypass occurred**, so this finding does not itself invalidate its accuracy results.

   No other P1 or P2 findings. The explicitly excluded B0 interrupt/cleanup limitation does not affect this verdict.

The `run.json` set matches the task’s **40 items**: mission texts, events, memories, current actions, allowed actions, strict labels and also-ok labels. Representing `-` as `null`, and expanding `all` into catalogue order, preserves the specified data.

Both arms follow the requested rendering and scoring:

- **A:** unchanged `picker.Gemma.prompt_ids()` and `decide()`, raw state block, letter options, `Answer: (`, temperature zero and `cache_prompt: false`. Extending `letter_ids` to A–E supports five options; preflight checks single-token encoding and distinctness.
- **B:** unchanged SYS and catalogue, state plus B0 tail in the user turn, template equality checked against `/apply-template` with thinking disabled, name-token scoring and `cache_prompt: true`.
- Both renormalise over **allowed actions only**. Item 4 excludes `greet` from scoring.
- REV reverses A’s options and B’s tail names. Keeping the state’s allowed-action line in catalogue order, using neutral gerunds and inserting a blank line before B’s tail are faithful fixed choices.

The metric implementation matches the definitions: strict and acceptable accuracy, per-label recall, confusion matrices, copy rate restricted to items whose strict label differs from current activity, near-ties using renormalised top-two margins strictly below 0.05, and order flips. The fixed PASS rule is implemented correctly.

The supplied choices and probabilities support these results:

| Arm | FWD strict / acceptable | REV strict / acceptable | Flips | Result |
|---|---|---|---|---|
| A | 14/40 / 20/40 | 14/40 / 21/40 | 19/40 | **FAIL** |
| B | 19/40 / 28/40 | 20/40 / 28/40 | 6/40 | **FAIL** |

Copy counts are **6/32, 6/32, 6/32 and 7/32**, respectively. Near-tie IDs match: A-FWD `[20, 24]`, A-REV `[10, 23]`, B-FWD `[31]`, B-REV `[1, 27]`. The reported flip lists, recall and confusion counts are consistent with the supplied rows.

The phone run is **VALID accuracy evidence on the supplied record**: 160 rows, no recorded errors, consistent PID/start identity, exit zero, no SIGKILL and matching start/end set hashes. Pre-call custody timing and actual execution remain supplied claims. Timings are correctly labelled **NOT MEASUREMENTS**.

B performs better on this frozen set and flips less often, but **both arms fail every component of the fixed acceptance rule**. Neither qualifies as a passing arm. This review grants no commit approval.