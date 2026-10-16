# PICKPROMPT-1: desk prompt selection

Research only. Uses the deployed `picker.Gemma` scoring transport and prompt assembly;
no production files, camera, motors, root or M2 runtime. No dependencies added.

Behaviour policy (verbatim from the task):

```text
Modes: EXPLORE, APPROACH, GREET, LOOK (same meaning as main.py). Python safety is unchanged and is NOT part
of the label (person-stop, obstacles, stale frame stay Python).
Missions in the set (exact text):
 MS1 "explore and map the rooms"
 MS2 "find a person and say hello"
 MS3 "map the rooms quietly and do not disturb people"
Rules, first match wins:
 R1 MS3 and a person is visible                                   -> EXPLORE
 R2 person visible, not greeted in the last 300 s, bucket close/very close -> GREET
 R3 person visible, not greeted in the last 300 s, bucket far/medium        -> APPROACH
 R4 person visible, greeted in the last 300 s                    -> EXPLORE
 R5 no person; no new room for >= 60 s (MS2: >= 30 s) and no look-around in the last 60 s -> LOOK
 R6 otherwise                                                    -> EXPLORE
```

`label(fields)` returns `(mode, rule)` without changing its input. Recent means
strictly less than 300/60 seconds: at exactly 300 seconds another greeting is due,
and at exactly 60 seconds a look-around is allowed. None means never. With several
people, a close person takes precedence over a far person. No identity tracking is
claimed; the greeting age is a situation-wide synthetic fact.

Seed 20261016 generates 120 items, 30 per label; each label splits 18 DEV / 12 TEST.
Rules R1/R4/R6 each get 10 items, R2/R3/R5 each get 30. Mission text is fixed.
The generator samples distinct scene lines from the owner's console, ignoring YOLO
timing metadata. It records the source line, original parsed objects, synthetic
bucket indices, and synthetic person replacement/removal. Every missing bucket is
assigned synthetically. Scene tuples in JSON are `(label, position, bucket)`;
`state()` inserts an unused confidence placeholder for `picker.situation()`'s
four-field legacy input. M2 lines are synthetic, in `relate.context()` format,
using only the first two scene objects and the fixed predicate `near`, or the two
specified empty-relation messages. All remaining situation facts are synthetic.
These balanced policy cases do not estimate real-world mission accuracy.

V0 is exactly `picker.situation() + "\n" + M2`, with production options and order.
V1 adds the three time facts; V2 adds explicit first-match rules; V3 also describes
options. V4 uses the highest macro recall among V1–V3 (tie: fewer median tokens,
then variant id for an otherwise unresolved tie), with contextual calibration.
V5 reverses V4's option order and is diagnostic only. There are no extra variants.
Each calibration prompt replaces every situation field value with `N/A`, including
M2; the fixed rule block, if present, stays. There is one baseline per eligible
list for V4, plus two separate reversed-list baselines for V5. Calibration divides
normalised eligible letter probabilities by the baseline and renormalises; using
full-vocabulary probabilities yields the same result because their common scale
cancels. JSONL records all eligible probabilities before and after calibration.

Selection is fixed: highest DEV macro recall among V0–V4 with zero R1 violations
and every mode recall >= 0.6; tie: fewer median prompt tokens (then variant id).
No qualifier yields `NO VARIANT QUALIFIES`; TEST stays untouched. Otherwise only
the winner is scored on TEST, once. Existing events prevent rerunning the same
folder. No prompt tuning after any Gemma call. A transport timeout may recover
pre-calibration DEV with `--resume-dev`; it preserves all saved decisions and the
frozen set, archives both source versions and never resumes or repeats TEST.
Initial-source and resumed-source hashes identify both scoring phases. Missing V0
speed timings are recovered by repeating only the speed pairs, asserting that
probabilities match the saved decisions. Speed timings now persist per pair.
The desk transport waits up to 120 seconds; production files and request bodies
are unchanged. All results, including failures, stay available for review.

The static tokenizer caches the state wrapper and all option suffixes at startup,
then makes one tokenize and one completion request per pick. Production makes
four tokenize and one completion request. Every variant's DEV prompt asserts exact
token parity; V0 also runs both scoring paths per DEV item and asserts matching
probabilities. Speed pairs always run production first, static second. This order,
thermal drift, and the resident agent make all reported timings NOT measurements.
Other variants use static assembly. Prompt-token p90 uses nearest rank; confusion
matrices have true labels as rows and predicted labels as columns. Empty predicted
classes have precision zero. Accuracy results use temperature 0, one predicted
token, top 1000 probabilities, and cache_prompt false, unchanged from production.

Run from the repository root (output folder must not already exist):

```bash
python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py
python3 -B benchmark/strategic_selector/picker_prompt/harness.py generate /termux-home/robot_runs/picker_prompt_<UTC>
python3 server_manager.py setup_q4
python3 -B benchmark/strategic_selector/picker_prompt/harness.py run /termux-home/robot_runs/picker_prompt_<UTC>
python3 server_manager.py stop
pgrep -af '[l]lama-server'
```

Always stop the server even if scoring fails. `freeze.json` records the set hash,
source console hash, harness hash, seed, scene pool and counts before any Gemma
call. `events.jsonl` records first-call, selection and TEST ordering. Set bytes and
harness bytes are checked before scoring; the set hash is checked again at the end.
Results stay outside Git. `selected_prompts.json` contains rendered examples, with
`<bos>` representing the tokenizer's BOS id; prompt pieces are tokenized separately,
exactly as in production, rather than tokenizing the rendered string as a whole.

The isolated review copy receives the completed run at `picker_prompt_results/`.
The optional saved-results test reads that folder to recompute metrics, calibration,
selection, speed medians, split counts, set/harness hashes and TEST event ordering.
It skips when the external run is absent. No test writes files. The owner's console
also contains `empty room, nothing detected`; this compound empty-scene line is
excluded by the object parser. Report both all distinct scene lines and usable
object-bearing lines, rather than treating them as the same count.
