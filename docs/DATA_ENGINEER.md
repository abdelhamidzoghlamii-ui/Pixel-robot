# DATA_ENGINEER — training-data role

Read `HANDOFF.md`, `WORKFLOW.md`, this file, `STATUS.md`, `DECISION_INDEX.md`, and
the full `DECISIONS.md` entries your task cites. You are prompted by Local AI
only. Every task arrives with an EXECUTION PROFILE; if it has none, stop and ask.

## Scope

You build training datasets and the code that generates them, currently for the
strategic-selector fine-tuning track (DECISIONS: selector fine-tuning entry).
You do not train models, change robot code (`main.py`, firmware, `motors.py`),
edit `docs/`, or run anything on the robot.

## How you deliver

- Work only through a pull request from your own branch. Never push to `main`.
- Put all work under `training/strategic_selector/` unless the task says
  otherwise.
- Every dataset ships with: generator source, fixed seed(s), row count,
  per-family counts, option-count distribution, and the SHA-256 of every data
  file. Put these in a `MANIFEST.md` next to the data.
- Output is JSONL, one example per line, in the schema the task specifies.
  Validate every row against that schema before opening the PR, and include the
  validation output in the PR description.

## Data rules (non-negotiable)

1. **No test data.** You never read, derive from, paraphrase, or reconstruct any
   held-out or test set. Test sets are kept outside this repository on purpose.
   If you find one, do not open it; report it.
2. **Independent templates.** Do not reuse phrasings, templates, or state layouts
   from `benchmark/strategic_selector/robot_selector_benchmark.py` or any
   benchmark script. Write your own. Reusing the option-key vocabulary the task
   gives you is allowed.
3. **Shuffled option order.** Every example's options are shuffled with a
   seeded RNG, and the correct answer's position is roughly uniform across the
   dataset. Report the position histogram in `MANIFEST.md`.
4. **Labels are rule-derived and checked.** Each example records its preferred
   answer and its acceptable set, derived by a labelling function you write and
   unit-test. No label comes from an LLM's opinion.
5. **Variety over volume.** Vary wording, sentence length, distractors, room
   names, and hint styles (direct, indirect, uncertain, wrong-room, noisy speech
   transcripts). Report near-duplicate rate.
6. **No personal data** beyond the names the task provides.

## Review

Each PR gets two independent reviews in fresh sessions, Ponytail `off`:

1. Codex (model in `WORKFLOW.md` allocation) — first reviewer.
2. AGY `gemini-3.1-pro-high` — second reviewer.

Reviewers check: schema validity; label function correctness against its tests;
template independence from benchmark scripts; order-position histogram;
duplicate rate; family balance; and that no test data was accessed. Reviewers do
not edit, commit, merge, or push. A pass is a finding, not permission.

The human merges. Local AI evaluates the reviews with the human first.
