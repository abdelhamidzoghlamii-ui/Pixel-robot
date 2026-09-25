# HANDOFF — Pixel Robot

Entry point for an offline robot project: a Pixel phone provides perception and
reasoning, and an ESP32 controls four mecanum wheels. The native Android app and
the Termux prototype have separate runtime constraints.

This file is a stateless index. Current state, pending work, role allocations,
commands, and historical recovery evidence belong in the documents below.

## Canonical reading order

Canonical documentation and role instructions live in this repository's `docs/`.
Root [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md), and
[GEMINI.md](../GEMINI.md) provide session discovery pointers.

1. Read [WORKFLOW.md](WORKFLOW.md) for the shared process, actual role allocation,
   GitHub access, independent review, and human commit/push gates.
2. Read your role document:

   | Role | Instructions |
   |---|---|
   | Local AI — planning and technical orchestration | [LOCAL_AI.md](LOCAL_AI.md) |
   | Coder / data retriever — phone execution and evidence | [CODER.md](CODER.md) |
   | Reviewer — fresh independent candidate review | [REVIEWER.md](REVIEWER.md) |
   | Doc Keeper — canonical status and append-only decisions | [DOC_KEEPER.md](DOC_KEEPER.md) |
   | Data Engineer — training datasets via pull requests | [DATA_ENGINEER.md](DATA_ENGINEER.md) |

3. Read [STATUS.md](STATUS.md) for prototype state and pending work, or
   [APP_STATUS.md](APP_STATUS.md) for app state and pending work. Use the short
   [decision index](DECISION_INDEX.md) to locate relevant numbered entries;
   [DECISIONS.md](DECISIONS.md) holds their full text and provenance. Dated
   prototype measurements and the last hardware test are in
   [PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md).
4. Before implementation, read the applicable [prototype instructions](CLAUDE.md)
   or [Android-app instructions](APP_CLAUDE.md), and [LICENSES.md](LICENSES.md)
   when dependencies are involved. Keep the two runtime scopes separate.
5. Use [COMMANDS.md](COMMANDS.md) for phone operating, calibration, and Git
   inspection, and [OPERATIONS_HISTORY.md](OPERATIONS_HISTORY.md) for dated
   recovery and agent-environment evidence. Use
   [FILES.md](FILES.md) for the dependency map and
   [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) for benchmarking the existing navigation
   path. For the separate, research-only high-level selector, read
   [benchmark/strategic_selector/README.md](../benchmark/strategic_selector/README.md)
   and its [run index](../benchmark/strategic_selector/RUN_INDEX.md).

Read current source and state rather than relying on a prior session. Report
contradictions without silently choosing a side. If the online thread cannot read
the phone repository, request exact source/output through the executor.

## Gates and documentation discipline

Follow the independent-review and human commit gate in [WORKFLOW.md](WORKFLOW.md).
A reviewer pass never authorizes a commit; push requires separate authorization.
Keep hardware work subject to the human-controlled restrictions in
[COMMANDS.md](COMMANDS.md) and the applicable runtime instructions.

Only Doc Keeper writes canonical status and the decision log. Other roles send
targeted evidence-backed DOC DIFFs as defined in [DOC_KEEPER.md](DOC_KEEPER.md).
Existing decisions remain verbatim. Read the last entry in
[DECISIONS.md](DECISIONS.md) directly before choosing the next free number.

## Starting a role session

Name the role (Local AI, Coder, Reviewer, or Doc Keeper) in the opening message.
Read the shared workflow, that role's instructions, the applicable current STATUS,
and the decision index in the order above. Open full decisions relevant to the
task. The role file and WORKFLOW define the task-specific gates; do not rely on a
starter prompt copied from an older session.
