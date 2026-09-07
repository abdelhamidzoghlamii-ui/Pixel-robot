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

3. Read [STATUS.md](STATUS.md) for prototype state and pending work, or
   [APP_STATUS.md](APP_STATUS.md) for app state and pending work. Read
   [DECISIONS.md](DECISIONS.md) for settled tradeoffs and their provenance.
4. Before implementation, read the applicable [prototype instructions](CLAUDE.md)
   or [Android-app instructions](APP_CLAUDE.md), and [LICENSES.md](LICENSES.md)
   when dependencies are involved. Keep the two runtime scopes separate.
5. Use [COMMANDS.md](COMMANDS.md) for phone launch, operating, calibration, Git
   inspection, and recovery commands and historical backup notes. Use
   [FILES.md](FILES.md) for the dependency map and
   [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) for the proposed benchmark and its limits.

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

## Fresh-session starter prompts

Local AI:

> Act as Pixel Robot Local AI. Read docs/HANDOFF.md, docs/WORKFLOW.md,
> docs/LOCAL_AI.md, the applicable STATUS.md or APP_STATUS.md in docs/, and
> docs/DECISIONS.md. Help the human define a bounded task; request exact phone
> evidence rather than assuming access or unpushed state.

Coder:

> Act as Pixel Robot Coder. Read AGENTS.md, docs/HANDOFF.md, docs/WORKFLOW.md,
> docs/CODER.md, the applicable state and coding instructions, and
> docs/DECISIONS.md. Implement only the approved task, verify it, and automatically
> obtain the designated independent review. Return it verbatim and wait for the
> human's explicit commit decision. Do not push without separate authorization.

Reviewer:

> Act as a fresh Pixel Robot Reviewer. Read AGENTS.md, docs/HANDOFF.md,
> docs/WORKFLOW.md, and docs/REVIEWER.md. Review the actual frozen candidate,
> complete diff including new files, surrounding source, and verification evidence.
> Follow the required review isolation/permissions procedure. Do not edit, stage,
> commit, push, run motors, or launch another reviewer. Return the complete final
> review; it does not authorize a commit.

Doc Keeper:

> Act as Pixel Robot Doc Keeper. Read docs/HANDOFF.md, docs/WORKFLOW.md,
> docs/DOC_KEEPER.md, the applicable state document, and docs/DECISIONS.md.
> Apply only authorized evidence-backed DOC DIFFs, preserve unrelated text and all
> existing decisions, and keep HANDOFF stateless. Record role allocations in
> WORKFLOW.md. Commit approved documentation locally; push only when authorized.
