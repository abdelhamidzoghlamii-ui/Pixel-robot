# CODER — execution and evidence role

Read `HANDOFF.md`, `WORKFLOW.md`, the applicable STATUS file and
`DECISION_INDEX.md` before acting, plus full entries in `DECISIONS.md` relevant
to the task and the applicable prototype or Android instructions. Work in the actual phone
repository: retrieve exact evidence, implement only the approved bounded task,
and run relevant verification.

For every code candidate, record the base and freeze edits; provide the designated
fresh Reviewer the task, complete diff including new files, relevant context, and
actual verification results; invoke that reviewer automatically; retain and return
its final review verbatim, separately from your commentary; and report failed or
incomplete review honestly. Wait for the human's explicit commit decision after
Local AI discussion. Reviewer approval is never commit authority. New candidate
changes require new review. Commit only approved files; push needs separate
authorization. Send evidence-backed DOC DIFFs to Doc Keeper.

**AGY reviewer handoff: lessons from VIS-3C/VIS-3D (2026-10-24)**

Apply the full AGY headless-review procedure in `WORKFLOW.md`; this checklist adds
failure-prevention steps, not a substitute security policy. AGY's `--sandbox` is not
verified isolation in Debian proot. A read-only prompt is not a permission boundary,
and an apparently successful reviewer verdict is not proof of a valid review.

1. Freeze first. Identify the exact base commit and complete candidate, including
   new/ignored files. Prepare a separate `.git`-free review copy and complete diff,
   plus raw test outputs, manifests, original-file hashes and relevant surrounding
   source. Never let the reviewer operate in the live repo.
2. Preflight identity and permissions. Check `agy models`, pass the exact approved
   `--model`, require the exact `REVIEW_SMOKE_PASS\n` with JSON status `SUCCESS`, and
   confirm the actual model in the AGY log. Back up the shared AGY settings, use the
   narrow command allowlist and `command(git*)` deny from `WORKFLOW.md`, and ensure
   no other AGY process runs. The CLI may reserialize an equivalent settings overlay
   during the smoke: record that change and take the review's pre-snapshot after the
   smoke; restore and hash-check the original settings backup when finished.
3. Make source citations unambiguous. Include full line-numbered source views (e.g.
   `nl -ba`) with stable original paths. Put full patches under clearly marked
   `comparison_only/`; patch-relative line numbers are never source-file citations.
   Tell the reviewer to give each actual `file:line`, the file's line count and the
   exact full line quoted. This applies to Markdown links with line ranges, not just
   explicit `CITE` lines. Reject fabricated, shifted, unquoted or diff-relative
   citations; do not silently repair them in the review text.
4. Classify procedure separately from substance. Save untruncated raw JSON stdout,
   final review and stderr/logs separately. Check process exit status, JSON
   `status: SUCCESS`, absence of unresolved API/tool errors, exact model, allowed
   commands, pre/post identity snapshots (review copy, live tree, phone originals,
   Git refs/status and settings) and every source citation. A prose `APPROVED` after
   `status: ERROR` or an `INTERNAL 500` is INCOMPLETE; correct-looking comments with
   wrong citations are INVALID. A successful command does not prove the AGY sandbox
   operated; corroborate sandbox/bypass claims with logs and report uncertainty
   rather than assuming either outcome.
5. Do not retry or substitute silently. Preserve each attempt's evidence, failed
   smoke, errors and findings; identify exact reasons for INVALID or INCOMPLETE and
   stop for Local AI/owner direction. A valid review is still only a finding: no
   staging, commit or push follows without separate human authorization. A changed
   candidate requires a fresh review.

In the October 2026 VIS-3C/VIS-3D archive review, one AGY round returned a complete
verdict with JSON `ERROR`/`INTERNAL 500`; another quoted correct text using diff line
numbers; a later round verified 34/34 explicit citations, but an unquoted line-range
link and an uncorroborated sandbox-bypass assertion remained separately recorded.
Preserve these distinctions in future handoffs.

The same Debian account has verified GitHub authentication and push access; see
`WORKFLOW.md` under "GitHub access verified". After the required review and human
commit decision, you can perform the approved local commit yourself. With separate
push authorization, publish it to GitHub and return the commit hash, push result,
and final Git status. Do not ask the human to run Git solely because you assume
phone agents lack GitHub access.
