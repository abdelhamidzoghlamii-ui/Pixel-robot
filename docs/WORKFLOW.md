# WORKFLOW — Pixel Robot roles and commit gate

This is the canonical collaboration workflow. It is procedural, not a Git hook or
an enforced sandbox unless a later decision says a specific mechanism was actually
implemented and tested. Start with `HANDOFF.md`, then this file, then the role file;
read `STATUS.md` and `DECISIONS.md` directly for current facts and numbering.
`AGENTS.md` is the common Codex/Claude/AGY entry point; `CLAUDE.md` and `GEMINI.md`
are minimal pointers. Prototype and app instructions remain separate.

## Roles and allocation

Roles are independent of providers and models. Record the actual assignment at
handoff; do not silently substitute the selected reviewer/model. If one provider
fills multiple roles, use separate role sessions. The coder never reviews its own
work in the same conversation.

| Role | Ordinary preference | Current allocation |
|---|---|---|
| Local AI: online-app planning and technical orchestration with the human | Local AI app thread | Fresh online thread |
| Coder/data retriever: actual phone repo, exact evidence, approved changes and checks | Claude Code | Codex |
| Reviewer: fresh independent candidate review | Codex | Gemini 3.1 Pro High through AGY |
| Doc Keeper: canonical status and append-only decisions | Gemini / AGY | Gemini / AGY after this handoff |

Claude Code is temporarily parked because the human reported a weekly limit; this
is availability, not a permanent technical restriction. The human reports paid
subscriptions to all three providers; no exact plan, allowance, or reset date is
assumed.

Local AI sets objectives, prepares bounded tasks and exact phone commands, requests
source/output from the executor, evaluates evidence and reviews, and helps the
human decide whether to commit. It does not assume phone access or unpushed state.
Coder/data retriever implements and verifies in the actual repository. Reviewer
examines actual candidate changes and surrounding code, not merely the coder's
summary. Doc Keeper owns `STATUS.md`/`APP_STATUS.md` and `DECISIONS.md`; all other
roles send targeted evidence-backed DOC DIFFs.

## Code-change and commit gate

For every code change: (1) Local AI and human define the task; (2) Coder implements
and verifies; (3) Coder automatically invokes the designated independent Reviewer,
without human relay; (4) request includes task, base commit, full candidate diff
including new files, surrounding context and actual checks; (5) freeze edits while
reviewing; (6) Reviewer does not edit, stage, commit, push, run motors, or launch
another reviewer; (7) retain diagnostics separately and return the final review
verbatim, untruncated and separate from Coder commentary; (8) stop for the human's
explicit commit decision after Local AI discussion; (9) any code change after review
requires new review; (10) commit only approved files and push only with separate
authorization; (11) send DOC DIFFs to Doc Keeper.

“Automatically” means no manual relay between applications. It never hides review
findings, failure, costs/limits, or permission errors. Never use blanket permission
bypasses. Reviewer approval is not authorization to commit.

## Phone sessions

From native Termux:
```bash
proot-distro login debian --bind /data/data/com.termux/files/home:/termux-home
```
Inside Debian, choose one alternative session:
```bash
cd /termux-home/robot
codex
```
```bash
cd /termux-home/robot
/root/.local/bin/agy
```
```bash
cd /termux-home/robot
claude
```
These are alternatives, not a sequence inside one prompt. Use separate terminals
or exit the current agent first. Native Termux's repository and
`/termux-home/robot` are the same bind-mounted files.

User-provided environment evidence: official `@openai/codex` 0.153.4 via npm with
Node 20.19.2/npm 9.2.0, launcher `/usr/local/bin/codex`; AGY at
`/root/.local/bin/agy`, version 1.1.27; Claude previously ran in Debian. AGY
started successfully on this Pixel, so it is not categorically unsupported.

## AGY headless reviewer

Installed model evidence: `agy models` lists
`gemini-3.1-pro-high  Gemini 3.1 Pro (High)`. The authorized smoke call:
```bash
agy -p 'Reply exactly: REVIEW_SMOKE_PASS. Do not inspect files, run commands, call tools, or make changes.' \
  --model gemini-3.1-pro-high --output-format json --sandbox --print-timeout 2m
```
returned `{"status":"SUCCESS","response":"REVIEW_SMOKE_PASS\n"}`.
The concrete invocation pattern is:
```bash
agy -p "$REVIEW_REQUEST" --model gemini-3.1-pro-high \
  --output-format json --sandbox --print-timeout 10m
```
`$REVIEW_REQUEST` contains the frozen task, base, complete diff including new
files, source context, check output, no-mutation rules, and a final-review request.
Preserve stdout as the complete JSON review and stderr as separate diagnostics. It
is a direct Coder action, not an executable helper.

Official [AGY headless](https://antigravity.google/docs/cli/headless/) documentation
says headless workspace reads and writes are auto-allowed by default. A prompt
alone is therefore not an enforced read-only boundary. `--sandbox` was smoke-tested,
but sandboxed workspace paths remain writable when permissions allow. Never use
`--dangerously-skip-permissions`.

A true read-only reviewer needs a dedicated AGY configuration or OS account, not
the shared Doc Keeper configuration. Its `settings.json` must use strict,
sandboxed permissions and deny mutation tools:
```json
{"enableTerminalSandbox":true,"toolPermission":"strict","permissions":{"deny":["write_file(*)","command(*)","mcp(*)","execute_url(*)"]}}
```
Official permission precedence is deny before ask before allow. The current shared
AGY settings have only a chosen model and trusted workspace, so this dedicated
deny policy is not provisioned or tested; do not add it globally because it would
block Doc Keeper. Until a separate setup is tested, review an isolated copy of the
exact frozen candidate to protect the live worktree. Isolation is not an enforced
read-only sandbox. Authentication, quota, model, or permission failures mean
incomplete review, never a pass. Sources: [permissions](https://antigravity.google/docs/cli/permissions/)
and [sandbox](https://antigravity.google/docs/cli/sandbox/).

## GitHub access verified

GitHub CLI authentication in Debian completed as `abdelhamidzoghlamii-ui`.
The phone executor successfully pushed commit `053d5a5` to `origin/main` at
`https://github.com/abdelhamidzoghlamii-ui/Pixel-robot.git`.

Coder and Doc Keeper sessions using this same Debian account can make local Git
commits and push authorized commits to GitHub, subject to their tool permissions
and valid credentials. Other accounts or environments do not inherit this access
automatically. A Git commit is local; a push publishes it to GitHub.

Local AI should ask the phone executor to perform authorized Git operations and
return the commit hash, push output, and final Git status. The online planning
thread does not gain phone access or visibility into unpushed work from this login.
Authentication establishes capability, not blanket authorization: Coder still
needs independent review and the human's commit decision; Doc Keeper commits
approved documentation; pushes require separate human authorization. Reviewer
remains read-only. If credentials fail, report the failure without exposing tokens.
