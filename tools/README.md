# Operator tools (copies)

Copied 2026-11-16 (`cp -p`, byte-identical) from the phone so they exist outside it. **The originals in `~` and
`~/bin` remain the ones in use**; docs keep referring to those paths. Editing a copy here changes nothing on the phone.

| File | Original | What it does | Where the docs use it |
|---|---|---|---|
| `kill.py` | `~/kill.py` | SIGKILL every `llama-server` found with `pgrep -x` (root) | `docs/COMMANDS.md:474` (`kill` alias); `docs/DECISIONS.md:766` (why `-x`) |
| `prep.py` | `~/prep.py` | free RAM by killing a hardcoded list of user apps, then drop caches (root) | `docs/COMMANDS.md:477` (`prep` alias) |
| `ramfree.py` | `~/ramfree.py` | `off`/`on`: disable/restrict heavy Google apps and restore them (root). Do **not** use `off` (#94) | `docs/COMMANDS.md:479` |
| `json_mechanism_test.py` | `~/json_mechanism_test.py` | read-only diagnostic: structured-action output mechanisms on synthetic text scenes | `docs/STATUS.md:28` |
| `actions.gbnf` | `~/actions.gbnf` | GBNF grammar for the old `llm.py` JSON actions | `docs/FILES.md:90` (`llm.py` needs it) |
| `robot-agent` | `~/bin/robot-agent` | menu: start Codex, AGY or Claude | not in `COMMANDS.md` |
| `robot-agy` | `~/bin/robot-agy` | start AGY in Debian proot from native Termux, cwd `~/robot` | not in `COMMANDS.md` |
| `robot-claude` | `~/bin/robot-claude` | start Claude Code in Debian proot from native Termux, cwd `~/robot` | not in `COMMANDS.md` |
| `robot-codex` | `~/bin/robot-codex` | start Codex in Debian proot from native Termux, cwd `~/robot` | not in `COMMANDS.md` |

Not copied: `~/bin/robot-chat` is a symlink to the tracked `robot-chat` (`docs/COMMANDS.md:272`);
`~/bin/robot-codex.bak` is an old backup (ignored `*.bak`).
