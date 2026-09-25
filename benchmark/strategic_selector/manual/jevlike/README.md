# robot-jevlike — manual selector test

Interactive terminal playground for the strategic-selector models. Research
only: offline, no motors, does not touch `main.py`. Not part of any automated
run; results are for a human to read, not a benchmark score.

You write a Jev-shaped request (one `INPUT`, any number of `QUESTION`s with
`-` choices) in `$EDITOR` (default `nano`), then compile it on one model or
compare several. Output per question: ranked probabilities, choice, decision
ms, tokens read vs the model's limit, tok/s, margin, confidence; per compile:
load time, total time, model RAM. Toggles: rerun with reversed choices, print
the exact text sent to the model. The editor template lists the advised
maximum choices/questions per model.

Each model runs as `worker.py` inside its own venv (JSON lines over
stdin/stdout), reusing the unmodified v3 harness adapters in `../../v3`
(`adapters.py`, `measure.py`), which must be present. One model is loaded at a
time; s1o starts and stops its own llama-server; nothing is left running on
quit or Ctrl-C. Worker logs and the last request live outside the repo in
`/termux-home/jevlike/`.

## Launch

`robot-jevlike`, from native Termux or inside Debian, after installing:

```bash
# inside Debian
cat > /usr/local/bin/robot-jevlike <<'SH'
#!/bin/bash
exec python3 /termux-home/robot/benchmark/strategic_selector/manual/jevlike/jevlike.py "$@"
SH
chmod +x /usr/local/bin/robot-jevlike

# native Termux wrapper ($PREFIX/bin), enters the Debian proot
cat > /data/data/com.termux/files/usr/bin/robot-jevlike <<'SH'
#!/data/data/com.termux/files/usr/bin/bash
exec /data/data/com.termux/files/usr/bin/proot-distro login debian --bind /data/data/com.termux/files/home:/termux-home -- /usr/local/bin/robot-jevlike "$@"
SH
chmod +x /data/data/com.termux/files/usr/bin/robot-jevlike
```

Parser check: `python3 test_jevlike.py`.
