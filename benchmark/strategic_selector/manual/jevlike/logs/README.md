# robot-jevlike run logs (2026-09-25)

Worker logs and the last request from manual `robot-jevlike` sessions, copied from `/termux-home/jevlike/`
on 2026-09-26. Status: **complete**. They are manual playground runs, not a benchmark: no score is
derived from them. One file per model worker (`<model>.log`, stderr of `worker.py`; `s1o.server.log` is
the llama-server log of the s1o worker) and `request.txt`, the last request typed in the editor.

The logs predate the 2026-09-26 cleanup, so they still include the `laya_micro` and `von12` entries
that were removed from the jevlike menu after their model files were deleted. `laya_multi.log` and
`s1o.log` are empty in the original. Not reviewed when recorded. [ARTIFACTS.md](ARTIFACTS.md) hashes
every file.
