# MIND-B0b — keeping picker and chat contexts warm: slot save/restore vs two slots

Research only (#146), extending MIND-B0 (`../b0_cache/`, imported unchanged; frozen set SHA-256 `dddb288a…` checked
before every launch). B0 showed the picker rebuilding 428 of 434 tokens after a chat in a separate context (R7). This
asks whether (L1) saving/restoring the picker slot or (L2) a second slot keeps both contexts warm, what each costs in
tokens and RAM, and whether picker probabilities stay unchanged. Counters, probabilities and slot file size: VALID
when the criteria's checks hold; timings NOT MEASUREMENTS; RAM INDICATIVE (agent resident allowed).

- Launches L0 (B0 reference), L1 (`--slot-save-path`), L2 (`--parallel 2`, ctx 4096), plans and criteria K0–K5:
  see the docstring of `b0b.py`. Each launch = one server start, one block.
- Server facts used (pinned b1609 binary and its source on the phone, read-only):
  - `--help` lists `--slot-save-path`, `-np/--parallel`, `-kvu/--kv-unified` (default enabled only when the slot
    count is auto), `-sps/--slot-prompt-similarity` (default 0.10), `--slots/--no-slots` (default enabled).
  - `/completion` honours `id_slot` (`server-context.cpp`: `task.id_slot = json_value(data, "id_slot", -1)`, then
    `get_slot_by_id`); the result reports `id_slot`.
  - Per-slot context: `n_ctx_seq = n_ctx / n_seq_max` (padded to 256) unless `kv_unified`; with `--parallel 2` and
    ctx 4096 the server logs `n_slots = 2, n_ctx_slot = 2048, kv_unified = 'false'` and `/slots` shows 2048 each.
  - `POST /slots/0?action=save|restore` with `{"filename": ...}` needs `--slot-save-path` (an existing directory);
    save writes the slot's KV and tokens (`llama_state_seq_save_file`), restore loads them and sets the slot's
    tokens; the responses give `n_saved`/`n_written` and `n_restored`/`n_read` and `save_ms`/`restore_ms`.
- Run: `python b0b.py L0`, then `L1`, then `L2` (Termux python, also from proot Debian); one at a time. Writes
  `~/robot_runs/mind_b0b/<UTC>_<L>/` (run.json, requests.jsonl, server.log, slots/ for L1, report.txt,
  artifacts.sha256).
- Offline tests: `python -B test_b0b.py` (fake slot server built on B0's fake; no phone).
- Known limitation carried over from B0: an interrupt or a server-stop error can skip report.txt and
  artifacts.sha256 (completed rows stay in requests.jsonl).
