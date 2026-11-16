# Deleted model files

| Date (UTC) | Path | Bytes | SHA-256 | Decision | Reason |
|---|---|---:|---|---|---|
| 2026-11-16 | `~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` | 4368439584 | `3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9` | owner decision 2026-11-16 (CLEANUP-C, D1) | used by nothing live |

Reference check before deletion (HEAD `6af0812`, `git grep -i mistral` outside `benchmark/` archives):
- `chat.py:16` lists it in `MODEL_FILES`; the standalone-chat menu shows only files that exist (`chat.py:10-13`), so
  the entry simply disappears from the menu.
- `test_e4b_mtp.py:23`, `test_tplfix.py:30,39` check the static list and the Mistral prompt format; no file is opened.
- `docs/STATUS.md:304` lists "Mistral 7B Q4_K_M" among installed chat models; stale after this deletion.
- No reference in `~/bin`, `~/*.py`, `~/*.sh`, `~/.bashrc`.

Not archived to Drive. Upstream source: `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (public GGUF quantization); verify a
re-download against the SHA-256 above.
