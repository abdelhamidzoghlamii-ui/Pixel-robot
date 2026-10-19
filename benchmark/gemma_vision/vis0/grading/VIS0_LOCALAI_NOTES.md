# VIS-0 — Local AI quality notes (INDICATIVE)

Run: benchmark/gemma_vision/vis0/runs/20261018T010701Z (vis0_sheet.pdf, SHA-256 5aa8cd5b…f990f).
The grading was done by Local AI on 2026-10-18 from the PDF. It covers 10 frames, was not blind, and is INDICATIVE only.
Timings are NOT measurements (agent resident).

## Template check (VALID, deterministic)

CONFIRMED. The Gemma 3 markers (`<start_of_turn>`, `<end_of_turn>`) are plain text for this GGUF. The Gemma 4 `<|turn>` / `<turn|>` markers are single special tokens.

Text replies across the 10 frames:
- OLD markers: 9 of 10 replies broke into repetition or were cut off.
- NEW markers (GGUF template): 10 of 10 were clean.

## Vision replies, per variant

| Variant | Wrong statements | Missed main subject | Notes |
|---|---|---|---|
| Photo, budget 70 | 1 ("pink chair" = a bag) | person missed in 1 frame | otherwise correct, short |
| Photo, budget 140 | 0 clear | people missed in 2 frames | slightly more detail |
| Photo, budget 280 | 1 ("knife and cutting board") | people missed in 3 frames | odd framing in 1 reply ("in front of the person"); not better than 140 |
| Photo + YOLO, budget 140 | 2 (both "sink" errors copied from YOLO in the bathroom frames) | none | best overall; very good on 8 of 10 |

## Conclusions (INDICATIVE)

- Photo + YOLO at budget 140 gave the best descriptions.
- A larger image budget did not improve quality on these frames.
- Vision repeats YOLO's labelling errors when the YOLO list is added to the prompt.
- The speed (about 21 s at budget 70, 54 s at 140, 109 s at 280 per image; NOT measurements) needs an owner-timed test (VIS-1). The BF16 mmproj on a CPU without BF16 support is a suspected cause.
