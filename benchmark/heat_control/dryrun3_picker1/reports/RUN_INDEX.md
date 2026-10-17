# ARCHIVE-2 run index — reports

Fixed labels below describe the related benchmark evidence, not the creation time or measured result of these historical code-task reports.

Fixed labels (related complete session where this is a report/partial-attempt folder):

Motors-off `run_mission.py --dry` with the picker and M2, code 2ce2b3b (PICKER-1), setup_q4 server, owner-run, native Termux, 2026-10-16 10:31:51–14:10:17Z, 3.64 h, stopped by Ctrl-C (Ctrl-C also stopped `tee`: the console log lacks the shutdown lines; the heat trace is complete). Owner stood in front of the camera about 3 times. 372 picks traced, 0 failures; 3 picks cancelled at an ACTIVE->PAUSE change, as designed. Every pick chose EXPLORE, including all 59 picks with a person in view (EXPLORE probability 0.93–0.98, median 0.95): the mode never changed. Gemma per pick median 2,767 ms (1,842–6,047; p90 3,289); M2 median 730 ms (693–1,130); prompt 116–214 tokens. Person-stop held a forward move 168 times (close 105, very close 63). Heat: 98 pauses, first after 176 s; 86 of 98 pauses began at the 60 s minimum ACTIVE time; ACTIVE share 0.48 of the run; skin peaked at 39.69 °C right after a minimum-length ACTIVE slot; Android status max 1 (LIGHT), never SEVERE. 3,457 ACTIVE cycles, cycle period 1.82 s; slot-median detect 141–171 ms (320) and 526–679 ms (640); max frame age 1.37 s; 0 camera stops; YOLO affinity cpus 4–5 in all 99 slots; M2 cpus 6–7, no re-pin. VALID as one motors-off functional and heat run; pick quality NOT EVALUATED there (see the desk tests); not a motors-live result. Phone checks (S7) are untimed functional checks with the agent resident, NOT measurements.

| Run/file | UTC start / provenance | Label |
|---|---|---|
| `picker1/frozen_manifest.json` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/frozen_manifest_r1.json` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/frozen_manifest_r2.json` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/frozen_manifest_r3.json` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/frozen_manifest_r4.json` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/request_r1.txt` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/request_r2.txt` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/request_r3.txt` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/request_r4.txt` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/review_r1.txt` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/review_r2.txt` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/review_r3.txt` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/review_r4.txt` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
| `picker1/PICKER1_CODER_REPORT.md` | UNKNOWN (historical record; no canonical UTC start) | Verbatim PICKER-1 coder/reviewer/request/manifest record; not a timed run. |
