**CHANGES REQUESTED — one P2 validation defect.** I found no discrepancy in the supplied run’s scores or benchmark FAIL conclusion.

This was a **source-only static review** of this packet. No tools, commands, file reads, edits, web/MCP calls, sub-agents or Ponytail were used. The packet specifies this approach because the Codex read-only sandbox previously could not execute commands in proot. All saved test outputs and phone-run artifacts below are **supplied evidence**; static review cannot prove runtime behaviour.

1. **P2 — Run validity can accept duplicate rows and missing methods.**

   In `benchmark/gemma_vision/bv_compare/bv.py` (**393 lines**):

   - **Line 242:** `        if len(lr) != len(methods) * len(IDS):`
   - **Line 221:** `        got = {x['id']: x for x in rows if x['method'] == m}`
   - **Line 229:** `    if not vis or 'Y' not in mets:`

   Validity checks the total row count per launch, but does not require exactly one row for every expected `(launch, method, id)`. Prediction construction silently collapses duplicate IDs, and the verdict accepts a subset of vision methods.

   For example, launch A could contain two complete V70 sequences and no VY70 sequence: its 122-row count would pass, VY70 would disappear from scoring, and the remaining methods could produce PASS. Require the exact expected row combinations, reject duplicates and unexpected combinations, and require all three vision methods for a complete benchmark verdict. Add an offline regression test.

   **Impact:** this weakens the runner’s VALID/completeness claim. The supplied digest contains the expected method/ID sequences, so this defect does **not** explain or alter the supplied run’s results.

2. **P3 — Known cleanup/finalization limitation, explicitly out of scope.**

   `benchmark/gemma_vision/bv_compare/README.md` (**23 lines**):

   - **Line 22:** `- Known limitation carried over from B0: an interrupt or a server-stop error can skip report.txt and`
   - **Line 23:** `  artifacts.sha256 (completed rows stay in requests.jsonl).`

   This limitation is disclosed and does not affect the supplied completed run.

The supplied key matches the owner’s key text, frame assignments and decisions: **61 selected IDs, 40 person photos**, with the specified room and distance labels. The source pins the reported key hash and checks custody before reading. I did not independently recompute hashes or establish when the key was written.

Method **Y implements the requested rules**: person presence from classes/detections, largest person box by area, the specified bucket map, room rules in the required priority, and class-only person distance as **UNAVAILABLE**, scored wrong. The four affected person IDs are 10, 12, 43 and 44. Detector provenance is supported by the supplied manifest, source excerpts and matching reported `detect_person.py` hashes: **yolo11s_640.onnx**, plain **640×640** resize, confidence **> 0.35**, per-class NMS IoU **0.45**.

The prompt matches the task verbatim. VY70 inserts the required detector line immediately before “Answer,” including the owner-approved unknown position/distance text. Request construction preserves frame bytes through base64 and uses the supplied VIS-3D payload with temperature 0, top_k 1, seed 0, max_tokens 32 and thinking disabled. Server construction matches BC’s underlying command with the b11716 binary and Q8 projector, fixes image tokens to 70 or 140, and adds neither a slot-save path nor MTP. Ordering and pause logic match the prescribed launches and blocks.

Parsing implements the documented, fixed normalization, including offered distance option text. Missing, duplicate or invalid field values become UNPARSED. Metrics correctly use 61 photos for room/person, **40 key-person photos for distance**, require person=yes for distance credit, and compare all three fields for all-three correctness. Distance confusion matrices cover all 61 and are labelled accordingly. The tie interpretation is documented and does not affect this run.

The supplied replies, Y detections and key support these numbers:

| Method | Room | Person | Distance | All three | UNPARSED room/person/distance |
|---|---:|---:|---:|---:|---:|
| Y | 44/61 | 61/61 | 18/40 | 29/61 | 0/0/0 |
| V70 | 31/61 | 61/61 | 16/40 | 9/61 | 0/0/0 |
| V140 | 40/61 | 61/61 | 19/40 | 17/61 | 0/0/0 |
| VY70 | 39/61 | 61/61 | 19/40 | 11/61 | 0/0/0 |

Y additionally has **4 UNAVAILABLE distances**. The supplied wrong-ID lists and confusion matrices agree with the replies. **Benchmark FAIL is correct:** V140 has the best vision room accuracy, **40/61 = 65.6%**, below 80%; its person accuracy equals Y’s.

The offline tests cover the task’s listed areas: Y priority and largest-box rules, class-only handling, parsing/UNPARSED, metrics, PASS, custody, and fake-server execution. They also cover payload and command construction. **11/11 passing is supplied evidence**, not a reviewer execution. They omit the completeness defect described above.

The supplied artifacts support treating this particular run as **VALID evidence for accuracy**, subject to the static-review limits: expected rows, consistent PID/start identity per launch, no recorded request errors, empty end-frame problems, and reported server exits **0 / 0 without SIGKILL**. Timings are correctly labelled **NOT MEASUREMENTS**. Base equality and clean ordinary git status are supplied evidence; the three candidate files remain ignored.

**Final verdict: CHANGES REQUESTED**, limited to the P2 completeness validation and its regression test. This is a review finding, not commit approval.