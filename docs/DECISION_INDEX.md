# Decision index — current orientation

This is a short reading guide, not another decision log. The numbered entries in
[DECISIONS.md](DECISIONS.md) remain the authority and retain their original text.
Read [STATUS.md](STATUS.md) or [APP_STATUS.md](APP_STATUS.md) for what is currently
implemented. Before appending a decision, read the last number in DECISIONS.md
itself. An entry labelled *decided* here may still be unimplemented.

| Area | Current orientation | Evidence |
|---|---|---|
| Prototype navigation | Python rules and sensors are intended to own low-level movement; removal of Gemma from the navigation loop is **decided, not implemented**. The as-coded triggers remain. | #104, #88, #108; STATUS Architecture |
| Strategic selector | Laya or Von as a high-level mission-script selector is **research only**. No chosen model, live map/sensor fusion or motor integration. | #109; [archive](../benchmark/strategic_selector/README.md) |
| Motor safety | Firmware watchdog is 1000 ms; the `nav_stuck` safety-move override was fixed in Python; LLM failure paths now select STOP. Hardware validation remains separate. | #47, #88, #108; STATUS Pending |
| Firmware and wiring | Corrected `mode2_auto.ino` and corner map are committed; what is flashed on the physical board is unconfirmed. | #43, #89; STATUS Architecture |
| Person stop | Coarse detector area bucket was chosen to replace uncalibrated centimetres, but is not implemented. | #78–#80; STATUS Pending |
| Thermal control | VIRTUAL-SKIN-CPU-GPU governs HAL CPU throttling; the live `main.py` pause still reads BIG at >80 °C. The replacement threshold is not decided. | #74–#75, #90–#93; STATUS Thermal/Pending |
| Model vision | Deployed image requests were effectively text-only; later projector experiments did not yield usable on-device LLM vision. YOLO remains perception. | #96, #105; STATUS Vision |
| Android app | Root-free USB and sustained inference spikes passed; the native app itself has no code yet. Prototype measurements do not transfer automatically. | #67–#69; APP_STATUS |
| Licensing | [LICENSES.md](LICENSES.md) owns the app allowlist. Its derived copy in APP_CLAUDE.md is intentional. | #66 |
| Collaboration | Independent Coder/Reviewer/human commit gate and per-task execution profiles are defined in [WORKFLOW.md](WORKFLOW.md). Push needs separate authorization. | #86–#87 |
| Strategic selector harness v3 | s1o dropped on speed; laya_en and von11 proceed; hardware timings provisional. | #112 |
| Data Engineer role | Selector moves to a fine-tuning track; new Data Engineer role created to generate the dataset. | #113 |

## Corrections worth knowing before citing an older entry

| Earlier record | Later correction or change |
|---|---|
| #33 front/rear motor wording, then #40 left/right mapping | #43 corrects the left-side corner assignment; #89 identifies the corrected committed firmware. |
| #52's avoidance escalation | #59 changes the counter logic; #62 limits Gemma consultation to once per blockage episode. The planned LLM-free nav direction is #104 and remains unimplemented. |
| #81 open `nav_stuck` safety defect | #88 records the code fix; #108 records the later STOP fallback for LLM failures. Do not treat #81 as an open code defect. |
| #84 says seven orphaned scripts were deleted | #85 verifies they remain tracked and explicitly defers deletion. |
| #77 records a built b2233 binary | #107 says its binary was replaced by b2351; the old source commit is still recoverable. |
| #95's negative `strings` check for Qwen support | #107 / STATUS report a successful Qwen3.5 text load on b2351; #105 separately records unsuccessful vision tests. |
| #99's proposed model-in-loop work order | #100–#102 tested output and rule defects; #104 changes the intended architecture. #109 is separate strategic research. |

For any claim outside these selected topics, search the numbered log and inspect
the later entries that cite it. This index makes no claim to classify all decisions.
