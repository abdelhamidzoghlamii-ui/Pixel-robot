# Frozen task given to the Coder (owner text, condensed by the Coder; requirements and test definitions kept)

PIXEL ROBOT — VIS-3D: Fair Performance and Vision Benchmark

Authorization: benchmark implementation, offline tests, and independent review only. Do not launch llama-server or
run timed inference; do not start motors, camera services or robot missions; do not change production code, models,
llama.cpp binaries or existing benchmark implementations; do not commit, push or modify canonical documentation; do
not automatically execute a sequence of hardware benchmarks.

Confirmed performance baseline VIS-3C, October 23, 2026: original b1609-dotprod MTP OFF 13.35 tok/s, MTP ON 18.55
tok/s, draft acceptance 38.2 %, both VALID, configuration SAME, conditions MATCHED; 13 historical prompts, ctx 4096,
temperature 0.7, n_predict 512, cores 4-7, sparse 5 s telemetry. Folders ~/robot_runs/chat_mtp_repro/20261023T124846Z_OFF
and 20261023T132032Z_ON. Earlier VIS-3B tests reported lower rates under intensive telemetry; monitoring interference
is a leading hypothesis, not a proven cause.

Objective: performance and usability of the original Gemma 4 E2B Q4_0 on the vision-capable b11716 binary, with
optional Q8 vision projector and Q8 MTP drafter. Four separately launchable tests:
A — b11716, Gemma E2B Q4_0, no projector, MTP OFF, frozen VIS-3C Oct 14 text workload; compare with 13.35 tok/s.
B — b11716, same model, Q8_0 projector loaded, MTP OFF, exactly the same text workload, no images. A vs B isolates
    projector residency. Measure load time, text throughput, prompt time, memory, thermal.
C — b11716, same model, Q8_0 projector and original Q8_0 E2B MTP drafter loaded, same text workload. B vs C = MTP
    with projector resident. C vs historical b1609 MTP-ON 18.55, clearly labelling projector residency. Verify actual
    speculative flags, model and drafter hashes, draft acceptance; never report MTP enabled merely because requested.
D — b11716, Q8 projector, MTP OFF; fixed existing local images with traceable provenance; no camera; correct Gemma 4
    multimodal request format and chat template; inspect existing image-token-budget evidence and choose a bounded,
    practical configuration; measure image-processing latency, total wall time, text decoding, prompt evaluation,
    loading and peak memory separately wherever instrumentation genuinely supports it; record failures, incomplete
    responses and unsupported metrics. Small image-quality assessment (identifiable objects, scene description,
    obvious hallucinations, robot-context errors) against verified contents or manual references, not filenames.
    Preserve the legacy Oct 14 template for speed only; use the verified Gemma 4 template for D. No MTP-with-image test.

Technical requirements:
1. Prefer small reusable components from reviewed VIS-3C and existing vision runners; no duplicate large framework.
2. Confirm b11716 source identity, build flags, effective command, loaded library paths and hashes; do not assume the
   Q8 projector path or hash from the BF16 VIS-0 archive.
3. Each test independently launchable; exactly one test per owner command; no follow-on launches.
4. Sparse telemetry comparable with VIS-3C (existing 5 s sampler); no dense VIS-3B monitor, repeated smaps_rollup or
   high-frequency polling during requests.
5. Preserve CPU affinity 4-7, four inference threads, frozen text-workload settings.
6. New server instance for every projector/MTP configuration.
7. Safety preflight: native Termux, no coding agents, charger unplugged, adequate battery, thermal status, safe battery
   temperature, CPU availability, single-run lock, clean shutdown, restoration of modified screen settings.
8. Save complete commands, model/projector/drafter hashes, build-library hashes, exact workload and response
   metadata, per-request timings, telemetry samples and an explicit validity label.
9. Historical-comparable token-weighted throughput for text tests; do not substitute the VIS median-of-request-speeds.
10. Report-only commands comparing A, B, C with each other and with the VIS-3C baseline; configuration and
    environmental comparability reported separately.
11. Distinguish speculative-verification-step counts from proxies such as generated - accepted.
12. No unverified claims that the projector causes a slowdown or that the new binary is faster or slower.
13. Offline tests with mocks and fixtures only: flags, hashing, workload identity, image request serialization, timing
    aggregation, invalid runs, timeouts, cleanup, report comparability.
14. Do not alter existing VIS-3C, VIS-3B or historical archived results.

Base: main HEAD 685a44be63aa88c30e09386ce8e5e196ebf11abc (= origin/main = GitHub). benchmark/* is git-ignored, so
the candidate (benchmark/gemma_vision/vis3d/, 5 new files) and the VIS-3C dependency
(benchmark/llm_objective_setting/conversation/chat_mtp_repro/, untracked, unchanged, reviewed in VIS-3C round 3) are
local only.
