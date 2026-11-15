CHANGES REQUESTED

| Scoped item | Status | Assessment |
|---|---|---|
| Round-1 finding 1 — cancellation after responses/scoring | Resolved | Permission and generation are checked after requests, restore and scoring; requested cancellation cases have tests. |
| Round-1 finding 2 — camera deadline | Resolved | A deadline check precedes each cycle; late detected changes are counted without offering events. |
| Round-1 finding 3 — startup barrier | Resolved | The scheduler starts after initial permission assessment. Active and initially forbidden cases have tests. |
| Round-1 finding 4 — resume permission order | Resolved | Fresh SEVERE assessment and permission restoration precede `pause_end`; failure forbids immediately. |
| Round-1 finding 5 — pause accounting | Partly resolved | Intent/completion are separated and failed restarts do not count as resumes. Successful restarts crossing the deadline can still overcount mission pause time; new finding 1 below. |
| Round-1 finding 6 — refusal through report | Resolved | Early refusal receives metrics; a test exercises actual refusal, serialization and reporting. |
| Round-1 finding 7 — integration tests | Partly resolved | The requested original boundaries are covered. Deadline-crossing pause metrics and concurrent re-pin records remain uncovered. |
| Round-1 finding 8 — start-reading time | Resolved | Once mission `t0` exists, the start reading receives its actual relative timestamp. |
| Re-pin change | Partly resolved | Re-pin points, per-thread server affinity, drift validity rules, group records and requested basic tests are present. Concurrent recording has a race; new finding 2 below. |

This review used only the supplied packet. No tools, commands, file reads, edits, web access or sub-agents were used, and no Ponytail skill or mode was applied. As specified, Codex’s read-only sandbox cannot execute commands in proot (#144 R3). Tests, smoke artifacts, hashes and repository-state claims are **supplied evidence**. Static inspection cannot prove runtime behavior.

1. **P1 — A successful restart crossing the deadline overcounts pause time in a potentially VALID arm.**

   `benchmark/mind_picker/b5_mission/b5.py:717` — **1,000 lines**:
   > `out.append((began, r['t']))`

   `benchmark/mind_picker/b5_mission/b5.py:641` — **1,000 lines**:
   > `rec['mission_end_s'] = min(m.rel(clock()), ctx['dur'])`

   `benchmark/mind_picker/b5_mission/b5.py:751` — **1,000 lines**:
   > `'active_share': round(1 - total / end_t, 3) if end_t else None,`

   The owner explicitly allows an underway transition to finish after the deadline. Its successful ACTIVE `done` row can therefore be later than `mission_end_s`. Closed pause intervals use that uncapped completion time, while the denominator is capped to mission duration.

   For example, a pause beginning at 100 seconds and a restart completing at 1,802 seconds yields 1,702 seconds of pause against an 1,800-second mission, although only 1,700 seconds occurred inside mission time. This can pass the 15-second overrun rule. More extreme boundary cases can produce negative ACTIVE share.

   **Concrete fix:** Intersect every pause interval with the mission interval `[0, end_t]` before summing. Preserve the actual transition completion timestamp and successful-resume record separately. Extend the fake-clock deadline-crossing test to assert pause duration and ACTIVE share when camera start itself completes after the deadline.

2. **P2 — Concurrent re-pin recording can attach another operation’s record to a watcher scan.**

   `benchmark/mind_picker/b5_mission/b5.py:439` — **1,000 lines**:
   > `self.events.append({'mono': round(time.monotonic(), 3), **ev})`

   `benchmark/mind_picker/b5_mission/b5.py:440` — **1,000 lines**:
   > `return self.events[-1]`

   `benchmark/mind_picker/b5_mission/b5.py:502` — **1,000 lines**:
   > `x['repins'] = [rp.server('watch') if p == 'llama_server' else rp.detector('watch') for p in sorted(drift)]`

   The append is locked, but selecting the returned event happens after releasing the lock. A camera-start operation on the mission thread can append between a watcher append and its return. The watcher then stores the camera-start event in its scan’s `repins`, potentially with the wrong trigger or process.

   The global event list retains both entries, so this does not itself demonstrate an incorrect final validity label. It does break the requested association between a drift scan and its corrective operation. Existing re-pin tests run these operations sequentially.

   **Concrete fix:** Construct a local event object, append that object under the lock, and return that same object. Add a deterministic concurrency test verifying that each caller receives its own event and that watcher samples retain the matching watch-triggered correction.

The original fixes otherwise match the owner’s decisions in the visible source. In particular, cancellation checks cover returned HTTP responses and post-score work; the startup callback follows `set_ok`; resume permits work before the reading; and the refusal path now produces report-compatible metrics. The new test source exercises the requested original cases, including exact eligible A marks, nonzero restart/read durations, final-response cancellation and refusal-to-report.

The re-pin implementation uses explicit thread IDs for server affinity changes and the detector’s existing explicit-thread helper. Calling it from the watcher does not inherently target the watcher itself. Drift is separated from other watcher hits, failed recorded re-pins invalidate, and consecutive scans are compared by process. The cpuset/cgroup values and summary contain JSON-compatible data. Apart from finding 2, no additional definite thread-safety or serialization defect was established from this packet.

The **supplied evidence** reports 66 passing offline tests. The Coder’s unarchived mutation-check claims cannot be independently assessed.

The **supplied S3 evidence** supports the smoke prerequisite and the principal smoke claims:

- A records two pauses and one successful resume; B records one of each, with subsequent camera cycles.
- A serves its initial mark at 0.001 seconds. Its four decisions match the eligible marks shown in this trace.
- B’s pending item starts at 123.458 seconds, 0.042 seconds after successful ACTIVE re-entry at 123.416.
- Reported pause totals reconcile with the supplied transition rows: A 64.437 seconds and B 65.291 seconds.
- Both arms report eight watcher scans, zero hits, zero drift and zero failed re-pins. Recorded group values remain `/top-app`.
- Start readings have negative relative timestamps. Supplied cleanup records report server exit zero, restored screen timeout and a stopped logger.

S3 contains no corrective re-pin, cancelled decision, real event or deadline-crossing transition. It therefore neither exercises the two new findings nor establishes those untested phone paths. The earlier INVALID T1 remains evidence motivating the re-pin change, not validation of the current implementation.

This static review cannot establish native affinity behavior under Android group changes, actual concurrent interleavings, hardware safety, capture cessation, thermal or battery behavior, complete transitive import behavior, or 30-minute sustainability. It cannot independently verify hashes, repository status, HEAD/origin equality or artifact checks. No commit, deployment or timed run is approved by this verdict.