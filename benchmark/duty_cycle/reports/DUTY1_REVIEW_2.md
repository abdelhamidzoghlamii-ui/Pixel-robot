1. **[P1] Preserve termination signals during exceptional selector draining** — [duty_cycle.py:306](/termux-home/robot/benchmark/duty_cycle/duty_cycle.py:306). Once `error` contains `CoresLost`, subsequent `SystemExit(143)` from SIGTERM is discarded by `error = error or e`, including in the final join loop. The original `CoresLost` then reaches the retry handler, which restarts the block despite the human’s stop request. Preserve termination exceptions while still cancelling/draining and joining the selector before exit. Add an offline test combining core loss, a pending selector, and SIGTERM; assert the selector finishes and no retry starts.

The rate‑1 pacing fix matches POWERMAP. Selector cleanup now joins before retry on the tested core-loss path and retains drain evidence.

HEAD, status, both POWERMAP hashes, and both changed DUTY1 hashes matched. Further inspection encountered the documented bubblewrap unsupported-host-mount error. Review completed from the supplied full candidate, context, and test output; tests were not independently rerun. No edits or hardware execution occurred.

**REQUEST CHANGES**
