# LICENSES

Every component shipped in, or copied into, the Pixel Robot Android app.

**Rule:** a dependency is recorded here **before** it is added, not after. A row with
a PENDING licence is a blocker, not a to-do.

```
allowed     MIT, Apache-2.0, BSD-2/3, Android platform APIs
forbidden   AGPL (any version), GPL
review      LGPL — case by case, never assumed acceptable
```

This file is canonical for the allowlist. `APP_CLAUDE.md` §0.5 carries a derived
copy for agent reliability (DECISIONS #66).

**Scope:** anything that ships inside the APK, plus any third-party source copied
into the repo. Build-time tooling that never ships (Android Studio, Gradle plugins,
Claude Code) is out of scope.

**What `verified on` means:** a human or agent opened the LICENSE file at the pinned
version and read it. It is *not* the date someone read a README badge, a package
index entry, or a blog post. PENDING is the honest value until that has happened.

---

## Shipped in the APK

| Component | Licence | Source / pinned version | Obligation | Verified on |
|---|---|---|---|---|
| llama.cpp | MIT | `ggml-org/llama.cpp` — version PENDING | Retain copyright and licence text in the app's open-source notices | PENDING |
| whisper.cpp | MIT | `ggml-org/whisper.cpp` — version PENDING | Retain copyright and licence text | PENDING |
| Gemma 4 E2B weights (`gemma-4-e2b-it-q4_k_m.gguf`) | Apache-2.0 | Google DeepMind, Gemma 4, released 2026-04-02. Redistributor of this specific GGUF requant: **PENDING** — see note 1 | Include Apache-2.0 text, state modifications, include NOTICE if one exists | 2026-08-16, family only |
| usb-serial-for-android | MIT | `mik3y/usb-serial-for-android` — pin **>= 3.10.0**, see note 2 | Retain copyright and licence text | 2026-08-16, repo licence field only |
| Android `TextToSpeech` | Platform API | Android SDK | None — nothing redistributed | 2026-08-16 |
| Android USB Host API | Platform API | Android SDK | None | 2026-08-16 |

## Copied source — the licence attaches to the copy

| Component | Licence | Source | Obligation | Verified on |
|---|---|---|---|---|
| `examples/llama.android` | MIT, as part of llama.cpp | `ggml-org/llama.cpp` | Same as llama.cpp; note in-repo which files were derived from it | PENDING |
| `SimpleUsbTerminal` | **PENDING** | `kai-morich/SimpleUsbTerminal` | PENDING — **do not copy any code until this row is filled in** | PENDING |
| `PARSE_SYS` prompt and JSON contract | Own work | Pixel Robot prototype, `main.py` | None | 2026-08-16 |

## Excluded — recorded so the trap is not rediscovered

| Component | Licence | Why excluded | Recorded |
|---|---|---|---|
| Ultralytics YOLO, `yolo11m.onnx` | AGPL-3.0 | AGPL is incompatible with a closed-source commercial product. Permissive replacements: YOLOX, NanoDet, TFLite EfficientDet-Lite, all Apache-2.0. The prototype's use of `yolo11m.onnx` (STATUS.md) does not carry into the app. | 2026-08-16 |
| OpenBot Android app | Apache-2.0 | Read for control-loop and camera-capture architecture only. No code reuse, so no obligation attaches. Listed to record that the question was asked and answered. | 2026-08-16 |

---

## Notes

**1. Gemma licensing changed at version 4, and the family licence is not the file
licence.** Gemma 1 through 3 shipped under Google's custom "Gemma Terms of Use" —
source-available, with a prohibited-use policy that passed through to derivative
models. Gemma 4, released 2026-04-02, ships under Apache-2.0, which is what makes it
usable here. Apache-2.0 is confirmed for the model family. What remains PENDING is
the provenance of the specific quantised GGUF on the device: the obligations that
attach to a shipped file are the redistributor's, not the upstream release page's.
Record the exact repository and revision the file came from before shipping it.

**2. `usb-serial-for-android` was LGPL-2.1 for most of its history.** Current master
and release 3.10.0 (December 2025) are MIT, and the repository's licence field reads
MIT. Older snapshots and the many forks still carry LGPL-2.1, whose relinking clause
is awkward for a closed-source Android app. Pin a version at or above the relicense
and read `LICENSE.txt` at that exact tag. For this dependency the name alone does not
establish the licence.

**3. Categories deliberately absent.** There is no row for a vision model, model-tier
weights, or any UI or charting library, because nothing in those categories has been
chosen. When one is chosen, the row is added before the dependency is.

**4. This file is an external artifact.** An investor or acquirer will ask for it.
Write rows to be read by someone with no context on this project.

**5. The llama.cpp and whisper.cpp versions do not come from the phone.**
`~/llama.cpp` is the prototype's Termux checkout; `git rev-parse` there would
produce a revision instantly and it would be the wrong one. The row records the
revision vendored into the app repo and compiled by the NDK, which does not exist
until the first build. Version and verified-on close separately: version at first
build, verified-on when someone reads `LICENSE` at that pinned commit.
