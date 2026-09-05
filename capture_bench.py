#!/usr/bin/env python3
"""
capture_bench.py - guided photo capture for the navigation benchmark.

Captures with the SAME command the robot uses. Copied verbatim from
main.py take_photo():

    line 31:  os.system(f'termux-camera-photo {path} 2>/dev/null')
    line 32:  time.sleep(0.5)
    line 33:  return os.path.exists(path) and os.path.getsize(path) > 1000

There is NO -c / camera-id flag and no other flag in the original, so
termux-camera-photo's default camera is used - matched here exactly (no
flag added), so these frames match what the live system sees.

After every shot the new file is run through detect_person.detect_scene()
and you choose keep / retake. A room-signature shot whose target object is
not detected is useless and defaults to retake.

Output:     bench_photos/  +  bench_photos/labels.csv
Resumable:  shots already listed in labels.csv are skipped.
Limits:     no root, no USB, no motor commands. Stdlib + detect_person only.
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_person import detect_scene

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(HERE, "bench_photos")
CSV_PATH = os.path.join(OUT_DIR, "labels.csv")
CSV_FIELDS = ["filename", "category", "target_object",
              "true_distance_cm", "detected", "detected_labels"]

# verbatim quote of the line copied from main.py:31
CAMERA_LINE = "os.system(f'termux-camera-photo {path} 2>/dev/null')"
SETTLE_S  = 0.5     # main.py:32
MIN_BYTES = 1000    # main.py:33

CAMERA_HEIGHT = "UNKNOWN"   # set in main(), written to the labels.csv header


# ── capture: identical path to main.py take_photo() ──────────────────
def capture(path):
    os.system(f'termux-camera-photo {path} 2>/dev/null')   # main.py:31
    time.sleep(SETTLE_S)                                    # main.py:32
    return os.path.exists(path) and os.path.getsize(path) > MIN_BYTES  # main.py:33


# ── shot list ───────────────────────────────────────────────────────
# DECISIONS #22: these object labels are keyed to room identity.
ROOM_SIGNATURES  = ["refrigerator", "couch", "tv", "bed", "toilet"]
PERSON_DISTANCES = [40, 60, 80, 100, 150, 200, 300]
PERSON_DIRS = [("facing", "facing the camera head-on"),
               ("sideon", "turned 90 degrees, side-on to the camera")]


def build_shots():
    shots = []

    for obj in ROOM_SIGNATURES:
        for n in (1, 2, 3):
            shots.append({
                "filename": f"roomsig_{obj}_{n}.jpg",
                "category": "room_signature",
                "target_object": obj,
                "nominal_distance_cm": None,
                "instruction":
                    f"ROOM SIGNATURE  {obj}  ({n}/3)\n"
                    f"  Frame the {obj} clearly, filling a good part of the view.\n"
                    f"  Change angle AND distance across the 3 shots\n"
                    f"  (e.g. 1: straight-on ~1 m, 2: ~45 deg, 3: far across the room).",
            })

    for d in PERSON_DISTANCES:
        for key, desc in PERSON_DIRS:
            shots.append({
                "filename": f"person_{key}_{d:03d}cm.jpg",
                "category": "person",
                "target_object": "person",
                "nominal_distance_cm": d,
                "instruction":
                    f"PERSON  ~{d} cm  ({desc})\n"
                    f"  Tape-measure lens -> person, then stand {desc}.\n"
                    f"  You will be asked for the real measured distance.",
            })

    for n in range(1, 9):
        shots.append({
            "filename": f"empty_{n}.jpg",
            "category": "empty",
            "target_object": "",
            "nominal_distance_cm": None,
            "instruction":
                f"EMPTY frame  ({n}/8)\n"
                f"  No person anywhere in view. Aim at a normal navigable area.\n"
                f"  Use a different room / heading than the earlier empty shots.",
        })

    for n in range(1, 6):
        shots.append({
            "filename": f"blocked_{n}.jpg",
            "category": "blocked",
            "target_object": "",
            "nominal_distance_cm": None,
            "instruction":
                f"BLOCKED path  ({n}/5)\n"
                f"  Something directly ahead within ~30 cm of the lens\n"
                f"  (wall, box, chair, closed door). Frame it as the robot\n"
                f"  would see it in the instant before it stops.",
        })

    return shots


# ── labels.csv ──────────────────────────────────────────────────────
def load_done():
    """Return (set of filenames already captured, saved camera height or None)."""
    if not os.path.exists(CSV_PATH):
        return set(), None
    with open(CSV_PATH, newline="") as f:
        lines = f.readlines()
    done, height, data = set(), None, []
    for ln in lines:
        if ln.startswith("#"):
            if "camera_height_cm=" in ln:
                height = ln.split("camera_height_cm=", 1)[1].split(";", 1)[0].strip()
            continue
        data.append(ln)
    for row in csv.DictReader(data):
        done.add(row["filename"])
    return done, height


def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        if new:
            # header comment: camera height must stay constant across sessions
            f.write(f"# camera_height_cm={CAMERA_HEIGHT} ; "
                    f"capture=termux-camera-photo <path> 2>/dev/null ; "
                    f"from capture_bench.py\n")
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


# ── detection ───────────────────────────────────────────────────────
def run_detect(path):
    """Run the real detect_scene(). Return list of label strings, or None on error."""
    try:
        results = detect_scene(path)
    except Exception as e:
        print(f"  detect_scene ERROR: {e}")
        return None
    if results:
        print("  detected: " +
              ", ".join(f"{r[0]}({r[1]:.2f},{r[2]})" for r in results))
    else:
        print("  detected: nothing")
    return [r[0] for r in results]


# ── one shot ────────────────────────────────────────────────────────
def ask_distance(nominal):
    while True:
        raw = input(f"  measured lens->person distance in cm "
                    f"[ENTER = nominal {nominal}, q = quit]: ").strip()
        if raw == "":
            return str(nominal)
        if raw.lower() == "q":
            return None
        try:
            return str(int(round(float(raw))))
        except ValueError:
            print("  ...number please")


def do_shot(shot, remaining):
    print("\n" + "=" * 64)
    print(f"[{remaining} shot(s) left]")
    print(shot["instruction"])
    print("=" * 64)

    path   = os.path.join(OUT_DIR, shot["filename"])
    cat    = shot["category"]
    target = shot["target_object"]

    true_dist = ""
    if cat == "person":
        true_dist = ask_distance(shot["nominal_distance_cm"])
        if true_dist is None:
            return "quit"

    while True:
        if input("  ENTER to capture  (q = quit and resume later): ").strip().lower() == "q":
            return "quit"

        if not capture(path):
            print("  !! capture failed (no file, or < 1000 bytes).")
            print("     Check Termux camera permission and that termux-api is installed.")
            continue

        labels = run_detect(path)

        if labels is None:
            hit, default_keep = "", True
        else:
            if cat in ("room_signature", "person"):
                hit = target in labels
            else:
                hit = len(labels) > 0
            default_keep = True
            if cat == "room_signature" and target not in labels:
                print(f"  !! '{target}' NOT detected. Room-signature shots are keyed "
                      f"to the detected object - this one is USELESS. Retake.")
                default_keep = False
            elif cat == "person" and "person" not in labels:
                print("  !! no 'person' detected - probably unusable.")
                default_keep = False
            elif cat == "empty" and "person" in labels:
                print("  !! 'person' detected in an EMPTY frame. Retake.")
                default_keep = False

        if default_keep:
            ans = input("  keep? [Y/n/q]: ").strip().lower()
            keep = ans not in ("n", "no", "retake")
        else:
            ans = input("  keep anyway? [y/N/q]: ").strip().lower()
            keep = ans in ("y", "yes", "keep")
        if ans == "q":
            return "quit"

        if not keep:
            print("  -> retake")
            try:
                os.remove(path)
            except OSError:
                pass
            continue

        append_row({
            "filename":        shot["filename"],
            "category":        cat,
            "target_object":   target,
            "true_distance_cm": true_dist,
            "detected":        hit,
            "detected_labels": ";".join(labels) if labels else "",
        })
        print(f"  saved {shot['filename']}  (detected={hit})")
        return "done"


# ── main ────────────────────────────────────────────────────────────
def main():
    global CAMERA_HEIGHT

    os.makedirs(OUT_DIR, exist_ok=True)
    done, saved_height = load_done()

    print("=" * 64)
    print("  NAV BENCHMARK - GUIDED PHOTO CAPTURE")
    print("=" * 64)
    print("  capture command (copied verbatim from main.py:31):")
    print(f"    {CAMERA_LINE}")
    print("""
  SETUP - do this before the first shot:
    * Mount the phone on the robot at its normal camera height, OR
      hold it at exactly that height.
    * Keep that height IDENTICAL for every shot and every session.
    * Frame every shot from the robot's viewpoint / heading.
    * A room-signature shot must actually contain its target object
      or it is discarded.
""")

    if saved_height:
        CAMERA_HEIGHT = saved_height
        print(f"  resuming - camera height from labels.csv: {CAMERA_HEIGHT} cm")
    else:
        CAMERA_HEIGHT = input("  camera height above floor (cm): ").strip() or "UNKNOWN"

    shots = build_shots()
    todo  = [s for s in shots if s["filename"] not in done]
    print(f"\n  total {len(shots)}  |  done {len(done)}  |  remaining {len(todo)}")

    if not todo:
        print("\n  Nothing left to capture. labels.csv is complete.")
        return

    for i, shot in enumerate(todo):
        remaining = len(todo) - i
        if do_shot(shot, remaining) == "quit":
            print(f"\n  stopped - {remaining} shot(s) still remaining. "
                  f"re-run capture_bench.py to resume.")
            return
        print(f"  {len(todo) - i - 1} shot(s) remaining")

    print("\n  All shots captured. bench_photos/labels.csv is complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  interrupted - progress is saved, re-run to resume.")
