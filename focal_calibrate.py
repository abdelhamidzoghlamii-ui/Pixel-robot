#!/usr/bin/env python3
"""
focal_calibrate.py - derive FOCAL_PX for detect_scene's single-photo person
distance estimate, from the person photos in bench_photos/ + labels.csv.

ANALYSIS ONLY. Reads stereo_depth.py, detect_person.py, bench_photos/labels.csv.
Does NOT modify FOCAL_PX or any other file. Prints UNKNOWN where the data does
not support a conclusion.

Model (stereo_depth.py, estimate_distance_single):
    dist_cm = real_h_cm * FOCAL_PX / height_px
  => FOCAL_PX = true_distance_cm * height_px / real_h_cm
"""

import csv
import os
import sys
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageOps
from detect_person import detect_scene
from stereo_depth import estimate_distance_single, REAL_HEIGHTS, FOCAL_PX

HERE       = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(HERE, "bench_photos", "labels.csv")
PHOTO_DIR  = os.path.join(HERE, "bench_photos")
STEREO_SRC = os.path.join(HERE, "stereo_depth.py")
REAL_H     = REAL_HEIGHTS["person"]        # cm
EXCLUDE_SUFFIX = "_300cm.jpg"              # test_photos-sourced rows (see section 2)


def rule(c="="):
    print(c * 72)


# ── 1. pinhole model ────────────────────────────────────────────────
def show_model():
    rule(); print("1. PINHOLE MODEL  (verbatim from stereo_depth.py)"); rule()
    src = open(STEREO_SRC).read().splitlines()
    for n in list(range(6, 14)) + list(range(28, 38)):   # 1-indexed
        print(f"{n:4d}| {src[n - 1]}")
    print()
    print(f"  REAL_HEIGHTS['person'] = {REAL_H} cm")
    print(f"  FOCAL_PX (current, left unchanged) = {FOCAL_PX}")
    print(f"  => per photo:  FOCAL_PX = true_distance_cm * h / {REAL_H}")
    print()


# ── 2. row selection ────────────────────────────────────────────────
def load_rows():
    hdr_notes = [l.rstrip() for l in open(CSV_PATH) if l.startswith("#")]
    data = [l for l in open(CSV_PATH) if not l.startswith("#")]
    rows = list(csv.DictReader(data))

    usable, excluded = [], []
    for r in rows:
        if r.get("category") != "person":
            continue
        fn  = r["filename"]
        raw = (r.get("true_distance_cm") or "").strip()
        if fn.endswith(EXCLUDE_SUFFIX):
            excluded.append((fn, raw, "sourced from test_photos/ - different capture "
                             "session, different camera height, distance approximate"))
            continue
        try:
            d = float(raw)
        except ValueError:
            excluded.append((fn, raw, "true_distance_cm is not a number"))
            continue
        usable.append({"fn": fn, "d": d})
    return usable, excluded, hdr_notes


# ── 3 + 5. detection + original dimensions ──────────────────────────
def measure(rec):
    path = os.path.join(PHOTO_DIR, rec["fn"])
    if not os.path.exists(path):
        rec["skip"] = "file missing"
        return rec
    rec["raw_wh"] = Image.open(path).size                       # stored orientation
    rec["wh"] = ImageOps.exif_transpose(Image.open(path)).size  # what detect_scene sees
    w, h = rec["wh"]
    rec["orient"] = "landscape" if w >= h else "portrait"
    try:
        results = detect_scene(path)
    except Exception as e:
        rec["skip"] = f"detect_scene error: {e}"
        return rec
    persons = [x for x in results if x[0] == "person"]
    if not persons:
        rec["skip"] = "no 'person' detected"
        return rec
    p = max(persons, key=lambda x: x[1])
    rec["conf"] = p[1]
    rec["h_px"] = p[7]                                          # field 7 = bbox height (640 frame)
    rec["focal"] = rec["d"] * rec["h_px"] / REAL_H
    return rec


# ── stats helpers ──────────────────────────────────────────────────
def summarise(vals):
    lo, hi, med = min(vals), max(vals), st.median(vals)
    return dict(
        n=len(vals), lo=lo, hi=hi, median=med, mean=st.fmean(vals),
        stdev=(st.pstdev(vals) if len(vals) > 1 else 0.0),
        spread_pct=(100 * (hi - lo) / med if med else float("nan")),
    )


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = st.fmean(xs), st.fmean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


# ── main ───────────────────────────────────────────────────────────
def main():
    show_model()

    usable, excluded, hdr_notes = load_rows()

    rule(); print("2. ROW SELECTION  (bench_photos/labels.csv)"); rule()
    print("  EXCLUDED:")
    for fn, raw, why in excluded:
        print(f"    - {fn}  (true_distance_cm={raw or 'blank'})")
        print(f"        {why}")
    print(f"\n  USABLE - category=person, numeric true_distance_cm: {len(usable)}")
    for r in sorted(usable, key=lambda r: r["d"]):
        print(f"    - {r['fn']:28s}  d = {r['d']:.0f} cm")

    prov = [l for l in hdr_notes
            if ("ESTIMATE" in l.upper() or "VISUAL" in l.upper() or "NOT tape" in l)]
    if prov:
        print("\n  !! PROVENANCE WARNING - labels.csv's own header says these distances")
        print("     are NOT tape-measured:")
        for l in prov:
            print(f"       {l}")
        print("     Any FOCAL_PX derived below inherits that uncertainty. The task's")
        print("     premise ('measured person photos') is NOT met by the file - the true")
        print("     calibration value is UNKNOWN; treat the numbers as indicative only.")
    print()

    for r in usable:
        measure(r)
    ok  = [r for r in usable if "skip" not in r]
    bad = [r for r in usable if "skip" in r]

    rule(); print("3. DETECTION  (real detect_scene(); 'person' row; field 7 = h)"); rule()
    for r in usable:
        if "skip" in r:
            print(f"  SKIP  {r['fn']:28s}  {r['skip']}")
        else:
            print(f"  ok    {r['fn']:28s}  conf={r['conf']:.2f}  h={r['h_px']}px  (640 frame)")
    if not ok:
        print("\n  No usable detections -> FOCAL_PX cannot be derived. UNKNOWN.")
        rule(); print("NOTE: FOCAL_PX was NOT changed."); rule()
        return

    rule(); print("5. ORIGINAL PIXEL DIMENSIONS  (before the 640x640 resize)"); rule()
    print("  detect_scene: Image.open -> ImageOps.exif_transpose -> resize((640,640)).")
    print("  The resize does NOT preserve aspect ratio: bbox h scales by 640/H_exif, so")
    print("  landscape and portrait frames are NOT comparable and are grouped below.\n")
    for r in ok:
        print(f"  {r['fn']:28s}  stored {r['raw_wh'][0]}x{r['raw_wh'][1]}  "
              f"exif {r['wh'][0]}x{r['wh'][1]}  -> {r['orient']}")

    groups = {}
    for r in ok:
        groups.setdefault(r["orient"], []).append(r)

    rule(); print("4. DERIVED FOCAL_PX  (per photo, grouped by orientation)"); rule()
    group_median = {}
    for g, rs in sorted(groups.items()):
        foc = [r["focal"] for r in rs]
        dst = [r["d"] for r in rs]
        s = summarise(foc)
        group_median[g] = s["median"]
        print(f"\n  --- {g}  (n={s['n']}, frame {rs[0]['wh'][0]}x{rs[0]['wh'][1]}) ---")
        for r in sorted(rs, key=lambda r: r["d"]):
            print(f"    d={r['d']:6.0f} cm   h={r['h_px']:4d} px   focal = {r['focal']:8.1f}")
        print(f"    median {s['median']:8.1f}    mean {s['mean']:8.1f}    "
              f"pop.stdev {s['stdev']:.1f}")
        print(f"    range  {s['lo']:.1f} .. {s['hi']:.1f}   "
              f"(spread = {s['spread_pct']:.0f}% of median)")
        r_df = pearson(dst, foc)
        if r_df is None:
            print("    distance <-> focal correlation: UNKNOWN (n<3 or zero variance)")
        else:
            print(f"    distance <-> focal correlation  r = {r_df:+.2f}")
            if abs(r_df) >= 0.7:
                print("    -> SYSTEMATIC DRIFT with distance. A valid pinhole model yields")
                print("       a CONSTANT focal; this does not. The bbox height is not")
                print("       tracking a fixed 170 cm real height (reclining/seated pose,")
                print("       clipped bbox, or wrong true_distance). NOT a usable calibration.")
            elif s["spread_pct"] > 40:
                print("    -> spread > 40% with no clean trend: noisy, NOT a usable calibration.")
            else:
                print("    -> reasonably flat; median is an indicative FOCAL_PX (see warning).")

    rule(); print("   RECOMMENDED FOCAL_PX"); rule()
    if len(group_median) == 1:
        g = next(iter(group_median))
        print(f"  single orientation ({g}): median focal = {group_median[g]:.0f}  "
              f"(indicative only - provenance warning applies)")
    else:
        print("  the set mixes orientations; the non-aspect-preserving resize makes one")
        print("  FOCAL_PX invalid for both. Per-group medians:")
        for g, m in sorted(group_median.items()):
            print(f"    {g}: {m:.0f}")
        print("  single canonical value: UNKNOWN")

    rule(); print("6. MEASURED distance  vs  estimate_distance_single()"); rule()
    print(f"  est@500 : live estimate_distance_single('person', h)  (FOCAL_PX={FOCAL_PX})")
    print("  est@new : REAL_H * (orientation-group median focal) / h")
    print("  err% relative to the (estimated) measured distance.\n")
    head = (f"  {'photo':28s} {'orient':9s} {'meas':>6s} {'h_px':>5s} "
            f"{'focal_i':>8s} {'est@500':>7s} {'err%':>6s} {'est@new':>7s} {'err%':>6s}")
    print(head); print("  " + "-" * (len(head) - 2))
    for r in sorted(ok, key=lambda r: (r["orient"], r["d"])):
        e500 = estimate_distance_single("person", r["h_px"])
        enew = round(REAL_H * group_median[r["orient"]] / r["h_px"])
        er5  = 100 * (e500 - r["d"]) / r["d"]
        ern  = 100 * (enew - r["d"]) / r["d"]
        print(f"  {r['fn']:28s} {r['orient']:9s} {r['d']:6.0f} {r['h_px']:5d} "
              f"{r['focal']:8.1f} {e500:7d} {er5:+6.0f} {enew:7d} {ern:+6.0f}")

    if bad:
        print(f"\n  {len(bad)} photo(s) skipped - see section 3.")

    rule(); print("NOTE: FOCAL_PX was NOT changed anywhere. This script only reports."); rule()


if __name__ == "__main__":
    main()
