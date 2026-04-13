import sys, os, time
sys.path.insert(0, '/data/data/com.termux/files/home/robot')
from detect_person import detect_scene

# ── Camera constants ──────────────────────────────────
# Pixel 7 main camera: 6.81mm focal length, 1/1.31" sensor
# At 640x640 inference size, estimated focal length in pixels:
FOCAL_PX   = 500   # pixels (calibrate later with known distance)
FRAME_SIZE = 640   # pixels

# Known real-world heights for distance estimation (cm)
REAL_HEIGHTS = {
    'person':       170,
    'chair':         90,
    'dining table':  75,
    'couch':         85,
    'refrigerator': 180,
    'tv':            60,
    'bed':           50,
    'toilet':        40,
    'potted plant':  40,
    'vase':          25,
    'cup':           10,
    'bottle':        25,
    'laptop':         3,
}

def estimate_distance_single(label, height_px):
    """
    Estimate distance from single photo using known object height.
    Returns distance in cm or None if unknown object.
    """
    if label not in REAL_HEIGHTS or height_px < 5:
        return None
    real_h = REAL_HEIGHTS[label]
    dist = (real_h * FOCAL_PX) / height_px
    return round(dist)

def match_objects(results_a, results_b):
    """
    Match objects between two photos by label.
    Returns list of (label, cx_a, cx_b, cy, w, h, conf)
    """
    matched = []
    used_b = set()

    for r_a in results_a:
        label_a = r_a[0]
        cx_a = r_a[4]
        cy_a = r_a[5]

        # Find closest matching label in photo B
        best_match = None
        best_dy = 999
        for i, r_b in enumerate(results_b):
            if i in used_b:
                continue
            if r_b[0] != label_a:
                continue
            # Same label — check vertical position is similar
            dy = abs(r_b[5] - cy_a)
            if dy < 80 and dy < best_dy:
                best_dy = dy
                best_match = (i, r_b)

        if best_match:
            i, r_b = best_match
            used_b.add(i)
            matched.append({
                'label':     label_a,
                'cx_a':      cx_a,
                'cx_b':      r_b[4],
                'cy':        cy_a,
                'w':         r_a[6],
                'h':         r_a[7],
                'conf':      r_a[1],
                'disparity': cx_a - r_b[4],  # positive = object shifted left in B
            })

    return matched

def stereo_distance(disparity_px, baseline_cm):
    """
    Calculate distance using stereo triangulation.
    distance = (focal × baseline) / disparity
    """
    if abs(disparity_px) < 2:
        return None  # too small disparity, unreliable
    return round((FOCAL_PX * baseline_cm) / abs(disparity_px))

def stereo_scan(photo_a, photo_b, baseline_cm=5.0):
    """
    Full stereo depth analysis.
    photo_a: taken at position X
    photo_b: taken at position X + baseline_cm (strafe right)
    Returns enhanced scene with distances.
    """
    print(f"[STEREO] Analyzing with {baseline_cm}cm baseline...")

    results_a = detect_scene(photo_a)
    results_b = detect_scene(photo_b)

    print(f"[STEREO] Photo A: {len(results_a)} objects")
    print(f"[STEREO] Photo B: {len(results_b)} objects")

    matched = match_objects(results_a, results_b)
    print(f"[STEREO] Matched: {len(matched)} objects")

    enhanced = []
    for m in matched:
        # Try stereo first
        dist_stereo = stereo_distance(m['disparity'], baseline_cm)
        # Fallback to single-photo estimate
        dist_single = estimate_distance_single(m['label'], m['h'])

        dist = dist_stereo if dist_stereo else dist_single
        method = 'stereo' if dist_stereo else 'estimated'

        pos = 'left' if m['cx_a'] < 213 else 'right' if m['cx_a'] > 427 else 'center'

        enhanced.append({
            'label':    m['label'],
            'conf':     m['conf'],
            'pos':      pos,
            'dist_cm':  dist,
            'method':   method,
            'disparity': m['disparity'],
            'cx':       m['cx_a'],
            'cy':       m['cy'],
        })

    enhanced.sort(key=lambda x: (x['dist_cm'] or 9999))
    return enhanced

def scene_with_depth(enhanced):
    """Convert enhanced scene to robot-readable string."""
    parts = []
    for e in enhanced:
        dist_str = f"{e['dist_cm']}cm" if e['dist_cm'] else "unknown dist"
        parts.append(f"{e['label']} {e['pos']} {dist_str}")
    return ', '.join(parts)

if __name__ == '__main__':
    # Test with two copies of the same photo (disparity=0)
    # In real use: photo_a and photo_b taken 5cm apart
    if len(sys.argv) >= 3:
        photo_a = sys.argv[1]
        photo_b = sys.argv[2]
        baseline = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    else:
        # Demo: use same photo twice to show single-photo estimation
        photo_a = 'test_photos/scene_test.jpg'
        photo_b = 'test_photos/scene_test.jpg'
        baseline = 5.0

    print("=== STEREO DEPTH TEST ===")
    enhanced = stereo_scan(photo_a, photo_b, baseline)

    print(f"\n{'Object':<15} {'Pos':<8} {'Dist':>8} {'Method':<10} {'Disparity':>10}")
    print('-'*55)
    for e in enhanced:
        dist_str = f"{e['dist_cm']}cm" if e['dist_cm'] else "N/A"
        print(f"{e['label']:<15} {e['pos']:<8} {dist_str:>8} {e['method']:<10} {e['disparity']:>10}px")

    print(f"\nRobot sees: \"{scene_with_depth(enhanced)}\"")

    # Also show single-photo estimates for all objects
    print("\n--- Single photo distance estimates ---")
    results = detect_scene(photo_a)
    for r in results:
        dist = estimate_distance_single(r[0], r[7])
        if dist:
            print(f"  {r[0]:<15} height={r[7]}px → ~{dist}cm")
