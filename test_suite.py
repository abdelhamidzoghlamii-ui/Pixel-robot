import os, time, json, subprocess
from datetime import datetime
from detect_person import detect_person

CAMERA = "/data/data/com.termux/files/usr/bin/termux-camera-photo"
PHOTO  = "/data/data/com.termux/files/home/test.jpg"
LOG    = os.path.expanduser("~/robot/test_results.json")

PEOPLE     = ["person1", "person2"]
DISTANCES  = ["close", "medium", "far"]
DIRECTIONS = ["front", "left", "right", "back"]
REPS       = 3
EMPTY_SHOTS = 10

results = []

def take_photo():
    os.system("su -c '%s %s'" % (CAMERA, PHOTO))
    time.sleep(1.0)

def prompt(msg):
    input(f"\n>>> {msg}\n    Press ENTER when ready...")

def run_test(label, person, distance, direction, rep):
    prompt(f"{person.upper()} | {distance} | {direction} | shot {rep+1}/{REPS}")
    take_photo()
    found, conf, pos = detect_person(PHOTO)
    result = {
        "label"    : label,
        "person"   : person,
        "distance" : distance,
        "direction": direction,
        "rep"      : rep + 1,
        "detected" : found,
        "confidence": round(conf, 3),
        "position" : pos,
        "correct"  : found  # person should be detected
    }
    status = "✓ DETECTED" if found else "✗ MISSED"
    print(f"    {status} | conf={conf:.2f} | pos={pos}")
    return result

def run_empty(rep):
    prompt(f"EMPTY ROOM — no person in frame | shot {rep+1}/{EMPTY_SHOTS}")
    take_photo()
    found, conf, pos = detect_person(PHOTO)
    result = {
        "label"    : "empty",
        "person"   : "none",
        "distance" : "n/a",
        "direction": "n/a",
        "rep"      : rep + 1,
        "detected" : found,
        "confidence": round(conf, 3),
        "position" : pos,
        "correct"  : not found  # should NOT be detected
    }
    status = "✗ FALSE POSITIVE" if found else "✓ CORRECTLY EMPTY"
    print(f"    {status} | conf={conf:.2f}")
    return result

print("=" * 55)
print("  YOLO DETECTION TEST SUITE")
print(f"  Model   : YOLOv8n")
print(f"  People  : {', '.join(PEOPLE)}")
print(f"  Photos  : {len(PEOPLE)*len(DISTANCES)*len(DIRECTIONS)*REPS + EMPTY_SHOTS} total")
print("=" * 55)

# Person tests
for person in PEOPLE:
    print(f"\n{'='*55}")
    print(f"  SUBJECT: {person.upper()}")
    print(f"{'='*55}")
    for distance in DISTANCES:
        for direction in DIRECTIONS:
            for rep in range(REPS):
                label = f"{person}_{distance}_{direction}"
                r = run_test(label, person, distance, direction, rep)
                results.append(r)

# Empty room tests
print(f"\n{'='*55}")
print(f"  EMPTY ROOM TESTS (false positive check)")
print(f"{'='*55}")
for rep in range(EMPTY_SHOTS):
    r = run_empty(rep)
    results.append(r)

# Save results
with open(LOG, "w") as f:
    json.dump(results, f, indent=2)

# Summary
print(f"\n{'='*55}")
print("  RESULTS SUMMARY")
print(f"{'='*55}")

person_results = [r for r in results if r["person"] != "none"]
empty_results  = [r for r in results if r["person"] == "none"]

total   = len(person_results)
correct = sum(1 for r in person_results if r["correct"])
avg_conf = sum(r["confidence"] for r in person_results if r["detected"]) / max(1, sum(1 for r in person_results if r["detected"]))

print(f"\n  Person detection:")
print(f"    Detected  : {correct}/{total} ({100*correct//total}%)")
print(f"    Avg conf  : {avg_conf:.2f}")

print(f"\n  By distance:")
for d in DISTANCES:
    sub = [r for r in person_results if r["distance"] == d]
    hit = sum(1 for r in sub if r["correct"])
    print(f"    {d:8s}: {hit}/{len(sub)} ({100*hit//len(sub)}%)")

print(f"\n  By direction:")
for d in DIRECTIONS:
    sub = [r for r in person_results if r["direction"] == d]
    hit = sum(1 for r in sub if r["correct"])
    print(f"    {d:8s}: {hit}/{len(sub)} ({100*hit//len(sub)}%)")

print(f"\n  By person:")
for p in PEOPLE:
    sub = [r for r in person_results if r["person"] == p]
    hit = sum(1 for r in sub if r["correct"])
    print(f"    {p:10s}: {hit}/{len(sub)} ({100*hit//len(sub)}%)")

fp = sum(1 for r in empty_results if not r["correct"])
print(f"\n  False positives : {fp}/{EMPTY_SHOTS}")
print(f"\n  Full results saved to: {LOG}")
print("=" * 55)
