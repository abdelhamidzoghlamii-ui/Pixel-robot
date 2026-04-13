import os, time, json
from detect_person import detect_person

CAMERA = "/data/data/com.termux/files/usr/bin/termux-camera-photo"
PHOTO_DIR = os.path.expanduser("~/robot/test_photos")
LOG = os.path.expanduser("~/robot/test_results_m.json")
os.makedirs(PHOTO_DIR, exist_ok=True)

PEOPLE = ["chiara", "abdel"]
DISTANCES = ["close", "far"]
DIRECTIONS = ["front", "left", "right", "back"]
results = []

def take_photo(path):
    os.system("/data/data/com.termux/files/usr/bin/termux-camera-photo " + path)
    time.sleep(1.5)

def prompt(msg):
    input("\n>>> " + msg + "\n    Press ENTER when ready...")

print("=" * 55)
print("  YOLO TEST SUITE - YOLOv8m")
print("  " + str(len(PEOPLE)*len(DISTANCES)*len(DIRECTIONS)) + " photos total")
print("=" * 55)

for person in PEOPLE:
    for distance in DISTANCES:
        for direction in DIRECTIONS:
            label = person + "_" + distance + "_" + direction
            photo_path = PHOTO_DIR + "/" + label + ".jpg"
            prompt(person.upper() + " | " + distance + " | " + direction)
            take_photo(photo_path)
            if not os.path.exists(photo_path) or os.path.getsize(photo_path) < 1000:
                print("    ERROR: photo not saved")
                results.append({"label": label, "detected": False, "confidence": 0, "position": "unknown", "error": "photo_failed"})
                continue
            found, conf, pos = detect_person(photo_path)
            result = {"label": label, "person": person, "distance": distance,
                      "direction": direction, "detected": found,
                      "confidence": round(conf, 3), "position": pos,
                      "photo": photo_path}
            status = "DETECTED conf=" + str(round(conf,2)) + " pos=" + pos if found else "MISSED"
            print("    " + status)
            results.append(result)

with open(LOG, "w") as f:
    json.dump(results, f, indent=2)

total = len([r for r in results if "error" not in r])
detected = sum(1 for r in results if r.get("detected"))
print("\n" + "=" * 55)
print("  SUMMARY: " + str(detected) + "/" + str(total) + " detected")
for d in DISTANCES:
    sub = [r for r in results if r.get("distance") == d]
    hit = sum(1 for r in sub if r.get("detected"))
    print("  " + d + ": " + str(hit) + "/" + str(len(sub)))
for p in PEOPLE:
    sub = [r for r in results if r.get("person") == p]
    hit = sum(1 for r in sub if r.get("detected"))
    print("  " + p + ": " + str(hit) + "/" + str(len(sub)))
print("  Photos saved to: " + PHOTO_DIR)
print("=" * 55)
