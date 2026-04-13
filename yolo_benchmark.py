import sys, os, time
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps

HOME       = "/data/data/com.termux/files/home"
PHOTO_DIR  = HOME + "/robot/test_photos"
CONF_THRES = 0.25

def get_temp():
    try:
        return int(os.popen('su -c "cat /sys/class/thermal/thermal_zone9/temp"').read().strip()) // 1000
    except:
        return 0

def load_image(path, size=640):
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    orig_w, orig_h = img.size
    scale = size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (size, size), (114, 114, 114))
    padded.paste(img, (0, 0))
    arr = np.array(padded).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr, scale, orig_w, orig_h

def run_yolo(session, img_array):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    return outputs[0]

def detect_person(session, path):
    arr, scale, orig_w, orig_h = load_image(path)
    out = run_yolo(session, arr)
    # Parse detections
    if out.shape[1] == 84:  # YOLOv8/11 format
        preds = out[0].T
    else:
        preds = out[0]
    
    person_found = False
    best_conf = 0.0
    position = "none"
    
    for pred in preds:
        if len(pred) >= 6:
            scores = pred[4:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
        else:
            cx, cy, w, h, conf, cls = pred[:6]
            cls = int(cls)
        
        if cls == 0 and conf > CONF_THRES:  # class 0 = person
            if conf > best_conf:
                best_conf = conf
                person_found = True
                cx = float(pred[0])
                if cx < 640/3:
                    position = "left"
                elif cx > 2*640/3:
                    position = "right"
                else:
                    position = "center"
    
    return person_found, best_conf, position

def benchmark_model(model_path):
    name = os.path.basename(model_path)
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"\n{'='*50}")
    print(f"  Testing: {name} ({size_mb:.0f}MB)")
    print(f"{'='*50}")
    
    # Load model
    t0 = time.time()
    session = ort.InferenceSession(model_path)
    load_time = round(time.time() - t0, 2)
    print(f"  Load time: {load_time}s")
    
    # Get all photos
    photos = sorted(os.listdir(PHOTO_DIR))
    person_photos = [p for p in photos if not p.startswith("empty") and p.endswith(".jpg")]
    empty_photos  = [p for p in photos if p.startswith("empty") and p.endswith(".jpg")]
    
    # Test person photos
    person_correct = 0
    person_times = []
    temp_start = get_temp()
    
    for photo in person_photos:
        path = PHOTO_DIR + "/" + photo
        t0 = time.time()
        found, conf, pos = detect_person(session, path)
        elapsed = round(time.time() - t0, 2)
        person_times.append(elapsed)
        if found:
            person_correct += 1
        else:
            print(f"  MISSED: {photo}")
    
    temp_after_person = get_temp()
    
    # Test empty photos
    empty_correct = 0
    empty_times = []
    
    for photo in empty_photos:
        path = PHOTO_DIR + "/" + photo
        t0 = time.time()
        found, conf, pos = detect_person(session, path)
        elapsed = round(time.time() - t0, 2)
        empty_times.append(elapsed)
        if not found:
            empty_correct += 1
        else:
            print(f"  FALSE POSITIVE: {photo} conf={conf:.2f}")
    
    temp_end = get_temp()
    
    # Results
    all_times = person_times + empty_times
    avg_time = round(sum(all_times) / len(all_times), 3)
    min_time = round(min(all_times), 3)
    max_time = round(max(all_times), 3)
    
    print(f"\n  ACCURACY:")
    print(f"    Person detection: {person_correct}/{len(person_photos)} ({round(person_correct/len(person_photos)*100)}%)")
    print(f"    Empty rooms:      {empty_correct}/{len(empty_photos)} ({round(empty_correct/len(empty_photos)*100)}%)")
    print(f"\n  SPEED:")
    print(f"    Avg: {avg_time}s | Min: {min_time}s | Max: {max_time}s")
    print(f"\n  THERMAL:")
    print(f"    Start: {temp_start}C | After persons: {temp_after_person}C | End: {temp_end}C | Rise: +{temp_end-temp_start}C")
    
    return {
        "name": name,
        "size_mb": round(size_mb),
        "load_time": load_time,
        "person_acc": f"{person_correct}/{len(person_photos)}",
        "empty_acc": f"{empty_correct}/{len(empty_photos)}",
        "avg_time": avg_time,
        "temp_rise": temp_end - temp_start
    }

# Find all YOLO models
models = sorted([
    f for f in os.listdir(HOME + "/robot")
    if f.endswith(".onnx") and os.path.getsize(HOME + "/robot/" + f) > 1000
])

print(f"Found {len(models)} models to test:")
for m in models:
    size = os.path.getsize(HOME + "/robot/" + m) / 1024 / 1024
    print(f"  {m} ({size:.0f}MB)")

results = []
for model in models:
    path = HOME + "/robot/" + model
    result = benchmark_model(path)
    results.append(result)
    time.sleep(5)  # cool down between models

# Final comparison
print(f"\n{'='*65}")
print(f"  FINAL COMPARISON")
print(f"{'='*65}")
print(f"  {'Model':<20} {'Size':>6} {'Person':>8} {'Empty':>7} {'Avg(s)':>7} {'Temp+':>6}")
print(f"  {'-'*63}")
for r in results:
    print(f"  {r['name']:<20} {r['size_mb']:>5}MB {r['person_acc']:>8} {r['empty_acc']:>7} {r['avg_time']:>7} {r['temp_rise']:>5}C")
