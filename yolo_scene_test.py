import sys
sys.path.insert(0, '/data/data/com.termux/files/home/robot')
import os, time
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

PHOTO = '/data/data/com.termux/files/home/robot/test_photos/scene_test.jpg'
ROBOT = '/data/data/com.termux/files/home/robot'

CLASSES = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck',
'boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird',
'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
'clock','vase','scissors','teddy bear','hair dryer','toothbrush']

print("Loading image...")
img = Image.open(PHOTO).convert('RGB')
img = ImageOps.exif_transpose(img)
print(f"Image size: {img.size}")
img = img.resize((640, 640))
arr = np.array(img).astype(np.float32) / 255.0
arr = arr.transpose(2, 0, 1)[np.newaxis]
print("Image ready\n")

models = sorted([f for f in os.listdir(ROBOT)
    if f.endswith('.onnx') and os.path.getsize(ROBOT+'/'+f) > 10000])

print(f"Found {len(models)} models\n")

for model_name in models:
    model_path = ROBOT + '/' + model_name
    size_mb = round(os.path.getsize(model_path)/1024/1024)
    print(f"=== {model_name} ({size_mb}MB) ===")
    
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    t0 = time.time()
    out = session.run(None, {input_name: arr})[0][0].T
    elapsed = round(time.time() - t0, 3)
    
    detections = []
    for pred in out:
        scores = pred[4:]
        cls = int(np.argmax(scores))
        conf = float(scores[cls])
        if conf > 0.25:
            cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
            area = (w * h) / (640 * 640)
            pos = 'left' if cx < 213 else 'right' if cx > 427 else 'center'
            dist = 'very close' if area > 0.3 else 'close' if area > 0.1 else 'medium' if area > 0.03 else 'far'
            detections.append((CLASSES[cls], round(conf,2), pos, dist))
    
    detections.sort(key=lambda x: -x[1])
    print(f"Time: {elapsed}s | Found: {len(detections)} objects")
    for d in detections:
        print(f"  {d[0]:<15} {d[1]}  {d[2]:<8} {d[3]}")
    
    seen = [f"{d[0]} {d[2]}" for d in detections[:6]]
    print(f'  -> "{", ".join(seen)}"')
    print()
