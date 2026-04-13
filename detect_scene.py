import sys, os, time
sys.path.insert(0, '/data/data/com.termux/files/home/robot')
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

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

MODEL = '/data/data/com.termux/files/home/robot/yolo11m.onnx'
CONF  = 0.35
IOU   = 0.45

def iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0

def nms(boxes, confs, iou_thresh=0.45):
    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep

def detect(image_path, model_path=MODEL):
    session = ort.InferenceSession(model_path)
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img = img.resize((640, 640))
    arr = np.array(img).astype(np.float32)/255.0
    arr = arr.transpose(2,0,1)[np.newaxis]
    
    t0 = time.time()
    out = session.run(None, {session.get_inputs()[0].name: arr})[0][0].T
    elapsed = round(time.time()-t0, 3)
    
    # Group by class then apply NMS
    by_class = {}
    for pred in out:
        scores = pred[4:]
        cls = int(np.argmax(scores))
        conf = float(scores[cls])
        if conf > CONF:
            cx,cy,w,h = float(pred[0]),float(pred[1]),float(pred[2]),float(pred[3])
            box = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append((box, conf, cx, cy, w, h))
    
    results = []
    for cls, items in by_class.items():
        boxes = [i[0] for i in items]
        confs = [i[1] for i in items]
        keep = nms(boxes, confs, IOU)
        best = items[keep[0]]  # keep highest confidence per class
        cx, cy, w, h = best[2], best[3], best[4], best[5]
        conf = best[1]
        area = (w*h)/(640*640)
        pos = 'left' if cx < 213 else 'right' if cx > 427 else 'center'
        dist = 'very close' if area>0.3 else 'close' if area>0.1 else 'medium' if area>0.03 else 'far'
        results.append((CLASSES[cls], round(conf,2), pos, dist))
    
    results.sort(key=lambda x: -x[1])
    return results, elapsed

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv)>1 else '/data/data/com.termux/files/home/robot/test_photos/scene_test.jpg'
    results, elapsed = detect(path)
    print(f"Detected in {elapsed}s:")
    for r in results:
        print(f"  {r[0]:<15} {r[1]}  {r[2]:<8} {r[3]}")
    sentence = ', '.join([f"{r[0]} {r[2]}" for r in results])
    print(f'\nRobot sees: "{sentence}"')
