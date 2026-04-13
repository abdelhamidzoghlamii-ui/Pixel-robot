import sys, os, time
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

_session = None

def get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL)
    return _session

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

def detect_scene(image_path):
    """
    Detect all objects in the scene.
    Returns list of (label, confidence, position, distance)
    """
    session = get_session()
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img = img.resize((640, 640))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]

    out = session.run(None, {session.get_inputs()[0].name: arr})[0][0].T

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
        best = items[keep[0]]
        cx, cy, w, h = best[2], best[3], best[4], best[5]
        conf = best[1]
        area = (w*h)/(640*640)
        pos = 'left' if cx < 213 else 'right' if cx > 427 else 'center'
        dist = 'very close' if area>0.3 else 'close' if area>0.1 else 'medium' if area>0.03 else 'far'
        results.append((CLASSES[cls], round(conf,2), pos, dist, round(cx), round(cy), round(w), round(h)))

    results.sort(key=lambda x: -x[1])
    return results

def detect_person(image_path):
    """
    Legacy function — returns (found, confidence, position)
    for backwards compatibility with existing code.
    """
    results = detect_scene(image_path)
    for r in results:
        label, conf, pos, dist = r[0], r[1], r[2], r[3]
        if label == 'person':
            return True, conf, pos
    return False, 0.0, 'none'

def scene_to_text(results, coords=False):
    """
    Convert detection results to a natural language sentence.
    coords=True adds pixel coordinates for stereo vision.
    """
    if not results:
        return "empty room, nothing detected"
    if coords:
        parts = [f"{r[0]} x={r[4]} y={r[5]} w={r[6]} h={r[7]} {r[2]} {r[3]}" for r in results]
    else:
        parts = [f"{r[0]} {r[2]}" for r in results]
    return ', '.join(parts)

def person_direction(results):
    """
    Return direction to move toward detected person.
    Returns 'LEFT', 'RIGHT', 'FORWARD', or None
    """
    for r in results:
        label, conf, pos, dist = r[0], r[1], r[2], r[3]
        if label == 'person':
            if pos == 'left':   return 'LEFT'
            if pos == 'right':  return 'RIGHT'
            if pos == 'center': return 'FORWARD'
    return None

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'test_photos/scene_test.jpg'
    t0 = time.time()
    results = detect_scene(path)
    elapsed = round(time.time()-t0, 3)
    print(f'Detected in {elapsed}s (640x640 frame):')
    print(f'  {"object":<15} {"conf":<5} {"pos":<8} {"dist":<12} {"cx":>5} {"cy":>5} {"w":>5} {"h":>5}')
    print('  ' + '-'*60)
    for r in results:
        print(f'  {r[0]:<15} {r[1]:<5} {r[2]:<8} {r[3]:<12} {r[4]:>5} {r[5]:>5} {r[6]:>5} {r[7]:>5}')
    print(f'\nScene: "{scene_to_text(results)}"')
    print(f'Coords: "{scene_to_text(results, coords=True)}"')
    direction = person_direction(results)
    if direction:
        print(f"Person direction: {direction}")
    else:
        print("No person detected")