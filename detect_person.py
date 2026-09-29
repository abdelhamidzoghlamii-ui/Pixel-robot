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

MODELS = {320: os.path.join(os.path.dirname(__file__), 'yolo11s_320.onnx'),
          640: os.path.join(os.path.dirname(__file__), 'yolo11s_640.onnx')}
CONF  = 0.35
IOU   = 0.45

class Detector:
    def __init__(self):
        self.sessions = {size: ort.InferenceSession(path) for size, path in MODELS.items()}

    def detect(self, frame, size=320):
        if size not in self.sessions:
            raise ValueError('size must be 320 or 640')
        if not isinstance(frame, Image.Image):
            with Image.open(frame) as source:
                frame = ImageOps.exif_transpose(source).convert('RGB')
        width, height = frame.size
        img = frame.resize((size, size)).convert('RGB')
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis]
        session = self.sessions[size]
        out = session.run(None, {session.get_inputs()[0].name: arr})[0][0].T
        by_class = {}
        for pred in out:
            scores = pred[4:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
            if conf > CONF:
                cx, cy, w, h = map(float, pred[:4])
                box = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
                by_class.setdefault(cls, []).append((box, conf))
        results = []
        for cls, items in by_class.items():
            for i in nms([v[0] for v in items], [v[1] for v in items], IOU):
                box, conf = items[i]
                x1, y1, x2, y2 = box
                results.append({'class_name': CLASSES[cls], 'class_id': cls,
                                'conf': conf,
                                'box_xyxy': (max(0, min(width, x1*width/size)),
                                             max(0, min(height, y1*height/size)),
                                             max(0, min(width, x2*width/size)),
                                             max(0, min(height, y2*height/size))),
                                'size_used': size})
        return sorted(results, key=lambda d: -d['conf'])

_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector

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

def scene_from_detections(detections, frame_width, frame_height):
    """Legacy tuple view used by navigation, stereo and text callers."""
    results = []
    seen = set()
    for d in detections:
        label = d['class_name']
        if label in seen:
            continue
        seen.add(label)
        x1, y1, x2, y2 = d['box_xyxy']
        cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        area = w*h/(frame_width*frame_height)
        pos = 'left' if cx < frame_width/3 else 'right' if cx > 2*frame_width/3 else 'center'
        dist = 'very close' if area>0.3 else 'close' if area>0.1 else 'medium' if area>0.03 else 'far'
        results.append((label, round(d['conf'], 2), pos, dist,
                        round(cx*640/frame_width), round(cy*640/frame_height),
                        round(w*640/frame_width), round(h*640/frame_height)))
    return results

def detect_scene(image_path, size=640):
    """
    Detect all objects in the scene.
    Returns list of (label, confidence, position, distance)
    """
    with Image.open(image_path) as source:
        frame = ImageOps.exif_transpose(source).convert('RGB')
    return scene_from_detections(get_detector().detect(frame, size), *frame.size)

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
    print(f'Detected in {elapsed}s (640-space legacy coordinates):')
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
