import sys, os, time, queue, threading
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

ROBOT_CPUS = {4, 5}       # MID cores (DECISIONS #128 power map)
ROBOT_INTRA_THREADS = 2   # per session; ORT counts the calling thread as one of them


# Copied from benchmark/power_map/power_map.py: thread_ids() (os.listdir instead of pathlib).
def thread_ids():
    return {int(t) for t in os.listdir(f'/proc/{os.getpid()}/task')}


# Copied from benchmark/power_map/power_map.py: the set + verify lines of PinnedDetector.loop()/build_detector();
# the tid 0 / main-thread refusal is added.
def pin(tid, cpus):
    if tid in (0, os.getpid()):  # 0 = the caller, pid = the main thread: never pinned
        raise RuntimeError(f'refusing to pin tid {tid}')
    os.sched_setaffinity(tid, cpus)
    if os.sched_getaffinity(tid) != cpus:
        raise RuntimeError(f'thread {tid} affinity did not take effect')


class PinnedDetector:
    """The robot's detector (#128): both yolo11s sessions with 2 intra-op / 1 inter-op threads; the 2 new ORT
    workers and one detector-only calling thread are pinned to ROBOT_CPUS and verified. Only these 3 threads are
    ever pinned: never the process, pid 0, the main thread or any other thread.
    Copied from benchmark/power_map/power_map.py: build_detector() ('mid' branch) and PinnedDetector. Changed:
    raises on any failure (no 'default' branch); each request carries its own reply queue, so a late answer can
    never reach a later request; check_affinity()/affinity()/config() added for the robot's trace; no close()
    (daemon thread, ends with the process)."""

    def __init__(self, cpus=ROBOT_CPUS):
        self.cpus = set(cpus)
        before = thread_ids()  # the diff sees only ORT workers: no other code starts threads meanwhile
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ROBOT_INTRA_THREADS
        opts.inter_op_num_threads = 1
        self.detector = Detector.__new__(Detector)
        self.detector.sessions = {size: ort.InferenceSession(path, sess_options=opts)
                                  for size, path in MODELS.items()}
        self.worker_tids = sorted(thread_ids() - before)
        expected = len(MODELS) * (ROBOT_INTRA_THREADS - 1)
        if len(self.worker_tids) != expected:
            raise RuntimeError(f'cannot identify ORT workers: expected {expected}, got {self.worker_tids}')
        for tid in self.worker_tids:
            pin(tid, self.cpus)
        self.requests, ready = queue.Queue(), queue.Queue()
        self.thread = threading.Thread(target=self.loop, args=(ready,), daemon=True)
        self.thread.start()
        ok, result = ready.get()
        if not ok:
            self.thread.join()
            raise result
        self.caller_tid = result

    def loop(self, ready):
        try:
            tid = threading.get_native_id()
            pin(tid, self.cpus)
            ready.put((True, tid))
        except BaseException as e:
            ready.put((False, e))
            return
        while True:
            frame, size, reply = self.requests.get()
            try:
                reply.put((True, self.detector.detect(frame, size)))
            except BaseException as e:
                reply.put((False, e))

    def detect(self, frame, size=320):
        reply = queue.Queue()
        self.requests.put((frame, size, reply))
        ok, result = reply.get()
        if not ok:
            raise result
        return result

    def affinity(self):
        """{tid: sorted cpus or error text} for the caller and workers; read only."""
        live, out = thread_ids(), {}
        for tid in [self.caller_tid] + self.worker_tids:
            try:
                if tid not in live:  # never touch a tid that left this process (it could be reused elsewhere)
                    raise ProcessLookupError(f'thread {tid} is gone')
                out[tid] = sorted(os.sched_getaffinity(tid))
            except OSError as e:
                out[tid] = f'{type(e).__name__}: {e}'
        return out

    def check_affinity(self):
        """Re-pin and verify any of the 3 threads whose affinity drifted (Android cpuset changes can reset it).
        Never raises: the result says ok or not."""
        before = self.affinity()
        repin = [tid for tid, cpus in before.items() if isinstance(cpus, list) and set(cpus) != self.cpus]
        errors = {tid: cpus for tid, cpus in before.items() if isinstance(cpus, str)}
        for tid in repin:
            try:
                pin(tid, self.cpus)
            except (OSError, RuntimeError) as e:
                errors[tid] = f'{type(e).__name__}: {e}'
        after = self.affinity() if repin else before
        return {'before': before, 'repinned': repin, 'after': after, 'ok': not errors, 'errors': errors}

    def config(self):
        return {'ort_version': ort.__version__,
                'sessions': {size: {'intra_op_num_threads': s.get_session_options().intra_op_num_threads,
                                    'inter_op_num_threads': s.get_session_options().inter_op_num_threads}
                             for size, s in self.detector.sessions.items()},
                'worker_tids': self.worker_tids, 'caller_tid': self.caller_tid,
                'requested_cpus': sorted(self.cpus), 'affinity': self.affinity()}


_detector = None

def get_detector(pinned=False):
    """Shared detector. pinned=True (the robot process) creates the PinnedDetector; later get_detector() calls,
    e.g. from detect_scene(), then return that same one. Without pinned it is a plain Detector(), as before."""
    global _detector
    if _detector is None:
        _detector = PinnedDetector() if pinned else Detector()
    elif pinned and not isinstance(_detector, PinnedDetector):
        raise RuntimeError('a default detector is already loaded; the robot needs one pinned session pair')
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
