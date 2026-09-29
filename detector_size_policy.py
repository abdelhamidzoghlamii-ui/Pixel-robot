"""Select one detector input size per frame."""
import time


class SizePolicy:
    def __init__(self, interval_s=5, person_height_fraction=1/3):
        self.interval_s = interval_s
        self.person_height_fraction = person_height_fraction
        self.mode = 'drive'
        self.last_large = time.monotonic()

    def set_mode(self, mode):
        if mode not in ('drive', 'person_search'):
            raise ValueError(mode)
        self.mode = mode

    def next_size(self, now=None):
        now = time.monotonic() if now is None else now
        if self.mode == 'person_search':
            return 640
        if now - self.last_large >= self.interval_s:
            self.last_large = now
            return 640
        return 320

    def observe(self, detections, frame_height):
        if self.mode == 'person_search' and any(
                d['class_name'] == 'person' and
                d['box_xyxy'][3] - d['box_xyxy'][1] >= frame_height*self.person_height_fraction
                for d in detections):
            self.mode = 'drive'
            self.last_large = time.monotonic()
