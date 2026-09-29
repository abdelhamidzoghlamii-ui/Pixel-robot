"""Fail-closed RobotCam JPEG reader; frame.json is diagnostic only."""
import io
import os
import re
import time

from PIL import Image, ImageOps

FRAME_DIR = '/storage/emulated/0/Download/robotcam'
COMMENT_RE = re.compile(rb'^robotcam session=([0-9a-f]{1,32}) frame=(\d{1,12}) '
                        rb'capture_boot_ms=(\d{1,15}) capture_wall_ms=(\d{1,15}) '
                        rb'clock=(sensor|arrival)$')


def read_frame(directory=FRAME_DIR, session=None, max_age=2.0, min_capture_boot_s=None):
    """Return status, session and upright image; only status='ok' is usable."""
    try:
        with open(os.path.join(directory, 'frame.jpg'), 'rb') as f:
            data = f.read()
    except OSError:
        return {'status': 'missing'}
    try:
        with Image.open(io.BytesIO(data)) as source:
            comment = source.info.get('comment', b'')
            image = ImageOps.exif_transpose(source).convert('RGB')
    except (OSError, ValueError, SyntaxError):
        return {'status': 'bad'}
    match = COMMENT_RE.fullmatch(comment) if isinstance(comment, bytes) else None
    if not match:
        return {'status': 'bad'}
    session_id = match.group(1).decode()
    if session is not None and session_id != session:
        return {'status': 'other_session'}
    captured_boot_s = int(match.group(3))/1000
    age = time.clock_gettime(time.CLOCK_BOOTTIME) - captured_boot_s
    if age < -0.2:
        return {'status': 'bad'}
    if age > max_age:
        return {'status': 'missing'}
    if min_capture_boot_s is not None and captured_boot_s <= min_capture_boot_s:
        return {'status': 'missing'}
    return {'status': 'ok', 'session': session_id, 'image': image,
            'frame': int(match.group(2)), 'age_s': age}
