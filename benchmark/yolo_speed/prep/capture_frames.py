# Capture 20 frames exactly as main.take_photo does (termux-camera-photo, default camera, then sleep 0.5),
# timing each call. Frames 01-10 timings = the "10 captures". Run with native Termux python.
import os, time, json
D = '/data/data/com.termux/files/home/yolo_bench/frames'
t = []
for i in range(1, 21):
    p = f'{D}/frame_{i:02d}.jpg'
    t0 = time.perf_counter()
    os.system(f'termux-camera-photo {p} 2>/dev/null')
    t1 = time.perf_counter()
    time.sleep(0.5)
    ok = os.path.exists(p) and os.path.getsize(p) > 1000
    t.append({'frame': i, 'call_s': round(t1 - t0, 3), 'take_photo_s': round(time.perf_counter() - t0, 3), 'ok': ok})
    print(t[-1], flush=True)
json.dump(t, open(f'{D}/../capture_timing.json', 'w'), indent=1)
