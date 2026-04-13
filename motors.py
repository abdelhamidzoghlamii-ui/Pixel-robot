import usb.core, usb.backend.libusb1, time, threading, os

LIBUSB   = "/data/data/com.termux/files/usr/lib/libusb-1.0.so"
VENDOR   = 0x1a86
PRODUCT  = 0x7523
EP_WRITE = 0x02
EP_READ  = 0x82

class Motors:
    def __init__(self):
        self.dev     = None
        self.state   = {"distance": 999, "alive": False, "connected": False}
        self._lock   = threading.Lock()
        self._running = False
        self._thread  = None

    def connect(self):
        backend = usb.backend.libusb1.get_backend(
            find_library=lambda x: LIBUSB)
        dev = usb.core.find(idVendor=VENDOR, idProduct=PRODUCT, backend=backend)
        if dev is None:
            raise RuntimeError("Arduino not found — check USB connection")
        dev.set_configuration()
        self.dev = dev

        # CH340 init — exact sequence from Wireshark capture
        def co(r, v, i): dev.ctrl_transfer(0x40, r, v, i, None)
        co(0xa1, 0xc39c, 0xd98a)
        time.sleep(0.1)
        co(0x9a, 0x0f2c, 0x0007)
        co(0xa4, 0x00df, 0x0000)
        co(0xa4, 0x009f, 0x0000)
        co(0x9a, 0x2727, 0x0000)
        co(0x9a, 0x1312, 0xb282)
        co(0x9a, 0x0f2c, 0x0008)
        co(0x9a, 0x2727, 0x0000)
        co(0x9a, 0x2727, 0x0000)
        time.sleep(0.5)

        # Clear buffer
        try: dev.read(EP_READ, 64, timeout=200)
        except: pass

        self.state["connected"] = True
        print("[MOTORS] Connected to Arduino")

        # Start background reader
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        buf = ""
        while self._running:
            try:
                data = self.dev.read(EP_READ, 64, timeout=500)
                buf += bytes(data).decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("DIST:"):
                        try:
                            self.state["distance"] = int(line.split(":")[1])
                        except:
                            pass
                    elif line == "ALIVE":
                        self.state["alive"] = True
                    elif line == "READY":
                        print("[MOTORS] Arduino ready")
            except:
                pass

    def send(self, cmd):
        with self._lock:
            try:
                self.dev.write(EP_WRITE, (cmd + "\n").encode())
                time.sleep(0.05)
            except Exception as e:
                print("[MOTORS] Send error:", e)

    def stop(self):
        self.send("STOP")

    def forward(self, speed=150, duration=0):
        self.send(f"FORWARD:{speed}")
        if duration > 0:
            time.sleep(duration)
            self.stop()

    def backward(self, speed=150, duration=0):
        self.send(f"BACK:{speed}")
        if duration > 0:
            time.sleep(duration)
            self.stop()

    def rotate_left(self, speed=120, duration=0):
        self.send(f"ROTATE_L:{speed}")
        if duration > 0:
            time.sleep(duration)
            self.stop()

    def rotate_right(self, speed=120, duration=0):
        self.send(f"ROTATE_R:{speed}")
        if duration > 0:
            time.sleep(duration)
            self.stop()

    def strafe_left(self, speed=150, duration=0):
        self.send(f"LEFT:{speed}")
        if duration > 0:
            time.sleep(duration)
            self.stop()

    def strafe_right(self, speed=150, duration=0):
        self.send(f"RIGHT:{speed}")
        if duration > 0:
            time.sleep(duration)
            self.stop()

    def servo(self, angle=90):
        angle = max(0, min(180, angle))
        self.send(f"SERVO:{angle}")

    def get_distance(self):
        return self.state["distance"]

    def navigate_safe(self, speed=150, duration=3.0, min_dist=20):
        self.send(f"FORWARD:{speed}")
        start = time.time()
        while time.time() - start < duration:
            dist = self.get_distance()
            if dist < min_dist:
                self.stop()
                print(f"[MOTORS] Obstacle at {dist}cm — stopping")
                time.sleep(0.3)
                self.rotate_left(120, 0.8)
                return "obstacle"
            time.sleep(0.05)
        self.stop()
        return "arrived"

    def disconnect(self):
        self._running = False
        self.stop()
        print("[MOTORS] Disconnected")

if __name__ == "__main__":
    import sys
    m = Motors()
    m.connect()
    time.sleep(1)

    print("Distance:", m.get_distance(), "cm")
    print("Testing PING...")
    m.send("PING")
    time.sleep(0.5)

    print("\nTesting forward 1 second...")
    m.forward(150, 1.0)
    time.sleep(0.5)

    print("Testing rotate left 0.5 seconds...")
    m.rotate_left(120, 0.5)
    time.sleep(0.5)

    print("Distance:", m.get_distance(), "cm")
    print("\nAll tests done!")
    m.disconnect()
