import usb.core, usb.util, usb.backend.libusb1, struct, time, threading, os, atexit

LIBUSB   = "/data/data/com.termux/files/usr/lib/libusb-1.0.so"
VENDOR   = 0x10c4          # CP2102 (was CH340 0x1a86)
PRODUCT  = 0xea60
EP_WRITE = 0x01            # was 0x02 on CH340
EP_READ  = 0x81            # was 0x82 on CH340
BAUD     = 115200

class Motors:
    def __init__(self):
        self.dev = None
        self.state = {"distance": 999, "alive": False, "connected": False,
                      "alive_at": 0.0, "dist_at": 0.0}
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def connect(self):
        backend = usb.backend.libusb1.get_backend(find_library=lambda x: LIBUSB)
        dev = usb.core.find(idVendor=VENDOR, idProduct=PRODUCT, backend=backend)
        if dev is None:
            raise RuntimeError("CP2102 not found — check USB connection")

        try:
            if dev.is_kernel_driver_active(0):
                dev.detach_kernel_driver(0)
        except Exception:
            pass

        dev.set_configuration()
        usb.util.claim_interface(dev, 0)
        self.dev = dev

        # CP210x vendor init. DTR/RTS deliberately left untouched —
        # asserting them resets the ESP32 on auto-reset boards.
        dev.ctrl_transfer(0x41, 0x00, 0x0001, 0, None)                   # IFC_ENABLE
        dev.ctrl_transfer(0x41, 0x1E, 0, 0, struct.pack('<I', BAUD))     # SET_BAUDRATE
        dev.ctrl_transfer(0x41, 0x03, 0x0800, 0, None)                   # LINE_CTL 8N1

        try: dev.read(EP_READ, 64, timeout=200)
        except Exception: pass

        self.state["connected"] = True
        atexit.register(self.disconnect)
        print("[MOTORS] Connected to ESP32 (CP2102)")

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
                            self.state["dist_at"] = time.time()
                        except Exception:
                            pass
                    elif line == "ALIVE":
                        self.state["alive"] = True
                        self.state["alive_at"] = time.time()
                    elif line == "READY":
                        print("[MOTORS] ESP32 ready")
            except Exception:
                pass

    def send(self, cmd):
        """Returns True if the bytes went out, False otherwise."""
        with self._lock:
            try:
                self.dev.write(EP_WRITE, (cmd + "\n").encode())
                time.sleep(0.05)
                return True
            except Exception as e:
                print("[MOTORS] Send error:", e)
                self.state["connected"] = False
                return False

    def ping(self, timeout=1.0):
        """True if the ESP32 answers ALIVE within timeout."""
        self.state["alive"] = False
        if not self.send("PING"):
            return False
        end = time.time() + timeout
        while time.time() < end:
            if self.state["alive"]:
                return True
            time.sleep(0.05)
        return False

    def stop(self):
        return self.send("STOP")

    def _hold(self, cmd, duration, every=0.2):
        """Send cmd, re-sending every `every` s so the ESP32 watchdog
        (WATCHDOG_MS = 1000) does not cut the move short, then STOP."""
        if not self.send(cmd):
            return False
        if duration > 0:
            end = time.time() + duration
            while time.time() < end:
                time.sleep(min(every, max(0.0, end - time.time())))
                if time.time() < end:
                    self.send(cmd)
            self.stop()
        return True

    def forward(self, speed=150, duration=0):
        return self._hold(f"FORWARD:{speed}", duration)

    def backward(self, speed=150, duration=0):
        return self._hold(f"BACK:{speed}", duration)

    def rotate_left(self, speed=120, duration=0):
        return self._hold(f"ROTATE_L:{speed}", duration)

    def rotate_right(self, speed=120, duration=0):
        return self._hold(f"ROTATE_R:{speed}", duration)

    def strafe_left(self, speed=150, duration=0):
        return self._hold(f"LEFT:{speed}", duration)

    def strafe_right(self, speed=150, duration=0):
        return self._hold(f"RIGHT:{speed}", duration)

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
        if self.dev is None:
            return
        self._running = False
        try: self.stop()
        except Exception: pass
        try: usb.util.release_interface(self.dev, 0)
        except Exception: pass
        try: usb.util.dispose_resources(self.dev)
        except Exception: pass
        self.dev = None
        self.state["connected"] = False
        print("[MOTORS] Disconnected")


if __name__ == "__main__":
    m = Motors()
    m.connect()
    time.sleep(1)
    print("PING:", "ALIVE" if m.ping() else "no reply")
    print("Distance:", m.get_distance(), "cm")
    print("\nForward 1s...")
    m.forward(150, 1.0)
    time.sleep(0.5)
    print("Rotate left 0.5s...")
    m.rotate_left(120, 0.5)
    time.sleep(0.5)
    m.disconnect()
