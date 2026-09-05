import usb.core, usb.util, usb.backend.libusb1, struct, time

LIB = "/data/data/com.termux/files/usr/lib/libusb-1.0.so"
b = usb.backend.libusb1.get_backend(find_library=lambda x: LIB)
d = usb.core.find(idVendor=0x10c4, idProduct=0xea60, backend=b)
if d is None: raise SystemExit("CP2102 not found")
try:
    if d.is_kernel_driver_active(0): d.detach_kernel_driver(0)
except Exception: pass
d.set_configuration(); usb.util.claim_interface(d, 0)
d.ctrl_transfer(0x41, 0x00, 0x0001, 0, None)
d.ctrl_transfer(0x41, 0x1E, 0, 0, struct.pack('<I', 115200))
d.ctrl_transfer(0x41, 0x03, 0x0800, 0, None)

print("raw lines for 5s — move your hand in front of the sensor now")
buf, end = "", time.time() + 5
while time.time() < end:
    try: buf += bytes(d.read(0x81, 64, timeout=200)).decode(errors="replace")
    except Exception: pass
    while "\n" in buf:
        line, buf = buf.split("\n", 1)
        if line.strip(): print(repr(line.strip()))
usb.util.release_interface(d, 0)
