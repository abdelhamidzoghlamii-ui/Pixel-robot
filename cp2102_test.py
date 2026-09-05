import usb.core, usb.util, usb.backend.libusb1, struct, time

LIBUSB = "/data/data/com.termux/files/usr/lib/libusb-1.0.so"
BAUD   = 115200

b = usb.backend.libusb1.get_backend(find_library=lambda x: LIBUSB)
d = usb.core.find(idVendor=0x10c4, idProduct=0xea60, backend=b)
if d is None: raise SystemExit("CP2102 not found")

try:
    if d.is_kernel_driver_active(0):
        d.detach_kernel_driver(0)
        print("detached kernel driver")
except Exception as e:
    print("detach skipped:", e)

d.set_configuration()
usb.util.claim_interface(d, 0)
print("interface claimed")

d.ctrl_transfer(0x41, 0x00, 0x0001, 0, None)
d.ctrl_transfer(0x41, 0x1E, 0, 0, struct.pack('<I', BAUD))
d.ctrl_transfer(0x41, 0x03, 0x0800, 0, None)

def drain(label, secs=2.0):
    print(f"--- {label} ---")
    end, got = time.time() + secs, b""
    while time.time() < end:
        try: got += bytes(d.read(0x81, 64, timeout=200))
        except Exception: pass
    print(repr(got) if got else "(nothing)")

drain("idle", 2.0)
d.write(0x01, b"PING\n")
drain("after PING", 2.0)
