import usb.core, usb.backend.libusb1, time

LIBUSB = "/data/data/com.termux/files/usr/lib/libusb-1.0.so"
backend = usb.backend.libusb1.get_backend(find_library=lambda x: LIBUSB)
dev = usb.core.find(idVendor=0x1a86, idProduct=0x7523, backend=backend)
dev.set_configuration()
print("Connected")

def co(r,v,i): dev.ctrl_transfer(0x40,r,v,i,None)

# Exact sequence from Wireshark capture
co(0xa1, 0xc39c, 0xd98a)  # packet 3147 - init
time.sleep(0.1)
co(0x9a, 0x0f2c, 0x0007)  # packet 3151
co(0xa4, 0x00df, 0x0000)  # packet 3153
co(0xa4, 0x009f, 0x0000)  # packet 3155
co(0x9a, 0x2727, 0x0000)  # packet 3160
co(0x9a, 0x1312, 0xb282)  # packet 3169 - baud rate high
co(0x9a, 0x0f2c, 0x0008)  # packet 3178 - baud rate low
co(0x9a, 0x2727, 0x0000)  # packet 3181
co(0x9a, 0x2727, 0x0000)  # packet 3186
time.sleep(0.5)

# Clear buffer
try: dev.read(0x82, 64, timeout=200)
except: pass

print("Listening for ALIVE messages...")
for i in range(8):
    try:
        data = dev.read(0x82, 64, timeout=1500)
        text = bytes(data).decode("utf-8", errors="replace")
        print("Got:", repr(text), "bytes:", list(data))
        if "ALIVE" in text or "OK" in text:
            print("SUCCESS!")
            break
    except:
        print(".", end="", flush=True)

print("\nSending PING...")
dev.write(0x02, b"PING\n")
time.sleep(1.0)
try:
    data = dev.read(0x82, 64, timeout=2000)
    print("Response:", repr(bytes(data).decode("utf-8", errors="replace")))
except:
    print("No response to PING")
