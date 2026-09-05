import http.server, socketserver, subprocess, re, json, time, sys
sys.path.insert(0, "/data/data/com.termux/files/home/robot")
from motors import Motors

PORT      = 8080
CAM       = "http://10.98.123.14:8081"   # IP Webcam
OBST_CM   = 25        # STATUS OBSTACLE_DIST — blocks forward. 0 = readout only
STALE_S   = 1.0       # no DIST: for this long -> treat as unknown

MOVES = ("FORWARD","BACK","LEFT","RIGHT","ROTATE_L","ROTATE_R",
         "FWD_L","FWD_R","BACK_L","BACK_R")
FWD   = ("FORWARD","FWD_L","FWD_R")       # refused when blocked

m = Motors()
m.connect()
time.sleep(1)
print("PING:", "ALIVE" if m.ping() else "NO REPLY")

PAGE = """<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Robot</title><style>
body{margin:0;background:#111;color:#eee;font:16px system-ui;
 -webkit-user-select:none;user-select:none;touch-action:none}
#vid{position:fixed;top:0;left:0;width:100%;height:100%;object-fit:cover;
 z-index:0;opacity:.55}
#wrap{max-width:420px;margin:0 auto;padding:12px;position:relative;z-index:1}
#dist{background:rgba(0,0,0,.55);border-radius:12px;padding:10px;
 text-align:center;margin-bottom:10px}
#dv{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums}
#du{font-size:12px;color:#8b95a3}
#bar2{height:6px;background:#333;border-radius:3px;margin-top:8px;overflow:hidden}
#fill{height:100%;width:0;background:#3d8f5f;transition:width .2s,background .2s}
.warn #dv,.warn #du{color:#ff6b5e}
.warn #fill{background:#ff6b5e}
.near #dv{color:#f0b429}
.near #fill{background:#f0b429}
#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
button{aspect-ratio:1;font-size:24px;border:0;border-radius:12px;
 background:rgba(42,42,42,.85);color:#eee;touch-action:none}
button:active{background:#0a7}
button.off{opacity:.3}
.stop{background:#833}
.rot{margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:8px}
.rot button{aspect-ratio:2.6}
#sp{margin-top:14px;background:rgba(0,0,0,.55);border-radius:12px;padding:8px}
input[type=range]{width:100%;height:36px}
#bar{margin-top:8px;text-align:center;color:#aaa;font-size:14px}
</style></head><body>
<img id="vid" src="__CAM__/video">
<audio src="__CAM__/audio.wav" autoplay></audio>
<div id="wrap">
<div id="dist"><div id="dv">--</div><div id="du">cm ahead</div>
<div id="bar2"><div id="fill"></div></div></div>
<div id="grid">
<button data-c="FWD_L">&#8598;</button>
<button data-c="FORWARD">&#9650;</button>
<button data-c="FWD_R">&#8599;</button>
<button data-c="LEFT">&#9664;</button>
<button data-c="STOP" class="stop">&#9632;</button>
<button data-c="RIGHT">&#9654;</button>
<button data-c="BACK_L">&#8601;</button>
<button data-c="BACK">&#9660;</button>
<button data-c="BACK_R">&#8600;</button>
</div>
<div class="rot">
<button data-c="ROTATE_L">&#8634;</button>
<button data-c="ROTATE_R">&#8635;</button>
</div>
<div id="sp">speed <span id="sv">130</span>
<input type="range" id="sl" min="60" max="200" value="130"></div>
<div id="bar">ready</div>
</div><script>
let cur=null, bar=document.getElementById('bar');
let sl=document.getElementById('sl'), sv=document.getElementById('sv');
let box=document.getElementById('dist'), dv=document.getElementById('dv');
let du=document.getElementById('du'), fill=document.getElementById('fill');
const FWD=['FORWARD','FWD_L','FWD_R'];
sl.addEventListener('input',()=>{sv.textContent=sl.value;});
function send(c){fetch('/cmd?c='+c+'&s='+sl.value)
  .then(r=>r.text()).then(t=>bar.textContent=t)
  .catch(e=>bar.textContent='link lost');}
setInterval(()=>{if(cur)send(cur);},200);
setInterval(()=>{fetch('/status').then(r=>r.json()).then(j=>{
  if(j.stale){dv.textContent='--';du.textContent='no sensor data';
    fill.style.width='0';box.className='';}
  else if(j.d<0){dv.textContent='4m+';du.textContent='clear';
    fill.style.width='100%';box.className='';}
  else{dv.textContent=j.d;du.textContent='cm ahead';
    fill.style.width=Math.min(100,j.d/1.5)+'%';
    box.className=j.blocked?'warn':(j.d<50?'near':'');}
  document.querySelectorAll('button').forEach(b=>{
    if(FWD.includes(b.dataset.c)) b.classList.toggle('off',!!j.blocked);});
}).catch(e=>{});},250);
document.querySelectorAll('button').forEach(b=>{
  const c=b.dataset.c;
  const go=e=>{e.preventDefault();
    if(c==='STOP'){cur=null;send('STOP');}else{cur=c;send(c);}};
  const end=e=>{e.preventDefault();if(c!=='STOP'){cur=null;send('STOP');}};
  b.addEventListener('touchstart',go);
  b.addEventListener('touchend',end);
  b.addEventListener('touchcancel',end);
  b.addEventListener('mousedown',go);
  b.addEventListener('mouseup',end);
});
</script></body></html>""".replace("__CAM__", CAM)

def dist_state():
    """(distance, stale, blocked). -1 = no echo (nothing within ~4m)."""
    d     = m.state["distance"]
    stale = (time.time() - m.state["dist_at"]) > STALE_S
    blocked = (not stale and OBST_CM > 0 and 0 <= d < OBST_CM)
    return d, stale, blocked

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="text/plain"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE, "text/html"); return

        if self.path.startswith("/status"):
            d, stale, blocked = dist_state()
            self._send(200, json.dumps({"d": d, "stale": stale,
                                        "blocked": blocked}), "application/json")
            return

        if not self.path.startswith("/cmd"):
            self._send(404, "no"); return

        q = dict(p.split("=", 1) for p in self.path.split("?")[-1].split("&") if "=" in p)
        c = q.get("c", "")
        try: s = max(0, min(200, int(q.get("s", 130))))
        except ValueError: s = 130

        if c == "STOP":
            self._send(200, "STOP" if m.send("STOP") else "SERIAL FAIL"); return
        if c not in MOVES:
            self._send(400, "bad cmd"); return

        _, _, blocked = dist_state()
        if blocked and c in FWD:
            m.send("STOP")
            self._send(200, "BLOCKED — obstacle"); return

        ok = m.send(f"{c}:{s}")
        self._send(200, f"{c} {s}" if ok else "SERIAL FAIL")

class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

print("\nListening on port", PORT)
try:
    out = subprocess.run(["/system/bin/ip", "-4", "addr"],
                         capture_output=True, text=True).stdout
    for ip in re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", out):
        if not ip.startswith("127."):
            print(f"   http://{ip}:{PORT}")
except Exception as e:
    print("   (could not list IPs:", e, ")")
print()

try:
    S(("0.0.0.0", PORT), H).serve_forever()
except KeyboardInterrupt:
    print("\nstopping")
    m.disconnect()
