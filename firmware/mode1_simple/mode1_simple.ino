// ============================================================
// MODE 1 — teleop. ESP32 hotspot + web UI + HC-SR04 readout.
// Written from scratch. No camera, Pixel not required.
//
//   1. Flash
//   2. Control phone: join WiFi "MecanumBot" / "12345678"
//   3. Open  http://192.168.4.1
//
// Power: ESP32 on USB (phone or powerbank), motors on the battery
// rail via the MX1508s. Buck->VIN stays off until caps are fitted
// (DECISIONS #38 / #39).
// ============================================================

#include <WiFi.h>
#include <WebServer.h>

// ---------------- WIRING (measured, supersedes DECISIONS #40) ----------------
// Verified by driving each channel alone:
//   ch0  P16/P17  -> RR   spins forward on positive
//   ch1  P18/P19  -> FR   spins forward on positive
//   ch2  P21/P22  -> FL   spins backward on positive
//   ch3  P23/P25  -> RL   spins backward on positive
// Note driver #2's channels are the reverse of what #40 records:
// chA (P21/P22) is FRONT left, chB (P23/P25) is REAR left.
const int RR = 0;
const int FR = 1;
const int FL = 2;
const int RL = 3;

// Left side spins backward on positive drive, so both left indices are
// inverted in software.
const bool INVERT[4] = { false, false, true, true };

const int PINS[4][2] = {
  {16, 17},   // 0 RR
  {18, 19},   // 1 FR
  {21, 22},   // 2 FL
  {23, 25}    // 3 RL
};

// ---------------- SENSOR (DECISIONS #42) ----------------
const int  TRIG = 27;
const int  ECHO = 26;              // via 1k/2k divider - ECHO is 5V
const unsigned long PING_EVERY_MS = 100;
const unsigned long ECHO_TIMEOUT  = 25000;   // us, ~4 m

// Block forward motion below this. Matches STATUS OBSTACLE_DIST (25 cm).
// Set to 0 to disable the block and keep the readout only.
const int STOP_DIST_CM = 25;

// ---------------- TUNING ----------------
const char* AP_SSID = "MecanumBot";
const char* AP_PASS = "12345678";

const int PWM_FREQ   = 1000;
const int PWM_RES    = 8;
const int MAX_SPEED  = 200;    // DECISIONS #37 - 6V motors on an 8.2V rail
const int TIMEOUT_MS = 600;    // no command -> stop

// ---------------- state ----------------
WebServer server(80);
unsigned long lastCmd  = 0;
unsigned long lastPing = 0;
int  distCm    = -1;           // -1 = no echo
bool blocked   = false;
bool moving    = false;

// ---------------- PWM ----------------
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  void pwmInit(int pin, int ch)    { ledcAttach(pin, PWM_FREQ, PWM_RES); }
  void pwmWrite(int pin, int duty) { ledcWrite(pin, duty); }
#else
  int chanOf(int pin) {
    for (int i = 0; i < 4; i++) {
      if (PINS[i][0] == pin) return i * 2;
      if (PINS[i][1] == pin) return i * 2 + 1;
    }
    return 0;
  }
  void pwmInit(int pin, int ch) {
    ledcSetup(ch, PWM_FREQ, PWM_RES);
    ledcAttachPin(pin, ch);
  }
  void pwmWrite(int pin, int duty) { ledcWrite(chanOf(pin), duty); }
#endif

void setMotor(int i, int speed) {
  if (INVERT[i]) speed = -speed;
  speed = constrain(speed, -255, 255);
  if (speed >= 0) {
    pwmWrite(PINS[i][1], 0);
    pwmWrite(PINS[i][0], speed);
  } else {
    pwmWrite(PINS[i][0], 0);
    pwmWrite(PINS[i][1], -speed);
  }
}

void stopAll() {
  for (int i = 0; i < 4; i++) {
    pwmWrite(PINS[i][0], 0);
    pwmWrite(PINS[i][1], 0);
  }
  moving = false;
}

// vy forward, vx strafe right, w rotate CW  (each -1..1)
// vx signs are flipped from the textbook mix: this chassis's rollers
// sit in the mirrored X pattern. Verified live.
void drive(float vy, float vx, float w, int speed) {
  float fl = vy - vx + w;
  float fr = vy + vx - w;
  float rl = vy + vx + w;
  float rr = vy - vx - w;

  float m = max(max(fabs(fl), fabs(fr)), max(fabs(rl), fabs(rr)));
  if (m > 1.0f) { fl /= m; fr /= m; rl /= m; rr /= m; }

  setMotor(FL, (int)(fl * speed));
  setMotor(FR, (int)(fr * speed));
  setMotor(RL, (int)(rl * speed));
  setMotor(RR, (int)(rr * speed));
  moving = true;
}

// ---------------- ultrasonic ----------------
void readDistance() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  unsigned long us = pulseIn(ECHO, HIGH, ECHO_TIMEOUT);
  distCm = (us == 0) ? -1 : (int)(us * 0.0343 / 2.0);

  blocked = (STOP_DIST_CM > 0 && distCm >= 0 && distCm < STOP_DIST_CM);
}

// ---------------- web page ----------------
const char PAGE[] PROGMEM = R"rawliteral(<!DOCTYPE html><html><head>
<meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,user-scalable=no">
<title>Mecanum</title>
<style>
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;padding:14px 12px 26px;background:#14171c;color:#e8ecf1;
 font-family:system-ui,-apple-system,sans-serif;text-align:center;
 user-select:none;-webkit-user-select:none;touch-action:manipulation}
h2{margin:2px 0 12px;font-size:19px;font-weight:600;letter-spacing:.3px}

.dist{max-width:330px;margin:0 auto 14px;background:#1c2129;border-radius:12px;
 padding:12px 14px;border:1px solid #2a313b}
.dv{font-size:30px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums}
.du{font-size:13px;color:#8b95a3;margin-top:2px}
.bar{height:7px;background:#252c36;border-radius:4px;margin-top:10px;overflow:hidden}
.fill{height:100%;width:0;background:#3d8f5f;border-radius:4px;
 transition:width .18s linear,background .18s}
.warn .dv{color:#ff6b5e}
.warn .fill{background:#ff6b5e}
.near .dv{color:#f0b429}
.near .fill{background:#f0b429}

.pad{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;
 max-width:330px;margin:0 auto}
button{background:#252d38;border:1px solid #313a47;border-radius:13px;
 color:#e8ecf1;font-size:25px;padding:19px 0;cursor:pointer;
 transition:background .06s,transform .06s}
button:active{background:#3f6fa8;transform:scale(.95)}
.stop{background:#7d2a2a;border-color:#963333;font-size:14px;font-weight:700;
 letter-spacing:1px}
.stop:active{background:#a83b3b}
.rot{display:flex;gap:9px;max-width:330px;margin:9px auto 0}
.rot button{flex:1;font-size:16px;padding:15px 0}
.off{opacity:.32;pointer-events:none}

.sp{max-width:330px;margin:16px auto 0}
input[type=range]{width:100%;height:26px;accent-color:#3f6fa8;background:none}
.sl{font-size:13px;color:#8b95a3;margin-top:2px}
.msg{font-size:12px;color:#ff6b5e;height:16px;margin-top:8px}
</style></head><body>

<h2>Mecanum Control</h2>

<div class=dist id=dbox>
  <div class=dv id=dv>--</div>
  <div class=du>cm ahead</div>
  <div class=bar><div class=fill id=fill></div></div>
</div>

<div class=pad>
  <button data-y=1  data-x=-1>&#8598;</button>
  <button data-y=1            id=fwd>&#8593;</button>
  <button data-y=1  data-x=1 >&#8599;</button>
  <button data-x=-1>&#8592;</button>
  <button class=stop data-stop=1>STOP</button>
  <button data-x=1>&#8594;</button>
  <button data-y=-1 data-x=-1>&#8601;</button>
  <button data-y=-1>&#8595;</button>
  <button data-y=-1 data-x=1>&#8600;</button>
</div>

<div class=rot>
  <button data-w=-1>&#8634; left</button>
  <button data-w=1>right &#8635;</button>
</div>

<div class=sp>
  <input type=range min=60 max=200 value=150 id=sp>
  <div class=sl id=sl>speed 150</div>
</div>

<div class=msg id=msg></div>

<script>
const sp=document.getElementById('sp'), sl=document.getElementById('sl'),
      dv=document.getElementById('dv'), fill=document.getElementById('fill'),
      dbox=document.getElementById('dbox'), msg=document.getElementById('msg');
let hold=null;

sp.oninput=()=>sl.textContent='speed '+sp.value;

function send(y,x,w){
  fetch(`/cmd?y=${y}&x=${x}&w=${w}&s=${sp.value}`)
    .then(r=>r.text())
    .then(t=>{ msg.textContent = (t=='blocked') ? 'obstacle - forward blocked' : ''; })
    .catch(()=>{});
}
function halt(){ clearInterval(hold); hold=null; fetch('/stop').catch(()=>{}); }

function press(e){
  const d=e.currentTarget.dataset;
  if(d.stop){ halt(); return; }
  const y=+d.y||0, x=+d.x||0, w=+d.w||0;
  send(y,x,w);
  clearInterval(hold);
  hold=setInterval(()=>send(y,x,w),150);
  e.preventDefault();
}

document.querySelectorAll('button').forEach(b=>{
  b.addEventListener('touchstart',press,{passive:false});
  b.addEventListener('mousedown',press);
  ['touchend','touchcancel','mouseup','mouseleave']
    .forEach(ev=>b.addEventListener(ev,halt));
});

// live distance
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(j=>{
    if(j.d<0){ dv.textContent='--'; fill.style.width='0'; dbox.className='dist'; }
    else{
      dv.textContent=j.d;
      fill.style.width=Math.min(100,j.d/1.5)+'%';
      dbox.className='dist'+(j.b?' warn':(j.d<50?' near':''));
    }
    document.getElementById('fwd').classList.toggle('off', !!j.b);
  }).catch(()=>{});
},250);
</script></body></html>)rawliteral";

// ---------------- handlers ----------------
void handleRoot() { server.send_P(200, "text/html", PAGE); }

void handleCmd() {
  float y = server.arg("y").toFloat();
  float x = server.arg("x").toFloat();
  float w = server.arg("w").toFloat();
  int   s = constrain(server.arg("s").toInt(), 0, MAX_SPEED);

  lastCmd = millis();

  // refuse forward into an obstacle; reverse/strafe/rotate stay allowed
  if (blocked && y > 0) {
    stopAll();
    server.send(200, "text/plain", "blocked");
    return;
  }

  drive(y, x, w, s);
  server.send(200, "text/plain", "ok");
}

void handleStop() {
  stopAll();
  lastCmd = 0;
  server.send(200, "text/plain", "stop");
}

void handleStatus() {
  char buf[48];
  snprintf(buf, sizeof(buf), "{\"d\":%d,\"b\":%d}", distCm, blocked ? 1 : 0);
  server.send(200, "application/json", buf);
}

// ---------------- setup / loop ----------------
void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 4; i++) {
    pwmInit(PINS[i][0], i * 2);
    pwmInit(PINS[i][1], i * 2 + 1);
  }
  stopAll();

  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  digitalWrite(TRIG, LOW);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.println("AP up - open http://192.168.4.1");

  server.on("/",       handleRoot);
  server.on("/cmd",    handleCmd);
  server.on("/stop",   handleStop);
  server.on("/status", handleStatus);
  server.begin();
}

void loop() {
  server.handleClient();

  if (millis() - lastPing >= PING_EVERY_MS) {
    lastPing = millis();
    readDistance();
    if (blocked && moving) stopAll();   // obstacle appeared mid-move
  }

  // failsafe - stop if the browser goes quiet
  if (lastCmd && millis() - lastCmd > TIMEOUT_MS) {
    stopAll();
    lastCmd = 0;
  }
}
