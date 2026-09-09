// MODE 2 — teleop / autonomous. Pixel 7 drives; ESP32 executes.
// Serial protocol matches motors.py.
//
//   in :  FORWARD:<spd>  BACK:<spd>  LEFT:<spd>  RIGHT:<spd>
//         ROTATE_L:<spd> ROTATE_R:<spd>
//         FWD_L:<spd>    FWD_R:<spd>   BACK_L:<spd>  BACK_R:<spd>
//         STOP  SERVO:<deg>  PING
//   out:  READY (boot)   ALIVE (reply to PING)   DIST:<cm>  (-1 = no echo)
//
// LEFT/RIGHT are strafes. FWD_L etc. are diagonals. Speeds 0-255, capped
// at MAX_SPEED. ESP32 on phone USB, motors on battery rail (DECISIONS #39).

// ---- BAUD ----
// Board is CP2102 (10c4:ea60), not CH340. motors.py sets the rate explicitly
// via CP210x SET_BAUDRATE (0x1E) as a 32-bit value, so this is a free choice.
// Verified working phone-side at 115200.
const long BAUD = 115200;

// ---- corner mapping — DECISIONS #43 (supersedes #40) ----
// Measured by driving each channel alone:
//   ch0  P16/P17 -> RR   ch1  P18/P19 -> FR
//   ch2  P21/P22 -> FL   ch3  P23/P25 -> RL
// Left motors spin backward on positive drive, so indices 2 and 3 are
// inverted in software. Matches mode1_simple.ino, which is verified.
const int RR = 0;
const int FR = 1;
const int FL = 2;
const int RL = 3;

const bool INVERT[4] = { false, false, true, true };

const int PINS[4][2] = {
  {16, 17},   // 0 RR
  {18, 19},   // 1 FR
  {21, 22},   // 2 FL
  {23, 25}    // 3 RL
};

const int PWM_FREQ  = 1000;
const int PWM_RES   = 8;
const int MAX_SPEED = 200;      // DECISIONS #37 — 6V-nominal motors on 8.2V

// Failsafe: stop if nothing arrives for this long. Teleop re-sends the held
// command at 5 Hz, so 1 s means a dropped WiFi link stops the chassis fast.
const unsigned long WATCHDOG_MS = 1000;

// ---- ultrasonic — DECISIONS #42 ----
// TRIG P27 direct (3.3V drive is enough). ECHO P26 via 1k/2k divider —
// mandatory, ECHO is a 5V output into a 3.3V pin. VCC from ESP32 5V pin.
#define HAS_ULTRASONIC 1
#if HAS_ULTRASONIC
  const int TRIG = 27;
  const int ECHO = 26;
  const unsigned long DIST_EVERY_MS = 200;
  const unsigned long ECHO_TIMEOUT  = 25000;   // us, ~4 m
  unsigned long lastDist = 0;
#endif

unsigned long lastCmd = 0;
bool moving = false;

// ---------- PWM ----------
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

// vy forward, vx strafe right, w rotate CW — each -1..1
// vx terms flipped from the textbook mix: this chassis's rollers sit in the
// mirrored X pattern. Verified live in mode1_simple.ino.
void drive(float vy, float vx, float w, int speed) {
  float fl = vy - vx + w;
  float fr = vy + vx - w;
  float rl = vy + vx + w;
  float rr = vy - vx - w;

  float m = max(max(fabs(fl), fabs(fr)), max(fabs(rl), fabs(rr)));
  if (m > 1.0) { fl /= m; fr /= m; rl /= m; rr /= m; }

  setMotor(FL, (int)(fl * speed));
  setMotor(FR, (int)(fr * speed));
  setMotor(RL, (int)(rl * speed));
  setMotor(RR, (int)(rr * speed));
  moving = true;
}

// ---------- command parsing ----------
void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  lastCmd = millis();

  String cmd = line;
  int spd = 150;                       // motors.py default
  int colon = line.indexOf(':');
  if (colon > 0) {
    cmd = line.substring(0, colon);
    spd = line.substring(colon + 1).toInt();
  }
  spd = constrain(spd, 0, MAX_SPEED);

  if      (cmd == "FORWARD")  drive( 1,  0,  0, spd);
  else if (cmd == "BACK")     drive(-1,  0,  0, spd);
  else if (cmd == "RIGHT")    drive( 0,  1,  0, spd);   // strafe right
  else if (cmd == "LEFT")     drive( 0, -1,  0, spd);   // strafe left
  else if (cmd == "ROTATE_R") drive( 0,  0,  1, spd);
  else if (cmd == "ROTATE_L") drive( 0,  0, -1, spd);
  else if (cmd == "FWD_R")    drive( 1,  1,  0, spd);   // diagonal
  else if (cmd == "FWD_L")    drive( 1, -1,  0, spd);
  else if (cmd == "BACK_R")   drive(-1,  1,  0, spd);
  else if (cmd == "BACK_L")   drive(-1, -1,  0, spd);
  else if (cmd == "STOP")     stopAll();
  else if (cmd == "PING")     Serial.println("ALIVE");
  else if (cmd == "SERVO")    { /* no servo fitted */ }
}

void setup() {
  Serial.begin(BAUD);

  for (int i = 0; i < 4; i++) {
    pwmInit(PINS[i][0], i * 2);
    pwmInit(PINS[i][1], i * 2 + 1);
  }
  stopAll();

#if HAS_ULTRASONIC
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  digitalWrite(TRIG, LOW);
#endif

  delay(300);
  Serial.println("READY");
}

void loop() {
  // serial commands
  static String buf = "";
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleLine(buf); buf = ""; }
    else if (c != '\r' && buf.length() < 64) buf += c;
  }

  // failsafe
  if (moving && millis() - lastCmd > WATCHDOG_MS) stopAll();

#if HAS_ULTRASONIC
  if (millis() - lastDist > DIST_EVERY_MS) {
    lastDist = millis();
    digitalWrite(TRIG, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG, LOW);
    unsigned long us = pulseIn(ECHO, HIGH, ECHO_TIMEOUT);
    // -1 = no echo, so the host can tell "nothing in range" from "very close"
    Serial.printf("DIST:%d\n", us == 0 ? -1 : (int)(us * 0.0343 / 2.0));
  }
#endif
}
