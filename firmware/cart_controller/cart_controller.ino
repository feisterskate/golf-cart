/*
 * Follow-me golf cart -- ESP32 motor controller
 *
 * Reads target position over USB serial from tracker.py, runs a proportional
 * follow controller, and drives two ZS-X11H BLDC drivers.
 *
 * SERIAL PROTOCOL (115200 baud, newline terminated)
 *   T,<distance_m>,<bearing_deg>   target seen
 *   S                              stop now
 *
 * Bearing sign: positive = target is to the RIGHT of centre.
 *
 * SAFETY: if no valid packet arrives for DEADMAN_MS, motors go to zero.
 * This is the single most important behaviour in this file. Do not disable it
 * while testing, and always bench-test with the wheels off the ground.
 */

// ---------------------------------------------------------------- pins
// GPIO25 and GPIO26 are the ESP32's only true DAC pins -- real analog out,
// 0-3.3V, no RC filter needed. The ZS-X11H speed input accepts 0-5V analog,
// so 3.3V max gives us roughly 66% of top speed. That is a feature here:
// a hardware speed cap you cannot accidentally code your way past.
const int PIN_L_SPEED = 25;   // DAC1 -> left driver speed input
const int PIN_R_SPEED = 26;   // DAC2 -> right driver speed input

const int PIN_L_DIR   = 32;   // left driver direction (ZF)
const int PIN_R_DIR   = 33;   // right driver direction
const int PIN_L_BRAKE = 27;   // left driver brake (EL) -- active per your board
const int PIN_R_BRAKE = 14;   // right driver brake

const int PIN_ESTOP   = 34;   // input-only pin. Switch to GND = run, open = stop.
                              // GPIO34 has NO internal pullup, so fit a real
                              // 10k pullup resistor to 3.3V on this line.

// ---------------------------------------------------------- tuning knobs
const float FOLLOW_DIST_M   = 2.5;   // how far behind you the cart holds
const float DIST_DEADBAND_M = 0.25;  // don't twitch inside this window
const float BEAR_DEADBAND_D = 4.0;   // ditto for steering

const float KP_DIST = 0.55;          // distance error -> forward speed
const float KP_BEAR = 0.020;         // bearing error  -> turn rate

const float V_MAX      = 0.55;       // 0..1 fraction of driver full scale
const float TURN_MAX   = 0.35;
const float SLEW_PER_S = 1.2;        // max change in output per second

const unsigned long DEADMAN_MS  = 300;
const unsigned long LOOP_MS     = 20;   // 50 Hz

// ------------------------------------------------------------- state
float g_dist = 0, g_bear = 0;
bool  g_have_target = false;
unsigned long g_last_packet = 0;

float g_out_l = 0, g_out_r = 0;   // current commanded, after slew limiting

char  g_buf[64];
int   g_len = 0;

// --------------------------------------------------------------- utils
float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Move `cur` toward `want`, no faster than SLEW_PER_S. Stops the cart from
// lurching when the marker is momentarily lost and reacquired.
float slew(float cur, float want, float dt) {
  float maxStep = SLEW_PER_S * dt;
  float d = want - cur;
  if (d >  maxStep) d =  maxStep;
  if (d < -maxStep) d = -maxStep;
  return cur + d;
}

// Send one wheel's command out. `cmd` is -1..+1.
void driveWheel(int pinSpeed, int pinDir, int pinBrake, float cmd) {
  bool reverse = cmd < 0;
  float mag = fabs(cmd);
  if (mag < 0.03) mag = 0;                 // kill driver buzz near zero

  digitalWrite(pinDir, reverse ? HIGH : LOW);
  digitalWrite(pinBrake, mag == 0 ? HIGH : LOW);   // brake engaged at rest

  dacWrite(pinSpeed, (int)(clampf(mag, 0, 1) * 255));
}

void allStop() {
  g_out_l = g_out_r = 0;
  driveWheel(PIN_L_SPEED, PIN_L_DIR, PIN_L_BRAKE, 0);
  driveWheel(PIN_R_SPEED, PIN_R_DIR, PIN_R_BRAKE, 0);
}

// --------------------------------------------------------- serial parse
void handleLine(char *s) {
  if (s[0] == 'S') {
    g_have_target = false;
    g_last_packet = millis();
    return;
  }
  if (s[0] != 'T') return;

  // T,<dist>,<bear>
  char *c1 = strchr(s, ',');
  if (!c1) return;
  char *c2 = strchr(c1 + 1, ',');
  if (!c2) return;

  float d = atof(c1 + 1);
  float b = atof(c2 + 1);

  // Reject nonsense rather than acting on it.
  if (d < 0.3 || d > 15.0) return;
  if (b < -70.0 || b > 70.0) return;

  g_dist = d;
  g_bear = b;
  g_have_target = true;
  g_last_packet = millis();
}

void pollSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (g_len > 0) { g_buf[g_len] = 0; handleLine(g_buf); g_len = 0; }
    } else if (g_len < (int)sizeof(g_buf) - 1) {
      g_buf[g_len++] = c;
    } else {
      g_len = 0;              // overlong garbage, resync
    }
  }
}

// ---------------------------------------------------------- controller
void computeCommand(float dt) {
  bool estop_ok  = (digitalRead(PIN_ESTOP) == LOW);
  bool fresh     = (millis() - g_last_packet) < DEADMAN_MS;

  if (!estop_ok || !fresh || !g_have_target) {
    g_out_l = slew(g_out_l, 0, dt);
    g_out_r = slew(g_out_r, 0, dt);
    return;
  }

  // forward speed from distance error
  float derr = g_dist - FOLLOW_DIST_M;
  if (fabs(derr) < DIST_DEADBAND_M) derr = 0;
  float v = clampf(KP_DIST * derr, -V_MAX * 0.4, V_MAX);
  // note the asymmetric clamp: reverse is deliberately much slower than forward

  // turn rate from bearing error
  float berr = g_bear;
  if (fabs(berr) < BEAR_DEADBAND_D) berr = 0;
  float w = clampf(KP_BEAR * berr, -TURN_MAX, TURN_MAX);

  // If we're basically at the right distance, still allow turning in place
  // so the cart squares up to you instead of drifting sideways.
  float want_l = clampf(v + w, -1, 1);
  float want_r = clampf(v - w, -1, 1);

  g_out_l = slew(g_out_l, want_l, dt);
  g_out_r = slew(g_out_r, want_r, dt);
}

// ------------------------------------------------------------ arduino
void setup() {
  Serial.begin(115200);

  pinMode(PIN_L_DIR, OUTPUT);
  pinMode(PIN_R_DIR, OUTPUT);
  pinMode(PIN_L_BRAKE, OUTPUT);
  pinMode(PIN_R_BRAKE, OUTPUT);
  pinMode(PIN_ESTOP, INPUT);      // external 10k pullup required on GPIO34

  allStop();
  delay(200);
  Serial.println("# cart controller ready");
}

void loop() {
  static unsigned long last = 0;
  pollSerial();

  unsigned long now = millis();
  if (now - last < LOOP_MS) return;
  float dt = (now - last) / 1000.0;
  last = now;

  computeCommand(dt);

  driveWheel(PIN_L_SPEED, PIN_L_DIR, PIN_L_BRAKE, g_out_l);
  driveWheel(PIN_R_SPEED, PIN_R_DIR, PIN_R_BRAKE, g_out_r);

  // telemetry at ~5 Hz, readable in the Serial Monitor
  static int n = 0;
  if (++n >= 10) {
    n = 0;
    Serial.printf("# d=%.2f b=%+.1f L=%+.2f R=%+.2f %s\n",
                  g_dist, g_bear, g_out_l, g_out_r,
                  g_have_target ? "TRK" : "---");
  }
}
