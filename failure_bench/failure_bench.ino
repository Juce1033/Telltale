/*
  Failure Bench — ESP32 firmware

  Samples INA226 (current) + MPU6050 (vibration) at a fixed 500 Hz and streams
  one line per sample over USB serial at 921600 baud:

    timestamp_ms,current_a,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z

  Units: amps, g, degrees/second. Lines starting with '#' are status messages
  (the Python scripts skip them).

  Motor control over the same serial port (send a line, e.g. from Serial Monitor):
    S60   -> run forward at 60% duty
    S0    -> stop
  Motor starts STOPPED on boot.

  Wiring (from the electrical brief):
    I2C:      SDA = GPIO 21, SCL = GPIO 22   (INA226 @ 0x40, MPU6050 @ 0x68)
    BTS7960:  RPWM = GPIO 25, LPWM = GPIO 26 (held low), R_EN/L_EN tied to 3.3V

  No external libraries needed. Board: "ESP32 Dev Module".
*/
#include <Wire.h>

// ---------- config ----------
const uint32_t SAMPLE_US  = 2000;      // 2000 us = 500 Hz. Must match FS in features.py
const uint32_t BAUD       = 921600;
const int      PIN_RPWM   = 25;
const int      PIN_LPWM   = 26;
const uint32_t PWM_FREQ   = 20000;     // 20 kHz: above hearing, no whine in the vibration data
const int      PWM_BITS   = 8;         // duty 0..255
const float    SHUNT_OHMS = 0.1;       // "R100" resistor on most INA226 modules; check yours

const uint8_t INA = 0x40;
const uint8_t MPU = 0x68;

// ---------- I2C helpers ----------
bool writeReg8(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool writeReg16(uint8_t addr, uint8_t reg, uint16_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val >> 8);
  Wire.write(val & 0xFF);
  return Wire.endTransmission() == 0;
}

bool readBytes(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

bool present(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

// ---------- motor ----------
void setMotor(int pct) {
  pct = constrain(pct, 0, 100);
  int duty = map(pct, 0, 100, 0, (1 << PWM_BITS) - 1);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PIN_RPWM, duty);
#else
  ledcWrite(0, duty);
#endif
  Serial.printf("# motor %d%%\n", pct);
}

void handleSerialCommands() {
  static char cmd[16];
  static uint8_t len = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      cmd[len] = 0;
      if (len > 1 && (cmd[0] == 'S' || cmd[0] == 's')) setMotor(atoi(cmd + 1));
      len = 0;
    } else if (len < sizeof(cmd) - 1) {
      cmd[len++] = c;
    }
  }
}

// ---------- setup ----------
void setup() {
  Serial.begin(BAUD);
  delay(300);

  // motor PWM, starts stopped
  pinMode(PIN_LPWM, OUTPUT);
  digitalWrite(PIN_LPWM, LOW);             // reverse never used
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PIN_RPWM, PWM_FREQ, PWM_BITS);
#else
  ledcSetup(0, PWM_FREQ, PWM_BITS);
  ledcAttachPin(PIN_RPWM, 0);
#endif
  setMotor(0);

  Wire.begin(21, 22);
  Wire.setClock(400000);                   // fast I2C so both reads fit in 2 ms

  bool inaOk = present(INA), mpuOk = present(MPU);
  Serial.printf("# INA226 0x40: %s\n", inaOk ? "found" : "MISSING - check wiring/ground");
  Serial.printf("# MPU6050 0x68: %s\n", mpuOk ? "found" : "MISSING - check wiring/ground");

  // MPU6050: wake up, +-2 g, +-250 dps, 184 Hz low-pass (prevents aliasing at 500 Hz)
  writeReg8(MPU, 0x6B, 0x00);              // PWR_MGMT_1: wake
  writeReg8(MPU, 0x1A, 0x01);              // CONFIG: DLPF 184 Hz
  writeReg8(MPU, 0x19, 0x00);              // SMPLRT_DIV: 1 kHz internal rate
  writeReg8(MPU, 0x1B, 0x00);              // GYRO_CONFIG: +-250 dps
  writeReg8(MPU, 0x1C, 0x00);              // ACCEL_CONFIG: +-2 g

  // INA226: no averaging, 140 us conversions, continuous shunt+bus
  writeReg16(INA, 0x00, 0x4007);

  Serial.println("# streaming: timestamp_ms,current_a,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z");
}

// ---------- main loop: fixed-rate sampling ----------
void loop() {
  static uint32_t next = micros();
  static uint32_t t0 = next;
  static uint32_t errors = 0;

  handleSerialCommands();

  if ((int32_t)(micros() - next) < 0) return;   // not time yet
  uint32_t scheduled = next;
  next += SAMPLE_US;
  if ((int32_t)(micros() - next) > 0) next = micros() + SAMPLE_US;   // fell behind: resync

  uint8_t m[14], s[2];
  bool ok = readBytes(MPU, 0x3B, m, 14) && readBytes(INA, 0x01, s, 2);
  if (!ok) {
    if (++errors % 500 == 1) Serial.printf("# I2C read error (%lu total)\n", errors);
    return;
  }

  int16_t ax = (m[0] << 8) | m[1], ay = (m[2] << 8) | m[3], az = (m[4] << 8) | m[5];
  int16_t gx = (m[8] << 8) | m[9], gy = (m[10] << 8) | m[11], gz = (m[12] << 8) | m[13];
  int16_t shunt = (s[0] << 8) | s[1];                 // 2.5 uV per bit

  float current = shunt * 2.5e-6f / SHUNT_OHMS;
  uint32_t t_ms = (scheduled - t0) / 1000;            // scheduled time = evenly spaced stamps

  Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n",
                t_ms, current,
                ax / 16384.0f, ay / 16384.0f, az / 16384.0f,
                gx / 131.0f, gy / 131.0f, gz / 131.0f);
}
