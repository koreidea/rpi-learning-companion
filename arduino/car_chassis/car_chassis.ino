/*
 * Car Chassis Motor Controller — Bluetooth SPP (ASCII Protocol)
 *
 * Arduino Uno receives ASCII commands from Raspberry Pi via HC-05 Bluetooth
 * and drives 4 DC motors through 2× L298N motor drivers.
 *
 * Protocol (from "Arduino Bluetooth Controller" app + Pi):
 *   Direction: 'F'=Forward 'B'=Backward 'L'=SpinLeft 'R'=SpinRight
 *              'G'=ForwardLeft 'H'=ForwardRight 'I'=BackwardLeft 'J'=BackwardRight
 *              'S'=Stop
 *   Speed:     '0'-'9' → mapped to PWM 0-255
 *
 * NOTE: The Pi bot is mounted FACING BACKWARD on the chassis.
 *       Direction reversal is handled on the Pi side (car.py).
 *       This sketch uses PHYSICAL car directions only.
 *
 * Wiring:
 *   L298N #1 (Front):  ENA=D5, IN1=D7, IN2=D8, ENB=D6, IN3=D4, IN4=D2
 *   L298N #2 (Rear):   ENA=D3, IN1=D9, IN2=D10, ENB=D11, IN3=D12, IN4=D13
 *   HC-05:             TX→Arduino RX (via SoftSerial D10,D11 or HW Serial)
 *
 * Motor layout (top view, physical car front at top):
 *   FL [Motor1]  ----  [Motor2] FR
 *        |                |
 *   RL [Motor3]  ----  [Motor4] RR
 */

// ─── L298N #1 — Front motors ────────────────────────────────────
#define FL_ENA  5    // Front-Left speed (PWM)
#define FL_IN1  7    // Front-Left direction
#define FL_IN2  8

#define FR_ENB  6    // Front-Right speed (PWM)
#define FR_IN1  4    // Front-Right direction
#define FR_IN2  2

// ─── L298N #2 — Rear motors ─────────────────────────────────────
#define RL_ENA  3    // Rear-Left speed (PWM)
#define RL_IN1  9    // Rear-Left direction
#define RL_IN2  10

#define RR_ENB  11   // Rear-Right speed (PWM)
#define RR_IN1  12   // Rear-Right direction
#define RR_IN2  13

// ─── Safety: auto-stop if no command received ────────────────────
#define TIMEOUT_MS  2000
unsigned long lastCmdTime = 0;

// Current speed (set by '0'-'9' commands)
int currentSpeed = 0;

// Current direction command
char currentDir = 'S';

// ─── Motor control helpers ───────────────────────────────────────

void setMotor(int in1, int in2, int en, int speed, bool forward) {
    if (speed == 0) {
        digitalWrite(in1, LOW);
        digitalWrite(in2, LOW);
        analogWrite(en, 0);
    } else if (forward) {
        digitalWrite(in1, HIGH);
        digitalWrite(in2, LOW);
        analogWrite(en, speed);
    } else {
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        analogWrite(en, speed);
    }
}

void setLeftMotors(int speed, bool forward) {
    setMotor(FL_IN1, FL_IN2, FL_ENA, speed, forward);
    setMotor(RL_IN1, RL_IN2, RL_ENA, speed, forward);
}

void setRightMotors(int speed, bool forward) {
    setMotor(FR_IN1, FR_IN2, FR_ENB, speed, forward);
    setMotor(RR_IN1, RR_IN2, RR_ENB, speed, forward);
}

void stopAll() {
    setLeftMotors(0, true);
    setRightMotors(0, true);
}

// ─── Apply direction with current speed ──────────────────────────

void applyDirection(char dir, int speed) {
    switch (dir) {
        case 'F':  // Physical forward
            setLeftMotors(speed, true);
            setRightMotors(speed, true);
            break;

        case 'B':  // Physical backward
            setLeftMotors(speed, false);
            setRightMotors(speed, false);
            break;

        case 'L':  // Physical spin left (left backward, right forward)
            setLeftMotors(speed, false);
            setRightMotors(speed, true);
            break;

        case 'R':  // Physical spin right (left forward, right backward)
            setLeftMotors(speed, true);
            setRightMotors(speed, false);
            break;

        case 'G':  // Forward-left (right wheels forward, left stopped)
            setLeftMotors(0, true);
            setRightMotors(speed, true);
            break;

        case 'H':  // Forward-right (left wheels forward, right stopped)
            setLeftMotors(speed, true);
            setRightMotors(0, true);
            break;

        case 'I':  // Backward-left (right wheels backward, left stopped)
            setLeftMotors(0, true);
            setRightMotors(speed, false);
            break;

        case 'J':  // Backward-right (left wheels backward, right stopped)
            setLeftMotors(speed, false);
            setRightMotors(0, true);
            break;

        case 'S':  // Stop
            stopAll();
            break;

        default:
            stopAll();
            break;
    }
}

// ─── Setup ──────────────────────────────────────────────────────

void setup() {
    Serial.begin(9600);

    // L298N #1 pins
    pinMode(FL_IN1, OUTPUT);
    pinMode(FL_IN2, OUTPUT);
    pinMode(FL_ENA, OUTPUT);
    pinMode(FR_IN1, OUTPUT);
    pinMode(FR_IN2, OUTPUT);
    pinMode(FR_ENB, OUTPUT);

    // L298N #2 pins
    pinMode(RL_IN1, OUTPUT);
    pinMode(RL_IN2, OUTPUT);
    pinMode(RL_ENA, OUTPUT);
    pinMode(RR_IN1, OUTPUT);
    pinMode(RR_IN2, OUTPUT);
    pinMode(RR_ENB, OUTPUT);

    stopAll();
    lastCmdTime = millis();
}

// ─── Main loop ──────────────────────────────────────────────────

void loop() {
    while (Serial.available() > 0) {
        char c = Serial.read();
        lastCmdTime = millis();

        // Speed digit: '0'-'9' → PWM 0-255
        if (c >= '0' && c <= '9') {
            currentSpeed = map(c - '0', 0, 9, 0, 255);
            // Re-apply current direction at new speed
            if (currentDir != 'S') {
                applyDirection(currentDir, currentSpeed);
            }
        }
        // Direction command
        else if (c == 'F' || c == 'B' || c == 'L' || c == 'R' ||
                 c == 'G' || c == 'H' || c == 'I' || c == 'J' || c == 'S') {
            currentDir = c;
            applyDirection(currentDir, currentSpeed);
        }
        // Ignore other characters (newlines, etc.)
    }

    // Safety timeout
    if (millis() - lastCmdTime > TIMEOUT_MS) {
        stopAll();
        currentDir = 'S';
    }
}
