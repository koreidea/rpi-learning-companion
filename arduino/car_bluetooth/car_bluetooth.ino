/*
 * RPi Learning Companion — Car Chassis Controller
 * Arduino Uno + HC-05 Bluetooth + L298N Motor Driver
 *
 * BOT IS MOUNTED FACING BACKWARD on the chassis.
 * Only FORWARD/BACKWARD are reversed in software.
 * LEFT/RIGHT stay the same — left is still left.
 *
 * Protocol (from Pi or phone app):
 *   'F' = Forward    'B' = Backward
 *   'L' = Spin Left  'R' = Spin Right
 *   'G' = Fwd-Left   'H' = Fwd-Right
 *   'I' = Bwd-Left   'J' = Bwd-Right
 *   'S' = Stop
 *   '0'-'9' = Speed (0=stop, 9=max)
 *
 * HC-05 on SoftwareSerial pins 10 (RX), 11 (TX)
 * L298N: ENA=5, IN1=7, IN2=8, ENB=6, IN3=4, IN4=2
 */

#include <SoftwareSerial.h>

SoftwareSerial BTSerial(10, 11);  // RX=10, TX=11

// ── Motor Driver Pins (L298N) — original wiring, no changes ──
#define ENA  5    // Left motors PWM
#define IN1  7    // Left motor dir A
#define IN2  8    // Left motor dir B
#define ENB  6    // Right motors PWM
#define IN3  4    // Right motor dir A
#define IN4  2    // Right motor dir B

// ── Speed ──
int currentSpeed = 200;  // Default PWM 0-255

void setup() {
  BTSerial.begin(9600);

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopAll();
}

void loop() {
  if (BTSerial.available()) {
    char cmd = BTSerial.read();

    // Speed commands: '0' to '9'
    if (cmd >= '0' && cmd <= '9') {
      currentSpeed = map(cmd - '0', 0, 9, 0, 255);
      return;
    }

    switch (cmd) {
      case 'F': goForward();     break;
      case 'B': goBackward();    break;
      case 'L': spinLeft();      break;
      case 'R': spinRight();     break;
      case 'G': forwardLeft();   break;
      case 'H': forwardRight();  break;
      case 'I': backwardLeft();  break;
      case 'J': backwardRight(); break;
      case 'S': stopAll();       break;
    }
  }
}

/*
 * Only F/B are reversed (physical backward = bot forward).
 * L/R/diagonals keep original turn direction.
 *
 * Original motor directions:
 *   IN1=HIGH, IN2=LOW  = left wheels physical forward
 *   IN1=LOW,  IN2=HIGH = left wheels physical backward
 *   IN3=HIGH, IN4=LOW  = right wheels physical forward
 *   IN3=LOW,  IN4=HIGH = right wheels physical backward
 */

void goForward() {
  // REVERSED: physical backward = bot moves toward its face
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
  analogWrite(ENA, currentSpeed);
  analogWrite(ENB, currentSpeed);
}

void goBackward() {
  // REVERSED: physical forward = bot moves away from its face
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  analogWrite(ENA, currentSpeed);
  analogWrite(ENB, currentSpeed);
}

void spinLeft() {
  // NOT reversed — left is still left
  // Left wheels backward, right wheels forward
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  analogWrite(ENA, currentSpeed);
  analogWrite(ENB, currentSpeed);
}

void spinRight() {
  // NOT reversed — right is still right
  // Left wheels forward, right wheels backward
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
  analogWrite(ENA, currentSpeed);
  analogWrite(ENB, currentSpeed);
}

void forwardLeft() {
  // Bot forward (physical backward) + curve left
  // Right wheels stop, left wheels physical backward
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW);
  analogWrite(ENA, currentSpeed);
  analogWrite(ENB, 0);
}

void forwardRight() {
  // Bot forward (physical backward) + curve right
  // Left wheels stop, right wheels physical backward
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  analogWrite(ENA, 0);
  analogWrite(ENB, currentSpeed);
}

void backwardLeft() {
  // Bot backward (physical forward) + curve left
  // Right wheels stop, left wheels physical forward
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW);
  analogWrite(ENA, currentSpeed);
  analogWrite(ENB, 0);
}

void backwardRight() {
  // Bot backward (physical forward) + curve right
  // Left wheels stop, right wheels physical forward
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  analogWrite(ENA, 0);
  analogWrite(ENB, currentSpeed);
}

void stopAll() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
}
