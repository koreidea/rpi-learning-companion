# Learning Companion Bot — Project Context

## Overview
A Raspberry Pi 5 based learning companion robot with a face display, voice interaction, servo hands, touch input, and a Bluetooth-controlled car attachment.

---

## Hardware Components

### Main Bot
| Component | Model | Interface |
|---|---|---|
| Single Board Computer | Raspberry Pi 5 | — |
| Display | 3.5" ILI9488 TFT LCD (480x320) | SPI |
| Touch Screen | XPT2046 (on display board) | SPI CE1 |
| Microphones | 2x INMP441 (stereo) | I2S |
| Speaker Amplifiers | 2x MAX98357A | I2S |
| Servo Motors | 2x SG90 (left & right hands) | Hardware PWM |
| Capacitive Touch | TTP223 | GPIO |
| Power | Power bank | USB-C |
| Shutdown | Limit switch | GPIO (gpio-shutdown removed) |

### Car Attachment
| Component | Model | Interface |
|---|---|---|
| Microcontroller | Arduino Uno | — |
| Motor Driver | L298N | PWM + Digital |
| Bluetooth Module | HC-05 | SoftwareSerial (pins 10, 11) |
| Motors | 4x DC motors | L298N |

---

## Pin Connections

### ILI9488 Display (SPI)
| Display Pin | RPi Pin | BCM |
|---|---|---|
| VCC | Pin 1 | 3.3V |
| GND | Pin 25 | GND |
| CS | Pin 24 | BCM8 (CE0) |
| RESET | Pin 22 | BCM25 |
| DC | Pin 16 | BCM23 |
| SDI (MOSI) | Pin 19 | BCM10 |
| SCK | Pin 23 | BCM11 |
| LED | Pin 1 | 3.3V |
| SDO (MISO) | Not connected | — |

### XPT2046 Touch (SPI)
| Touch Pin | RPi Pin | BCM |
|---|---|---|
| VCC | Pin 1 | 3.3V |
| GND | Pin 25 | GND |
| CS | Pin 26 | BCM7 (CE1) |
| IRQ (PEN) | Pin 18 | BCM24 |
| MOSI | Pin 19 | BCM10 (shared) |
| MISO | Pin 21 | BCM9 (shared) |
| SCK | Pin 23 | BCM11 (shared) |

### INMP441 Microphones (I2S) — Stereo
| Mic Pin | RPi Pin | BCM | Notes |
|---|---|---|---|
| VDD | Pin 1 | 3.3V | Both mics |
| GND | Pin 6 | GND | Both mics |
| SD | Pin 38 | BCM20 | Both mics (shared data) |
| WS | Pin 35 | BCM19 | Both mics |
| SCK | Pin 40 | BCM21 | Both mics |
| L/R | GND | — | Mic 1 = Left |
| L/R | 3.3V | — | Mic 2 = Right |

### MAX98357A Speakers (I2S)
| Speaker Pin | RPi Pin | BCM |
|---|---|---|
| VIN | Pin 4 | 5V |
| GND | Pin 6 | GND |
| DIN | Pin 40 | BCM21 (shared) |
| BCLK | Pin 12 | BCM18 |
| LRC | Pin 35 | BCM19 (shared) |

### SG90 Servo Motors (Hardware PWM)
| Servo | RPi Pin | BCM |
|---|---|---|
| Left Servo Signal | Pin 32 | BCM12 (PWM0) |
| Right Servo Signal | Pin 33 | BCM13 (PWM1) |
| VCC (both) | Pin 2 | 5V |
| GND (both) | Pin 6 | GND |

### TTP223 Capacitive Touch
| Pin | RPi Pin | BCM |
|---|---|---|
| VCC | Pin 1 | 3.3V |
| GND | Pin 6 | GND |
| OUT | Pin 11 | BCM17 |

### Limit Switch (Shutdown)
| Pin | RPi Pin | BCM |
|---|---|---|
| One terminal | Pin 7 | BCM4 |
| Other terminal | Pin 6 | GND |

> Note: gpio-shutdown was removed. Pi 5 does not support GPIO wake from halt.

---

## Software Stack

### OS & Libraries
- Raspberry Pi OS (64-bit)
- Python 3
- `spidev` — SPI communication
- `gpiod` — GPIO control (RPi.GPIO not supported on Pi 5)
- `lgpio` — Hardware PWM for servos
- `loguru` — Logging

### Key Files
```
rpi/
├── core/
│   └── main.py              # Main bot logic, speech pipeline
├── display/
│   └── tft_display.py       # ILI9488 driver (480x320, 18-bit RGB666)
├── modules/
│   └── servos.py            # SG90 servo controller with speaking gestures
└── ...
```

---

## Key Implementation Details

### ILI9488 Display
- **Resolution**: 480x320
- **Color format**: 18-bit RGB666 (3 bytes per pixel)
- **MADCTL**: `0x28` (180° rotation, BGR mode)
- **SPI**: BCM8 (CE0), up to 40MHz
- **Eye color**: Dark orange `#CC5500`

### Servo Controller (`modules/servos.py`)
```python
GPIO_CHIP = 4
PIN_LEFT  = 12   # BCM12 — PWM0
PIN_RIGHT = 13   # BCM13 — PWM1
PWM_FREQ  = 50   # Hz
DUTY_MIN  = 2.5  # 0°
DUTY_MAX  = 7.5  # 90°
MAX_ANGLE = 90
```
- Left servo is **mirror-mounted**: uses `MAX_ANGLE - angle` for duty cycle
- Rest position: left = MAX_ANGLE (90°), right = 0°
- Speaking gestures: 6 patterns with ease-in-out sine transitions
- `start_speaking_gestures()` / `stop_speaking_gestures()` hooked into `main.py`

### Speaking Gestures (in `main.py`)
```python
# In _stream_response():
self._audio_capture.pause()
self._servos.start_speaking_gestures()
# ... speech pipeline ...
self._servos.stop_speaking_gestures()

# In _speak_text():
self._servos.start_speaking_gestures()
audio_data = await self._tts.synthesize(text)
if not self.state.interrupt_event.is_set():
    await self._audio_player.play(audio_data)
self._servos.stop_speaking_gestures()
```

### XPT2046 Touch
- SPI CE1 (BCM7)
- PEN IRQ pin: BCM24 with `Bias.PULL_UP` (active low)

---

## Car Attachment

### Arduino Pin Connections (L298N)
| Arduino Pin | L298N Pin |
|---|---|
| D5 (PWM) | ENA |
| D6 (PWM) | ENB |
| D7 | IN1 |
| D8 | IN2 |
| D4 | IN3 |
| D2 | IN4 |
| D10 | HC-05 TX |
| D11 | HC-05 RX |

### HC-05 Bluetooth Module
| HC-05 Pin | Arduino Pin |
|---|---|
| TX | D10 |
| RX | D11 |
| VCC | 5V |
| GND | GND |

### Arduino Sketch — Command Protocol
| Command | Action |
|---|---|
| `F` | Both forward |
| `B` | Both backward |
| `L` | Spin left |
| `R` | Spin right |
| `G` | Forward-Left (right wheels only) |
| `H` | Forward-Right (left wheels only) |
| `I` | Backward-Left (right wheels only) |
| `J` | Backward-Right (left wheels only) |
| `S` | Stop |
| `0`-`9` | Speed (maps to PWM 0–255) |

### Speed Mapping
| Slider | PWM |
|---|---|
| 0 | 0 |
| 1 | 28 |
| 2 | 56 |
| 3 | 84 |
| 4 | 112 |
| 5 | 140 |
| 6 | 168 |
| 7 | 196 |
| 8 | 224 |
| 9 | 255 |

### Mobile App
- **App**: Arduino Bluetooth Control (Android)
- **Connection**: Phone → HC-05 Bluetooth Classic
- Buttons send: F, B, L, R, S (stop on release)
- Diagonal buttons send: G, H, I, J
- Slider (0–9) sends speed character

### Arduino Sketch (Full)
```cpp
#include <SoftwareSerial.h>
SoftwareSerial BT(10, 11);

#define ENA 5
#define ENB 6
#define IN1 7
#define IN2 8
#define IN3 4
#define IN4 2

int speed = 255;

void setup() {
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  Serial.begin(9600);
  BT.begin(9600);
  stopMotors();
}

void loop() {
  if (BT.available()) {
    char c = BT.read();
    switch (c) {
      case 'F': forward();       break;
      case 'B': backward();      break;
      case 'L': spinLeft();      break;
      case 'R': spinRight();     break;
      case 'G': forwardLeft();   break;
      case 'H': forwardRight();  break;
      case 'I': backwardLeft();  break;
      case 'J': backwardRight(); break;
      case 'S': stopMotors();    break;
      case '0': speed = 0;   break;
      case '1': speed = 28;  break;
      case '2': speed = 56;  break;
      case '3': speed = 84;  break;
      case '4': speed = 112; break;
      case '5': speed = 140; break;
      case '6': speed = 168; break;
      case '7': speed = 196; break;
      case '8': speed = 224; break;
      case '9': speed = 255; break;
    }
  }
}

void forward() {
  analogWrite(ENA, speed); analogWrite(ENB, speed);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}
void backward() {
  analogWrite(ENA, speed); analogWrite(ENB, speed);
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
}
void spinLeft() {
  analogWrite(ENA, speed); analogWrite(ENB, speed);
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}
void spinRight() {
  analogWrite(ENA, speed); analogWrite(ENB, speed);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
}
void forwardLeft() {
  analogWrite(ENA, 0);     analogWrite(ENB, speed);
  digitalWrite(IN1, LOW);  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}
void forwardRight() {
  analogWrite(ENA, speed); analogWrite(ENB, 0);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW);
}
void backwardLeft() {
  analogWrite(ENA, 0);     analogWrite(ENB, speed);
  digitalWrite(IN1, LOW);  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
}
void backwardRight() {
  analogWrite(ENA, speed); analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW);
}
void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  analogWrite(ENA, 0);    analogWrite(ENB, 0);
}
```

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| RPi.GPIO not compatible with Pi 5 | Use `gpiod` library instead |
| gpiod "Device or resource busy" | Stop `rpi-bot` service before testing |
| ILI9488 white screen | Check DC pin (BCM23), verify 3.3V on LED |
| Left servo opposite direction | Use `MAX_ANGLE - angle` for left servo duty |
| XPT2046 PEN IRQ floating/toggling | Add `Bias.PULL_UP` in gpiod LineSettings |
| Toggle switch kept triggering shutdown | Use momentary/limit switch instead |
| Pi 5 GPIO wake from halt not supported | Removed gpio-shutdown entirely |
| lgpio "GPIO not set as output" | Call `lgpio.gpio_claim_output()` before `tx_pwm()` |
| Display MADCTL mirrored text | Use `0x28` instead of `0xA8` |

---

## Bot ↔ Car Integration

### How it works
- At boot, Pi automatically tries to connect to car (uses saved MAC or scans for HC-05)
- If not found at boot, user connects manually from **Settings → Car**
- Connection status shows in the menu: `Disconnected` / `Connecting...` / `Connected`
- Once connected, child can control car by **voice commands**

### Settings Menu — Car Item
| State | Display | Action on Select |
|---|---|---|
| Not connected | `Disconnected` | Starts scanning & connecting |
| Connecting | `Connecting...` | Shows "Still connecting, please wait" |
| Connected | `Connected` | Disconnects the car |

### Updated Settings Menu Order (7 items)
| Index | Item | Action |
|---|---|---|
| 0 | Volume | Cycle 20→40→60→80→100→20% |
| 1 | Mode | Toggle Offline/Online |
| 2 | Microphone | Toggle On/Off |
| 3 | Projector | Toggle On/Off |
| 4 | Flashcards | Launch flashcard mode |
| 5 | **Car** | **Connect/Disconnect HC-05** |
| 6 | Sleep | Disable mic |

### Files Changed for Integration
| File | Change |
|---|---|
| `core/state.py` | Added `car_connected`, `car_connecting`, `car_mac` fields |
| `modules/car.py` | Updated to ASCII text protocol matching Arduino sketch |
| `display/tft_display.py` | Added Car item, adjusted item height to 37px for 7 items |
| `core/main.py` | Menu count 6→7, `_connect_car_module` syncs state, added `_disconnect_car_module`, Car case in `_handle_menu_select_safe` |

### Car Command Protocol (Pi → Arduino via Bluetooth)
| Char | Action |
|---|---|
| `F` | Forward |
| `B` | Backward |
| `L` | Spin Left |
| `R` | Spin Right |
| `G` | Forward-Left (right wheels only) |
| `H` | Forward-Right (left wheels only) |
| `I` | Backward-Left (right wheels only) |
| `J` | Backward-Right (left wheels only) |
| `S` | Stop |
| `0`-`9` | Speed (sent before direction char) |

Speed mapping in car.py: `speed_char = str(round(speed * 9 / 255))`

---

## Pending / Future Features
- [x] Voice commands for car control (e.g., "move forward", "turn left")
- [x] Pi-side Bluetooth code to send motor commands to HC-05
- [x] Settings menu Car connect/disconnect
- [ ] "Follow the kid" mode using dual INMP441 mics for sound direction detection
- [ ] Speed control via voice
