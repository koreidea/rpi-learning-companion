# Learning Companion Bot — Project Context

## Overview
A Raspberry Pi 5 based AI learning companion robot for children (3-6) with an animated face display, voice interaction, servo hands, touch input, and a Bluetooth-controlled car attachment. Long-term vision: **Modular AI Robotics Kit** — brain module stays constant, body is reconfigurable by the child.

---

## Hardware Components

### Main Bot
| Component | Model | Interface |
|---|---|---|
| Single Board Computer | Raspberry Pi 5 (8GB) | — |
| Display | 3.5" ILI9488 TFT LCD (480x320) | SPI |
| Touch Screen | XPT2046 (on display board) | SPI CE1 |
| Microphone | INMP441 | I2S |
| Speaker Amplifiers | 2x MAX98357A (stereo) | I2S |
| Speakers | 2x 4ohm 5W | Driven by MAX98357A |
| Servo Motors | 2x SG90 (left & right hands) | Hardware PWM (BCM12, BCM13) |
| Capacitive Touch | TTP223 | GPIO (BCM17) |
| Power | Power bank | USB-C |
| Shutdown | Limit switch | GPIO (BCM4) |

### Car Attachment (Bluetooth, wireless)
| Component | Model | Interface |
|---|---|---|
| Microcontroller | Arduino Uno | — |
| Motor Driver | L298N | PWM + Digital |
| Bluetooth Module | HC-05 | SoftwareSerial (pins 10, 11) |
| Motors | 4x DC motors (4WD) | L298N |
| Power | 7.4-12V LiPo battery | L298N 12V input |

**Note:** Bot is mounted **facing backward** on car chassis — all left/right directions are swapped in software.

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

### XPT2046 Touch (SPI)
| Touch Pin | RPi Pin | BCM |
|---|---|---|
| CS | Pin 26 | BCM7 (CE1) |
| IRQ (PEN) | Pin 18 | BCM24 |
| MOSI/MISO/SCK | Shared with display | BCM10/9/11 |

### INMP441 Microphone (I2S)
| Mic Pin | RPi Pin | BCM |
|---|---|---|
| SD (DOUT) | Pin 38 | BCM20 |
| WS (LRC) | Pin 35 | BCM19 |
| SCK (BCLK) | Pin 40 | BCM21 |
| L/R | GND | Left channel |

### MAX98357A Speakers (I2S)
| Speaker Pin | RPi Pin | BCM |
|---|---|---|
| VIN | Pin 4 | 5V |
| GND | Pin 6 | GND |
| DIN | Pin 40 | BCM21 |
| BCLK | Pin 12 | BCM18 |
| LRC | Pin 35 | BCM19 |
Both SD pins → VIN (Right channel — I2S overlay sends audio on Right channel only).
GAIN pins → floating (15dB, loudest).

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
| OUT | Pin 11 | BCM17 |

### Limit Switch (Shutdown)
| Pin | RPi Pin | BCM |
|---|---|---|
| One terminal | Pin 7 | BCM4 |
| Other terminal | Pin 6 | GND |

### GPIO Budget
- **Used:** BCM4,7,8,9,10,11 (SPI+switch), BCM12,13 (servos), BCM17 (touch), BCM18,19,20,21 (I2S), BCM23,24,25 (display) = 16 pins
- **Free:** BCM0,1,2,3,5,6,14,15,16,22,26,27 = 12 pins
- **I2C (BCM2/3)** reserved for modular accessories

---

## Software Stack

### Pipeline
```
Wake Word (OpenWakeWord) → VAD (Silero) → STT (Whisper.cpp) → LLM (streaming) → Sentence Buffer → TTS (Piper) → Speaker
```

### Libraries
- Python 3 + asyncio (single process, multi-thread)
- FastAPI (API + serves React app on port 8080, HTTPS on 8443)
- OpenWakeWord — "hey jarvis" wake word (threshold 0.4)
- Silero-VAD — voice activity detection
- Whisper.cpp — tiny.en multilingual STT (77MB)
- llama-cpp-python — offline LLM (Qwen2.5-3B-Instruct-Q4_K_M, 2.0GB)
- Piper TTS — en_US-lessac-medium, hi_IN-rohan-medium, te_IN-maya-medium
- YOLOv8-nano + PaddleOCR — vision
- React + Vite + Tailwind — parent dashboard
- lgpio — hardware PWM for servos
- spidev — SPI for display
- cryptography — AES-256 encryption for data at rest

### Online Mode
When mode="online", uses ChatGPT/Gemini/Claude APIs instead of local Qwen model.

---

## Project Structure & Key Files

```
rpi/                              # Everything running ON the Pi
├── core/
│   ├── main.py                   # Main orchestrator — streaming pipeline, car commands, dance, follow
│   ├── config.py                 # ConfigManager with Pydantic models
│   └── state.py                  # SharedState (BotState enum, remote_text, car status)
├── audio/
│   ├── audio_capture.py          # Mic input (I2S INMP441)
│   ├── audio_player.py           # aplay playback, mono-to-stereo, software volume
│   ├── wake_word.py              # OpenWakeWord detector
│   ├── vad.py                    # Silero VAD
│   ├── sentence_buffer.py        # Streaming TTS sentence chunking
│   ├── songs/                    # 10 real sung nursery rhyme WAVs
│   └── songs_mp3/                # MP3 versions (gitignored)
├── display/
│   ├── tft_display.py            # ILI9488 driver, animated face (EMO/Cozmo-style)
│   ├── touch.py                  # XPT2046 touch input
│   ├── spi_lock.py               # SPI bus lock (display + touch share bus)
│   └── games/                    # Snake, Pong, Tetris, Breakout, Space Invaders
├── modules/
│   ├── car.py                    # Bluetooth car driver (HC-05 via /dev/rfcomm0)
│   ├── follow.py                 # HSV color tracking follow mode (yellow wheel)
│   ├── servos.py                 # SG90 servo controller (speaking gestures, tickle, dance)
│   └── projector.py              # HDMI projector module
├── llm/
│   ├── base.py                   # BaseLLM abstract + LLMRouter
│   └── prompts.py                # System prompts
├── vision/
│   ├── camera.py                 # Pi Camera
│   └── image_generator.py        # Image generation
├── api/
│   ├── server.py                 # FastAPI app factory
│   └── routes/
│       ├── control.py            # /api/control/send-text, /api/control/wake
│       ├── dashboard.py          # Dashboard data API
│       └── settings.py           # Settings API
├── privacy/                      # DPDP compliance
├── scripts/
│   ├── generate_flashcards.py
│   ├── generate_songs.py
│   └── download_models.sh
├── assets/
│   └── projector/flashcards/     # Animal, color, fruit, shape flashcard PNGs
└── requirements.txt

app/                              # React parent dashboard
├── src/components/
│   └── Dashboard.jsx             # Status panel, remote voice, settings
└── dist/                         # Built static files (served by Pi)

android_app/                      # Flutter companion app
├── lib/
│   ├── activities/               # 60+ learning activities across 20 skill categories
│   ├── core/                     # Orchestrator, LLM, config, state
│   ├── audio/                    # Wake word, VAD, STT, TTS, songs
│   ├── bluetooth/                # Car chassis BT control
│   ├── ui/                       # Face, home, parent dashboard, progress, setup
│   └── vision/                   # Camera, follow mode, person detection
└── pubspec.yaml

arduino/
├── car_bluetooth/                # HC-05 Bluetooth car firmware
└── car_chassis/                  # Basic car chassis firmware
```

---

## Deployment

### Pi: `hairobo@192.168.0.174`
```bash
# Deploy a file
scp <file> hairobo@192.168.0.174:/home/hairobo/rpi-bot/rpi/<path>

# Deploy React app
cd app && npm run build
scp -r app/dist/* hairobo@192.168.0.174:/home/hairobo/rpi-bot/app/dist/

# Restart service
ssh hairobo@192.168.0.174 "sudo systemctl restart rpi-bot"

# Check logs
ssh hairobo@192.168.0.174 "sudo journalctl -u rpi-bot --since '15 seconds ago' --no-pager -l"
```

- Service: `rpi-bot.service` (systemd)
- Parent dashboard: `http://raspberrypi.local:8080`
- HTTPS (for Web Speech API): `https://raspberrypi.local:8443` (self-signed cert)

---

## Car Control System

### Architecture
- Pi connects to Arduino via Bluetooth SPP (`/dev/rfcomm0`)
- ASCII protocol: single character commands (F/B/L/R/S/G/H/I/J + speed 0-9)
- Bot mounted **backward** on chassis — left/right swapped in `_parse_single_car_command()`

### Voice Commands
- "move forward for 5 seconds" → forward with duration
- "turn left for 10 seconds" → spin left 2s + forward remaining 8s
- "forward 5 seconds then left 3 seconds then right 4 seconds" → sequential execution
- "reverse" → replays last command with directions swapped and sequence reversed
- "dance" → synchronized car wiggles + servo hands + excited display
- "follow" → HSV yellow color tracking mode
- "stop" → stops everything

### Duration Parsing
- Regex matches "for 5 seconds", "5 seconds", spoken numbers ("five seconds")
- Turn commands: spin for 2s, then forward for remaining time
- Sequential commands: split on "then", flatten nested sequences

### Reverse Command
- `_REVERSE_ACTION` dict swaps forward↔backward, spin_left↔spin_right
- Reverses sequence order so bot returns to start position

### Arduino Pin Connections (L298N)
| Arduino Pin | L298N Pin |
|---|---|
| D5 (PWM) | ENA |
| D6 (PWM) | ENB |
| D7 | IN1 |
| D8 | IN2 |
| D4 | IN3 |
| D2 | IN4 |
| D10 | HC-05 TX (SoftwareSerial) |
| D11 | HC-05 RX (SoftwareSerial) |

---

## Follow Mode (rpi/modules/follow.py)

### HSV Yellow Color Tracking
- Target: Yellow wheel (H:18-45, S:80-255, V:80-255)
- Dead zone: 120px (wide to prevent jitter)
- Rolling average: 5 frames for cx and width
- Detection: HSV mask → morphological cleanup → largest contour → bounding rect

### Steering Logic (directions swapped for backward-mounted bot)
- Too close → backward
- Close + centered → stop
- Off-center → spin (left/right swapped)
- Far → forward or diagonal (swapped)
- Search: spin-stop-check pattern (0.3s spin, 0.8s pause, detect, repeat)

---

## Dance Mode

### Synchronized Three-System Dance (~5.5s)
1. **Car** (`car.dance()`): Wiggles (4x L-R) → Rock (4x F-B) → Fast spin → More wiggles → Finale spin
2. **Servo Hands** (`servos.dance()`): Alternating pumps → Up-down waves → Windmill → Rapid pumps → Triumphant arms high
3. **Display** (`BotState.DANCING`): Hot pink eyes, wiggle side-to-side, squash-stretch, bouncy mouth, floating hearts, 2.5x sparkles

All three run concurrently — servo dance in background thread, car dance awaited, display driven by state.

---

## Servo Controller (rpi/modules/servos.py)

### Hardware
```python
GPIO_CHIP = 4       # Pi 5 gpiochip
PIN_LEFT  = 12      # BCM12 — PWM0
PIN_RIGHT = 13      # BCM13 — PWM1
PWM_FREQ  = 50      # Hz (SG90)
DUTY_MIN  = 2.5     # 0 degrees
DUTY_MAX  = 7.5     # 90 degrees
MAX_ANGLE = 90
```
- Left servo is **mirror-mounted**: uses `MAX_ANGLE - angle` for duty cycle
- Rest position: left = 90 degrees (MAX_ANGLE), right = 0 degrees

### Gesture Types
| Method | When | Description |
|---|---|---|
| `start_speaking_gestures()` | During TTS playback | 6 random patterns: emphasis, alternate, nod, wide open, one hand, wave explain |
| `tickle_wave()` | On touch long-press | Sine sweep 0-90-0 with wiggle oscillation, 2.5s |
| `dance()` | On "dance" command | 5-phase energetic animation synced to car dance, ~5.5s |

---

## Display — Animated Face (tft_display.py)

### Architecture
- PIL-based rendering → ILI9488 via SPI (18-bit RGB666)
- Dataclasses: `EyeParams`, `FaceParams`, `AnimState`, `Sparkle`, `Heart`
- Smooth lerp interpolation between states
- Adaptive FPS: IDLE=1, LOOK=10, ACTIVE=10, TRANSITION=12, BLINK=12

### State Expressions
| State | Color | Eyes | Extra |
|---|---|---|---|
| **ready** | Orange | 120x140, pupils 0.40 | Idle look-around, sparkles |
| **listening** | Blue | 120x160, dilated 0.55, looking up | Pulsing glow + concentric rings |
| **processing** | Yellow | 120x100, look up-right | Processing dots |
| **speaking** | Purple | 120x100, happy curved eyelids | Animated mouth, sparkles |
| **dancing** | Hot Pink | 200x200, dilated 0.50 | Wiggle, squash-stretch, hearts, 2.5x sparkles, bouncy mouth |
| **error** | Red | 100x100 | X marks, flat mouth |
| **loading** | Yellow | 120x65, half-open | Loading spinner |
| **sleeping** | — | Thin lines | Floating zzZ |

### Tickle Animation
- Triggered by TTP223 long-press
- Warm pink eyes, rapid wiggle, floating hearts, bouncy mouth
- Duration: 2.5s with fade-out in last 0.5s

### Display Games
- Snake, Pong, Tetris, Breakout, Space Invaders
- Controlled via XPT2046 touch
- Accessible from settings menu

---

## Remote Voice Control

### How it works
1. User opens `https://raspberrypi.local:8443` on phone
2. Taps mic button → Web Speech API captures speech → sends text to Pi
3. Pi receives via `POST /api/control/send-text`
4. Sets `state.remote_text` and triggers `wake_event`
5. Main loop picks up text, skips wake word + STT, routes to intent pipeline
6. `_from_remote` flag prevents follow-up listening after phone commands

### HTTPS
- Self-signed cert on port 8443 (required for Web Speech API on iOS/Android)
- HTTP on port 8080 for local dashboard access

### iOS Mic Fix
- `recognition.abort()` (not `.stop()`) to release mic hardware immediately
- Null out `recognitionRef.current` on result/error/end

---

## Parent Dashboard (app/src/components/Dashboard.jsx)

### Features
- Live status panel (bot state, current activity)
- Remote Voice panel: mic button (Web Speech API), text input, Wake Bot button
- Companion toggle (enable/disable bot)
- Settings controls
- State colors: ready=orange, listening=blue, processing=yellow, speaking=purple, dancing=pink, error=red

---

## ALSA Config (~/.asoundrc on Pi)
```
pcm.!default { type asym; playback.pcm "speaker"; capture.pcm "mic" }
pcm.speaker { type plug; slave.pcm "hw:2,0" }
pcm.mic { type plug; slave { pcm "hw:2,0"; format S32_LE; rate 48000; channels 2 } }
ctl.!default { type hw; card 2 }
```
Card 2 = snd_rpi_googlevoicehat_soundcard (I2S overlay)

---

## Android/Flutter App (android_app/)

### Features
- 60+ learning activities across 20 skill categories (critical thinking, creativity, communication, etc.)
- Animated face display (same expression system as Pi)
- Bluetooth car control with joystick overlay
- Follow mode (camera-based person detection)
- Offline/online LLM support
- Parent dashboard with PIN gate, conversation history, model manager
- Progress tracking with skill radar chart

---

## Modular System (Future)

### Planned Modules
| Module | MCU | Connection | Status |
|---|---|---|---|
| Car Chassis (4WD) | Arduino Uno + HC-05 | Bluetooth SPP | **Done** |
| Battery Projector | None (HDMI) | HDMI + detect | Planned |
| Servo Joints | Arduino/ESP32 | Bluetooth/I2C | Future |
| Sensor Pods | MCU | Bluetooth/I2C | Future |
| Gripper | MCU | Bluetooth/I2C | Future |

### Design Principles
- Wireless first (Bluetooth SPP/BLE)
- Self-identifying modules
- Auto-discovery on connect
- Separate power per module
- Child-safe snap-fit/magnetic connectors

### Possible Configurations
- Desk buddy (brain only — current)
- Driving car (brain + wheel base)
- Story projector (brain + projector)
- Robotic arm (brain + servo joints + gripper)
- Robot dog (brain + 4 servo legs)
- Humanoid (brain + 2 arms + 2 legs)
- Explorer (brain + wheels + sensors + gripper)

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| RPi.GPIO not compatible with Pi 5 | Use `gpiod` / `lgpio` library |
| Bot turns wrong direction on car | Mounted backward — swap L/R in parser |
| Voice says opposite direction | Swap speech labels in responses dict |
| Follow mode too fast spin | Spin-stop-check pattern (0.3s spin, 0.8s pause) |
| Follow mode camera L→car R | Swap all directions in `_steer_to_center` |
| Follow mode jitter | Wide dead zone (120px) + 5-frame rolling average |
| Sequence commands not executing | Flatten nested sequences in `_parse_car_command` |
| Web Speech API needs HTTPS | Self-signed cert on port 8443 |
| Remote commands trigger listening | `_from_remote` flag overrides `follow_up` |
| Dance display at 1 FPS | Add "dancing" to FPS_ACTIVE and needs_redraw state lists |
| iOS mic icon stays on | Use `recognition.abort()` + null ref on end |
| ILI9488 white screen | Check DC pin (BCM23), verify 3.3V on LED |
| Left servo opposite direction | Use `MAX_ANGLE - angle` for left servo duty |
| lgpio "GPIO not set as output" | Call `lgpio.gpio_claim_output()` before `tx_pwm()` |

---

## Song Library (rpi/audio/songs/)
10 real sung nursery rhymes (WAV 16-bit mono 22050Hz, auto-converted to stereo):
twinkle_twinkle, baa_baa_black_sheep, row_row_row_your_boat, london_bridge, head_shoulders_knees_and_toes, humpty_dumpty, jack_and_jill, mary_had_a_little_lamb, old_macdonald, itsy_bitsy_spider

Song interrupt: `asyncio.wait FIRST_COMPLETED` races playback vs wake word detection.
Stop commands: "stop", "quiet", "enough" etc. in English/Hindi/Telugu.
