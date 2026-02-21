# RPi Learning Companion — Device Software

Everything in this folder runs on the Raspberry Pi 5.

## Quick Start

```bash
# 1. Copy this folder to your Pi
scp -r rpi/ pi@raspberrypi:~/rpi-bot/

# 2. SSH into the Pi and run setup
ssh pi@raspberrypi
cd ~/rpi-bot
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Start the bot
sudo systemctl start rpi-bot
sudo systemctl enable rpi-bot   # auto-start on boot

# 4. Open the parent app
# On your phone/laptop: http://raspberrypi.local:8080
```

## Structure

- `core/` — Main orchestrator, config, state management
- `audio/` — Wake word, VAD, STT, TTS, audio capture/playback
- `llm/` — Local and cloud LLM providers, safety filters
- `vision/` — Camera, object detection, OCR
- `api/` — FastAPI server + routes (serves the parent app)
- `privacy/` — Encryption, consent, audit logging
- `models/` — Downloaded AI models (gitignored)
- `data/` — Runtime data (encrypted, gitignored)
- `scripts/` — Setup and deployment scripts
- `services/` — systemd service files

## Development

```bash
# Run directly (without systemd)
cd rpi
source .venv/bin/activate
python -m core.main

# Run tests
pytest tests/
```

## Models Required

Run `scripts/download_models.sh` or the models will download automatically:

| Model | Size | Purpose |
|-------|------|---------|
| whisper-tiny.en | 39MB | Speech-to-text |
| Llama 3.2 1B Q4_K_M | 700MB | Local LLM |
| Piper en_US-lessac-medium | 100MB | Text-to-speech |
| YOLOv8n | 6MB | Object detection |
