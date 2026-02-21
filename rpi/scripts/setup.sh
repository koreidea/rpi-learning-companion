#!/bin/bash
# =============================================================
# RPi Learning Companion — Full Setup Script
# Run this on the Raspberry Pi 5 after copying the rpi/ folder
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPI_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$RPI_DIR/.venv"

echo "========================================"
echo "  RPi Learning Companion Setup"
echo "========================================"

# ----- 1. System dependencies -----
echo ""
echo "[1/7] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    portaudio19-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev libpng-dev \
    bluetooth bluez pulseaudio-module-bluetooth \
    libcamera-dev python3-picamera2 \
    ffmpeg \
    cmake build-essential

# ----- 2. Python virtual environment -----
echo ""
echo "[2/7] Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel

# ----- 3. Python dependencies -----
echo ""
echo "[3/7] Installing Python dependencies..."
pip install -r "$RPI_DIR/requirements.txt"

# ----- 4. Download models -----
echo ""
echo "[4/7] Downloading AI models..."
bash "$SCRIPT_DIR/download_models.sh"

# ----- 5. Bluetooth speaker setup -----
echo ""
echo "[5/7] Configuring Bluetooth audio..."
# Enable Bluetooth
sudo systemctl enable bluetooth
sudo systemctl start bluetooth

echo "  To pair your Bluetooth speaker:"
echo "    1. Put your speaker in pairing mode"
echo "    2. Run: bluetoothctl"
echo "    3. Type: scan on"
echo "    4. Find your speaker and type: pair XX:XX:XX:XX:XX:XX"
echo "    5. Type: connect XX:XX:XX:XX:XX:XX"
echo "    6. Type: trust XX:XX:XX:XX:XX:XX"
echo "    7. Type: quit"

# ----- 6. Create data directories -----
echo ""
echo "[6/7] Creating data directories..."
mkdir -p "$RPI_DIR/data/sessions"

# ----- 7. Install systemd service -----
echo ""
echo "[7/7] Installing systemd service..."
sudo cp "$RPI_DIR/services/rpi-bot.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rpi-bot

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "  To start the bot:"
echo "    sudo systemctl start rpi-bot"
echo ""
echo "  To view logs:"
echo "    journalctl -u rpi-bot -f"
echo ""
echo "  Parent app will be available at:"
echo "    http://$(hostname).local:8080"
echo ""
echo "  Don't forget to pair your Bluetooth speaker!"
echo "========================================"
