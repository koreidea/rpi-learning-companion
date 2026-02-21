#!/bin/bash
# Quick start script for development/testing
# For production, use systemd: sudo systemctl start rpi-bot

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPI_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$RPI_DIR/.venv"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Start the bot
cd "$RPI_DIR"
python -m core.main
