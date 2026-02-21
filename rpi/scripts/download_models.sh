#!/bin/bash
# =============================================================
# Download all required AI models for offline operation
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPI_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$RPI_DIR/models"

echo "========================================"
echo "  Downloading AI Models"
echo "========================================"

# ----- Wake Word Models -----
echo ""
echo "[1/5] Wake word models (OpenWakeWord)..."
mkdir -p "$MODELS_DIR/wake_word"
# OpenWakeWord downloads its own models automatically on first use.
echo "  → Will be downloaded on first run."

# ----- STT Model (Whisper tiny.en) -----
echo ""
echo "[2/5] Speech-to-Text model (Whisper tiny.en)..."
mkdir -p "$MODELS_DIR/stt"
WHISPER_MODEL="$MODELS_DIR/stt/ggml-tiny.en.bin"
if [ ! -f "$WHISPER_MODEL" ]; then
    echo "  Downloading ggml-tiny.en.bin (~39MB)..."
    curl -L -o "$WHISPER_MODEL" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
    echo "  → Done."
else
    echo "  → Already exists."
fi

# ----- LLM Model (Llama 3.2 1B Q4_K_M) -----
echo ""
echo "[3/5] Local LLM model (Llama 3.2 1B Q4_K_M)..."
mkdir -p "$MODELS_DIR/llm"
LLM_MODEL="$MODELS_DIR/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
if [ ! -f "$LLM_MODEL" ]; then
    echo "  Downloading Llama-3.2-1B-Instruct-Q4_K_M.gguf (~700MB)..."
    curl -L -o "$LLM_MODEL" \
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    echo "  → Done."
else
    echo "  → Already exists."
fi

# ----- TTS Voice (Piper — en_US-lessac-medium) -----
echo ""
echo "[4/5] Text-to-Speech voice (Piper en_US-lessac-medium)..."
mkdir -p "$MODELS_DIR/tts"
TTS_MODEL="$MODELS_DIR/tts/en_US-lessac-medium.onnx"
TTS_CONFIG="$MODELS_DIR/tts/en_US-lessac-medium.onnx.json"
if [ ! -f "$TTS_MODEL" ]; then
    echo "  Downloading Piper voice model (~100MB)..."
    curl -L -o "$TTS_MODEL" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    curl -L -o "$TTS_CONFIG" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    echo "  → Done."
else
    echo "  → Already exists."
fi

# ----- Vision Model (YOLOv8n) -----
echo ""
echo "[5/5] Vision model (YOLOv8-nano)..."
mkdir -p "$MODELS_DIR/vision"
YOLO_MODEL="$MODELS_DIR/vision/yolov8n.pt"
if [ ! -f "$YOLO_MODEL" ]; then
    echo "  YOLOv8n will be downloaded automatically on first use by ultralytics."
    echo "  → Skipping manual download."
else
    echo "  → Already exists."
fi

echo ""
echo "========================================"
echo "  All models downloaded!"
echo "========================================"
echo ""
echo "  Disk space used:"
du -sh "$MODELS_DIR"/*/ 2>/dev/null || echo "  (empty directories)"
echo ""
