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
echo "[1/6] Wake word models (OpenWakeWord)..."
mkdir -p "$MODELS_DIR/wake_word"
# OpenWakeWord downloads its own models automatically on first use.
echo "  -> Will be downloaded on first run."

# ----- STT Model (Whisper tiny — multilingual) -----
echo ""
echo "[2/6] Speech-to-Text model (Whisper tiny multilingual)..."
mkdir -p "$MODELS_DIR/stt"
WHISPER_MODEL="$MODELS_DIR/stt/ggml-tiny.bin"
if [ ! -f "$WHISPER_MODEL" ]; then
    echo "  Downloading ggml-tiny.bin (~77MB, multilingual: en+hi)..."
    curl -L -o "$WHISPER_MODEL" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
    echo "  -> Done."
else
    echo "  -> Already exists."
fi

# ----- LLM Model (Qwen 2.5 3B Instruct Q4_K_M -- primary) -----
echo ""
echo "[3/6] Local LLM model (Qwen 2.5 3B Instruct Q4_K_M)..."
mkdir -p "$MODELS_DIR/llm"
QWEN_MODEL="$MODELS_DIR/llm/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
if [ ! -f "$QWEN_MODEL" ]; then
    echo "  Downloading Qwen2.5-3B-Instruct-Q4_K_M.gguf (~2.0GB)..."
    curl -L -o "$QWEN_MODEL" \
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    echo "  -> Done."
else
    echo "  -> Already exists."
fi

# Llama 3.2 1B kept as fallback (smaller, faster boot)
LLM_FALLBACK="$MODELS_DIR/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
if [ ! -f "$LLM_FALLBACK" ]; then
    echo "  Downloading Llama-3.2-1B fallback (~700MB)..."
    curl -L -o "$LLM_FALLBACK" \
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    echo "  -> Done."
else
    echo "  -> Fallback already exists."
fi

# ----- TTS Voice — English (Piper en_US-lessac-medium) -----
echo ""
echo "[4/6] Text-to-Speech voice — English (Piper en_US-lessac-medium)..."
mkdir -p "$MODELS_DIR/tts"
TTS_EN_MODEL="$MODELS_DIR/tts/en_US-lessac-medium.onnx"
TTS_EN_CONFIG="$MODELS_DIR/tts/en_US-lessac-medium.onnx.json"
if [ ! -f "$TTS_EN_MODEL" ]; then
    echo "  Downloading English Piper voice (~100MB)..."
    curl -L -o "$TTS_EN_MODEL" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    curl -L -o "$TTS_EN_CONFIG" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    echo "  -> Done."
else
    echo "  -> Already exists."
fi

# ----- TTS Voice — Hindi (Piper hi_IN-rohan-medium) -----
echo ""
echo "[5/6] Text-to-Speech voice — Hindi (Piper hi_IN-rohan-medium)..."
TTS_HI_MODEL="$MODELS_DIR/tts/hi_IN-rohan-medium.onnx"
TTS_HI_CONFIG="$MODELS_DIR/tts/hi_IN-rohan-medium.onnx.json"
if [ ! -f "$TTS_HI_MODEL" ]; then
    echo "  Downloading Hindi Piper voice (~63MB)..."
    curl -L -o "$TTS_HI_MODEL" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx"
    curl -L -o "$TTS_HI_CONFIG" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json"
    echo "  -> Done."
else
    echo "  -> Already exists."
fi

# ----- TTS Voice — Telugu (Piper te_IN-maya-medium) -----
echo ""
echo "[6/7] Text-to-Speech voice — Telugu (Piper te_IN-maya-medium)..."
TTS_TE_MODEL="$MODELS_DIR/tts/te_IN-maya-medium.onnx"
TTS_TE_CONFIG="$MODELS_DIR/tts/te_IN-maya-medium.onnx.json"
if [ ! -f "$TTS_TE_MODEL" ]; then
    echo "  Downloading Telugu Piper voice (~63MB)..."
    curl -L -o "$TTS_TE_MODEL" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/maya/medium/te_IN-maya-medium.onnx"
    curl -L -o "$TTS_TE_CONFIG" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/maya/medium/te_IN-maya-medium.onnx.json"
    echo "  -> Done."
else
    echo "  -> Already exists."
fi

# ----- Vision Model (YOLOv8n) -----
echo ""
echo "[7/7] Vision model (YOLOv8-nano)..."
mkdir -p "$MODELS_DIR/vision"
YOLO_MODEL="$MODELS_DIR/vision/yolov8n.pt"
if [ ! -f "$YOLO_MODEL" ]; then
    echo "  YOLOv8n will be downloaded automatically on first use by ultralytics."
    echo "  -> Skipping manual download."
else
    echo "  -> Already exists."
fi

echo ""
echo "========================================"
echo "  All models downloaded!"
echo "========================================"
echo ""
echo "  Disk space used:"
du -sh "$MODELS_DIR"/*/ 2>/dev/null || echo "  (empty directories)"
echo ""
