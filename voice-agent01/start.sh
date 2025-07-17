#!/bin/bash

# RunPod startup script for voice agent
set -e

echo "🚀 Starting Voice Agent on RunPod H100..."

# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check environment variables
if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo "❌ ELEVENLABS_API_KEY not set"
    exit 1
fi

if [ -z "$ELEVENLABS_VOICE_ID" ]; then
    echo "❌ ELEVENLABS_VOICE_ID not set"
    exit 1
fi

echo "✅ Environment variables configured"

# Pre-download Whisper model (optional, happens on first use anyway)
echo "📥 Pre-loading Whisper model..."
python -c "
from faster_whisper import WhisperModel
model = WhisperModel('large-v3', device='cuda', compute_type='float16')
print('✅ Whisper model loaded')
"

# Start the application
echo "🎯 Starting FastAPI application..."
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-5050} --workers 1
