# RunPod H100 Deployment Guide

## Quick Start

### 1. Deploy Options

#### Option A: Direct GitHub Deployment (Recommended for Mac users)

1. Push your code to GitHub
2. In RunPod, use **GitHub Container Registry** or **Build from Git**
3. RunPod will build the image on their Linux servers with GPU support

#### Option B: Pre-built Image (Easiest)

Use the pre-built image: `golo005/voice-agent:latest`

#### Option C: Build Locally (Linux/Windows with Docker)

```bash
# Build the image (requires Linux or Windows with WSL2)
docker build -t golo005/voice-agent:latest .

# Push to Docker Hub
docker push golo005/voice-agent:latest
```

**Note for Mac users**: Building CUDA images locally on Mac often fails due to architecture differences. Use Option A or B instead.

### 2. Deploy on RunPod

1. Go to [RunPod](https://runpod.io) → **Secure Cloud GPU**
2. Select **NVIDIA H100 80GB** instance
3. Configure:
   - **Container Image**: `golo005/voice-agent:latest`
   - **Container Disk**: 20GB (for Whisper models)
   - **Ports**: Add `5050` (TCP)
   - **Environment Variables**:
     - `ELEVENLABS_API_KEY`: Your ElevenLabs API key
     - `ELEVENLABS_VOICE_ID`: Your ElevenLabs voice ID
     - `PORT`: 5050

4. Launch the pod

### 3. Access Your Application

Your app will be available at:
```
https://<pod-id>-5050.proxy.runpod.net
```

### 4. Configure Twilio Webhook

Point your Twilio webhook to:
```
https://<pod-id>-5050.proxy.runpod.net/incoming-call
```

## Advanced Setup

### Option A: With Ollama on Same Pod

1. Use the provided `docker-compose.yml`
2. Or manually add Ollama container in RunPod
3. Set `OLLAMA_HOST=localhost:11434` in environment

### Option B: Persistent Storage

To avoid re-downloading Whisper models on each restart:

1. In RunPod, add a **Network Volume**
2. Mount it to `/root/.cache/huggingface`
3. First run will download models (~2GB), subsequent runs will be faster

## Environment Variables

Required:
- `ELEVENLABS_API_KEY`: Your ElevenLabs API key
- `ELEVENLABS_VOICE_ID`: Your ElevenLabs voice ID

Optional:
- `PORT`: Application port (default: 5050)
- `OLLAMA_HOST`: Ollama server URL (default: localhost:11434)

## Monitoring

Check logs in RunPod console:
```bash
# Inside the pod terminal
docker logs -f <container-id>
```

## Troubleshooting

### GPU Not Detected
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Out of Memory
- Reduce Whisper model size in `stt.py`: change `MODEL_NAME = "large-v3"` to `"medium"`
- The H100 80GB should handle large models easily

### Slow First Request
- First Whisper inference downloads the model (~2GB)
- Use persistent storage to cache models between restarts

## Cost Optimization

- Use **Spot Instances** for development (cheaper but can be interrupted)
- Use **On-Demand** for production
- Consider auto-scaling based on usage patterns
