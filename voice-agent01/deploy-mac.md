# Mac Deployment Guide for RunPod

## Why Mac Can't Build CUDA Images Locally

- **No NVIDIA GPU**: Macs use Apple Silicon or Intel GPUs, not NVIDIA
- **Architecture mismatch**: Mac ARM64 vs Linux x86_64 for GPU containers  
- **CUDA unavailable**: Can't test CUDA runtime during build

## Recommended Deployment Methods for Mac Users

### Method 1: GitHub Integration (Easiest)

1. **Create GitHub Repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial voice agent commit"
   git remote add origin https://github.com/golo005/voice-agent.git
   git push -u origin main
   ```

2. **Deploy from GitHub in RunPod**:
   - Go to RunPod → Secure Cloud GPU
   - Select "Deploy from GitHub"
   - Repository: `golo005/voice-agent`
   - Branch: `main`
   - RunPod will build on their Linux servers with GPU support

### Method 2: Use Pre-built Image

Simply use: `golo005/voice-agent:latest` in RunPod

### Method 3: Build on Linux VM/Server

If you have access to a Linux machine:

```bash
# On Linux machine
git clone https://github.com/golo005/voice-agent.git
cd voice-agent
docker build -t golo005/voice-agent:latest .
docker push golo005/voice-agent:latest
```

## RunPod Configuration

1. **Container Image**: `golo005/voice-agent:latest`
2. **GPU**: NVIDIA H100 80GB
3. **Ports**: 5050 (TCP)
4. **Environment Variables**:
   - `ELEVENLABS_API_KEY`
   - `ELEVENLABS_VOICE_ID`
   - `PORT=5050`

## Testing Locally on Mac (Without GPU)

You can test the basic FastAPI functionality:

```bash
# Install dependencies
pip install -r requirements.txt

# Run without GPU (will use CPU for Whisper)
python app.py
```

**Note**: This will be very slow for speech processing, but good for testing API endpoints.

## Alternative: GitHub Actions Build

Create `.github/workflows/docker.yml`:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
      
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: golo005
        password: ${{ secrets.DOCKER_PASSWORD }}
        
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: golo005/voice-agent:latest
        platforms: linux/amd64
```

This automatically builds and pushes when you commit to GitHub!
