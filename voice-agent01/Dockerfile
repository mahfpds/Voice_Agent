# CUDA 12.3 runtime → supports H100
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic build tools + ffmpeg for Whisper resampling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Copy your source code
COPY . /app
WORKDIR /app

# Create directory for Whisper model cache
RUN mkdir -p /root/.cache/huggingface

# Expose FastAPI port
EXPOSE 5050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5050/ || exit 1

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5050"]
