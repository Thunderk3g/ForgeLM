# LLMForge Shared ML Base Image
# Optimized CUDA base with PyTorch, Transformers, and Unsloth for ML services
# This base image is shared by: training-service, chat-comparison-service, model-management-service

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set CUDA environment
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    gcc \
    g++ \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install common dependencies first (shared layer)
COPY src/docker/shared-base/requirements-common.txt /tmp/requirements-common.txt
RUN pip install --no-cache-dir --timeout=300 --retries=3 -r /tmp/requirements-common.txt

# Install PyTorch with CUDA support (largest dependencies) - FIXED: Pin to compatible cu121 version for Unsloth stability
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install ML-specific dependencies (Unsloth, Transformers, etc.)
COPY src/docker/shared-ml-base/requirements-ml.txt /tmp/requirements-ml.txt
RUN pip install --no-cache-dir --timeout=600 --retries=2 -r /tmp/requirements-ml.txt

# Set environment variables for ML services
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV CUDA_LAUNCH_BLOCKING=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create non-root user
RUN groupadd -r llmforge && useradd -r -g llmforge llmforge

# Label for identification
LABEL maintainer="LLMForge"
LABEL description="Shared ML base image with CUDA, PyTorch, Transformers, and Unsloth"
LABEL version="1.0.0"