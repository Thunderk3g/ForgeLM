# Training Service - ML Optimized Service (~800MB)
# Self-contained with GPU support and ML libraries

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

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
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install Training Service specific dependencies
COPY microservices/training_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

# Copy only Training Service code and its dependencies
COPY microservices/training_service/src/ ./src/
COPY microservices/orchestrator/ ./microservices/orchestrator/
COPY shared/llmforge_contracts/ ./shared/llmforge_contracts/
COPY configs/ ./configs/

# Create non-root user and setup directories
RUN groupadd -r trainer && useradd -r -g trainer trainer && \
    mkdir -p logs data models/base models/finetuned datasets && \
    mkdir -p /home/trainer/.cache && \
    chown -R trainer:trainer /app /home/trainer/.cache

USER trainer

# Set ML environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV CUDA_LAUNCH_BLOCKING=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

EXPOSE 8001

CMD ["python", "-m", "src.main"]