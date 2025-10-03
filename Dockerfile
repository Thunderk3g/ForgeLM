# Multi-stage Dockerfile for LLMForge
# This builds a development environment with all services
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_LAUNCH_BLOCKING=0
ENV TOKENIZERS_PARALLELISM=false

# Create user for security
RUN groupadd -r llmforge && useradd -r -g llmforge llmforge

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    vim \
    htop \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy shared requirements files
COPY src/docker/shared-base/requirements-common.txt ./
COPY src/docker/shared-ml-base/requirements-ml.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements-common.txt
RUN pip3 install --no-cache-dir -r requirements-ml.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/datasets
RUN chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose ports for all services
EXPOSE 8000 8001 8002 8003 8004 8005 8080 3000

# Default command (can be overridden)
CMD ["python3", "-c", "print('LLMForge container ready. Use docker-compose to start services.')"]