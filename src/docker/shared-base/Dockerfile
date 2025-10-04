# LLMForge Shared Base Image
# Contains common dependencies used by all microservices

FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install common dependencies
COPY src/docker/shared-base/requirements-common.txt /tmp/requirements-common.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=3 -r /tmp/requirements-common.txt

# Set working directory
WORKDIR /app

# Create non-root user
RUN groupadd -r llmforge && useradd -r -g llmforge llmforge

# This base image will be used by other services
LABEL maintainer="LLMForge Team"
LABEL description="Shared base image for LLMForge microservices"