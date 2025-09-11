FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV TRANSFORMERS_NO_FLASH_ATTENTION=1

# Install system dependencies (cached layer - rarely changes)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Create directory structure early (cached layer)
RUN mkdir -p /app/data/input /app/data/prepared /app/models/base /app/models/finetuned

# Upgrade pip (small, stable layer)
RUN pip install --no-cache-dir --upgrade pip

# Layer 1: Base stable dependencies (rarely change)
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Layer 2: Heavy ML/AI dependencies (change less frequently than app deps)
COPY requirements-ml.txt .
# Install build dependencies first for packages that need compilation
RUN pip install --no-cache-dir packaging wheel ninja
RUN pip install --no-cache-dir -r requirements-ml.txt

# Layer 3: Application and web framework dependencies (moderate change frequency)
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Download language models (cached layer - only when ML deps change)
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Copy application code last (changes most frequently)
COPY *.py ./
COPY --chown=root:root . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]