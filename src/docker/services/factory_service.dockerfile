# Factory Service Dockerfile - CPU only, no ML base image required.
# Build context is the repository root (see docker-compose.yml).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image small; sentence-transformers needs it for the
# "encoder" and "finetuned" students. The static and TF-IDF students do not use it.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY src/microservices/factory_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The forgelm core package.
COPY pyproject.toml /opt/forgelm/pyproject.toml
COPY forgelm /opt/forgelm/forgelm
RUN touch /opt/forgelm/README.md && pip install --no-cache-dir --no-deps /opt/forgelm

COPY src/microservices/factory_service/src /app/src
WORKDIR /app/src

RUN useradd --create-home llmforge && mkdir -p /app/models/factory /app/logs && chown -R llmforge:llmforge /app
USER llmforge

EXPOSE 8006
CMD ["python", "main.py"]
