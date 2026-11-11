# Factory Service Dockerfile - CPU only, no ML base image required.
# Build context is the repository root (see docker-compose.yml).
#
# Needs BuildKit (the default builder since Docker Engine 23 and the only one
# Docker Compose v2 uses) for the RUN --mount lines below. With the legacy
# builder (DOCKER_BUILDKIT=0) the build stops with "the --mount option
# requires BuildKit".
FROM python:3.11-slim

# HF_HOME matches docker-compose.yml and the Kubernetes manifest, so encoder
# weights downloaded at run time land on the mounted cache volume and are not
# fetched again by the next container. Without a mount they still land in a
# directory this user can write. TOKENIZERS_PARALLELISM=false stops the
# Hugging Face tokenizers from starting their own thread pool inside uvicorn's
# worker threads (and from printing a fork warning on every request).
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/hf_cache \
    TOKENIZERS_PARALLELISM=false

# curl is used by the compose health check. The user gets fixed ids 1000:1000
# because the Kubernetes manifest runs the pod as that uid and gid
# (runAsUser/fsGroup), and files written to a shared volume should have the
# same owner whichever way the service was started. The directories are
# created here, before the large layers, so that this layer is cheap to change.
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 llmforge \
    && useradd --uid 1000 --gid 1000 --create-home llmforge \
    && mkdir -p /app/models/factory /app/models/hf_cache /app/logs \
    && chown -R llmforge:llmforge /app

# CPU-only PyTorch keeps the image small; sentence-transformers needs it for the
# "encoder" and "finetuned" students. The static and TF-IDF students do not use it.
# It is installed first and in the same layer as the other requirements:
# sentence-transformers would otherwise pull the much larger CUDA build from
# PyPI, and one layer instead of two avoids a second full copy of
# site-packages on storage drivers that copy each layer (vfs).
# The requirements file is bind-mounted rather than copied, so it adds no layer.
RUN --mount=type=bind,source=src/microservices/factory_service/requirements.txt,target=/tmp/requirements.txt \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# The forgelm core package, installed without dependencies (the requirements
# above already cover what the factory endpoints import). The sources are
# bind-mounted for the install only: the installed copy in site-packages is
# the one the service imports, so the image does not need a second one.
# README.md is excluded by .dockerignore but named in pyproject.toml, hence
# the empty placeholder.
RUN --mount=type=bind,source=pyproject.toml,target=/tmp/forgelm/pyproject.toml \
    --mount=type=bind,source=forgelm,target=/tmp/forgelm/forgelm \
    touch /tmp/forgelm/README.md \
    && pip install --no-cache-dir --no-deps /tmp/forgelm \
    && rm -rf /tmp/forgelm/build /tmp/forgelm/*.egg-info /tmp/forgelm/README.md

COPY --chown=llmforge:llmforge src/microservices/factory_service/src /app/src
WORKDIR /app/src

# Numeric, so that Kubernetes can check runAsNonRoot against the image itself.
USER 1000:1000

EXPOSE 8006
CMD ["python", "main.py"]
