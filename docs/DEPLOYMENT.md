# Deployment

How to build and run the platform services. What each service does is in
the README ("The platform services") and in [ARCHITECTURE.md](ARCHITECTURE.md).
Configuration lives in [`.env.example`](../.env.example): copy it to `.env`.

## Docker Compose

Build the two base images first, then the services:

```bash
docker compose build base ml-base
docker compose build
```

`base` (`src/docker/shared-base`) is the Python image the CPU services build
on; `ml-base` (`src/docker/shared-ml-base`) is a separate image, built from
`nvidia/cuda` with PyTorch, for the training, chat-comparison and
model-management services. Its `depends_on: base` in compose only orders the
builds; `ml-base` does not layer on `base`. The service images
that share modules install `src/shared/llmforge_common` (see each
`src/docker/services/*.dockerfile`), so rebuild them after changing it. The factory service builds on `python:3.11-slim`
and needs neither base image.

Without an NVIDIA GPU, start only the CPU services:

```bash
docker compose up -d api-gateway orchestrator data-generation-service visualization-service factory-service
```

The GPU services need an NVIDIA GPU, its driver and the NVIDIA Container
Toolkit on the host. Two overlays change their resources:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d    # a single workstation or laptop
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d   # a larger host
```

Read the overlay for the current numbers rather than copying them. The
training service sizes itself from its cgroup memory limit, falling back to
`CONTAINER_MEMORY_LIMIT_GB` (`utils/hardware_detection.py`); keep that
variable equal to the container's memory limit. The prod overlay takes it
from `CONTAINER_MEMORY_GB` in the shell or `.env`.

The dashboard is not containerised: `cd frontend && npm install
--legacy-peer-deps && npm run dev` serves it on port 5173 and proxies `/api`
to the gateway on port 8000.

## Kubernetes

`deployments/kubernetes/` has manifests for the namespace, config maps,
storage, monitoring, orchestrator, gateway, training service and factory
service. Apply the shared objects first:

```bash
kubectl apply -f deployments/kubernetes/namespace.yaml \
  -f deployments/kubernetes/configmaps.yaml -f deployments/kubernetes/storage.yaml
kubectl apply -f deployments/kubernetes/
```

- Create the `llmforge-secrets` Secret in the `llmforge` namespace with the
  keys the manifests' `secretKeyRef` entries name.
- The manifests pull `llmforge/<service>:latest`; tag and push the images
  built above to a registry the cluster can reach.
- `training-service.yaml` requests `nvidia.com/gpu` and selects nodes
  labelled `nvidia.com/gpu: "true"`. The cluster needs the NVIDIA device
  plugin and GPU nodes carrying that label (on EKS, a GPU node group).
- It sets `CONTAINER_MEMORY_LIMIT_GB` from the pod's own memory limit, so
  changing the limit is enough.

## Known GPU issues

Observed on an RTX 3050 4 GB laptop; neither can be re-checked without a GPU.

- With PyTorch Inductor enabled, training hung at step 0 while compile
  workers ran (PyTorch 2.8 with Unsloth). `docker-compose.yml` disables it
  (`TORCH_COMPILE_DISABLE=1`, `TORCHINDUCTOR_*`); keep those settings.
- After the training service's memory reservation was once lowered from 12G
  to 3G, the container then crashed with exit code 139 (SIGSEGV) during model
  loading and Triton compilation. Do not lower the reservation.
