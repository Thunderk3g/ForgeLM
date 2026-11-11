"""llmforge_common: the one copy of the modules the LLMForge microservices share.

These modules used to be copied, byte for byte, into each service's src/common,
src/utils, src/services, src/models and src/config directories. A fix made in one
copy silently missed the others. Each of those files is now a few-line shim that
points at the module here, so a service's imports are unchanged:

    from common.gpu_client import GPUClient          # in a service; still works
    # common/gpu_client.py does:
    #     from llmforge_common import gpu_client as _impl
    #     sys.modules[__name__] = _impl

After the shim runs, common.gpu_client *is* llmforge_common.gpu_client: one module
object, one set of classes, one copy of any module-level state.

Module                  Old path in a service             Third-party imports
gpu_orchestrator        common/gpu_orchestrator.py        aiohttp (redis optional)
gpu_client              common/gpu_client.py              aiohttp
gpu_monitoring          common/gpu_monitoring.py          aiohttp, psutil (torch, pynvml optional)
gpu_inference_wrapper   common/gpu_inference_wrapper.py   torch, aiohttp
service_orchestrator    services/service_orchestrator.py  aiohttp, fastapi, uvicorn (redis optional)
research_logger         utils/research_logger.py          whatever the service's models.schemas needs
memory                  utils/memory.py                   psutil (torch optional)
performance             utils/performance.py              none
logging                 utils/logging.py                  none
model_config            utils/model_config.py             pyyaml
schemas                 models/schemas.py                 pydantic
entities                models/entities.py                none
settings                config/settings.py                none

Rules for code in this package:

- A module here imports another module here relatively (gpu_client imports
  .gpu_orchestrator).
- research_logger and model_config import models.schemas and utils.logging by those
  absolute names, exactly as they did inside a service. The names resolve in the
  service that is running (its src directory is on sys.path), so each service
  supplies its own. That matters: the data generation service has its own, different
  models/schemas.py, and any service may replace a shim with a file of its own.
- Nothing is imported here at package level. A light service that only needs
  performance must not pay for, or fail on, torch.

Files that only look like copies were left in their services: the training
service's service_orchestrator.py, the visualization service's research_logger.py
and the chat comparison service's entities.py each differ from the shared version.
"""

__version__ = "1.0.0"
