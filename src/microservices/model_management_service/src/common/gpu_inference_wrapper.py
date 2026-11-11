# Shim. The real module is llmforge_common.gpu_inference_wrapper (src/shared/llmforge_common);
# this file keeps the old import path working by making it the same module object.
import sys

try:
    from llmforge_common import gpu_inference_wrapper as _impl
except ModuleNotFoundError as exc:
    if exc.name != "llmforge_common":
        raise  # the package is installed; one of the module's own imports is missing
    raise ImportError("llmforge_common is not installed: rebuild the base image, "
                      "or run pip install -e src/shared/llmforge_common") from exc

sys.modules[__name__] = _impl
