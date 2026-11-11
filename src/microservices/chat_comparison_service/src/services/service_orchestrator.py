# Shim. The real module is llmforge_common.service_orchestrator (src/shared/llmforge_common);
# this file keeps the old import path working by making it the same module object.
import sys

try:
    import llmforge_common  # noqa: F401  (fails first, with the message below, if absent)
    if __name__ == "__main__":  # run as a script: start the orchestrator, as before
        import runpy
        runpy.run_module("llmforge_common.service_orchestrator", run_name="__main__")
    else:
        from llmforge_common import service_orchestrator as _impl
        sys.modules[__name__] = _impl
except ModuleNotFoundError as exc:
    if exc.name != "llmforge_common":
        raise  # the package is installed; one of the module's own imports is missing
    raise ImportError("llmforge_common is not installed: rebuild the base image, "
                      "or run pip install -e src/shared/llmforge_common") from exc
