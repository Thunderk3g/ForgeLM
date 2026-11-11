"""Install script for llmforge_common, the modules every LLMForge microservice shares.

Each image installs it next to its requirements, into the same environment:

    COPY src/shared/llmforge_common /opt/llmforge_common
    RUN pip install --no-cache-dir --no-deps /opt/llmforge_common

For a checkout: pip install -e src/shared/llmforge_common
"""

from setuptools import setup

setup(
    name="llmforge-common",
    version="1.0.0",
    description="Modules shared by the LLMForge microservices (one copy instead of one per service)",
    packages=["llmforge_common"],
    # Deliberately empty. The modules need different things (gpu_inference_wrapper
    # needs torch, schemas needs pydantic, performance needs nothing), and a light
    # service must not be made to install torch. Each image already installs what
    # its service imports, and the package is installed with --no-deps so it cannot
    # disturb a pinned ML stack. llmforge_common/__init__.py lists the needs per module.
    install_requires=[],
    python_requires=">=3.10",
)
