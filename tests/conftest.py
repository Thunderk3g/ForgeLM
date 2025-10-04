"""
Pytest configuration and fixtures for LLMForge GPU testing suite.
"""

import pytest
import asyncio
import logging
import os
from typing import Generator

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL for testing services."""
    return os.getenv("BASE_URL", "http://localhost")

@pytest.fixture(scope="session") 
def service_ports() -> dict:
    """Service port mappings."""
    return {
        "training": 8001,
        "chat-comparison": 8002,
        "model-management": 8005,
        "api-gateway": 8000,
        "orchestrator": 8080
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as load test"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as stress test"
    )
    config.addinivalue_line(
        "markers", "memory: mark test as memory test"
    )
    config.addinivalue_line(
        "markers", "mps: mark test as MPS functionality test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )