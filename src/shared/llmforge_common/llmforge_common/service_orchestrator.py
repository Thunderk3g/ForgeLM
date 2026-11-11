"""
Client helper for the LLMForge service orchestrator.

Services call register_with_orchestrator once at start-up. The orchestrator itself
(registry, load balancer, GPU lock) lives in src/microservices/orchestrator.
"""

from typing import Any, Dict

import aiohttp


async def register_with_orchestrator(
    orchestrator_url: str,
    service_name: str,
    service_id: str,
    host: str,
    port: int,
    health_endpoint: str = "/health",
    metadata: Dict[str, Any] = None
):
    """Helper function to register a service with the orchestrator"""
    async with aiohttp.ClientSession() as session:
        data = {
            "service_name": service_name,
            "service_id": service_id,
            "host": host,
            "port": port,
            "health_endpoint": health_endpoint,
            "metadata": metadata or {}
        }
        async with session.post(f"{orchestrator_url}/register", json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Registration failed: {response.status}")

