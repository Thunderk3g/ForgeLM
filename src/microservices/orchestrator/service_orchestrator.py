"""
LLMForge Service Orchestrator

A Python-based agent orchestrator that manages microservices discovery, load balancing,
circuit breaker patterns, and health monitoring for the LLMForge platform.

This orchestrator implements patterns from agent-based systems and service mesh
architectures to provide intelligent service coordination.
"""

import asyncio
import time
import json
import logging
import uuid
import os
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
from fastapi import WebSocket
import weakref

# Try Redis import, make it optional
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not available - running without persistence")


# Service Health States
class ServiceState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"

# Load Balancing Strategies
class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    HEALTH_AWARE = "health_aware"


@dataclass
class ServiceInstance:
    """Individual service instance with health and performance metrics"""
    id: str
    name: str
    host: str
    port: int
    health_endpoint: str = "/health"
    state: ServiceState = ServiceState.UNKNOWN
    last_health_check: Optional[datetime] = None
    response_time_ms: float = 0.0
    active_connections: int = 0
    error_count: int = 0
    success_count: int = 0
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 1.0
    
    def mark_success(self, response_time_ms: float):
        self.success_count += 1
        self.response_time_ms = response_time_ms
        self.state = ServiceState.HEALTHY
        self.last_health_check = datetime.now()
    
    def mark_error(self):
        self.error_count += 1
        self.state = ServiceState.UNHEALTHY


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    service_name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    half_open_calls: int = 0
    
    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and \
               datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
                return True
            return False
        else:  # HALF_OPEN
            return self.half_open_calls < self.half_open_max_calls
    
    def on_success(self):
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
        self.half_open_calls = 0
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
        elif self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
        
        if self.state == "HALF_OPEN":
            self.half_open_calls += 1


class ServiceRegistry:
    """Service registry with automatic discovery and health monitoring"""
    
    def __init__(self, consul_client: Optional[Any] = None):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.consul = consul_client
        self.logger = logging.getLogger(__name__)
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the service registry with health monitoring"""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        if self.consul:
            await self._discover_services_from_consul()
    
    async def stop(self):
        """Stop the service registry"""
        if self._health_check_task:
            self._health_check_task.cancel()
    
    def register_service(self, service: ServiceInstance):
        """Register a new service instance"""
        if service.name not in self.services:
            self.services[service.name] = []
            self.circuit_breakers[service.name] = CircuitBreaker(service.name)
        
        # Remove existing instance with same ID
        self.services[service.name] = [s for s in self.services[service.name] if s.id != service.id]
        self.services[service.name].append(service)
        
        self.logger.info(f"Registered service: {service.name} at {service.url}")
    
    def deregister_service(self, service_name: str, service_id: str):
        """Deregister a service instance"""
        if service_name in self.services:
            self.services[service_name] = [
                s for s in self.services[service_name] if s.id != service_id
            ]
            if not self.services[service_name]:
                del self.services[service_name]
                if service_name in self.circuit_breakers:
                    del self.circuit_breakers[service_name]
    
    def get_healthy_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get healthy instances of a service"""
        if service_name not in self.services:
            return []
        
        return [s for s in self.services[service_name] 
                if s.state in [ServiceState.HEALTHY, ServiceState.DEGRADED]]
    
    async def _health_check_loop(self):
        """Continuous health monitoring of registered services"""
        while True:
            try:
                await asyncio.sleep(30)  # Health check every 30 seconds
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered services"""
        tasks = []
        for service_name, instances in self.services.items():
            for instance in instances:
                task = asyncio.create_task(self._check_instance_health(instance))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_instance_health(self, instance: ServiceInstance):
        """Check health of a single service instance"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(
                    f"{instance.url}{instance.health_endpoint}",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        instance.mark_success(response_time)
                        health_data = await response.json()
                        instance.metadata.update(health_data)
                    else:
                        instance.mark_error()
        except Exception as e:
            instance.mark_error()
            self.logger.warning(f"Health check failed for {instance.name}: {e}")
    
    async def _discover_services_from_consul(self):
        """Discover services from Consul"""
        if not self.consul:
            return
        
        try:
            services = await self.consul.catalog.services()
            for service_name, tags in services[1].items():
                if service_name.startswith('llmforge-'):
                    service_info = await self.consul.catalog.service(service_name)
                    for service in service_info[1]:
                        instance = ServiceInstance(
                            id=service['ServiceID'],
                            name=service_name,
                            host=service['Address'],
                            port=service['ServicePort'],
                            metadata={'tags': tags}
                        )
                        self.register_service(instance)
        except Exception as e:
            self.logger.error(f"Consul service discovery failed: {e}")


class LoadBalancer:
    """Intelligent load balancer with multiple strategies"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_AWARE):
        self.strategy = strategy
        self._round_robin_counters: Dict[str, int] = {}
    
    def select_instance(self, service_name: str, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select the best instance based on the load balancing strategy"""
        if not instances:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(service_name, instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            return self._weighted_response_time_selection(instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(instances)
        elif self.strategy == LoadBalancingStrategy.HEALTH_AWARE:
            return self._health_aware_selection(instances)
        else:
            return instances[0]  # Fallback
    
    def _round_robin_selection(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Simple round-robin selection"""
        if service_name not in self._round_robin_counters:
            self._round_robin_counters[service_name] = 0
        
        index = self._round_robin_counters[service_name] % len(instances)
        self._round_robin_counters[service_name] += 1
        return instances[index]
    
    def _weighted_round_robin_selection(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round-robin based on instance weights"""
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return instances[0]
        
        # Implement weighted selection logic
        import random
        r = random.uniform(0, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if r <= current_weight:
                return instance
        
        return instances[-1]  # Fallback
    
    def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least active connections"""
        return min(instances, key=lambda x: x.active_connections)
    
    def _health_aware_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance based on health metrics"""
        # Score based on success rate, response time, and active connections
        def health_score(instance: ServiceInstance) -> float:
            success_weight = 0.5
            response_time_weight = 0.3
            connection_weight = 0.2
            
            success_score = instance.success_rate
            # Lower response time is better (inverted)
            response_score = 1.0 / (1.0 + instance.response_time_ms / 1000.0)
            # Lower connections is better (inverted)
            connection_score = 1.0 / (1.0 + instance.active_connections)
            
            return (success_weight * success_score + 
                   response_time_weight * response_score + 
                   connection_weight * connection_score)
        
        return max(instances, key=health_score)


class ServiceOrchestrator:
    """Main orchestrator class for managing microservices"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.registry = ServiceRegistry()
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.HEALTH_AWARE)
        self.logger = logging.getLogger(__name__)
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Set[WebSocket] = set()
        
        # Service mesh configuration
        self.service_mesh_enabled = self.config.get('service_mesh_enabled', False)
        
        # Redis client for distributed locking
        if REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host='redis',
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.logger.info("Redis client initialized for GPU locking")
        else:
            self.redis_client = None
            self.logger.warning("Redis not available - GPU locking disabled")

    async def stop(self):
        """Gracefully stop orchestrator components"""
        await self.registry.stop()

    async def _broadcast_service_update(self):
        """Broadcast service updates to connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        message = {
            "type": "service_update",
            "services": {
                name: [{"id": i.id, "state": i.state.value, "url": i.url} for i in instances]
                for name, instances in self.registry.services.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except:
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        self.websocket_connections -= disconnected


# Example usage and service registration helper
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


# Global orchestrator instance
_orchestrator_instance: Optional[ServiceOrchestrator] = None

async def get_orchestrator() -> ServiceOrchestrator:
    """Get or create the global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ServiceOrchestrator()
        # Start the registry
        await _orchestrator_instance.registry.start()
    return _orchestrator_instance
