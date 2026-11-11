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
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import weakref

# Each service used to keep its own copy of this module, and __name__ there gave
# the logger name below. It is fixed now that the one copy lives in llmforge_common,
# so log lines, and anything that searches or filters them, read as they did.
_LOGGER_NAME = "services.service_orchestrator"

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
        self.logger = logging.getLogger(_LOGGER_NAME)
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
        self.logger = logging.getLogger(_LOGGER_NAME)
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Set[WebSocket] = set()
        
        # Service mesh configuration
        self.service_mesh_enabled = self.config.get('service_mesh_enabled', False)
        
        # Initialize FastAPI app
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with orchestrator endpoints"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.registry.start()
            self.logger.info("Service orchestrator started")
            yield
            # Shutdown
            await self.registry.stop()
            self.logger.info("Service orchestrator stopped")
        
        app = FastAPI(
            title="LLMForge Service Orchestrator",
            description="Python LLM orchestrator for microservices coordination",
            version="1.0.0",
            lifespan=lifespan
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    name: len(instances) for name, instances in self.registry.services.items()
                }
            }
        
        # Service registration endpoint
        @app.post("/register")
        async def register_service(
            service_name: str,
            service_id: str,
            host: str,
            port: int,
            health_endpoint: str = "/health",
            metadata: Dict[str, Any] = None
        ):
            instance = ServiceInstance(
                id=service_id,
                name=service_name,
                host=host,
                port=port,
                health_endpoint=health_endpoint,
                metadata=metadata or {}
            )
            self.registry.register_service(instance)
            await self._broadcast_service_update()
            return {"status": "registered", "service": service_name, "id": service_id}
        
        # Service discovery endpoint
        @app.get("/discover/{service_name}")
        async def discover_service(service_name: str):
            instances = self.registry.get_healthy_instances(service_name)
            selected = self.load_balancer.select_instance(service_name, instances)
            if selected:
                selected.active_connections += 1
                return {
                    "service": service_name,
                    "instance": {
                        "id": selected.id,
                        "url": selected.url,
                        "metadata": selected.metadata
                    }
                }
            else:
                raise HTTPException(status_code=503, detail=f"Service {service_name} unavailable")
        
        # Proxy request endpoint
        @app.api_route("/proxy/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def proxy_request(service_name: str, path: str, request):
            circuit_breaker = self.registry.circuit_breakers.get(service_name)
            if circuit_breaker and not circuit_breaker.can_execute():
                raise HTTPException(status_code=503, detail=f"Circuit breaker open for {service_name}")
            
            instances = self.registry.get_healthy_instances(service_name)
            selected = self.load_balancer.select_instance(service_name, instances)
            
            if not selected:
                raise HTTPException(status_code=503, detail=f"Service {service_name} unavailable")
            
            try:
                # Proxy the request
                async with aiohttp.ClientSession() as session:
                    url = f"{selected.url}/{path}"
                    method = request.method
                    headers = dict(request.headers)
                    
                    # Remove host header to avoid conflicts
                    headers.pop('host', None)
                    
                    body = None
                    if method in ['POST', 'PUT', 'PATCH']:
                        body = await request.body()
                    
                    start_time = time.time()
                    async with session.request(
                        method=method,
                        url=url,
                        headers=headers,
                        data=body,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        response_time = (time.time() - start_time) * 1000
                        content = await response.read()
                        
                        selected.mark_success(response_time)
                        if circuit_breaker:
                            circuit_breaker.on_success()
                        
                        return Response(
                            content=content,
                            status_code=response.status,
                            headers=dict(response.headers),
                            media_type=response.content_type
                        )
            
            except Exception as e:
                selected.mark_error()
                if circuit_breaker:
                    circuit_breaker.on_failure()
                
                self.logger.error(f"Proxy request failed for {service_name}: {e}")
                raise HTTPException(status_code=502, detail=f"Service {service_name} error: {str(e)}")
            
            finally:
                selected.active_connections = max(0, selected.active_connections - 1)
        
        # WebSocket for real-time updates
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.add(websocket)
            try:
                while True:
                    await websocket.receive_text()  # Keep connection alive
            except WebSocketDisconnect:
                self.websocket_connections.discard(websocket)
        
        # Service metrics endpoint
        @app.get("/metrics")
        async def get_metrics():
            metrics = {}
            for service_name, instances in self.registry.services.items():
                service_metrics = {
                    "instance_count": len(instances),
                    "healthy_instances": len([i for i in instances if i.state == ServiceState.HEALTHY]),
                    "avg_response_time": sum(i.response_time_ms for i in instances) / len(instances) if instances else 0,
                    "total_requests": sum(i.success_count + i.error_count for i in instances),
                    "success_rate": sum(i.success_rate for i in instances) / len(instances) if instances else 0
                }
                circuit_breaker = self.registry.circuit_breakers.get(service_name)
                if circuit_breaker:
                    service_metrics["circuit_breaker_state"] = circuit_breaker.state
                metrics[service_name] = service_metrics
            
            return metrics
        
        return app
    
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
    
    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the orchestrator server"""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


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


if __name__ == "__main__":
    # Run the orchestrator
    orchestrator = ServiceOrchestrator()
    asyncio.run(orchestrator.start())