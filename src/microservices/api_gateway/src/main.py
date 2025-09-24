"""
LLMForge API Gateway

Central entry point for all microservices with intelligent routing,
load balancing, authentication, and request aggregation.
"""

import asyncio
import os
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
import aiohttp

from fastapi import FastAPI, Request, HTTPException, Depends, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn

from services.service_orchestrator import register_with_orchestrator
from utils.logging import get_logger
from utils.performance import performance_monitor
from config.settings import settings

# Prometheus metrics
gateway_requests_total = Counter('gateway_requests_total', 'Total gateway requests', ['method', 'service'])
gateway_request_duration_seconds = Histogram('gateway_request_duration_seconds', 'Gateway request duration')
gateway_errors_total = Counter('gateway_errors_total', 'Total gateway errors', ['service', 'error_type'])
active_connections = Gauge('gateway_active_connections', 'Active gateway connections')

# Security
security = HTTPBearer(auto_error=False)

logger = get_logger(__name__)

# Service routing configuration
SERVICE_ROUTES = {
    # Training service routes
    '/api/v1/fine-tune': 'llmforge-training',
    '/api/v1/training': 'llmforge-training',
    '/fine-tune': 'llmforge-training',  # Backward compatibility
    
    # Chat comparison service routes  
    '/api/v1/sessions': 'llmforge-chat-comparison',
    '/api/v1/compare': 'llmforge-chat-comparison',
    '/api/v1/models/pairs': 'llmforge-chat-comparison',
    '/api/v1/chat': 'llmforge-chat-comparison',
    '/chat': 'llmforge-chat-comparison',  # Backward compatibility
    
    # Data generation service routes
    '/api/v1/generate': 'llmforge-data-generation',
    '/api/v1/templates': 'llmforge-data-generation',
    '/api/v1/data-generation': 'llmforge-data-generation',
    '/api/v1/prepare-data': 'llmforge-data-generation',
    '/api/data-generation': 'llmforge-data-generation',
    '/data-generation': 'llmforge-data-generation',  # Backward compatibility
    '/prepare-data': 'llmforge-data-generation',  # Backward compatibility
    
    # Visualization service routes
    '/api/v1/analytics': 'llmforge-visualization',
    '/api/v1/dashboard': 'llmforge-visualization',
    '/api/v1/visualization': 'llmforge-visualization',
    '/api/visualization': 'llmforge-visualization',
    '/visualization': 'llmforge-visualization',  # Backward compatibility
    
    # Model management service routes
    '/api/v1/models': 'llmforge-model-management',
    '/api/v1/cache': 'llmforge-model-management',
    '/models': 'llmforge-model-management',  # Backward compatibility
    '/select-model': 'llmforge-model-management',  # Backward compatibility
    
    # Config routes (could be handled by any service, let's use model-management for now)
    '/api/v1/config': 'llmforge-model-management',
    '/config': 'llmforge-model-management',  # Backward compatibility
    
    # Agent routes (part of chat comparison service)
    '/api/v1/agents': 'llmforge-chat-comparison',
    '/agents': 'llmforge-chat-comparison',  # Backward compatibility
    
    # Chat comparison service routes
    '/chat': 'llmforge-chat-comparison',
}

# Service port mappings for direct container access
SERVICE_PORTS = {
    'llmforge-training': 8001,
    'llmforge-chat-comparison': 8002,
    'llmforge-data-generation': 8003,
    'llmforge-visualization': 8004,
    'llmforge-model-management': 8005,
}

# Service discovery client
class ServiceDiscoveryClient:
    def __init__(self, orchestrator_url: str):
        self.orchestrator_url = orchestrator_url
        self.service_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 30  # 30 seconds
        self.last_cache_update: Dict[str, float] = {}
        # Only allow localhost fallback in non-production by default
        env = os.getenv('ENV', os.getenv('APP_ENV', 'development')).lower()
        default_localhost = 'false' if env in ['prod', 'production'] else 'true'
        self.allow_localhost_fallback = os.getenv('ALLOW_LOCALHOST_FALLBACK', default_localhost).lower() == 'true'
    
    async def get_service_instance(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get a healthy service instance"""
        current_time = time.time()
        
        # Check cache first
        if (service_name in self.service_cache and 
            service_name in self.last_cache_update and
            current_time - self.last_cache_update[service_name] < self.cache_ttl):
            return self.service_cache[service_name]
        
        # Try orchestrator first
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.orchestrator_url}/discover/{service_name}",
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    if response.status == 200:
                        service_info = await response.json()
                        self.service_cache[service_name] = service_info
                        self.last_cache_update[service_name] = current_time
                        return service_info
        except Exception as e:
            logger.warning(f"Orchestrator unavailable for {service_name}: {e}")
        
        # Fallback to direct container networking
        if service_name in SERVICE_PORTS:
            port = SERVICE_PORTS[service_name]
            # Try the container/service DNS name first; include localhost only if allowed (non-prod)
            hosts = [service_name]
            if self.allow_localhost_fallback:
                hosts.append('localhost')
            for host in hosts:
                service_url = f"http://{host}:{port}"
                try:
                    # Test if service is healthy
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{service_url}/health",
                            timeout=aiohttp.ClientTimeout(total=2)
                        ) as response:
                            if response.status == 200:
                                service_info = {
                                    'instance': {
                                        'url': service_url,
                                        'host': host,
                                        'port': port,
                                        'healthy': True
                                    },
                                    'service_name': service_name
                                }
                                self.service_cache[service_name] = service_info
                                self.last_cache_update[service_name] = current_time
                                logger.info(f"Using direct connection for {service_name}: {service_url}")
                                return service_info
                except Exception as e:
                    logger.debug(f"Failed to connect directly to {service_url}: {e}")
                    continue
        
        logger.error(f"No healthy instance found for service {service_name}")
        return None

# Global service discovery client
service_discovery: Optional[ServiceDiscoveryClient] = None


async def register_service():
    """Register API Gateway with the orchestrator"""
    orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8080')
    service_id = f"api-gateway-{os.getpid()}"
    
    try:
        await register_with_orchestrator(
            orchestrator_url=orchestrator_url,
            service_name="llmforge-api-gateway",
            service_id=service_id,
            # Advertised host should be reachable by other containers/services
            host=os.getenv('SERVICE_ADVERTISED_HOST', 'llmforge-api-gateway'),
            port=int(os.getenv('SERVICE_PORT', '8000')),
            health_endpoint="/health",
            metadata={
                "type": "api_gateway",
                "routes": list(SERVICE_ROUTES.keys()),
                "rate_limiting": True
            }
        )
        logger.info(f"Registered API Gateway with orchestrator as {service_id}")
    except Exception as e:
        logger.warning(f"Failed to register with orchestrator: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the API Gateway"""
    global service_discovery
    
    # Startup
    logger.info("Starting LLMForge API Gateway...")
    logger.info(f"Configured routes: {list(SERVICE_ROUTES.keys())}")
    
    orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8080')
    service_discovery = ServiceDiscoveryClient(orchestrator_url)
    
    # Register with orchestrator
    await register_service()
    
    # Start performance monitoring
    performance_monitor.start_monitoring()
    
    logger.info("API Gateway started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway...")
    performance_monitor.stop_monitoring()
    logger.info("API Gateway shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LLMForge API Gateway",
    description="Central API Gateway for LLMForge microservices",
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


def get_service_name(path: str) -> Optional[str]:
    """Determine target service based on request path"""
    import re
    
    # Strip locale prefixes like /en-GB/, /en/, etc.
    locale_pattern = r'^/[a-z]{2}(-[A-Z]{2})?/'
    if re.match(locale_pattern, path):
        path = re.sub(locale_pattern, '/', path)
    
    logger.info(f"[ROUTE DEBUG] Checking path: '{path}'")
    for route_prefix, service_name in SERVICE_ROUTES.items():
        if path.startswith(route_prefix):
            logger.info(f"[ROUTE DEBUG] Matched '{route_prefix}' -> '{service_name}'")
            return service_name
    logger.warning(f"[ROUTE DEBUG] No match for path: '{path}'")
    return None


async def authenticate_request(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Authenticate request (optional for now)"""
    # For now, just return None to allow all requests
    # In production, implement proper JWT validation
    return None


@app.get("/health")
async def health_check():
    """API Gateway health check"""
    try:
        return {
            "status": "healthy",
            "service": "api_gateway", 
            "version": "1.0.0",
            "timestamp": time.time(),
            "services_cache_size": len(service_discovery.service_cache) if service_discovery else 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(
            content=f'{{"status": "unhealthy", "error": "{str(e)}"}}',
            status_code=503,
            media_type="application/json"
        )


@app.get("/")
async def root():
    """API Gateway root endpoint"""
    return {
        "message": "LLMForge API Gateway",
        "version": "1.0.0",
        "documentation": "/docs",
        "available_services": list(set(SERVICE_ROUTES.values()))
    }


async def proxy_request(
    request: Request,
    service_name: str,
    target_path: str,
    user: Dict[str, Any] = Depends(authenticate_request)
) -> Response:
    """Proxy request to target microservice"""
    gateway_requests_total.labels(method=request.method, service=service_name).inc()
    active_connections.inc()
    
    start_time = time.time()
    
    try:
        # Get service instance
        if not service_discovery:
            gateway_errors_total.labels(service=service_name, error_type="no_discovery").inc()
            raise HTTPException(status_code=503, detail="Service discovery not available")
        
        service_info = await service_discovery.get_service_instance(service_name)
        if not service_info:
            gateway_errors_total.labels(service=service_name, error_type="service_unavailable").inc()
            raise HTTPException(status_code=503, detail=f"Service {service_name} unavailable")
        
        # Build target URL (normalize slashes)
        instance = service_info['instance']
        base_url = instance['url'].rstrip('/')
        normalized_path = '/' + target_path.lstrip('/')
        target_url = f"{base_url}{normalized_path}"
        
        # Prepare request data
        headers = dict(request.headers)
        # Remove host header to avoid conflicts
        headers.pop('host', None)
        
        body = None
        if request.method in ['POST', 'PUT', 'PATCH']:
            body = await request.body()
        
        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body,
                params=dict(request.query_params),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                content = await response.read()
                
                # Record metrics
                duration = time.time() - start_time
                gateway_request_duration_seconds.observe(duration)
                
                # Return response
                return Response(
                    content=content,
                    status_code=response.status,
                    headers=dict(response.headers),
                    media_type=response.content_type
                )
    
    except aiohttp.ClientError as e:
        gateway_errors_total.labels(service=service_name, error_type="client_error").inc()
        logger.error(f"Client error proxying to {service_name}: {e}")
        raise HTTPException(status_code=502, detail=f"Service communication error")
    
    except asyncio.TimeoutError:
        gateway_errors_total.labels(service=service_name, error_type="timeout").inc()
        logger.error(f"Timeout proxying to {service_name}")
        raise HTTPException(status_code=504, detail="Service timeout")
    
    except Exception as e:
        gateway_errors_total.labels(service=service_name, error_type="unknown").inc()
        logger.error(f"Unknown error proxying to {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal gateway error")
    
    finally:
        active_connections.dec()


# Handler for /api/{path} routes (without version prefix) - must come before /api/v1
@app.api_route("/api/{version}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def api_versioned_proxy(version: str, path: str, request: Request):
    """Proxy versioned API requests to appropriate microservices"""
    # Only handle v1 for now
    if version != "v1":
        # Try as a non-versioned /api/ route
        full_path = f"/api/{version}/{path}" if path else f"/api/{version}"
        logger.debug(f"Looking for service for path: {full_path}")
        service_name = get_service_name(full_path)
        
        if not service_name:
            # Also try with just /api/{version}
            full_path = f"/api/{version}"
            logger.debug(f"Trying shorter path: {full_path}")
            service_name = get_service_name(full_path)
            if not service_name:
                logger.error(f"No service found for path: {full_path}")
                raise HTTPException(status_code=404, detail=f"No service found for path: {full_path}")
            # For routes like /api/visualization, strip the matched prefix and forward the rest
            target_path = f"/{path}" if path else ""
        else:
            # For routes like /api/visualization/something, strip /api/visualization and forward /something
            target_path = f"/{path}" if path else ""
        
        return await proxy_request(request, service_name, target_path)
    
    # Handle v1 routes
    full_path = f"/api/v1/{path}"
    service_name = get_service_name(full_path)
    
    if not service_name:
        raise HTTPException(status_code=404, detail=f"No service found for path: {full_path}")
    
    # Find the matched route to strip its suffix from the path
    matched_route = None
    for route_prefix, svc_name in SERVICE_ROUTES.items():
        if full_path.startswith(route_prefix) and svc_name == service_name:
            matched_route = route_prefix
            break
    
    # Strip the matched route prefix and keep what comes after
    if matched_route:
        # Remove /api/v1/<service> prefix, keep the rest
        # e.g., /api/v1/visualization/preprocessing/visualize -> /preprocessing/visualize
        prefix_to_remove = matched_route.replace('/api/v1/', '')
        logger.info(f"[PATH DEBUG] matched_route={matched_route}, prefix_to_remove={prefix_to_remove}, path={path}")
        if path.startswith(prefix_to_remove):
            target_path = '/' + path[len(prefix_to_remove):].lstrip('/')
            logger.info(f"[PATH DEBUG] Stripped prefix, target_path={target_path}")
        else:
            target_path = f"/{path}" if path else ""
            logger.info(f"[PATH DEBUG] No prefix match, target_path={target_path}")
    else:
        target_path = f"/{path}" if path else ""
        logger.info(f"[PATH DEBUG] No matched route, target_path={target_path}")
    
    return await proxy_request(request, service_name, target_path)


# Backward compatibility route handlers (without /api/v1 prefix)
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def legacy_proxy(path: str, request: Request):
    """Proxy legacy API requests to appropriate microservices for backward compatibility"""
    # Skip internal routes and routes handled by other handlers
    if path in ["health", "services", "routes", "metrics"] or path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    
    full_path = f"/{path}"
    service_name = get_service_name(full_path)
    
    if not service_name:
        raise HTTPException(status_code=404, detail=f"No service found for path: {full_path}")
    
    # Use the full path as target path for backward compatibility
    return await proxy_request(request, service_name, path)


@app.get("/services")
async def list_services():
    """List all available services"""
    if not service_discovery:
        raise HTTPException(status_code=503, detail="Service discovery not available")
    
    services = {}
    for service_name in set(SERVICE_ROUTES.values()):
        try:
            service_info = await service_discovery.get_service_instance(service_name)
            if service_info:
                services[service_name] = {
                    "status": "available",
                    "instance": service_info['instance'],
                    "routes": [route for route, svc in SERVICE_ROUTES.items() if svc == service_name]
                }
            else:
                services[service_name] = {
                    "status": "unavailable",
                    "routes": [route for route, svc in SERVICE_ROUTES.items() if svc == service_name]
                }
        except Exception as e:
            services[service_name] = {
                "status": "error",
                "error": str(e),
                "routes": [route for route, svc in SERVICE_ROUTES.items() if svc == service_name]
            }
    
    return {"services": services}


@app.get("/routes")
async def list_routes():
    """List all available routes"""
    return {
        "routes": SERVICE_ROUTES,
        "description": "Map of API routes to microservices"
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


# Aggregated endpoints that combine multiple services
@app.get("/api/v1/status/overview")
async def get_system_overview():
    """Get system-wide status overview"""
    if not service_discovery:
        raise HTTPException(status_code=503, detail="Service discovery not available")
    
    overview = {
        "timestamp": time.time(),
        "services": {},
        "system_health": "unknown"
    }
    
    healthy_services = 0
    total_services = 0
    
    for service_name in set(SERVICE_ROUTES.values()):
        total_services += 1
        try:
            service_info = await service_discovery.get_service_instance(service_name)
            if service_info:
                # Try to get health status from the service
                instance = service_info['instance']
                health_url = f"{instance['url']}/health"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            overview['services'][service_name] = {
                                "status": "healthy",
                                "health": health_data
                            }
                            healthy_services += 1
                        else:
                            overview['services'][service_name] = {"status": "unhealthy"}
            else:
                overview['services'][service_name] = {"status": "unavailable"}
        except Exception as e:
            overview['services'][service_name] = {"status": "error", "error": str(e)}
    
    # Determine overall system health
    if healthy_services == total_services:
        overview['system_health'] = "healthy"
    elif healthy_services > total_services / 2:
        overview['system_health'] = "degraded"
    else:
        overview['system_health'] = "unhealthy"
    
    overview['service_summary'] = {
        "total": total_services,
        "healthy": healthy_services,
        "unhealthy": total_services - healthy_services
    }
    
    return overview


@app.post("/api/v1/batch")
async def batch_requests(requests: List[Dict[str, Any]]):
    """Execute multiple requests in batch"""
    results = []
    
    for req_data in requests:
        try:
            method = req_data.get('method', 'GET')
            path = req_data.get('path', '')
            data = req_data.get('data', {})
            
            # This is a simplified batch implementation
            # In production, you'd want more sophisticated batching
            service_name = get_service_name(path)
            if service_name:
                # Execute the request (simplified)
                result = {"path": path, "status": "queued"}
            else:
                result = {"path": path, "status": "invalid_path"}
            
            results.append(result)
        except Exception as e:
            results.append({"path": req_data.get('path', 'unknown'), "status": "error", "error": str(e)})
    
    return {"batch_results": results}


if __name__ == "__main__":
    # Run the API Gateway
    port = int(os.getenv('SERVICE_PORT', '8000'))
    host = os.getenv('SERVICE_HOST', '0.0.0.0')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )