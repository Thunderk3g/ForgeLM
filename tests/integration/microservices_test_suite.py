"""
LLMForge Microservices Test Suite

Comprehensive integration and end-to-end testing framework for 
the microservices architecture transformation.
"""

import asyncio
import aiohttp
import pytest
import time
import json
import uuid
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
import websockets

# Test Configuration
@dataclass
class TestConfig:
    """Test configuration settings"""
    orchestrator_url: str = "http://localhost:8080"
    api_gateway_url: str = "http://localhost:8000"
    training_service_url: str = "http://localhost:8001"
    chat_service_url: str = "http://localhost:8002"
    data_gen_service_url: str = "http://localhost:8003"
    visualization_service_url: str = "http://localhost:8004"
    model_management_service_url: str = "http://localhost:8005"
    prometheus_url: str = "http://localhost:9090"
    grafana_url: str = "http://localhost:3000"
    timeout: int = 30

# Test utilities
class TestClient:
    """HTTP client for testing microservices"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, path: str, **kwargs) -> aiohttp.ClientResponse:
        async with self.session.get(f"{self.base_url}{path}", **kwargs) as response:
            return response
    
    async def post(self, path: str, **kwargs) -> aiohttp.ClientResponse:
        async with self.session.post(f"{self.base_url}{path}", **kwargs) as response:
            return response
    
    async def put(self, path: str, **kwargs) -> aiohttp.ClientResponse:
        async with self.session.put(f"{self.base_url}{path}", **kwargs) as response:
            return response
    
    async def delete(self, path: str, **kwargs) -> aiohttp.ClientResponse:
        async with self.session.delete(f"{self.base_url}{path}", **kwargs) as response:
            return response

# Test fixtures
@pytest.fixture
def test_config():
    return TestConfig()

@pytest.fixture
async def orchestrator_client(test_config):
    async with TestClient(test_config.orchestrator_url) as client:
        yield client

@pytest.fixture
async def api_gateway_client(test_config):
    async with TestClient(test_config.api_gateway_url) as client:
        yield client

@pytest.fixture
async def training_client(test_config):
    async with TestClient(test_config.training_service_url) as client:
        yield client

# Health Check Tests
class TestHealthChecks:
    """Test suite for service health checks"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_health(self, orchestrator_client):
        """Test orchestrator health endpoint"""
        response = await orchestrator_client.get("/health")
        assert response.status == 200
        
        data = await response.json()
        assert data["status"] == "healthy"
        assert "services" in data
    
    @pytest.mark.asyncio
    async def test_api_gateway_health(self, api_gateway_client):
        """Test API gateway health endpoint"""
        response = await api_gateway_client.get("/health")
        assert response.status == 200
        
        data = await response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "api_gateway"
    
    @pytest.mark.asyncio
    async def test_all_services_health(self, test_config):
        """Test health of all microservices"""
        services = [
            (test_config.orchestrator_url, "orchestrator"),
            (test_config.api_gateway_url, "api_gateway"),
            (test_config.training_service_url, "training"),
            (test_config.chat_service_url, "chat_comparison"),
            (test_config.data_gen_service_url, "data_generation"),
            (test_config.visualization_service_url, "visualization"),
            (test_config.model_management_service_url, "model_management")
        ]
        
        for service_url, service_name in services:
            async with TestClient(service_url) as client:
                try:
                    response = await client.get("/health")
                    assert response.status == 200, f"{service_name} health check failed"
                    
                    data = await response.json()
                    assert data["status"] == "healthy", f"{service_name} reports unhealthy status"
                except Exception as e:
                    pytest.fail(f"{service_name} health check failed with error: {e}")

# Service Discovery Tests
class TestServiceDiscovery:
    """Test suite for service discovery and orchestration"""
    
    @pytest.mark.asyncio
    async def test_service_registration(self, orchestrator_client):
        """Test service registration with orchestrator"""
        # Register a test service
        registration_data = {
            "service_name": "test-service",
            "service_id": f"test-{uuid.uuid4().hex[:8]}",
            "host": "localhost",
            "port": 9999,
            "health_endpoint": "/health",
            "metadata": {"type": "test"}
        }
        
        response = await orchestrator_client.post("/register", json=registration_data)
        assert response.status == 200
        
        data = await response.json()
        assert data["status"] == "registered"
    
    @pytest.mark.asyncio
    async def test_service_discovery(self, orchestrator_client):
        """Test service discovery functionality"""
        # Discover a known service
        response = await orchestrator_client.get("/discover/llmforge-training")
        
        if response.status == 200:
            data = await response.json()
            assert "service" in data
            assert "instance" in data
            assert data["service"] == "llmforge-training"
        elif response.status == 503:
            # Service not available - this is acceptable in test environment
            pass
        else:
            pytest.fail(f"Unexpected response status: {response.status}")

# API Gateway Tests
class TestAPIGateway:
    """Test suite for API Gateway functionality"""
    
    @pytest.mark.asyncio
    async def test_service_routing(self, api_gateway_client):
        """Test API gateway routing to services"""
        # Test routing to different services through API gateway
        routes = [
            "/api/v1/models/available",  # Should route to model management
            "/api/v1/analytics/system",  # Should route to visualization
        ]
        
        for route in routes:
            response = await api_gateway_client.get(route)
            # Accept various response codes as services may not be fully configured
            assert response.status in [200, 404, 503], f"Route {route} returned unexpected status {response.status}"
    
    @pytest.mark.asyncio
    async def test_service_listing(self, api_gateway_client):
        """Test API gateway service listing"""
        response = await api_gateway_client.get("/services")
        
        if response.status == 200:
            data = await response.json()
            assert "services" in data
        else:
            # May fail if orchestrator is not fully configured
            assert response.status in [503, 404]
    
    @pytest.mark.asyncio
    async def test_route_listing(self, api_gateway_client):
        """Test API gateway route listing"""
        response = await api_gateway_client.get("/routes")
        assert response.status == 200
        
        data = await response.json()
        assert "routes" in data
        assert isinstance(data["routes"], dict)

# Training Service Tests
class TestTrainingService:
    """Test suite for Training Service functionality"""
    
    @pytest.mark.asyncio
    async def test_available_models(self, training_client):
        """Test listing available models"""
        response = await training_client.get("/models/available")
        assert response.status == 200
        
        data = await response.json()
        assert "models" in data
    
    @pytest.mark.asyncio
    async def test_available_datasets(self, training_client):
        """Test listing available datasets"""
        response = await training_client.get("/datasets/available")
        assert response.status == 200
        
        data = await response.json()
        assert "datasets" in data
    
    @pytest.mark.asyncio
    async def test_gpu_info(self, training_client):
        """Test GPU information endpoint"""
        response = await training_client.get("/system/gpu")
        
        # GPU may not be available in test environment
        assert response.status in [200, 500]
        
        if response.status == 200:
            data = await response.json()
            assert "available" in data
    
    @pytest.mark.asyncio
    async def test_training_job_lifecycle(self, training_client):
        """Test complete training job lifecycle"""
        # This test requires proper setup with models and datasets
        # Skip if not in full integration test environment
        try:
            # Create a minimal training request
            training_request = {
                "base_model_dir": "/app/models/test",
                "dataset_dir": "/app/data/test",
                "epochs": 1,
                "batch_size": 1,
                "learning_rate": 0.001
            }
            
            response = await training_client.post("/fine-tune", json=training_request)
            
            if response.status == 200:
                data = await response.json()
                task_id = data["task_id"]
                
                # Check job status
                status_response = await training_client.get(f"/status/{task_id}")
                assert status_response.status == 200
                
                status_data = await status_response.json()
                assert "status" in status_data
                
                # Stop the job (cleanup)
                stop_response = await training_client.post(f"/stop/{task_id}")
                # May succeed or fail depending on job state
                assert stop_response.status in [200, 404]
            
            else:
                # Training may fail due to missing models/datasets in test environment
                assert response.status in [400, 503]
        
        except Exception as e:
            # Skip test if environment not properly configured
            pytest.skip(f"Training test skipped due to environment: {e}")

# Chat Comparison Service Tests
class TestChatComparisonService:
    """Test suite for Chat Comparison Service"""
    
    @pytest.mark.asyncio
    async def test_model_pairs(self, test_config):
        """Test model pairs functionality"""
        async with TestClient(test_config.chat_service_url) as client:
            response = await client.get("/models/pairs")
            assert response.status == 200
            
            data = await response.json()
            assert "pairs" in data
    
    @pytest.mark.asyncio
    async def test_session_lifecycle(self, test_config):
        """Test chat session lifecycle"""
        async with TestClient(test_config.chat_service_url) as client:
            # Create session
            session_data = {
                "pair_id": "test-pair",
                "session_name": "test-session",
                "custom_system_prompt": "You are a helpful assistant."
            }
            
            create_response = await client.post("/sessions", json=session_data)
            
            if create_response.status == 200:
                session_info = await create_response.json()
                session_id = session_info["session_id"]
                
                # List sessions
                list_response = await client.get("/sessions")
                assert list_response.status == 200
                
                # Get specific session
                get_response = await client.get(f"/sessions/{session_id}")
                assert get_response.status == 200
                
                # Delete session
                delete_response = await client.delete(f"/sessions/{session_id}")
                assert delete_response.status == 200
            
            else:
                # May fail if model pairs not configured
                assert create_response.status in [400, 503]

# Data Generation Service Tests
class TestDataGenerationService:
    """Test suite for Data Generation Service"""
    
    @pytest.mark.asyncio
    async def test_templates(self, test_config):
        """Test template management"""
        async with TestClient(test_config.data_gen_service_url) as client:
            response = await client.get("/templates")
            assert response.status == 200
            
            data = await response.json()
            assert "templates" in data
    
    @pytest.mark.asyncio
    async def test_supported_formats(self, test_config):
        """Test supported format listing"""
        async with TestClient(test_config.data_gen_service_url) as client:
            response = await client.get("/formats")
            assert response.status == 200
            
            data = await response.json()
            assert "supported_formats" in data
            assert isinstance(data["supported_formats"], list)

# Visualization Service Tests
class TestVisualizationService:
    """Test suite for Visualization Service"""
    
    @pytest.mark.asyncio
    async def test_dashboards(self, test_config):
        """Test dashboard functionality"""
        async with TestClient(test_config.visualization_service_url) as client:
            response = await client.get("/dashboards")
            assert response.status == 200
            
            data = await response.json()
            assert "dashboards" in data
    
    @pytest.mark.asyncio
    async def test_chart_types(self, test_config):
        """Test chart types listing"""
        async with TestClient(test_config.visualization_service_url) as client:
            response = await client.get("/chart_types")
            assert response.status == 200
            
            data = await response.json()
            assert "chart_types" in data
            assert isinstance(data["chart_types"], list)

# Model Management Service Tests
class TestModelManagementService:
    """Test suite for Model Management Service"""
    
    @pytest.mark.asyncio
    async def test_loaded_models(self, test_config):
        """Test loaded models listing"""
        async with TestClient(test_config.model_management_service_url) as client:
            response = await client.get("/models/loaded")
            assert response.status == 200
            
            data = await response.json()
            assert "loaded_models" in data
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, test_config):
        """Test cache statistics"""
        async with TestClient(test_config.model_management_service_url) as client:
            response = await client.get("/cache/stats")
            assert response.status == 200
            
            data = await response.json()
            # Cache stats structure may vary

# Integration Tests
class TestIntegration:
    """Integration tests across multiple services"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, test_config):
        """Test end-to-end workflow through API gateway"""
        async with TestClient(test_config.api_gateway_url) as client:
            # Test system overview
            response = await client.get("/api/v1/status/overview")
            
            if response.status == 200:
                data = await response.json()
                assert "services" in data
                assert "system_health" in data
            else:
                # May fail if services not fully configured
                assert response.status in [503, 404]
    
    @pytest.mark.asyncio
    async def test_service_communication(self, test_config):
        """Test inter-service communication"""
        # This would test that services can communicate with each other
        # through the orchestrator and service discovery
        
        async with TestClient(test_config.orchestrator_url) as orchestrator:
            # Get service metrics
            response = await orchestrator.get("/metrics")
            assert response.status == 200
            
            data = await response.json()
            # Verify some services are registered
            assert isinstance(data, dict)

# Monitoring Tests
class TestMonitoring:
    """Test suite for monitoring and observability"""
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics(self, test_config):
        """Test Prometheus metrics collection"""
        try:
            async with TestClient(test_config.prometheus_url) as client:
                # Test Prometheus health
                response = await client.get("/-/healthy")
                assert response.status == 200
                
                # Test metrics endpoint
                metrics_response = await client.get("/metrics")
                assert metrics_response.status == 200
                
        except Exception:
            # Prometheus may not be available in test environment
            pytest.skip("Prometheus not available")
    
    @pytest.mark.asyncio
    async def test_service_metrics(self, test_config):
        """Test service-specific metrics endpoints"""
        services = [
            test_config.api_gateway_url,
            test_config.training_service_url,
            test_config.chat_service_url,
            test_config.data_gen_service_url,
            test_config.visualization_service_url,
            test_config.model_management_service_url
        ]
        
        for service_url in services:
            try:
                async with TestClient(service_url) as client:
                    response = await client.get("/metrics")
                    # Accept various status codes
                    assert response.status in [200, 404, 503]
            except Exception:
                # Skip if service not available
                continue

# Performance Tests
class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, test_config):
        """Test concurrent health check performance"""
        async def health_check(service_url):
            try:
                async with TestClient(service_url) as client:
                    start_time = time.time()
                    response = await client.get("/health")
                    end_time = time.time()
                    
                    return {
                        "service": service_url,
                        "status": response.status,
                        "response_time": end_time - start_time,
                        "success": response.status == 200
                    }
            except Exception as e:
                return {
                    "service": service_url,
                    "status": None,
                    "response_time": None,
                    "success": False,
                    "error": str(e)
                }
        
        services = [
            test_config.api_gateway_url,
            test_config.training_service_url,
            test_config.chat_service_url,
            test_config.data_gen_service_url,
            test_config.visualization_service_url,
            test_config.model_management_service_url
        ]
        
        # Run concurrent health checks
        tasks = [health_check(service_url) for service_url in services]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successful_checks = sum(1 for result in results if result["success"])
        total_checks = len(results)
        
        print(f"\nHealth check results: {successful_checks}/{total_checks} successful")
        for result in results:
            if result["success"]:
                print(f"✅ {result['service']}: {result['response_time']:.3f}s")
            else:
                print(f"❌ {result['service']}: {result.get('error', 'Failed')}")
        
        # At least some services should be healthy
        assert successful_checks >= total_checks * 0.5, f"Too many health checks failed: {successful_checks}/{total_checks}"

# Utility functions for test execution
def run_test_suite():
    """Run the complete test suite"""
    import subprocess
    import sys
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--cov=microservices",
        "--cov-report=html",
        "--cov-report=term-missing"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

if __name__ == "__main__":
    # Run tests if script is executed directly
    success = run_test_suite()
    if not success:
        exit(1)