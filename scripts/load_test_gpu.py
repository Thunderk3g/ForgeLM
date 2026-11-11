"""
Load Testing Suite for MPS-enabled GPU Services

Tests concurrent requests, performance under sustained load,
response time consistency, and error rate monitoring.

Usage, against a stack started with `docker-compose up -d`:
    python scripts/load_test_gpu.py [basic|intensive|endurance]
It needs aiohttp (`pip install aiohttp`), which pyproject.toml does not declare.
It targets http://localhost on ports 8001 (training), 8002 (chat comparison)
and 8005 (model management): /health, /system/gpu and /gpu/info.
"""

import asyncio
import aiohttp
import time
import statistics
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    concurrent_users: int = 10
    requests_per_user: int = 20
    ramp_up_seconds: int = 30
    test_duration_seconds: int = 300  # 5 minutes
    error_rate_threshold: float = 0.05  # 5%
    response_time_threshold_ms: int = 5000
    target_rps: int = 50  # requests per second

@dataclass 
class RequestResult:
    """Result of a single request"""
    service: str
    endpoint: str
    status_code: int
    response_time_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LoadTestResults:
    """Aggregated load test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    requests_per_second: float
    duration_seconds: float
    errors_by_service: Dict[str, int]
    response_times_by_service: Dict[str, List[float]]

class GPULoadTester:
    """Load tester for GPU services"""
    
    def __init__(self, base_url: str = "http://localhost", config: LoadTestConfig = None):
        self.base_url = base_url
        self.config = config or LoadTestConfig()
        self.results: List[RequestResult] = []
        self.sessions: List[aiohttp.ClientSession] = []
        
        # Test endpoints for each service
        self.test_endpoints = {
            "training": [
                ("/health", "GET"),
                ("/models/available", "GET"),
                ("/system/gpu", "GET"),
                ("/jobs", "GET"),
                ("/system/hardware", "GET")
            ],
            "chat-comparison": [
                ("/health", "GET"),
                ("/models/pairs", "GET"),
                ("/cache/stats", "GET"),
                ("/sessions", "GET")
            ],
            "model-management": [
                ("/health", "GET"),
                ("/models/available", "GET"),
                ("/models/loaded", "GET"),
                ("/cache/stats", "GET"),
                ("/gpu/info", "GET"),
                ("/storage/stats", "GET")
            ]
        }
        
        self.service_ports = {
            "training": 8001,
            "chat-comparison": 8002,
            "model-management": 8005
        }
    
    async def __aenter__(self):
        """Create HTTP sessions for concurrent testing"""
        timeout = aiohttp.ClientTimeout(total=self.config.response_time_threshold_ms / 1000)
        
        for _ in range(self.config.concurrent_users):
            session = aiohttp.ClientSession(timeout=timeout)
            self.sessions.append(session)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close all HTTP sessions"""
        for session in self.sessions:
            await session.close()
    
    async def make_request(self, session: aiohttp.ClientSession, service: str, 
                          endpoint: str, method: str = "GET") -> RequestResult:
        """Make a single request and record results"""
        port = self.service_ports[service]
        url = f"{self.base_url}:{port}{endpoint}"
        
        start_time = time.time()
        
        try:
            if method == "GET":
                async with session.get(url) as response:
                    await response.text()  # Read response body
                    end_time = time.time()
                    
                    response_time_ms = (end_time - start_time) * 1000
                    success = 200 <= response.status < 300
                    
                    return RequestResult(
                        service=service,
                        endpoint=endpoint,
                        status_code=response.status,
                        response_time_ms=response_time_ms,
                        success=success,
                        error=None if success else f"HTTP {response.status}"
                    )
            else:
                # Add support for POST requests if needed
                raise NotImplementedError(f"Method {method} not implemented")
                
        except asyncio.TimeoutError:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return RequestResult(
                service=service,
                endpoint=endpoint,
                status_code=0,
                response_time_ms=response_time_ms,
                success=False,
                error="Timeout"
            )
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return RequestResult(
                service=service,
                endpoint=endpoint,
                status_code=0,
                response_time_ms=response_time_ms,
                success=False,
                error=str(e)
            )
    
    async def user_simulation(self, user_id: int, session: aiohttp.ClientSession):
        """Simulate a single user's load testing behavior"""
        user_results = []
        
        # Stagger user start times for ramp-up
        delay = (user_id / self.config.concurrent_users) * self.config.ramp_up_seconds
        await asyncio.sleep(delay)
        
        logger.info(f"User {user_id} starting load test")
        
        # Run requests for the configured duration
        start_time = time.time()
        request_count = 0
        
        while (time.time() - start_time) < self.config.test_duration_seconds:
            # Select random service and endpoint
            service = random.choice(list(self.service_ports.keys()))
            endpoint_info = random.choice(self.test_endpoints[service])
            endpoint, method = endpoint_info
            
            # Make the request
            result = await self.make_request(session, service, endpoint, method)
            user_results.append(result)
            request_count += 1
            
            # Rate limiting to achieve target RPS
            if self.config.target_rps > 0:
                expected_interval = 1.0 / (self.config.target_rps / self.config.concurrent_users)
                await asyncio.sleep(max(0, expected_interval - 0.1))  # Small buffer
        
        logger.info(f"User {user_id} completed {request_count} requests")
        return user_results
    
    async def run_load_test(self) -> LoadTestResults:
        """Run the complete load test"""
        logger.info(f"Starting load test with {self.config.concurrent_users} concurrent users")
        logger.info(f"Test duration: {self.config.test_duration_seconds} seconds")
        logger.info(f"Target RPS: {self.config.target_rps}")
        
        start_time = time.time()
        
        # Create tasks for all users
        tasks = []
        for i in range(self.config.concurrent_users):
            session = self.sessions[i % len(self.sessions)]
            task = asyncio.create_task(self.user_simulation(i, session))
            tasks.append(task)
        
        # Wait for all users to complete
        user_results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Flatten results
        all_results = []
        for user_results in user_results_lists:
            if isinstance(user_results, list):
                all_results.extend(user_results)
            else:
                logger.error(f"User task failed: {user_results}")
        
        self.results = all_results
        
        # Calculate aggregated results
        return self._calculate_results(total_duration)
    
    def _calculate_results(self, duration_seconds: float) -> LoadTestResults:
        """Calculate aggregated test results"""
        if not self.results:
            return LoadTestResults(
                total_requests=0, successful_requests=0, failed_requests=0,
                error_rate=1.0, avg_response_time_ms=0, median_response_time_ms=0,
                p95_response_time_ms=0, p99_response_time_ms=0,
                min_response_time_ms=0, max_response_time_ms=0,
                requests_per_second=0, duration_seconds=duration_seconds,
                errors_by_service={}, response_times_by_service={}
            )
        
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        error_rate = failed_requests / total_requests if total_requests > 0 else 1.0
        
        # Response time statistics
        response_times = [r.response_time_ms for r in self.results]
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        
        sorted_times = sorted(response_times)
        p95_index = int(0.95 * len(sorted_times))
        p99_index = int(0.99 * len(sorted_times))
        
        p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
        p99_response_time = sorted_times[min(p99_index, len(sorted_times) - 1)]
        
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        requests_per_second = total_requests / duration_seconds if duration_seconds > 0 else 0
        
        # Errors by service
        errors_by_service = {}
        response_times_by_service = {}
        
        for service in self.service_ports.keys():
            service_results = [r for r in self.results if r.service == service]
            errors_by_service[service] = sum(1 for r in service_results if not r.success)
            response_times_by_service[service] = [r.response_time_ms for r in service_results]
        
        return LoadTestResults(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_rate=error_rate,
            avg_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            requests_per_second=requests_per_second,
            duration_seconds=duration_seconds,
            errors_by_service=errors_by_service,
            response_times_by_service=response_times_by_service
        )
    
    def validate_results(self, results: LoadTestResults) -> Dict[str, bool]:
        """Validate results against success criteria"""
        validations = {
            "error_rate_acceptable": results.error_rate <= self.config.error_rate_threshold,
            "response_time_acceptable": results.p95_response_time_ms <= self.config.response_time_threshold_ms,
            "requests_completed": results.total_requests > 0,
            "some_requests_successful": results.successful_requests > 0
        }
        
        return validations


async def run_basic_load_test():
    """Run basic load test with default configuration"""
    config = LoadTestConfig(
        concurrent_users=5,
        test_duration_seconds=60,  # 1 minute test
        target_rps=20
    )
    
    async with GPULoadTester(config=config) as tester:
        results = await tester.run_load_test()
        
        print("\n" + "="*60)
        print("BASIC LOAD TEST RESULTS")
        print("="*60)
        print(f"Total Requests: {results.total_requests}")
        print(f"Successful: {results.successful_requests}")
        print(f"Failed: {results.failed_requests}")
        print(f"Error Rate: {results.error_rate:.2%}")
        print(f"Average Response Time: {results.avg_response_time_ms:.1f}ms")
        print(f"95th Percentile: {results.p95_response_time_ms:.1f}ms")
        print(f"Requests/Second: {results.requests_per_second:.1f}")
        
        print("\nErrors by Service:")
        for service, error_count in results.errors_by_service.items():
            total_service_requests = len(results.response_times_by_service.get(service, []))
            error_rate = error_count / total_service_requests if total_service_requests > 0 else 0
            print(f"  {service}: {error_count}/{total_service_requests} ({error_rate:.1%})")
        
        # Validate results
        validations = tester.validate_results(results)
        print("\nValidation Results:")
        for check, passed in validations.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")
        
        return results, validations


async def run_intensive_load_test():
    """Run intensive load test for stress testing"""
    config = LoadTestConfig(
        concurrent_users=20,
        test_duration_seconds=300,  # 5 minutes
        target_rps=100,
        ramp_up_seconds=60
    )
    
    async with GPULoadTester(config=config) as tester:
        results = await tester.run_load_test()
        
        print("\n" + "="*60)
        print("INTENSIVE LOAD TEST RESULTS")
        print("="*60)
        print(f"Total Requests: {results.total_requests}")
        print(f"Successful: {results.successful_requests}")
        print(f"Failed: {results.failed_requests}")
        print(f"Error Rate: {results.error_rate:.2%}")
        print(f"Average Response Time: {results.avg_response_time_ms:.1f}ms")
        print(f"Median Response Time: {results.median_response_time_ms:.1f}ms")
        print(f"95th Percentile: {results.p95_response_time_ms:.1f}ms")
        print(f"99th Percentile: {results.p99_response_time_ms:.1f}ms")
        print(f"Min Response Time: {results.min_response_time_ms:.1f}ms")
        print(f"Max Response Time: {results.max_response_time_ms:.1f}ms")
        print(f"Requests/Second: {results.requests_per_second:.1f}")
        print(f"Test Duration: {results.duration_seconds:.1f}s")
        
        print("\nDetailed Results by Service:")
        for service, times in results.response_times_by_service.items():
            if times:
                avg_time = statistics.mean(times)
                median_time = statistics.median(times)
                error_count = results.errors_by_service.get(service, 0)
                error_rate = error_count / len(times) if times else 0
                
                print(f"  {service}:")
                print(f"    Requests: {len(times)}")
                print(f"    Errors: {error_count} ({error_rate:.1%})")
                print(f"    Avg Response: {avg_time:.1f}ms")
                print(f"    Median Response: {median_time:.1f}ms")
        
        # Validate results
        validations = tester.validate_results(results)
        print("\nValidation Results:")
        for check, passed in validations.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")
        
        # Overall assessment
        overall_pass = all(validations.values())
        print(f"\nOVERALL ASSESSMENT: {'PASS' if overall_pass else 'FAIL'}")
        
        return results, validations


async def run_endurance_test():
    """Run endurance test with consistent load"""
    config = LoadTestConfig(
        concurrent_users=10,
        test_duration_seconds=1800,  # 30 minutes
        target_rps=30,
        ramp_up_seconds=120
    )
    
    print(f"Starting 30-minute endurance test...")
    print(f"This will run {config.concurrent_users} concurrent users for {config.test_duration_seconds/60:.1f} minutes")
    
    async with GPULoadTester(config=config) as tester:
        results = await tester.run_load_test()
        
        print("\n" + "="*60)
        print("ENDURANCE TEST RESULTS")
        print("="*60)
        print(f"Test Duration: {results.duration_seconds/60:.1f} minutes")
        print(f"Total Requests: {results.total_requests}")
        print(f"Error Rate: {results.error_rate:.2%}")
        print(f"Average Response Time: {results.avg_response_time_ms:.1f}ms")
        print(f"95th Percentile: {results.p95_response_time_ms:.1f}ms")
        print(f"Requests/Second: {results.requests_per_second:.1f}")
        
        # Check for performance degradation over time
        # (This would require time-series analysis in a more advanced implementation)
        
        validations = tester.validate_results(results)
        overall_pass = all(validations.values())
        print(f"\nENDURANCE TEST: {'PASS' if overall_pass else 'FAIL'}")
        
        return results, validations


if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        if len(sys.argv) > 1:
            test_type = sys.argv[1].lower()
            
            if test_type == "basic":
                await run_basic_load_test()
            elif test_type == "intensive":
                await run_intensive_load_test()
            elif test_type == "endurance":
                await run_endurance_test()
            else:
                print("Unknown test type. Use: basic, intensive, or endurance")
        else:
            print("Running basic load test...")
            await run_basic_load_test()
    
    asyncio.run(main())