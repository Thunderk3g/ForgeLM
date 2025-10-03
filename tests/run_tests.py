"""
Test Runner for LLMForge MPS GPU Testing Suite

Orchestrates execution of all test suites and provides comprehensive reporting.
"""

import asyncio
import sys
import logging
import time
from datetime import datetime
from typing import Dict, Any, List
import json
import os

# Import test modules
from test_gpu_services import GPUServiceTester
from load_test_gpu import GPULoadTester, LoadTestConfig
from stress_test import GPUStressTester, StressTestConfig
from memory_leak_test import MemoryStressTester
from test_mps_functionality import MPSTester

logger = logging.getLogger(__name__)

class TestSuiteRunner:
    """Orchestrates execution of all test suites"""
    
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for GPU services"""
        logger.info("Running Integration Tests...")
        
        async with GPUServiceTester(self.base_url) as tester:
            results = {
                "health_checks": [],
                "gpu_accessibility": [],
                "concurrent_usage": None,
                "memory_allocation": [],
                "response_times": {}
            }
            
            # Health checks for all services
            from test_gpu_services import GPU_SERVICES
            for service in GPU_SERVICES:
                health_result = await tester.health_check(service)
                results["health_checks"].append(health_result)
            
            # GPU accessibility verification
            for service in GPU_SERVICES:
                gpu_result = await tester.check_gpu_info(service)
                results["gpu_accessibility"].append(gpu_result)
            
            # Concurrent usage test
            concurrent_tasks = []
            for service in GPU_SERVICES:
                task = asyncio.create_task(tester.test_gpu_computation(service))
                concurrent_tasks.append(task)
            
            concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            results["concurrent_usage"] = {
                "services_tested": len(GPU_SERVICES),
                "successful": sum(1 for r in concurrent_results if not isinstance(r, Exception) and r.get("status") == "success"),
                "results": concurrent_results
            }
            
            # Response time testing
            for service in GPU_SERVICES:
                start_time = time.time()
                health_result = await tester.health_check(service)
                response_time = (time.time() - start_time) * 1000
                results["response_times"][service.name] = response_time
            
            return results
    
    async def run_load_tests(self) -> Dict[str, Any]:
        """Run load tests"""
        logger.info("Running Load Tests...")
        
        # Basic load test
        basic_config = LoadTestConfig(
            concurrent_users=5,
            test_duration_seconds=60,
            target_rps=20
        )
        
        async with GPULoadTester(self.base_url, basic_config) as tester:
            basic_results = await tester.run_load_test()
            basic_validations = tester.validate_results(basic_results)
        
        # Intensive load test
        intensive_config = LoadTestConfig(
            concurrent_users=15,
            test_duration_seconds=120,
            target_rps=50
        )
        
        async with GPULoadTester(self.base_url, intensive_config) as tester:
            intensive_results = await tester.run_load_test()
            intensive_validations = tester.validate_results(intensive_results)
        
        return {
            "basic_load_test": {
                "results": basic_results,
                "validations": basic_validations,
                "passed": all(basic_validations.values())
            },
            "intensive_load_test": {
                "results": intensive_results,
                "validations": intensive_validations,
                "passed": all(intensive_validations.values())
            }
        }
    
    async def run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests"""
        logger.info("Running Stress Tests...")
        
        config = StressTestConfig(
            test_duration_minutes=10,  # Shorter for automated testing
            max_concurrent_users=20,
            ramp_up_minutes=2,
            memory_pressure_enabled=True
        )
        
        tester = GPUStressTester(self.base_url, config)
        results = await tester.run_stress_test()
        
        return {
            "results": results,
            "passed": results.stability_score >= 60,  # 60% threshold for automated tests
            "assessment": "STABLE" if results.stability_score >= 80 else "MODERATE" if results.stability_score >= 60 else "UNSTABLE"
        }
    
    async def run_memory_tests(self) -> Dict[str, Any]:
        """Run memory leak detection tests"""
        logger.info("Running Memory Tests...")
        
        tester = MemoryStressTester(self.base_url)
        reports = await tester.run_memory_leak_test(test_duration_minutes=10)
        
        # Assess overall memory health
        total_leaks = sum(1 for report in reports.values() if report.leak_detected)
        memory_health = "GOOD" if total_leaks == 0 else "MODERATE" if total_leaks <= 1 else "POOR"
        
        return {
            "reports": reports,
            "total_services_tested": len(reports),
            "services_with_leaks": total_leaks,
            "memory_health": memory_health,
            "passed": total_leaks <= 1  # Allow 1 minor leak
        }
    
    async def run_mps_tests(self) -> Dict[str, Any]:
        """Run MPS-specific tests"""
        logger.info("Running MPS Tests...")
        
        tester = MPSTester(self.base_url)
        
        # MPS daemon status
        mps_status = await tester.check_mps_daemon_status()
        
        # Directory checks
        dir_status = await tester.check_mps_directories()
        
        # Concurrent access test
        concurrent_result = await tester.test_concurrent_gpu_access()
        
        # Resource isolation test (abbreviated)
        isolation_result = await tester.test_resource_isolation()
        
        # Memory allocation test
        allocation_result = await tester.test_mps_memory_allocation()
        
        # Calculate MPS health score
        scores = {
            "daemon_running": mps_status.daemon_running,
            "concurrent_access": concurrent_result["concurrent_access_possible"],
            "all_services_accessible": concurrent_result["all_services_accessible"],
            "resource_isolation": not any(r.get("other_services_affected", False) for r in isolation_result.values())
        }
        
        mps_score = sum(scores.values()) / len(scores) * 100
        
        return {
            "mps_status": mps_status,
            "directory_status": dir_status,
            "concurrent_access": concurrent_result,
            "resource_isolation": isolation_result,
            "memory_allocation": allocation_result,
            "mps_score": mps_score,
            "passed": mps_score >= 50,  # 50% threshold for MPS functionality
            "assessment": "EXCELLENT" if mps_score >= 75 else "GOOD" if mps_score >= 50 else "NEEDS_ATTENTION"
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        self.start_time = datetime.now()
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE GPU TESTING SUITE")
        logger.info("="*60)
        
        try:
            # Run all test suites
            self.results["integration"] = await self.run_integration_tests()
            self.results["load"] = await self.run_load_tests()
            self.results["stress"] = await self.run_stress_tests()
            self.results["memory"] = await self.run_memory_tests()
            self.results["mps"] = await self.run_mps_tests()
            
            self.end_time = datetime.now()
            test_duration = (self.end_time - self.start_time).total_seconds() / 60
            
            # Calculate overall assessment
            test_results = {
                "integration_passed": self._assess_integration_tests(),
                "load_passed": self.results["load"]["basic_load_test"]["passed"] and self.results["load"]["intensive_load_test"]["passed"],
                "stress_passed": self.results["stress"]["passed"],
                "memory_passed": self.results["memory"]["passed"],
                "mps_passed": self.results["mps"]["passed"]
            }
            
            overall_score = sum(test_results.values()) / len(test_results) * 100
            
            self.results["summary"] = {
                "test_duration_minutes": test_duration,
                "individual_results": test_results,
                "overall_score": overall_score,
                "overall_passed": overall_score >= 70,  # 70% threshold for overall pass
                "assessment": self._get_overall_assessment(overall_score),
                "recommendations": self._get_recommendations()
            }
            
            return self.results
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            self.results["error"] = str(e)
            return self.results
    
    def _assess_integration_tests(self) -> bool:
        """Assess integration test results"""
        integration = self.results["integration"]
        
        # Check health
        healthy_services = sum(1 for h in integration["health_checks"] if h.get("status") == "healthy")
        
        # Check GPU accessibility
        gpu_accessible = sum(1 for g in integration["gpu_accessibility"] if g.get("status") == "success")
        
        # Check concurrent usage
        concurrent_ok = integration["concurrent_usage"]["successful"] > 0
        
        return healthy_services >= 2 and gpu_accessible >= 1 and concurrent_ok
    
    def _get_overall_assessment(self, score: float) -> str:
        """Get overall assessment based on score"""
        if score >= 90:
            return "EXCELLENT - All systems fully operational"
        elif score >= 80:
            return "VERY_GOOD - Minor issues detected"
        elif score >= 70:
            return "GOOD - Some issues require attention"
        elif score >= 50:
            return "MODERATE - Significant issues detected"
        else:
            return "POOR - Major issues require immediate attention"
    
    def _get_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Integration recommendations
        if not self._assess_integration_tests():
            recommendations.append("Check service health and GPU accessibility")
        
        # Load test recommendations
        if not self.results["load"]["basic_load_test"]["passed"]:
            recommendations.append("Investigate load test failures - check error rates and response times")
        
        # Stress test recommendations
        if not self.results["stress"]["passed"]:
            recommendations.append("System stability issues under stress - review resource limits")
        
        # Memory recommendations
        if not self.results["memory"]["passed"]:
            recommendations.append("Memory leaks detected - investigate GPU memory management")
        
        # MPS recommendations
        if not self.results["mps"]["passed"]:
            recommendations.append("MPS functionality issues - verify MPS daemon and configuration")
        
        if not recommendations:
            recommendations.append("All tests passed - system is performing well")
        
        return recommendations
    
    def print_summary(self):
        """Print test summary"""
        if "summary" not in self.results:
            print("Test results not available")
            return
        
        summary = self.results["summary"]
        
        print("\n" + "="*60)
        print("COMPREHENSIVE GPU TESTING SUITE RESULTS")
        print("="*60)
        print(f"Test Duration: {summary['test_duration_minutes']:.1f} minutes")
        print(f"Overall Score: {summary['overall_score']:.1f}/100")
        print(f"Overall Result: {'PASS' if summary['overall_passed'] else 'FAIL'}")
        print(f"Assessment: {summary['assessment']}")
        
        print("\nIndividual Test Results:")
        for test_name, passed in summary["individual_results"].items():
            status = "PASS" if passed else "FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        print("\nDetailed Results:")
        
        # Integration results
        integration = self.results["integration"]
        healthy_count = sum(1 for h in integration["health_checks"] if h.get("status") == "healthy")
        print(f"  Integration: {healthy_count}/{len(integration['health_checks'])} services healthy")
        
        # Load test results
        load = self.results["load"]
        basic_error_rate = load["basic_load_test"]["results"].error_rate
        intensive_error_rate = load["intensive_load_test"]["results"].error_rate
        print(f"  Load Tests: Basic {basic_error_rate:.1%} errors, Intensive {intensive_error_rate:.1%} errors")
        
        # Stress test results
        stress = self.results["stress"]
        print(f"  Stress Test: {stress['results'].stability_score:.1f}/100 stability score")
        
        # Memory test results
        memory = self.results["memory"]
        print(f"  Memory Tests: {memory['services_with_leaks']}/{memory['total_services_tested']} services with leaks")
        
        # MPS test results
        mps = self.results["mps"]
        print(f"  MPS Tests: {mps['mps_score']:.1f}/100 functionality score")
        
        print("\nRecommendations:")
        for rec in summary["recommendations"]:
            print(f"  - {rec}")
        
        print("="*60)
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gpu_test_results_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        results_copy = self._serialize_results(self.results)
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {filename}")
        return filename
    
    def _serialize_results(self, obj):
        """Recursively serialize objects for JSON"""
        if isinstance(obj, dict):
            return {key: self._serialize_results(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_results(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._serialize_results(obj.__dict__)
        elif isinstance(obj, (datetime, time)):
            return str(obj)
        else:
            return obj


async def run_quick_test():
    """Run quick test suite (abbreviated versions)"""
    runner = TestSuiteRunner()
    
    logger.info("Running Quick Test Suite...")
    
    # Run only integration and basic load tests
    results = {}
    results["integration"] = await runner.run_integration_tests()
    results["mps_basic"] = await runner.run_mps_tests()
    
    # Print quick summary
    print("\nQUICK TEST RESULTS:")
    integration_ok = runner._assess_integration_tests() if "integration" in runner.results else False
    mps_ok = results["mps_basic"]["passed"]
    
    print(f"Integration Tests: {'PASS' if integration_ok else 'FAIL'}")
    print(f"MPS Functionality: {'PASS' if mps_ok else 'FAIL'}")
    
    overall_ok = integration_ok and mps_ok
    print(f"Quick Assessment: {'SYSTEM_READY' if overall_ok else 'ISSUES_DETECTED'}")
    
    return results


async def main():
    """Main test runner"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "quick":
            await run_quick_test()
        elif test_type == "integration":
            runner = TestSuiteRunner()
            results = await runner.run_integration_tests()
            print(f"Integration tests completed: {len(results)} test categories")
        elif test_type == "load":
            runner = TestSuiteRunner()
            results = await runner.run_load_tests()
            print(f"Load tests completed")
        elif test_type == "stress":
            runner = TestSuiteRunner()
            results = await runner.run_stress_tests()
            print(f"Stress tests completed: {results['assessment']}")
        elif test_type == "memory":
            runner = TestSuiteRunner()
            results = await runner.run_memory_tests()
            print(f"Memory tests completed: {results['memory_health']}")
        elif test_type == "mps":
            runner = TestSuiteRunner()
            results = await runner.run_mps_tests()
            print(f"MPS tests completed: {results['assessment']}")
        elif test_type == "full":
            runner = TestSuiteRunner()
            results = await runner.run_all_tests()
            runner.print_summary()
            runner.save_results()
        else:
            print("Usage: python run_tests.py [quick|integration|load|stress|memory|mps|full]")
    else:
        # Default: run full test suite
        runner = TestSuiteRunner()
        results = await runner.run_all_tests()
        runner.print_summary()
        runner.save_results()


if __name__ == "__main__":
    asyncio.run(main())