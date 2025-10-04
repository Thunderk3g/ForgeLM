# LLMForge MPS GPU Testing Suite

Comprehensive testing suite for validating MPS-enabled GPU sharing implementation in LLMForge services.

## Overview

This testing suite validates all aspects of the MPS (Multi-Process Service) GPU sharing implementation across three GPU services:
- **Training Service** (port 8001): Primary GPU user with 70% MPS allocation
- **Chat Comparison Service** (port 8002): Shared GPU access with 15% MPS allocation  
- **Model Management Service** (port 8005): Shared GPU access with 15% MPS allocation

## Test Suites

### 1. Integration Tests (`test_gpu_services.py`)
Tests basic functionality and health of all GPU services.

**Key Tests:**
- Health checks for all services
- GPU accessibility verification
- Concurrent GPU usage testing
- GPU memory allocation verification
- Response time consistency
- MPS configuration indicators

**Usage:**
```bash
# Run integration tests
python -m pytest tests/test_gpu_services.py -v

# Run specific test
python tests/test_gpu_services.py
```

### 2. Load Testing (`load_test_gpu.py`)
Tests performance under sustained concurrent load.

**Features:**
- Configurable concurrent users and request rates
- Response time analysis (95th/99th percentiles)
- Error rate monitoring (<5% threshold)
- Service-specific load distribution

**Usage:**
```bash
# Basic load test (5 users, 1 minute)
python tests/load_test_gpu.py basic

# Intensive load test (20 users, 5 minutes)
python tests/load_test_gpu.py intensive

# Endurance test (10 users, 30 minutes)
python tests/load_test_gpu.py endurance
```

### 3. Stress Testing (`stress_test.py`)
Extended duration testing with high concurrency and memory pressure.

**Features:**
- Extended test duration (15+ minutes)
- High concurrency simulation (up to 50 users)
- Memory pressure scenarios
- System resource monitoring
- Service recovery testing

**Usage:**
```bash
# Basic stress test (15 minutes)
python tests/stress_test.py basic

# Extended stress test (30 minutes)
python tests/stress_test.py extended
```

### 4. Memory Leak Detection (`memory_leak_test.py`)
Monitors GPU memory usage patterns to detect leaks.

**Features:**
- Continuous memory monitoring
- Leak detection with <100MB threshold
- Service-specific memory profiling
- MPS memory sharing validation

**Usage:**
```bash
# Memory leak detection
python tests/memory_leak_test.py leak

# MPS memory sharing test
python tests/memory_leak_test.py mps
```

### 5. MPS Functionality Tests (`test_mps_functionality.py`)
Validates MPS-specific features and GPU sharing.

**Features:**
- MPS daemon status verification
- Shared GPU access validation
- Resource isolation testing
- Service restart recovery
- Memory allocation compliance

**Usage:**
```bash
# Comprehensive MPS test
python tests/test_mps_functionality.py

# Pytest-based tests
python tests/test_mps_functionality.py pytest
```

## Comprehensive Test Runner

The `run_tests.py` script orchestrates all test suites and provides unified reporting.

### Usage

```bash
# Run all tests
python tests/run_tests.py full

# Quick validation
python tests/run_tests.py quick

# Individual test suites
python tests/run_tests.py integration
python tests/run_tests.py load
python tests/run_tests.py stress
python tests/run_tests.py memory
python tests/run_tests.py mps
```

### Test Results

Results are automatically saved to JSON files with timestamps:
- `gpu_test_results_YYYYMMDD_HHMMSS.json`

## Success Criteria Validation

The test suite validates all implementation success criteria:

### ✅ Service Startup and Stability
- All services start without errors
- Services remain running for 1+ hour
- No container restarts during testing

### ✅ GPU Access and MPS Sharing
- All three services can access GPU simultaneously
- MPS daemon is running and accessible
- Resource isolation between services

### ✅ Functional Testing
- Training service: Model management and fine-tuning operations
- Chat comparison: Dual model inference and comparison
- Model management: Model loading, caching, and lifecycle

### ✅ Performance Requirements
- Error rate <5% under normal load
- Consistent response times (<5 seconds for health checks)
- Concurrent request handling capability

### ✅ Memory Management
- No memory leaks detected (<100MB increase threshold)
- GPU memory sharing within MPS limits
- Proper memory cleanup after operations

### ✅ Monitoring and Logging
- Health endpoints respond correctly
- GPU information endpoints accessible
- System metrics collection working

## Installation

1. Install dependencies:
```bash
pip install -r tests/requirements.txt
```

2. Ensure services are running:
```bash
docker-compose up -d
```

3. Wait for services to initialize (30-60 seconds)

## Configuration

### Environment Variables
- `BASE_URL`: Base URL for services (default: http://localhost)
- `LOG_LEVEL`: Logging level (default: INFO)

### Test Configuration
Each test suite has configurable parameters:

**Load Testing:**
- `concurrent_users`: Number of concurrent users
- `test_duration_seconds`: Test duration
- `target_rps`: Target requests per second
- `error_rate_threshold`: Acceptable error rate

**Stress Testing:**
- `test_duration_minutes`: Test duration in minutes
- `max_concurrent_users`: Maximum concurrent users
- `memory_pressure_enabled`: Enable memory pressure testing

**Memory Testing:**
- `leak_threshold_mb`: Memory leak threshold in MB
- `monitoring_interval`: Monitoring interval in seconds

## Troubleshooting

### Common Issues

1. **Services not accessible**
   - Check if services are running: `docker-compose ps`
   - Verify ports are not blocked: `netstat -an | grep 800[1-5]`
   - Wait for service initialization

2. **MPS tests failing**
   - Check if running in Docker environment with proper GPU access
   - Verify NVIDIA MPS is installed and configured
   - Check MPS directories exist: `/tmp/nvidia-mps`, `/var/log/nvidia-mps`

3. **High error rates**
   - Verify GPU memory is sufficient
   - Check for resource contention
   - Review service logs: `docker-compose logs [service-name]`

4. **Memory leak false positives**
   - GPU memory may fluctuate during model loading
   - Consider baseline memory usage
   - Run longer tests for accurate detection

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python tests/run_tests.py quick
```

### Manual Service Testing

Test individual services manually:
```bash
# Health checks
curl http://localhost:8001/health
curl http://localhost:8002/health  
curl http://localhost:8005/health

# GPU information
curl http://localhost:8001/system/gpu
curl http://localhost:8005/gpu/info
```

## Continuous Integration

For CI/CD integration, use the quick test:
```bash
# Returns exit code 0 on success, 1 on failure
python tests/run_tests.py quick
```

## Expected Performance Benchmarks

### Baseline Performance
- Health check response time: <100ms
- GPU info endpoint: <500ms
- Service startup time: <60 seconds

### Load Test Thresholds
- Error rate: <5%
- 95th percentile response time: <5 seconds
- Concurrent users supported: 10+

### Stress Test Thresholds
- Stability score: >80/100 for production
- Memory increase: <200MB over 15 minutes
- Service recovery: <30 seconds

## Support

For issues with the testing suite:
1. Check the troubleshooting section above
2. Review service logs for specific errors
3. Verify GPU and MPS configuration
4. Test individual components separately

The testing suite is designed to be robust and provide clear diagnostic information for any failures detected.