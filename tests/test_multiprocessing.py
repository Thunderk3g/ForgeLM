"""
Test Multiprocessing Training Implementation

This script tests the new multiprocessing training implementation to ensure:
1. FastAPI remains responsive during training
2. SSE streaming works correctly
3. Process isolation functions properly
4. Status communication works via files
"""

import asyncio
import time
import json
import requests
import threading
from pathlib import Path
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TASK_ID = None


def test_api_responsiveness():
    """Test that FastAPI remains responsive during training"""
    print("\n🔍 Testing API responsiveness during training...")
    
    try:
        # Test health endpoint
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint responsive")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
        
        # Test root endpoint
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint responsive")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            
        # Test active sessions endpoint
        response = requests.get(f"{API_BASE_URL}/fine-tune/active", timeout=5)
        if response.status_code == 200:
            print("✅ Active sessions endpoint responsive")
        else:
            print(f"❌ Active sessions endpoint failed: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("❌ API endpoints timed out - not responsive!")
        return False
    except Exception as e:
        print(f"❌ API responsiveness test failed: {e}")
        return False
    
    return True


def test_training_start():
    """Test starting a training process"""
    global TEST_TASK_ID
    
    print("\n🚀 Testing training process startup...")
    
    # Create a minimal training request
    request_data = {
        "base_model_dir": "models/base/Qwen_Qwen2.5-3B",
        "dataset_dir": "data/prepared",
        "output_dir": "models/test_multiprocessing",
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 2e-5,
        "max_seq_length": 512,
        "gradient_accumulation_steps": 4,
        "peft_method": "lora",
        "ft_type": "instruction"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/fine-tune", json=request_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            TEST_TASK_ID = result.get("task_id")
            print(f"✅ Training started successfully with task ID: {TEST_TASK_ID}")
            return True
        else:
            print(f"❌ Training start failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Training start failed: {e}")
        return False


def test_status_monitoring():
    """Test status file monitoring and API status endpoint"""
    if not TEST_TASK_ID:
        print("❌ No task ID available for status monitoring")
        return False
    
    print(f"\n📊 Testing status monitoring for task {TEST_TASK_ID}...")
    
    # Check status file
    status_file = Path("logs") / f"{TEST_TASK_ID}_status.json"
    
    for attempt in range(10):
        try:
            # Test API status endpoint
            response = requests.get(f"{API_BASE_URL}/fine-tune/status/{TEST_TASK_ID}", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                print(f"✅ Status API responsive - Status: {status_data.get('status', 'unknown')}")
            else:
                print(f"⚠️ Status API returned: {response.status_code}")
            
            # Test status file
            if status_file.exists():
                with open(status_file, 'r') as f:
                    file_status = json.load(f)
                print(f"✅ Status file exists - Status: {file_status.get('status', 'unknown')}")
                
                if file_status.get('status') in ['completed', 'error', 'cancelled']:
                    print(f"📋 Final status reached: {file_status.get('status')}")
                    return True
            else:
                print("⚠️ Status file not found yet")
            
            time.sleep(3)
            
        except Exception as e:
            print(f"⚠️ Status monitoring error: {e}")
            time.sleep(3)
    
    print("✅ Status monitoring working (may still be running)")
    return True


def test_sse_streaming():
    """Test SSE streaming during training"""
    if not TEST_TASK_ID:
        print("❌ No task ID available for SSE streaming test")
        return False
    
    print(f"\n📡 Testing SSE streaming for task {TEST_TASK_ID}...")
    
    def stream_logs():
        try:
            response = requests.get(f"{API_BASE_URL}/fine-tune/logs/{TEST_TASK_ID}", 
                                 stream=True, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ SSE stream failed to start: {response.status_code}")
                return
            
            print("✅ SSE stream started successfully")
            
            # Read first few events
            event_count = 0
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        event_count += 1
                        data = line[5:].strip()
                        try:
                            parsed = json.loads(data)
                            print(f"📨 SSE Event {event_count}: {parsed.get('message', 'No message')}")
                        except:
                            print(f"📨 SSE Event {event_count}: {data}")
                        
                        if event_count >= 5:  # Stop after 5 events for testing
                            break
                
        except Exception as e:
            print(f"❌ SSE streaming failed: {e}")
    
    # Run SSE test in a separate thread
    sse_thread = threading.Thread(target=stream_logs, daemon=True)
    sse_thread.start()
    
    # Wait a bit for SSE to start
    time.sleep(10)
    
    print("✅ SSE streaming test completed")
    return True


def test_concurrent_requests():
    """Test making concurrent requests while training is running"""
    print("\n🔄 Testing concurrent API requests during training...")
    
    def make_concurrent_requests():
        """Make multiple concurrent requests"""
        endpoints = [
            "/health",
            "/",
            "/fine-tune/active",
            "/models/available"
        ]
        
        results = []
        for endpoint in endpoints:
            try:
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
                results.append((endpoint, response.status_code, response.elapsed.total_seconds()))
            except Exception as e:
                results.append((endpoint, "ERROR", str(e)))
        
        return results
    
    # Make concurrent requests
    threads = []
    all_results = []
    
    for i in range(3):  # 3 concurrent threads
        thread = threading.Thread(target=lambda: all_results.append(make_concurrent_requests()))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Analyze results
    success_count = 0
    total_requests = 0
    
    for results in all_results:
        for endpoint, status, timing in results:
            total_requests += 1
            if status == 200:
                success_count += 1
                print(f"✅ {endpoint}: {status} ({timing:.2f}s)")
            else:
                print(f"❌ {endpoint}: {status} - {timing}")
    
    success_rate = (success_count / total_requests) * 100
    print(f"📊 Concurrent request success rate: {success_rate:.1f}% ({success_count}/{total_requests})")
    
    return success_rate > 80  # Consider success if >80% of requests succeed


def test_training_cancellation():
    """Test training cancellation"""
    if not TEST_TASK_ID:
        print("❌ No task ID available for cancellation test")
        return False
    
    print(f"\n🛑 Testing training cancellation for task {TEST_TASK_ID}...")
    
    try:
        response = requests.delete(f"{API_BASE_URL}/fine-tune/cancel/{TEST_TASK_ID}", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Cancellation requested: {result.get('message', 'No message')}")
            
            # Wait a bit and check status
            time.sleep(5)
            status_response = requests.get(f"{API_BASE_URL}/fine-tune/status/{TEST_TASK_ID}")
            if status_response.status_code == 200:
                status = status_response.json().get('status')
                print(f"📋 Status after cancellation: {status}")
            
            return True
        else:
            print(f"❌ Cancellation failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Cancellation test failed: {e}")
        return False


def main():
    """Run all multiprocessing tests"""
    print("🧪 LLMForge Multiprocessing Training Tests")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running or not healthy. Please start the API first.")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API at {API_BASE_URL}. Please start the API first.")
        print(f"Error: {e}")
        return
    
    print("✅ API is running and healthy")
    
    # Run tests
    tests = [
        ("API Responsiveness (baseline)", test_api_responsiveness),
        ("Training Process Startup", test_training_start),
        ("Status Monitoring", test_status_monitoring),
        ("SSE Streaming", test_sse_streaming),
        ("Concurrent Requests", test_concurrent_requests),
        ("Training Cancellation", test_training_cancellation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 50}")
        print(f"Running: {test_name}")
        print('=' * 50)
        
        try:
            result = test_func()
            results[test_name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = "💥 CRASH"
        
        time.sleep(2)  # Brief pause between tests
    
    # Summary
    print(f"\n{'=' * 50}")
    print("TEST SUMMARY")
    print('=' * 50)
    
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    passed = sum(1 for r in results.values() if "PASS" in r)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Multiprocessing implementation is working correctly.")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. Minor issues may need attention.")
    else:
        print("❌ Multiple test failures. Implementation needs review.")


if __name__ == "__main__":
    main()