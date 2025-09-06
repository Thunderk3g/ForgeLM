"""
GPU Resource Management and Isolation Tests

This script validates that the multiprocessing implementation properly handles:
1. CUDA context isolation between processes
2. GPU memory management
3. Process cleanup and resource release
4. Multiple concurrent training processes
"""

import os
import sys
import time
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Set spawn method for CUDA compatibility
mp.set_start_method('spawn', force=True)


def check_gpu_availability():
    """Check if GPU is available and get basic info"""
    print("🔍 Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {device_count} device(s)")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
            return True, device_count
        else:
            print("⚠️ CUDA not available - GPU tests will be limited")
            return False, 0
            
    except ImportError:
        print("❌ PyTorch not available - cannot test GPU")
        return False, 0


def test_cuda_context_isolation():
    """Test that CUDA contexts are properly isolated between processes"""
    print("\n🔐 Testing CUDA context isolation...")
    
    def worker_process(worker_id: int, results_queue: mp.Queue):
        """Worker process that initializes CUDA and reports status"""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Initialize CUDA context
                device = torch.cuda.current_device()
                torch.cuda.empty_cache()
                
                # Create a small tensor to initialize context
                tensor = torch.randn(1000, 1000, device='cuda')
                memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                
                result = {
                    "worker_id": worker_id,
                    "pid": os.getpid(),
                    "device": device,
                    "memory_allocated_mb": memory_allocated,
                    "success": True,
                    "error": None
                }
                
                # Clean up
                del tensor
                torch.cuda.empty_cache()
                
            else:
                result = {
                    "worker_id": worker_id,
                    "pid": os.getpid(),
                    "success": True,
                    "error": "CUDA not available",
                    "device": None,
                    "memory_allocated_mb": 0
                }
            
            results_queue.put(result)
            
        except Exception as e:
            results_queue.put({
                "worker_id": worker_id,
                "pid": os.getpid(),
                "success": False,
                "error": str(e)
            })
    
    # Test with multiple processes
    num_workers = 3
    processes = []
    results_queue = mp.Queue()
    
    # Start worker processes
    for i in range(num_workers):
        process = mp.Process(target=worker_process, args=(i, results_queue))
        process.start()
        processes.append(process)
    
    # Collect results
    results = []
    for _ in range(num_workers):
        try:
            result = results_queue.get(timeout=30)
            results.append(result)
        except:
            results.append({"error": "Timeout", "success": False})
    
    # Wait for processes to finish
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
    
    # Analyze results
    success_count = sum(1 for r in results if r.get("success", False))
    unique_pids = set(r.get("pid") for r in results if r.get("pid"))
    
    print(f"✅ Process isolation: {success_count}/{num_workers} processes succeeded")
    print(f"✅ Unique PIDs: {len(unique_pids)} (expected: {num_workers})")
    
    for result in results:
        if result.get("success"):
            print(f"   Worker {result['worker_id']}: PID {result['pid']}, "
                  f"GPU memory: {result.get('memory_allocated_mb', 0):.1f}MB")
        else:
            print(f"   Worker {result.get('worker_id', '?')}: Failed - {result.get('error', 'Unknown')}")
    
    return success_count == num_workers and len(unique_pids) == num_workers


def test_memory_cleanup():
    """Test that GPU memory is properly cleaned up after process termination"""
    print("\n🧹 Testing GPU memory cleanup after process termination...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available - skipping memory cleanup test")
            return True
        
        # Get initial memory state
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        def memory_intensive_worker():
            """Worker that allocates significant GPU memory"""
            import torch
            
            # Allocate large tensors
            tensors = []
            for i in range(10):
                tensor = torch.randn(1000, 1000, device='cuda')
                tensors.append(tensor)
            
            # Report memory usage
            memory_used = torch.cuda.memory_allocated() / (1024**2)
            print(f"   Worker allocated {memory_used:.1f}MB GPU memory")
            
            # Keep the process alive for a bit
            time.sleep(5)
        
        # Start memory-intensive process
        print("   Starting memory-intensive worker process...")
        process = mp.Process(target=memory_intensive_worker)
        process.start()
        
        # Wait for it to finish
        process.join(timeout=15)
        
        if process.is_alive():
            print("   Terminating worker process...")
            process.terminate()
            process.join(timeout=5)
        
        # Check memory after process cleanup
        time.sleep(2)  # Allow time for cleanup
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        memory_diff = final_memory - initial_memory
        
        print(f"   Initial GPU memory: {initial_memory / (1024**2):.1f}MB")
        print(f"   Final GPU memory: {final_memory / (1024**2):.1f}MB")
        print(f"   Memory difference: {memory_diff / (1024**2):.1f}MB")
        
        # Consider test passed if memory difference is small (< 100MB)
        cleanup_successful = abs(memory_diff) < 100 * 1024 * 1024
        
        if cleanup_successful:
            print("✅ Memory cleanup successful")
        else:
            print("⚠️ Potential memory leak detected")
        
        return cleanup_successful
        
    except ImportError:
        print("❌ PyTorch not available for memory cleanup test")
        return False
    except Exception as e:
        print(f"❌ Memory cleanup test failed: {e}")
        return False


def test_concurrent_training_processes():
    """Test running multiple training processes concurrently"""
    print("\n🔄 Testing concurrent training processes...")
    
    def simulate_training_worker(worker_id: int, duration: int, results_queue: mp.Queue):
        """Simulate a training process"""
        try:
            import torch
            
            start_time = time.time()
            
            # Simulate CUDA initialization
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # Simulate training workload
                for step in range(duration):
                    # Create and process tensors
                    data = torch.randn(100, 100, device='cuda')
                    result = torch.matmul(data, data.t())
                    
                    # Simulate gradient updates
                    loss = result.sum()
                    
                    time.sleep(0.1)  # Simulate processing time
            else:
                # CPU simulation
                for step in range(duration):
                    data = torch.randn(100, 100)
                    result = torch.matmul(data, data.t())
                    time.sleep(0.1)
            
            elapsed = time.time() - start_time
            
            results_queue.put({
                "worker_id": worker_id,
                "pid": os.getpid(),
                "success": True,
                "elapsed_time": elapsed,
                "steps_completed": duration
            })
            
        except Exception as e:
            results_queue.put({
                "worker_id": worker_id,
                "pid": os.getpid(),
                "success": False,
                "error": str(e)
            })
    
    # Start multiple concurrent "training" processes
    num_workers = 3
    duration_steps = 20
    processes = []
    results_queue = mp.Queue()
    
    print(f"   Starting {num_workers} concurrent training simulations...")
    start_time = time.time()
    
    for i in range(num_workers):
        process = mp.Process(target=simulate_training_worker, args=(i, duration_steps, results_queue))
        process.start()
        processes.append(process)
    
    # Collect results
    results = []
    for _ in range(num_workers):
        try:
            result = results_queue.get(timeout=60)
            results.append(result)
        except:
            results.append({"error": "Timeout", "success": False})
    
    # Wait for all processes
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
    
    total_elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r.get("success", False))
    
    print(f"   Total time for concurrent execution: {total_elapsed:.2f}s")
    print(f"   Successful processes: {success_count}/{num_workers}")
    
    for result in results:
        if result.get("success"):
            print(f"   Worker {result['worker_id']}: {result['elapsed_time']:.2f}s, "
                  f"{result['steps_completed']} steps")
        else:
            print(f"   Worker {result.get('worker_id', '?')}: Failed - {result.get('error', 'Unknown')}")
    
    return success_count == num_workers


def test_process_lifecycle():
    """Test proper process lifecycle management"""
    print("\n🔄 Testing process lifecycle management...")
    
    # Import the actual training worker to test lifecycle
    try:
        from services.training_worker import ProcessManager
        
        process_manager = ProcessManager()
        
        # Create a minimal request for testing
        test_request = {
            "base_model_dir": "models/base/Qwen_Qwen2.5-3B",
            "dataset_dir": "data/prepared",
            "output_dir": "models/test_lifecycle",
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-5,
            "max_seq_length": 256,
            "gradient_accumulation_steps": 4,
            "peft_method": "lora",
            "ft_type": "instruction"
        }
        
        task_id = "test_lifecycle_001"
        
        print("   Starting test training process...")
        success = process_manager.start_training_process(task_id, test_request)
        
        if success:
            print("✅ Process started successfully")
            
            # Check process status
            time.sleep(5)
            status = process_manager.get_process_status(task_id)
            print(f"   Process status: {status}")
            
            # Cancel the process
            print("   Cancelling process...")
            cancelled = process_manager.cancel_training(task_id)
            
            if cancelled:
                print("✅ Process cancelled successfully")
                
                # Verify cleanup
                time.sleep(2)
                final_status = process_manager.get_process_status(task_id)
                print(f"   Final status: {final_status}")
                
                return True
            else:
                print("❌ Failed to cancel process")
                return False
        else:
            print("❌ Failed to start process")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import training worker: {e}")
        return False
    except Exception as e:
        print(f"❌ Process lifecycle test failed: {e}")
        return False


def main():
    """Run all GPU isolation and resource management tests"""
    print("🧪 GPU Resource Management and Isolation Tests")
    print("=" * 60)
    
    # Check GPU availability
    gpu_available, device_count = check_gpu_availability()
    
    # Define tests
    tests = [
        ("CUDA Context Isolation", test_cuda_context_isolation),
        ("Memory Cleanup", test_memory_cleanup),
        ("Concurrent Training Processes", test_concurrent_training_processes),
        ("Process Lifecycle Management", test_process_lifecycle)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print('=' * 60)
        
        try:
            result = test_func()
            results[test_name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = "💥 CRASH"
        
        time.sleep(2)  # Brief pause between tests
    
    # Summary
    print(f"\n{'=' * 60}")
    print("GPU RESOURCE MANAGEMENT TEST SUMMARY")
    print('=' * 60)
    
    print(f"GPU Available: {'✅ Yes' if gpu_available else '❌ No'} ({device_count} devices)")
    print()
    
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    passed = sum(1 for r in results.values() if "PASS" in r)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All GPU resource tests passed! Process isolation is working correctly.")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. Minor GPU resource issues may need attention.")
    else:
        print("❌ Multiple GPU resource test failures. Review process isolation implementation.")
    
    # Recommendations
    print(f"\n📋 Recommendations:")
    if not gpu_available:
        print("• Install CUDA and PyTorch with GPU support for full GPU testing")
    if passed < total:
        print("• Review process cleanup and resource management code")
        print("• Check CUDA context handling in worker processes")
        print("• Monitor GPU memory usage during training")


if __name__ == "__main__":
    main()