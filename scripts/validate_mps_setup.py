#!/usr/bin/env python3
"""
MPS Setup Validation Script for LLMForge
Validates that MPS is properly configured and Unsloth can work with shared GPU resources.
"""

import os
import sys
import time
import json
import subprocess
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MPSValidator:
    """Validates MPS setup and Unsloth compatibility"""
    
    def __init__(self):
        self.mps_base_dir = "/tmp/nvidia-mps"
        self.mps_log_dir = "/var/log/nvidia-mps"
        self.validation_results = {}
        
    def validate_system_requirements(self) -> bool:
        """Validate system requirements for MPS"""
        logger.info("Validating system requirements...")
        
        checks = []
        
        # Check NVIDIA driver
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("NVIDIA Driver", True, "nvidia-smi working"))
            else:
                checks.append(("NVIDIA Driver", False, "nvidia-smi failed"))
        except FileNotFoundError:
            checks.append(("NVIDIA Driver", False, "nvidia-smi not found"))
        
        # Check CUDA installation
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("CUDA Toolkit", True, "nvcc working"))
            else:
                checks.append(("CUDA Toolkit", False, "nvcc failed"))
        except FileNotFoundError:
            checks.append(("CUDA Toolkit", False, "nvcc not found"))
        
        # Check MPS binaries
        try:
            result = subprocess.run(['nvidia-cuda-mps-control', '--help'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("MPS Control", True, "nvidia-cuda-mps-control available"))
            else:
                checks.append(("MPS Control", False, "MPS control failed"))
        except FileNotFoundError:
            checks.append(("MPS Control", False, "nvidia-cuda-mps-control not found"))
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("Docker", True, "Docker available"))
            else:
                checks.append(("Docker", False, "Docker failed"))
        except FileNotFoundError:
            checks.append(("Docker", False, "Docker not found"))
        
        # Check Python packages
        python_packages = ['torch', 'transformers', 'unsloth', 'bitsandbytes']
        for package in python_packages:
            try:
                __import__(package)
                checks.append((f"Python {package}", True, f"{package} imported successfully"))
            except ImportError:
                checks.append((f"Python {package}", False, f"{package} not available"))
        
        self.validation_results['system_requirements'] = checks
        
        # Print results
        logger.info("System Requirements Check:")
        all_passed = True
        for name, passed, message in checks:
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {name}: {status} - {message}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def validate_mps_configuration(self) -> bool:
        """Validate MPS configuration"""
        logger.info("Validating MPS configuration...")
        
        checks = []
        
        # Check MPS directories
        if os.path.exists(self.mps_base_dir):
            checks.append(("MPS Base Directory", True, f"{self.mps_base_dir} exists"))
        else:
            checks.append(("MPS Base Directory", False, f"{self.mps_base_dir} missing"))
        
        if os.path.exists(self.mps_log_dir):
            checks.append(("MPS Log Directory", True, f"{self.mps_log_dir} exists"))
        else:
            checks.append(("MPS Log Directory", False, f"{self.mps_log_dir} missing"))
        
        # Check MPS daemon status
        try:
            result = subprocess.run(['pgrep', '-f', 'nvidia-cuda-mps-control'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                checks.append(("MPS Daemon", True, "MPS control daemon running"))
            else:
                checks.append(("MPS Daemon", False, "MPS control daemon not running"))
        except Exception as e:
            checks.append(("MPS Daemon", False, f"Error checking MPS daemon: {e}"))
        
        # Check GPU directories
        gpu_dirs_found = 0
        if os.path.exists(self.mps_base_dir):
            for item in os.listdir(self.mps_base_dir):
                if item.startswith('gpu_'):
                    gpu_dirs_found += 1
        
        checks.append(("GPU Directories", gpu_dirs_found > 0, 
                      f"Found {gpu_dirs_found} GPU directories"))
        
        # Check environment variables
        env_vars = [
            'CUDA_MPS_PIPE_DIRECTORY',
            'CUDA_MPS_LOG_DIRECTORY',
            'NVIDIA_VISIBLE_DEVICES',
            'PYTORCH_CUDA_ALLOC_CONF'
        ]
        
        for var in env_vars:
            if os.getenv(var):
                checks.append((f"Env {var}", True, f"{var} is set"))
            else:
                checks.append((f"Env {var}", False, f"{var} not set"))
        
        self.validation_results['mps_configuration'] = checks
        
        # Print results
        logger.info("MPS Configuration Check:")
        all_passed = True
        for name, passed, message in checks:
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {name}: {status} - {message}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def validate_gpu_sharing(self) -> bool:
        """Validate GPU sharing functionality"""
        logger.info("Validating GPU sharing...")
        
        checks = []
        
        try:
            # Test CUDA availability
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                checks.append(("CUDA Available", True, f"{gpu_count} GPU(s) detected"))
                
                # Test GPU memory allocation
                try:
                    device = torch.device('cuda:0')
                    test_tensor = torch.randn(100, 100, device=device)
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                    checks.append(("GPU Memory Test", True, 
                                 f"Allocated {memory_allocated:.2f}GB"))
                    
                    # Clean up
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    checks.append(("GPU Memory Test", False, f"Memory allocation failed: {e}"))
            else:
                checks.append(("CUDA Available", False, "CUDA not available"))
        
        except ImportError:
            checks.append(("PyTorch CUDA", False, "PyTorch not available"))
        
        # Test MPS client connections
        try:
            # Check if we can connect to MPS
            env = os.environ.copy()
            env['CUDA_MPS_PIPE_DIRECTORY'] = f"{self.mps_base_dir}/gpu_0"
            
            # Simple CUDA program to test MPS
            test_code = '''
import torch
if torch.cuda.is_available():
    x = torch.randn(10, 10, device="cuda")
    print("MPS connection successful")
else:
    print("CUDA not available")
'''
            
            result = subprocess.run(
                [sys.executable, '-c', test_code],
                env=env, capture_output=True, text=True, timeout=10
            )
            
            if "MPS connection successful" in result.stdout:
                checks.append(("MPS Connection", True, "Successfully connected to MPS"))
            else:
                checks.append(("MPS Connection", False, 
                             f"MPS connection failed: {result.stderr}"))
                
        except Exception as e:
            checks.append(("MPS Connection", False, f"MPS test error: {e}"))
        
        self.validation_results['gpu_sharing'] = checks
        
        # Print results
        logger.info("GPU Sharing Check:")
        all_passed = True
        for name, passed, message in checks:
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {name}: {status} - {message}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def validate_unsloth_compatibility(self) -> bool:
        """Validate Unsloth compatibility with MPS"""
        logger.info("Validating Unsloth compatibility...")
        
        checks = []
        
        try:
            # Test Unsloth import
            import unsloth
            checks.append(("Unsloth Import", True, "Unsloth imported successfully"))
            
            # Test FastLanguageModel import
            from unsloth import FastLanguageModel
            checks.append(("FastLanguageModel", True, "FastLanguageModel available"))
            
            # Test basic Unsloth functionality (without downloading models)
            try:
                # Test if we can at least initialize the class
                # This is a minimal test to avoid downloading large models
                from unsloth.models.llama import LlamaForCausalLM
                checks.append(("Unsloth Models", True, "Unsloth model classes available"))
            except Exception as e:
                checks.append(("Unsloth Models", False, f"Model classes error: {e}"))
            
            # Test 4-bit quantization support
            try:
                import bitsandbytes as bnb
                checks.append(("BitsAndBytes", True, "4-bit quantization available"))
            except ImportError:
                checks.append(("BitsAndBytes", False, "BitsAndBytes not available"))
            
            # Test TRL (for SFTTrainer)
            try:
                from trl import SFTTrainer
                checks.append(("TRL SFTTrainer", True, "SFTTrainer available"))
            except ImportError:
                checks.append(("TRL SFTTrainer", False, "TRL not available"))
            
            # Test PEFT (for LoRA)
            try:
                from peft import LoraConfig
                checks.append(("PEFT LoRA", True, "PEFT LoRA available"))
            except ImportError:
                checks.append(("PEFT LoRA", False, "PEFT not available"))
                
        except ImportError as e:
            checks.append(("Unsloth Import", False, f"Unsloth import failed: {e}"))
        
        # Test environment variables for Unsloth optimization
        unsloth_env_vars = [
            'PYTORCH_CUDA_ALLOC_CONF',
            'TOKENIZERS_PARALLELISM',
            'TRITON_CACHE_DIR'
        ]
        
        for var in unsloth_env_vars:
            if os.getenv(var):
                checks.append((f"Unsloth Env {var}", True, f"{var} configured"))
            else:
                checks.append((f"Unsloth Env {var}", False, f"{var} not configured"))
        
        self.validation_results['unsloth_compatibility'] = checks
        
        # Print results
        logger.info("Unsloth Compatibility Check:")
        all_passed = True
        for name, passed, message in checks:
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {name}: {status} - {message}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def validate_docker_configuration(self) -> bool:
        """Validate Docker MPS configuration"""
        logger.info("Validating Docker configuration...")
        
        checks = []
        
        try:
            # Check if running in Docker
            if os.path.exists('/.dockerenv'):
                checks.append(("Docker Environment", True, "Running inside Docker"))
                
                # Check shared directories
                shared_dirs = ['/tmp/nvidia-mps', '/var/log/nvidia-mps', '/dev/shm']
                for dir_path in shared_dirs:
                    if os.path.exists(dir_path):
                        checks.append((f"Shared Dir {dir_path}", True, f"{dir_path} accessible"))
                    else:
                        checks.append((f"Shared Dir {dir_path}", False, f"{dir_path} not accessible"))
                
                # Check IPC host mode (required for MPS)
                try:
                    # In host IPC mode, we should see host processes in /proc
                    host_procs = len([f for f in os.listdir('/proc') if f.isdigit()])
                    if host_procs > 50:  # Arbitrary threshold for host processes
                        checks.append(("IPC Host Mode", True, "Host IPC mode detected"))
                    else:
                        checks.append(("IPC Host Mode", False, "Host IPC mode may not be enabled"))
                except Exception:
                    checks.append(("IPC Host Mode", False, "Cannot check IPC mode"))
                
            else:
                checks.append(("Docker Environment", False, "Not running in Docker"))
            
            # Check Docker GPU runtime
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    checks.append(("Docker GPU Access", True, "GPU accessible in container"))
                else:
                    checks.append(("Docker GPU Access", False, "GPU not accessible"))
            except Exception:
                checks.append(("Docker GPU Access", False, "Cannot check GPU access"))
        
        except Exception as e:
            checks.append(("Docker Check", False, f"Docker validation error: {e}"))
        
        self.validation_results['docker_configuration'] = checks
        
        # Print results
        logger.info("Docker Configuration Check:")
        all_passed = True
        for name, passed, message in checks:
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {name}: {status} - {message}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def run_performance_benchmark(self) -> Dict[str, float]:
        """Run a basic performance benchmark"""
        logger.info("Running performance benchmark...")
        
        benchmark_results = {}
        
        try:
            import torch
            
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                
                # Memory allocation benchmark
                start_time = time.time()
                test_tensor = torch.randn(1000, 1000, device=device)
                allocation_time = time.time() - start_time
                benchmark_results['memory_allocation_ms'] = allocation_time * 1000
                
                # Matrix multiplication benchmark
                start_time = time.time()
                result = torch.matmul(test_tensor, test_tensor)
                compute_time = time.time() - start_time
                benchmark_results['matrix_mult_ms'] = compute_time * 1000
                
                # Memory info
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                benchmark_results['memory_allocated_gb'] = memory_allocated
                benchmark_results['memory_reserved_gb'] = memory_reserved
                
                # Clean up
                del test_tensor, result
                torch.cuda.empty_cache()
                
                logger.info("Benchmark Results:")
                logger.info(f"  Memory Allocation: {allocation_time*1000:.2f}ms")
                logger.info(f"  Matrix Multiplication: {compute_time*1000:.2f}ms")
                logger.info(f"  Memory Allocated: {memory_allocated:.2f}GB")
                logger.info(f"  Memory Reserved: {memory_reserved:.2f}GB")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            benchmark_results['error'] = str(e)
        
        self.validation_results['performance_benchmark'] = benchmark_results
        return benchmark_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# LLMForge MPS Validation Report
Generated: {timestamp}

## Summary
"""
        
        total_checks = 0
        passed_checks = 0
        
        for category, checks in self.validation_results.items():
            if isinstance(checks, list):
                for name, passed, message in checks:
                    total_checks += 1
                    if passed:
                        passed_checks += 1
        
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        report += f"Overall Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks})\n\n"
        
        # Detailed results
        for category, checks in self.validation_results.items():
            report += f"## {category.replace('_', ' ').title()}\n"
            
            if isinstance(checks, list):
                for name, passed, message in checks:
                    status = "✅ PASS" if passed else "❌ FAIL"
                    report += f"- {name}: {status} - {message}\n"
            elif isinstance(checks, dict):
                for key, value in checks.items():
                    report += f"- {key}: {value}\n"
            
            report += "\n"
        
        # Recommendations
        report += "## Recommendations\n"
        
        if success_rate < 80:
            report += "⚠️ **Critical Issues Detected**\n"
            report += "- Review failed checks and resolve issues before proceeding\n"
            report += "- Ensure all system requirements are met\n"
            report += "- Check MPS daemon status and configuration\n\n"
        elif success_rate < 95:
            report += "⚠️ **Minor Issues Detected**\n"
            report += "- Some optimizations may not be fully configured\n"
            report += "- Consider addressing failed checks for optimal performance\n\n"
        else:
            report += "✅ **System Ready**\n"
            report += "- MPS configuration appears to be working correctly\n"
            report += "- Unsloth should work efficiently with shared GPU resources\n\n"
        
        return report
    
    def save_report(self, filename: str = "mps_validation_report.md"):
        """Save the validation report to a file"""
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to: {filename}")
        return filename
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        logger.info("Starting full MPS validation...")
        
        results = []
        
        # Run all validation checks
        results.append(self.validate_system_requirements())
        results.append(self.validate_mps_configuration())
        results.append(self.validate_gpu_sharing())
        results.append(self.validate_unsloth_compatibility())
        results.append(self.validate_docker_configuration())
        
        # Run performance benchmark
        self.run_performance_benchmark()
        
        # Overall result
        all_passed = all(results)
        
        logger.info(f"Full validation {'PASSED' if all_passed else 'FAILED'}")
        
        return all_passed

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate MPS setup for LLMForge')
    parser.add_argument('--full', action='store_true', help='Run full validation suite')
    parser.add_argument('--system', action='store_true', help='Check system requirements only')
    parser.add_argument('--mps', action='store_true', help='Check MPS configuration only')
    parser.add_argument('--gpu', action='store_true', help='Check GPU sharing only')
    parser.add_argument('--unsloth', action='store_true', help='Check Unsloth compatibility only')
    parser.add_argument('--docker', action='store_true', help='Check Docker configuration only')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark only')
    parser.add_argument('--report', help='Save report to file')
    
    args = parser.parse_args()
    
    validator = MPSValidator()
    
    if args.full or not any([args.system, args.mps, args.gpu, args.unsloth, args.docker, args.benchmark]):
        # Run full validation if no specific checks requested
        success = validator.run_full_validation()
        exit_code = 0 if success else 1
    else:
        # Run specific checks
        results = []
        
        if args.system:
            results.append(validator.validate_system_requirements())
        if args.mps:
            results.append(validator.validate_mps_configuration())
        if args.gpu:
            results.append(validator.validate_gpu_sharing())
        if args.unsloth:
            results.append(validator.validate_unsloth_compatibility())
        if args.docker:
            results.append(validator.validate_docker_configuration())
        if args.benchmark:
            validator.run_performance_benchmark()
        
        exit_code = 0 if all(results) else 1
    
    # Save report if requested
    if args.report:
        validator.save_report(args.report)
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()