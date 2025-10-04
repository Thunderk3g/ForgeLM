#!/usr/bin/env python3
"""
GPU Memory Manager for LLMForge with MPS Support
Manages GPU memory allocation, monitoring, and cleanup for Unsloth fine-tuning operations.
"""

import os
import sys
import time
import json
import signal
import logging
import subprocess
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import argparse

try:
    import psutil
    import GPUtil
    import torch
    import redis
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install psutil GPUtil torch redis")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/llmforge-gpu-manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GPUMemoryInfo:
    """GPU memory information structure"""
    gpu_id: int
    name: str
    total_memory: float  # GB
    used_memory: float   # GB
    free_memory: float   # GB
    utilization: float   # %
    temperature: float   # C
    mps_active: bool
    mps_clients: int
    container_assignments: List[str]
    timestamp: str

@dataclass
class ContainerGPUUsage:
    """Container GPU usage information"""
    container_name: str
    container_id: str
    gpu_memory_usage: float  # GB
    gpu_utilization: float   # %
    mps_thread_percentage: float
    status: str
    service_type: str  # training, inference, model_management

class GPUMemoryManager:
    """Manages GPU memory allocation and monitoring for LLMForge MPS setup"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.running = True
        self.monitor_interval = 10  # seconds
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = datetime.now()
        
        # MPS configuration
        self.mps_base_dir = "/tmp/nvidia-mps"
        self.mps_log_dir = "/var/log/nvidia-mps"
        
        # Service priorities and memory limits
        self.service_config = {
            'training-service': {
                'priority': 1,
                'memory_limit_gb': 12,
                'thread_percentage': 70,
                'gpu_preference': 0
            },
            'chat-comparison-service': {
                'priority': 2,
                'memory_limit_gb': 4,
                'thread_percentage': 15,
                'gpu_preference': 0
            },
            'model-management-service': {
                'priority': 3,
                'memory_limit_gb': 4,
                'thread_percentage': 15,
                'gpu_preference': 0
            }
        }
        
        # Initialize signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down GPU manager...")
        self.running = False
    
    def get_gpu_info(self) -> List[GPUMemoryInfo]:
        """Get current GPU information including MPS status"""
        gpu_info = []
        
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                # Check MPS status
                mps_active = self._is_mps_active(gpu.id)
                mps_clients = self._get_mps_client_count(gpu.id)
                container_assignments = self._get_container_assignments(gpu.id)
                
                info = GPUMemoryInfo(
                    gpu_id=gpu.id,
                    name=gpu.name,
                    total_memory=gpu.memoryTotal / 1024,  # Convert MB to GB
                    used_memory=gpu.memoryUsed / 1024,
                    free_memory=gpu.memoryFree / 1024,
                    utilization=gpu.load * 100,
                    temperature=gpu.temperature,
                    mps_active=mps_active,
                    mps_clients=mps_clients,
                    container_assignments=container_assignments,
                    timestamp=datetime.now().isoformat()
                )
                gpu_info.append(info)
                
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
        
        return gpu_info
    
    def _is_mps_active(self, gpu_id: int) -> bool:
        """Check if MPS is active for a specific GPU"""
        try:
            pipe_dir = f"{self.mps_base_dir}/gpu_{gpu_id}"
            return os.path.exists(pipe_dir) and os.listdir(pipe_dir)
        except Exception:
            return False
    
    def _get_mps_client_count(self, gpu_id: int) -> int:
        """Get number of active MPS clients for a GPU"""
        try:
            pipe_dir = f"{self.mps_base_dir}/gpu_{gpu_id}"
            if not os.path.exists(pipe_dir):
                return 0
            
            # Count .pipe files which represent active clients
            pipe_files = [f for f in os.listdir(pipe_dir) if f.endswith('.pipe')]
            return len(pipe_files)
        except Exception:
            return 0
    
    def _get_container_assignments(self, gpu_id: int) -> List[str]:
        """Get list of containers assigned to a specific GPU"""
        try:
            # Query Redis for container assignments
            key = f"gpu:{gpu_id}:containers"
            containers = self.redis_client.smembers(key)
            return list(containers) if containers else []
        except Exception as e:
            logger.debug(f"Error getting container assignments for GPU {gpu_id}: {e}")
            return []
    
    def get_container_gpu_usage(self) -> List[ContainerGPUUsage]:
        """Get GPU usage information for LLMForge containers"""
        container_usage = []
        
        try:
            # Get Docker containers
            result = subprocess.run(
                ['docker', 'ps', '--format', 'json'],
                capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                container_info = json.loads(line)
                container_name = container_info.get('Names', '')
                
                # Filter LLMForge GPU services
                if any(service in container_name for service in 
                      ['training', 'chat-comparison', 'model-management']):
                    
                    usage = self._get_container_metrics(container_info)
                    if usage:
                        container_usage.append(usage)
                        
        except Exception as e:
            logger.error(f"Error getting container GPU usage: {e}")
        
        return container_usage
    
    def _get_container_metrics(self, container_info: Dict) -> Optional[ContainerGPUUsage]:
        """Get detailed metrics for a specific container"""
        try:
            container_name = container_info.get('Names', '')
            container_id = container_info.get('ID', '')
            status = container_info.get('Status', '')
            
            # Determine service type
            service_type = 'unknown'
            if 'training' in container_name:
                service_type = 'training'
            elif 'chat-comparison' in container_name:
                service_type = 'inference'
            elif 'model-management' in container_name:
                service_type = 'model_management'
            
            # Get GPU memory usage from container stats
            try:
                stats_result = subprocess.run(
                    ['docker', 'stats', '--no-stream', '--format', 
                     'table {{.Container}}\t{{.MemUsage}}', container_id],
                    capture_output=True, text=True, timeout=5
                )
                
                # Parse memory usage (this is system memory, not GPU memory)
                # For GPU memory, we'd need nvidia-ml-py integration
                gpu_memory_usage = 0.0
                gpu_utilization = 0.0
                
                # Get MPS thread percentage from service config
                config = self.service_config.get(f"{service_type}-service", {})
                mps_thread_percentage = config.get('thread_percentage', 0)
                
                return ContainerGPUUsage(
                    container_name=container_name,
                    container_id=container_id,
                    gpu_memory_usage=gpu_memory_usage,
                    gpu_utilization=gpu_utilization,
                    mps_thread_percentage=mps_thread_percentage,
                    status=status,
                    service_type=service_type
                )
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout getting stats for container {container_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting container metrics: {e}")
            return None
    
    def monitor_memory_pressure(self) -> Dict[str, float]:
        """Monitor GPU memory pressure and return alerts"""
        alerts = {}
        
        try:
            gpu_info = self.get_gpu_info()
            
            for gpu in gpu_info:
                memory_usage_pct = (gpu.used_memory / gpu.total_memory) * 100
                
                # Alert thresholds
                if memory_usage_pct > 90:
                    alerts[f"gpu_{gpu.gpu_id}_critical"] = memory_usage_pct
                    logger.warning(f"GPU {gpu.gpu_id} critical memory usage: {memory_usage_pct:.1f}%")
                elif memory_usage_pct > 80:
                    alerts[f"gpu_{gpu.gpu_id}_warning"] = memory_usage_pct
                    logger.info(f"GPU {gpu.gpu_id} high memory usage: {memory_usage_pct:.1f}%")
                
                # Temperature alerts
                if gpu.temperature > 85:
                    alerts[f"gpu_{gpu.gpu_id}_temp_critical"] = gpu.temperature
                    logger.warning(f"GPU {gpu.gpu_id} critical temperature: {gpu.temperature}°C")
                elif gpu.temperature > 80:
                    alerts[f"gpu_{gpu.gpu_id}_temp_warning"] = gpu.temperature
                
        except Exception as e:
            logger.error(f"Error monitoring memory pressure: {e}")
        
        return alerts
    
    def cleanup_dead_containers(self):
        """Clean up GPU assignments for dead containers"""
        try:
            # Get current running containers
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True, text=True, check=True
            )
            running_containers = set(result.stdout.strip().split('\n'))
            
            # Clean up Redis assignments for dead containers
            for gpu_id in range(torch.cuda.device_count() if torch.cuda.is_available() else 1):
                key = f"gpu:{gpu_id}:containers"
                assigned_containers = self.redis_client.smembers(key)
                
                for container in assigned_containers:
                    if container not in running_containers:
                        self.redis_client.srem(key, container)
                        logger.info(f"Cleaned up dead container {container} from GPU {gpu_id}")
                        
        except Exception as e:
            logger.error(f"Error cleaning up dead containers: {e}")
    
    def restart_mps_if_needed(self):
        """Restart MPS service if it's not running properly"""
        try:
            gpu_info = self.get_gpu_info()
            
            for gpu in gpu_info:
                if not gpu.mps_active and gpu.container_assignments:
                    logger.warning(f"MPS not active for GPU {gpu.gpu_id} but containers assigned. Restarting MPS...")
                    
                    # Restart MPS for this GPU
                    self._restart_mps_for_gpu(gpu.gpu_id)
                    
        except Exception as e:
            logger.error(f"Error checking MPS status: {e}")
    
    def _restart_mps_for_gpu(self, gpu_id: int):
        """Restart MPS service for a specific GPU"""
        try:
            # Set environment for specific GPU
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            env['CUDA_MPS_PIPE_DIRECTORY'] = f"{self.mps_base_dir}/gpu_{gpu_id}"
            env['CUDA_MPS_LOG_DIRECTORY'] = f"{self.mps_log_dir}/gpu_{gpu_id}"
            
            # Stop existing MPS for this GPU
            subprocess.run(['echo', 'quit'], input='\n', text=True, env=env, timeout=5)
            time.sleep(2)
            
            # Start MPS control daemon
            subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, timeout=10)
            
            # Configure MPS settings based on service assignments
            self._configure_mps_for_gpu(gpu_id, env)
            
            logger.info(f"Successfully restarted MPS for GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"Error restarting MPS for GPU {gpu_id}: {e}")
    
    def _configure_mps_for_gpu(self, gpu_id: int, env: Dict[str, str]):
        """Configure MPS settings for a specific GPU"""
        try:
            # Get assigned containers for this GPU
            containers = self._get_container_assignments(gpu_id)
            
            if not containers:
                return
            
            # Determine primary service type
            primary_service = None
            for container in containers:
                if 'training' in container:
                    primary_service = 'training-service'
                    break
            
            if not primary_service:
                primary_service = 'chat-comparison-service'  # Default to inference
            
            config = self.service_config.get(primary_service, {})
            
            # Set active thread percentage
            thread_percentage = config.get('thread_percentage', 50)
            cmd = f"echo 'set_active_thread_percentage {thread_percentage}'"
            subprocess.run(cmd, shell=True, env=env, timeout=5)
            
            # Set memory limit
            memory_limit = config.get('memory_limit_gb', 4) * 1024  # Convert to MB
            cmd = f"echo 'set_default_device_pinned_mem_limit {memory_limit}'"
            subprocess.run(cmd, shell=True, env=env, timeout=5)
            
            logger.info(f"Configured MPS for GPU {gpu_id}: {thread_percentage}% threads, {memory_limit}MB memory")
            
        except Exception as e:
            logger.error(f"Error configuring MPS for GPU {gpu_id}: {e}")
    
    def log_gpu_status(self):
        """Log current GPU status to Redis and file"""
        try:
            gpu_info = self.get_gpu_info()
            container_usage = self.get_container_gpu_usage()
            alerts = self.monitor_memory_pressure()
            
            # Create status report
            status_report = {
                'timestamp': datetime.now().isoformat(),
                'gpus': [asdict(gpu) for gpu in gpu_info],
                'containers': [asdict(container) for container in container_usage],
                'alerts': alerts
            }
            
            # Store in Redis with TTL
            self.redis_client.setex(
                'llmforge:gpu_status', 
                300,  # 5 minute TTL
                json.dumps(status_report)
            )
            
            # Log summary
            total_memory = sum(gpu.total_memory for gpu in gpu_info)
            used_memory = sum(gpu.used_memory for gpu in gpu_info)
            utilization = (used_memory / total_memory * 100) if total_memory > 0 else 0
            
            logger.info(f"GPU Status: {len(gpu_info)} GPUs, {utilization:.1f}% memory used, "
                       f"{len(container_usage)} containers, {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Error logging GPU status: {e}")
    
    def run_monitor(self):
        """Main monitoring loop"""
        logger.info("Starting GPU memory manager monitoring...")
        
        while self.running:
            try:
                # Log current status
                self.log_gpu_status()
                
                # Periodic cleanup
                if datetime.now() - self.last_cleanup > timedelta(seconds=self.cleanup_interval):
                    self.cleanup_dead_containers()
                    self.restart_mps_if_needed()
                    self.last_cleanup = datetime.now()
                
                # Wait for next iteration
                time.sleep(self.monitor_interval)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
        
        logger.info("GPU memory manager stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='LLMForge GPU Memory Manager')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--monitor-interval', type=int, default=10, help='Monitor interval in seconds')
    parser.add_argument('--cleanup-interval', type=int, default=300, help='Cleanup interval in seconds')
    parser.add_argument('--status', action='store_true', help='Show current status and exit')
    
    args = parser.parse_args()
    
    try:
        manager = GPUMemoryManager(args.redis_host, args.redis_port)
        manager.monitor_interval = args.monitor_interval
        manager.cleanup_interval = args.cleanup_interval
        
        if args.status:
            # Show current status
            gpu_info = manager.get_gpu_info()
            container_usage = manager.get_container_gpu_usage()
            alerts = manager.monitor_memory_pressure()
            
            print("\n=== LLMForge GPU Status ===")
            for gpu in gpu_info:
                print(f"GPU {gpu.gpu_id} ({gpu.name}): "
                      f"{gpu.used_memory:.1f}GB/{gpu.total_memory:.1f}GB "
                      f"({gpu.utilization:.1f}% util, {gpu.temperature}°C)")
                print(f"  MPS: {'Active' if gpu.mps_active else 'Inactive'} "
                      f"({gpu.mps_clients} clients)")
                print(f"  Containers: {', '.join(gpu.container_assignments) or 'None'}")
            
            print(f"\n=== Container Usage ===")
            for container in container_usage:
                print(f"{container.container_name} ({container.service_type}): "
                      f"{container.mps_thread_percentage}% threads, {container.status}")
            
            if alerts:
                print(f"\n=== Alerts ===")
                for alert, value in alerts.items():
                    print(f"{alert}: {value}")
            
            print()
        else:
            # Run monitoring
            manager.run_monitor()
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()