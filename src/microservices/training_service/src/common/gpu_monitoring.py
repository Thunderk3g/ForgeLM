"""
GPU Monitoring and Health Check System

Provides comprehensive monitoring for GPU orchestration including:
- Real-time GPU utilization tracking
- Lock contention monitoring  
- Performance metrics collection
- Health check integration
- Alert system for GPU issues
"""

import asyncio
import time
import json
import logging
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import aiohttp

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from .gpu_client import GPUClient


@dataclass
class GPUMetrics:
    """GPU performance and utilization metrics"""
    timestamp: str
    gpu_id: int = 0
    utilization_gpu: float = 0.0
    utilization_memory: float = 0.0
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    memory_free_mb: int = 0
    temperature_gpu: int = 0
    power_draw: float = 0.0
    power_limit: float = 0.0
    compute_processes: List[Dict] = None
    
    def __post_init__(self):
        if self.compute_processes is None:
            self.compute_processes = []


@dataclass
class GPULockMetrics:
    """GPU lock orchestration metrics"""
    timestamp: str
    total_locks_active: int = 0
    total_requests_pending: int = 0
    average_wait_time: float = 0.0
    lock_contention_ratio: float = 0.0
    locks_by_operation: Dict[str, int] = None
    locks_by_service: Dict[str, int] = None
    
    def __post_init__(self):
        if self.locks_by_operation is None:
            self.locks_by_operation = {}
        if self.locks_by_service is None:
            self.locks_by_service = {}


class GPUMonitor:
    """Comprehensive GPU monitoring system"""
    
    def __init__(self, orchestrator_url: str = "http://orchestrator:8080"):
        self.orchestrator_url = orchestrator_url
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage (in-memory with rotation)
        self.gpu_metrics_history = deque(maxlen=1000)  # Last 1000 measurements
        self.lock_metrics_history = deque(maxlen=1000)
        
        # Monitoring configuration
        self.monitoring_interval = 30  # seconds
        self.alert_thresholds = {
            "gpu_utilization_high": 90.0,
            "memory_utilization_high": 85.0,
            "temperature_high": 80,  # Celsius
            "lock_wait_time_high": 300.0,  # seconds
            "lock_contention_high": 0.8  # ratio
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._alerts_active: Dict[str, datetime] = {}
        
        # Capabilities detection
        self.nvml_available = NVML_AVAILABLE
        self.torch_available = TORCH_AVAILABLE
        
        if not self.nvml_available:
            self.logger.warning("NVML not available - GPU hardware monitoring disabled")
        if not self.torch_available:
            self.logger.warning("PyTorch not available - CUDA monitoring disabled")
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        if self._monitoring_task:
            return
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("GPU monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("GPU monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect GPU hardware metrics
                gpu_metrics = await self.collect_gpu_metrics()
                if gpu_metrics:
                    self.gpu_metrics_history.append(gpu_metrics)
                    await self._check_gpu_alerts(gpu_metrics)
                
                # Collect GPU orchestration metrics
                lock_metrics = await self.collect_lock_metrics()
                if lock_metrics:
                    self.lock_metrics_history.append(lock_metrics)
                    await self._check_lock_alerts(lock_metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def collect_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Collect GPU hardware metrics using NVML and PyTorch"""
        try:
            metrics = GPUMetrics(timestamp=datetime.now().isoformat())
            
            if self.nvml_available:
                # Use NVML for detailed GPU metrics
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.utilization_gpu = util.gpu
                metrics.utilization_memory = util.memory
                
                # Memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics.memory_total_mb = memory_info.total // (1024 * 1024)
                metrics.memory_used_mb = memory_info.used // (1024 * 1024)
                metrics.memory_free_mb = memory_info.free // (1024 * 1024)
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics.temperature_gpu = temp
                except:
                    pass
                
                # Power info
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                    metrics.power_draw = power_draw
                    metrics.power_limit = power_limit
                except:
                    pass
                
                # Running processes
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for proc in procs:
                        try:
                            proc_info = {
                                "pid": proc.pid,
                                "used_memory_mb": proc.usedGpuMemory // (1024 * 1024)
                            }
                            # Try to get process name
                            try:
                                process = psutil.Process(proc.pid)
                                proc_info["name"] = process.name()
                                proc_info["command"] = " ".join(process.cmdline()[:3])
                            except:
                                pass
                            metrics.compute_processes.append(proc_info)
                        except:
                            continue
                except:
                    pass
            
            elif self.torch_available and torch.cuda.is_available():
                # Fallback to PyTorch for basic metrics
                metrics.memory_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                metrics.memory_used_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
                metrics.memory_free_mb = metrics.memory_total_mb - metrics.memory_used_mb
                
                if metrics.memory_total_mb > 0:
                    metrics.utilization_memory = (metrics.memory_used_mb / metrics.memory_total_mb) * 100
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting GPU metrics: {e}")
            return None
    
    async def collect_lock_metrics(self) -> Optional[GPULockMetrics]:
        """Collect GPU orchestration lock metrics"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.orchestrator_url}/metrics") as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    gpu_locks = data.get("gpu_lock", {})
                    
                    metrics = GPULockMetrics(timestamp=datetime.now().isoformat())
                    
                    # Count active locks and pending requests
                    for operation, lock_info in gpu_locks.items():
                        if isinstance(lock_info, dict):
                            if lock_info.get("locked", False):
                                metrics.total_locks_active += 1
                                holder = lock_info.get("holder", "unknown")
                                metrics.locks_by_service[holder] = metrics.locks_by_service.get(holder, 0) + 1
                                metrics.locks_by_operation[operation] = metrics.locks_by_operation.get(operation, 0) + 1
                    
                    # Get pending requests count if available
                    metrics.total_requests_pending = data.get("pending_requests", 0)
                    
                    # Calculate lock contention ratio
                    if metrics.total_locks_active > 0:
                        total_operations = len([op for op in gpu_locks.keys() if isinstance(gpu_locks[op], dict)])
                        if total_operations > 0:
                            metrics.lock_contention_ratio = metrics.total_locks_active / total_operations
                    
                    return metrics
                    
        except Exception as e:
            self.logger.error(f"Error collecting lock metrics: {e}")
            return None
    
    async def _check_gpu_alerts(self, metrics: GPUMetrics):
        """Check GPU metrics against alert thresholds"""
        alerts = []
        
        # GPU utilization alert
        if metrics.utilization_gpu > self.alert_thresholds["gpu_utilization_high"]:
            alerts.append({
                "type": "gpu_utilization_high",
                "message": f"GPU utilization at {metrics.utilization_gpu}%",
                "severity": "warning"
            })
        
        # Memory utilization alert
        if metrics.utilization_memory > self.alert_thresholds["memory_utilization_high"]:
            alerts.append({
                "type": "memory_utilization_high", 
                "message": f"GPU memory utilization at {metrics.utilization_memory}%",
                "severity": "warning"
            })
        
        # Temperature alert
        if metrics.temperature_gpu > self.alert_thresholds["temperature_high"]:
            alerts.append({
                "type": "temperature_high",
                "message": f"GPU temperature at {metrics.temperature_gpu}°C",
                "severity": "critical" if metrics.temperature_gpu > 85 else "warning"
            })
        
        # Process alerts if any
        for alert in alerts:
            await self._handle_alert(alert)
    
    async def _check_lock_alerts(self, metrics: GPULockMetrics):
        """Check lock metrics against alert thresholds"""
        alerts = []
        
        # Lock contention alert
        if metrics.lock_contention_ratio > self.alert_thresholds["lock_contention_high"]:
            alerts.append({
                "type": "lock_contention_high",
                "message": f"High GPU lock contention: {metrics.lock_contention_ratio:.2f}",
                "severity": "warning"
            })
        
        # Pending requests alert
        if metrics.total_requests_pending > 5:
            alerts.append({
                "type": "pending_requests_high",
                "message": f"High number of pending GPU requests: {metrics.total_requests_pending}",
                "severity": "warning"
            })
        
        # Process alerts if any
        for alert in alerts:
            await self._handle_alert(alert)
    
    async def _handle_alert(self, alert: Dict[str, Any]):
        """Handle alert notifications"""
        alert_key = alert["type"]
        now = datetime.now()
        
        # Debounce alerts (only fire once per 5 minutes)
        if alert_key in self._alerts_active:
            if now - self._alerts_active[alert_key] < timedelta(minutes=5):
                return
        
        self._alerts_active[alert_key] = now
        
        # Log the alert
        if alert["severity"] == "critical":
            self.logger.critical(f"GPU Alert: {alert['message']}")
        else:
            self.logger.warning(f"GPU Alert: {alert['message']}")
        
        # Could extend to send notifications, webhooks, etc.
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU health status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
            "capabilities": {
                "nvml_available": self.nvml_available,
                "torch_available": self.torch_available
            },
            "gpu_status": "unknown",
            "orchestration_status": "unknown",
            "alerts": {},
            "recent_metrics": {}
        }
        
        # Get latest GPU metrics
        if self.gpu_metrics_history:
            latest_gpu = self.gpu_metrics_history[-1]
            status["recent_metrics"]["gpu"] = asdict(latest_gpu)
            
            # Determine GPU health status
            if latest_gpu.temperature_gpu > 85:
                status["gpu_status"] = "critical"
            elif (latest_gpu.utilization_gpu > 90 or 
                  latest_gpu.utilization_memory > 85 or 
                  latest_gpu.temperature_gpu > 80):
                status["gpu_status"] = "warning"
            else:
                status["gpu_status"] = "healthy"
        
        # Get latest lock metrics
        if self.lock_metrics_history:
            latest_lock = self.lock_metrics_history[-1]
            status["recent_metrics"]["locks"] = asdict(latest_lock)
            
            # Determine orchestration health status
            if latest_lock.lock_contention_ratio > 0.8 or latest_lock.total_requests_pending > 10:
                status["orchestration_status"] = "warning"
            else:
                status["orchestration_status"] = "healthy"
        
        # Add active alerts
        status["alerts"] = {
            alert_type: alert_time.isoformat() 
            for alert_type, alert_time in self._alerts_active.items()
            if datetime.now() - alert_time < timedelta(minutes=10)
        }
        
        return status
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get summarized metrics for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        # Filter recent metrics
        recent_gpu_metrics = [
            m for m in self.gpu_metrics_history 
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        recent_lock_metrics = [
            m for m in self.lock_metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        summary = {
            "period_minutes": minutes,
            "gpu_summary": {},
            "lock_summary": {},
            "data_points": {
                "gpu_metrics": len(recent_gpu_metrics),
                "lock_metrics": len(recent_lock_metrics)
            }
        }
        
        # GPU metrics summary
        if recent_gpu_metrics:
            gpu_utils = [m.utilization_gpu for m in recent_gpu_metrics]
            mem_utils = [m.utilization_memory for m in recent_gpu_metrics]
            temps = [m.temperature_gpu for m in recent_gpu_metrics if m.temperature_gpu > 0]
            
            summary["gpu_summary"] = {
                "avg_gpu_utilization": sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                "max_gpu_utilization": max(gpu_utils) if gpu_utils else 0,
                "avg_memory_utilization": sum(mem_utils) / len(mem_utils) if mem_utils else 0,
                "max_memory_utilization": max(mem_utils) if mem_utils else 0,
                "avg_temperature": sum(temps) / len(temps) if temps else 0,
                "max_temperature": max(temps) if temps else 0
            }
        
        # Lock metrics summary
        if recent_lock_metrics:
            contention_ratios = [m.lock_contention_ratio for m in recent_lock_metrics]
            pending_counts = [m.total_requests_pending for m in recent_lock_metrics]
            
            summary["lock_summary"] = {
                "avg_contention_ratio": sum(contention_ratios) / len(contention_ratios),
                "max_contention_ratio": max(contention_ratios),
                "avg_pending_requests": sum(pending_counts) / len(pending_counts),
                "max_pending_requests": max(pending_counts),
                "total_measurements": len(recent_lock_metrics)
            }
        
        return summary


# Global monitor instance
_gpu_monitor: Optional[GPUMonitor] = None

async def get_gpu_monitor() -> GPUMonitor:
    """Get or create global GPU monitor instance"""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
        await _gpu_monitor.start_monitoring()
    return _gpu_monitor


# Health check integration
async def gpu_health_check() -> Dict[str, Any]:
    """GPU health check for service health endpoints"""
    try:
        monitor = await get_gpu_monitor()
        health_status = await monitor.get_health_status()
        
        # Determine overall health
        gpu_ok = health_status["gpu_status"] in ["healthy", "warning"]
        orchestration_ok = health_status["orchestration_status"] in ["healthy", "warning"]
        monitoring_ok = health_status["monitoring_active"]
        
        overall_healthy = gpu_ok and orchestration_ok and monitoring_ok
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "gpu_orchestration": health_status,
            "checks": {
                "gpu_hardware": gpu_ok,
                "gpu_orchestration": orchestration_ok,
                "monitoring_active": monitoring_ok
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "checks": {
                "gpu_hardware": False,
                "gpu_orchestration": False,
                "monitoring_active": False
            }
        }