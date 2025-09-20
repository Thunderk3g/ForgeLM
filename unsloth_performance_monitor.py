#!/usr/bin/env python3
"""
Unsloth Performance Monitor for RTX 4060 8GB
Real-time monitoring of training performance and memory usage
"""

import time
import torch
import psutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class UnslothPerformanceMonitor:
    """Monitor Unsloth training performance and resource usage"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.start_time = time.time()
        self.metrics_history = []
        self.alerts = []
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
        }
        
        # GPU metrics
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            metrics.update({
                "gpu_memory_allocated_gb": gpu_memory_allocated,
                "gpu_memory_reserved_gb": gpu_memory_reserved,
                "gpu_memory_total_gb": gpu_memory_total,
                "gpu_utilization_pct": (gpu_memory_allocated / gpu_memory_total) * 100,
                "gpu_name": torch.cuda.get_device_properties(0).name
            })
            
            # Check for memory pressure on RTX 4060 8GB
            if gpu_memory_total <= 8.5 and gpu_memory_allocated > 7.0:
                self.alerts.append({
                    "type": "memory_pressure",
                    "message": f"High GPU memory usage: {gpu_memory_allocated:.1f}GB / {gpu_memory_total:.1f}GB",
                    "timestamp": metrics["timestamp"]
                })
        
        # System metrics
        system_memory = psutil.virtual_memory()
        metrics.update({
            "system_memory_used_gb": system_memory.used / (1024**3),
            "system_memory_available_gb": system_memory.available / (1024**3),
            "system_memory_percent": system_memory.percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1)
        })
        
        # Training metrics from status file if available
        status_file = Path("logs") / f"{self.task_id}_status.json"
        if status_file.exists():
            try:
                with open(status_file) as f:
                    status_data = json.load(f)
                
                metrics.update({
                    "training_status": status_data.get("status"),
                    "current_step": status_data.get("current_step"),
                    "total_steps": status_data.get("total_steps"),
                    "current_epoch": status_data.get("current_epoch"),
                    "loss": status_data.get("loss"),
                    "learning_rate": status_data.get("learning_rate"),
                    "progress_percentage": status_data.get("progress_percentage", 0)
                })
                
                # Calculate training speed
                if status_data.get("current_step") and metrics["elapsed_time"] > 0:
                    steps_per_second = status_data["current_step"] / metrics["elapsed_time"]
                    metrics["steps_per_second"] = steps_per_second
                    
                    # Estimate completion time
                    if status_data.get("total_steps"):
                        remaining_steps = status_data["total_steps"] - status_data["current_step"]
                        eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                        metrics["eta_minutes"] = eta_seconds / 60
                
            except Exception as e:
                metrics["status_file_error"] = str(e)
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_performance_issues(self, metrics: Dict[str, Any]) -> None:
        """Check for performance issues and generate alerts"""
        
        # Check for OOM risk
        if metrics.get("gpu_utilization_pct", 0) > 95:
            self.alerts.append({
                "type": "oom_risk",
                "message": "GPU memory utilization >95% - OOM risk high",
                "recommendation": "Consider reducing batch_size or max_seq_length",
                "timestamp": metrics["timestamp"]
            })
        
        # Check training speed
        if metrics.get("steps_per_second", 0) < 0.1:
            self.alerts.append({
                "type": "slow_training",
                "message": f"Training speed very slow: {metrics.get('steps_per_second', 0):.3f} steps/sec",
                "recommendation": "Check if packing is enabled and LoRA rank is appropriate",
                "timestamp": metrics["timestamp"]
            })
        
        # Check for training stall
        if len(self.metrics_history) > 10:
            recent_steps = [m.get("current_step", 0) for m in self.metrics_history[-5:]]
            if len(set(recent_steps)) == 1 and recent_steps[0] > 0:
                self.alerts.append({
                    "type": "training_stall",
                    "message": "Training appears to have stalled",
                    "timestamp": metrics["timestamp"]
                })
    
    def print_status(self, metrics: Dict[str, Any]) -> None:
        """Print current status to console"""
        print(f"\n🦥 Unsloth Performance Monitor - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # Training progress
        if metrics.get("training_status"):
            progress = metrics.get("progress_percentage", 0)
            print(f"📊 Status: {metrics['training_status']} ({progress:.1f}%)")
            
            if metrics.get("current_step") and metrics.get("total_steps"):
                print(f"📈 Steps: {metrics['current_step']}/{metrics['total_steps']}")
            
            if metrics.get("loss"):
                print(f"📉 Loss: {metrics['loss']:.4f}")
            
            if metrics.get("steps_per_second"):
                print(f"⚡ Speed: {metrics['steps_per_second']:.3f} steps/sec")
            
            if metrics.get("eta_minutes"):
                print(f"⏱️  ETA: {metrics['eta_minutes']:.1f} minutes")
        
        # Resource usage
        if metrics.get("gpu_memory_allocated_gb"):
            gpu_usage = metrics["gpu_utilization_pct"]
            gpu_allocated = metrics["gpu_memory_allocated_gb"]
            gpu_total = metrics["gpu_memory_total_gb"]
            
            print(f"🖥️  GPU: {gpu_allocated:.1f}GB / {gpu_total:.1f}GB ({gpu_usage:.1f}%)")
            
            # Memory usage bar
            bar_length = 30
            filled = int(bar_length * gpu_usage / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"     [{bar}]")
        
        # System resources
        sys_mem_pct = metrics.get("system_memory_percent", 0)
        cpu_pct = metrics.get("cpu_percent", 0)
        print(f"💾 RAM: {sys_mem_pct:.1f}% | CPU: {cpu_pct:.1f}%")
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if 
                        (datetime.now() - datetime.fromisoformat(a["timestamp"])).seconds < 300]
        
        if recent_alerts:
            print("\n⚠️  Recent Alerts:")
            for alert in recent_alerts[-3:]:  # Show last 3 alerts
                print(f"   {alert['type']}: {alert['message']}")
                if alert.get("recommendation"):
                    print(f"   💡 {alert['recommendation']}")
    
    def save_metrics(self) -> None:
        """Save metrics history to file"""
        metrics_file = Path("logs") / f"{self.task_id}_performance_metrics.json"
        metrics_file.parent.mkdir(exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump({
                "task_id": self.task_id,
                "start_time": self.start_time,
                "metrics_history": self.metrics_history,
                "alerts": self.alerts
            }, f, indent=2)
    
    def run_monitoring(self, interval: int = 30, max_duration: int = 3600) -> None:
        """Run continuous monitoring"""
        print(f"🚀 Starting Unsloth performance monitoring for task: {self.task_id}")
        print(f"📊 Monitoring interval: {interval} seconds")
        print(f"⏱️  Max duration: {max_duration} seconds")
        
        try:
            while time.time() - self.start_time < max_duration:
                metrics = self.collect_metrics()
                self.check_performance_issues(metrics)
                self.print_status(metrics)
                
                # Save metrics periodically
                if len(self.metrics_history) % 10 == 0:
                    self.save_metrics()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        finally:
            self.save_metrics()
            print(f"💾 Metrics saved to logs/{self.task_id}_performance_metrics.json")

def main():
    """Main monitoring function"""
    import sys
    
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
    else:
        task_id = f"monitor_{int(time.time())}"
    
    monitor = UnslothPerformanceMonitor(task_id)
    monitor.run_monitoring()

if __name__ == "__main__":
    main()