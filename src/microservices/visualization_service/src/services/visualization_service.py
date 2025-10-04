"""
Visualization Service

Provides comprehensive data visualization capabilities for training analytics,
dataset analysis, and model comparison visualization.
"""

import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from datasets import load_from_disk, Dataset
from concurrent.futures import ThreadPoolExecutor

# Enhanced visualization dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import pandas as pd
except ImportError as e:
    px = None
    go = None
    PlotlyJSONEncoder = None
    sns = None
    plt = None
    PCA = None
    TSNE = None
    pd = None

# NOTE: Cross-service imports are removed to maintain microservice boundaries
# These services should be accessed via HTTP API calls through the orchestrator
# from services.training_service import TrainingService
# from services.comparison_service import ComparisonService  
# from services.template_service import TemplateService
from utils.research_logger import research_logger
from utils.memory import get_system_resource_info
from utils.logging import get_logger


class VisualizationError(Exception):
    """Custom exception for visualization-related errors"""
    def __init__(self, message: str, error_type: str = "visualization_error", details: Optional[Dict] = None):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        super().__init__(message)


class VisualizationService:
    """Enhanced service for comprehensive visualization data preparation and analytics"""
    
    def __init__(self, 
                 training_service: Optional[Any] = None,
                 comparison_service: Optional[Any] = None,
                 template_service: Optional[Any] = None):
        self.logger = get_logger(__name__)
        
        # Inject dependencies (will be properly initialized in app.py)
        self.training_service = training_service
        self.comparison_service = comparison_service
        self.template_service = template_service
        
        # Cache for frequently accessed visualization data
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        self.logger.info("VisualizationService initialized")
    
    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if still valid"""
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self._cache_timeout:
                return cache_entry['data']
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        return None
    
    def _set_cached_data(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Set data in cache with timestamp"""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    async def get_training_analytics(self, task_id: str, limit: int = 100) -> Dict[str, Any]:
        """Get comprehensive training analytics for visualization"""
        if not task_id or not isinstance(task_id, str):
            raise VisualizationError(
                "Invalid task_id provided",
                error_type="invalid_parameter",
                details={"task_id": task_id, "type": type(task_id)}
            )
        
        if not isinstance(limit, int) or limit < 1:
            raise VisualizationError(
                "Invalid limit parameter",
                error_type="invalid_parameter", 
                details={"limit": limit, "valid_range": "1-10000"}
            )
        
        try:
            # Check cache first
            cache_key = f"training_analytics_{task_id}_{limit}"
            cached_result = self._get_cached_data(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Get training visualization data from research logger
            training_metrics = research_logger.get_training_visualization_data(task_id, limit)
            if not training_metrics:
                raise VisualizationError(
                    f"No training metrics found for task {task_id}",
                    error_type="data_not_found",
                    details={"task_id": task_id}
                )
            
            # Get resource metrics
            resource_metrics = research_logger.get_recent_resource_metrics(task_id, minutes=60)
            
            # Calculate derived analytics
            analytics = self._calculate_training_analytics(training_metrics, resource_metrics)
            
            return {
                "task_id": task_id,
                "training_metrics": training_metrics,
                "resource_metrics": resource_metrics,
                "analytics": analytics,
                "timestamp": datetime.now().isoformat()
            }
            
        except VisualizationError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            self.logger.error(f"Failed to get training analytics for {task_id}: {e}")
            raise VisualizationError(
                f"Failed to generate training analytics for task {task_id}",
                error_type="analytics_generation_failed",
                details={"original_error": str(e), "task_id": task_id}
            )
    
    def _calculate_training_analytics(self, training_metrics: List[Dict], resource_metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate derived training analytics for visualization"""
        if not training_metrics:
            return {}
        
        try:
            # Extract loss values for trend analysis
            losses = [m['metrics'].get('loss') for m in training_metrics if m['metrics'].get('loss') is not None]
            learning_rates = [m['metrics'].get('learning_rate') for m in training_metrics if m['metrics'].get('learning_rate') is not None]
            
            analytics = {
                "loss_trend": self._calculate_trend(losses) if losses else 0,
                "loss_volatility": np.std(losses) if len(losses) > 1 else 0,
                "current_loss": losses[-1] if losses else None,
                "best_loss": min(losses) if losses else None,
                "avg_learning_rate": np.mean(learning_rates) if learning_rates else None,
                "total_steps": len(training_metrics),
                "training_stability": self._calculate_stability_score(losses) if len(losses) > 5 else None
            }
            
            # Resource utilization analytics
            if resource_metrics:
                gpu_usage = [r['gpu_utilization'] for r in resource_metrics if 'gpu_utilization' in r]
                memory_usage = [r['memory_pressure'] for r in resource_metrics if 'memory_pressure' in r]
                cpu_usage = [r['cpu_load'] for r in resource_metrics if 'cpu_load' in r]
                
                analytics.update({
                    "avg_gpu_utilization": np.mean(gpu_usage) if gpu_usage else 0,
                    "peak_gpu_utilization": max(gpu_usage) if gpu_usage else 0,
                    "avg_memory_pressure": np.mean(memory_usage) if memory_usage else 0,
                    "avg_cpu_load": np.mean(cpu_usage) if cpu_usage else 0,
                    "resource_efficiency_score": self._calculate_efficiency_score(gpu_usage, memory_usage, cpu_usage)
                })
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error calculating training analytics: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1, where -1 is strongly decreasing)"""
        if len(values) < 2:
            return 0
        
        try:
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            return -np.sign(z[0])  # Negative slope is good for loss
        except:
            return 0
    
    def _calculate_stability_score(self, losses: List[float]) -> float:
        """Calculate training stability score (0-1, higher is more stable)"""
        if len(losses) < 5:
            return 0
        
        try:
            # Calculate rolling variance
            window_size = min(10, len(losses) // 2)
            rolling_vars = []
            
            for i in range(window_size, len(losses)):
                window = losses[i-window_size:i]
                rolling_vars.append(np.var(window))
            
            if not rolling_vars:
                return 0
            
            # Stability is inverse of normalized variance
            avg_var = np.mean(rolling_vars)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                return 0
            
            normalized_var = avg_var / (avg_loss ** 2)
            stability = 1 / (1 + normalized_var)
            
            return min(1.0, max(0.0, stability))
            
        except:
            return 0
    
    def _calculate_efficiency_score(self, gpu_usage: List[float], memory_usage: List[float], cpu_usage: List[float]) -> float:
        """Calculate resource efficiency score (0-1)"""
        try:
            scores = []
            
            if gpu_usage:
                # GPU efficiency: prefer high utilization (70-90% is optimal)
                avg_gpu = np.mean(gpu_usage)
                if 70 <= avg_gpu <= 90:
                    gpu_score = 1.0
                elif avg_gpu < 70:
                    gpu_score = avg_gpu / 70
                else:
                    gpu_score = max(0.5, (100 - avg_gpu) / 10)
                scores.append(gpu_score)
            
            if memory_usage:
                # Memory efficiency: prefer moderate usage (50-80% is optimal)
                avg_memory = np.mean(memory_usage)
                if 50 <= avg_memory <= 80:
                    memory_score = 1.0
                elif avg_memory < 50:
                    memory_score = avg_memory / 50
                else:
                    memory_score = max(0.3, (100 - avg_memory) / 20)
                scores.append(memory_score)
            
            if cpu_usage:
                # CPU efficiency: prefer moderate usage
                avg_cpu = np.mean(cpu_usage)
                if avg_cpu <= 80:
                    cpu_score = 1.0
                else:
                    cpu_score = max(0.5, (100 - avg_cpu) / 20)
                scores.append(cpu_score)
            
            return np.mean(scores) if scores else 0
            
        except:
            return 0
    
    async def get_dataset_analytics(self, dataset_path: str) -> Dict[str, Any]:
        """Get comprehensive dataset analytics for visualization"""
        cache_key = f"dataset_analytics_{dataset_path}"
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if datetime.now().timestamp() - timestamp < self._cache_timeout:
                return cached_data
        
        try:
            dataset_path_obj = Path(dataset_path)
            if not dataset_path_obj.exists():
                return {"error": f"Dataset path not found: {dataset_path}"}
            
            self.logger.info(f"Analyzing dataset: {dataset_path}")
            
            # Load dataset
            dataset = load_from_disk(dataset_path)
            
            # Handle DatasetDict (with train/val splits) vs single Dataset
            if hasattr(dataset, 'keys'):
                dataset = dataset['train']
            
            # Basic statistics
            total_samples = len(dataset)
            
            # Analyze text lengths and token distributions
            analytics_data = await self._analyze_dataset_content(dataset)
            
            # Template format analysis
            template_analysis = await self._analyze_template_formats(dataset)
            
            # Quality metrics
            quality_metrics = await self._analyze_dataset_quality(dataset)
            
            result = {
                "dataset_path": dataset_path,
                "total_samples": total_samples,
                "analytics": analytics_data,
                "template_analysis": template_analysis,
                "quality_metrics": quality_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            self._cache[cache_key] = (result, datetime.now().timestamp())
            
            # Log analytics for future reference
            research_logger.log_dataset_analytics(dataset_path, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze dataset {dataset_path}: {e}")
            return {
                "dataset_path": dataset_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_dataset_content(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze dataset content for token distributions and statistics (non-blocking)"""
        try:
            sample_size = min(1000, len(dataset))
            
            # Process in parallel batches to prevent blocking
            def analyze_batch(start_idx: int, end_idx: int) -> Dict[str, List]:
                """Analyze a batch of dataset samples"""
                text_lengths = []
                token_counts = []
                
                for i in range(start_idx, min(end_idx, sample_size)):
                    try:
                        row = dataset[i]
                        
                        # Combine all text fields
                        text_content = ""
                        if 'instruction' in row:
                            text_content += str(row['instruction'])
                        if 'input' in row:
                            text_content += " " + str(row['input'])
                        if 'output' in row:
                            text_content += " " + str(row['output'])
                        if 'response' in row:
                            text_content += " " + str(row['response'])
                        if 'text' in row:
                            text_content = str(row['text'])
                        
                        # Basic token approximation (words * 1.3)
                        word_count = len(text_content.split())
                        token_count = int(word_count * 1.3)
                        
                        text_lengths.append(len(text_content))
                        token_counts.append(token_count)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing dataset row {i}: {e}")
                        continue
                
                return {
                    "text_lengths": text_lengths,
                    "token_counts": token_counts
                }
            
            # Process dataset in parallel batches
            batch_size = 100
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                tasks = []
                for i in range(0, sample_size, batch_size):
                    task = loop.run_in_executor(
                        executor, analyze_batch, i, i + batch_size
                    )
                    tasks.append(task)
                
                # Wait for all batches to complete
                batch_results = await asyncio.gather(*tasks)
            
            # Combine results from all batches
            all_text_lengths = []
            all_token_counts = []
            
            for batch_result in batch_results:
                all_text_lengths.extend(batch_result["text_lengths"])
                all_token_counts.extend(batch_result["token_counts"])
            
            if not all_text_lengths or not all_token_counts:
                return {"error": "No valid samples found in dataset"}
            
            # Calculate statistics
            length_stats = {
                "mean": np.mean(all_text_lengths),
                "median": np.median(all_text_lengths),
                "std_dev": np.std(all_text_lengths),
                "min": min(all_text_lengths),
                "max": max(all_text_lengths)
            }
            
            token_stats = {
                "mean": np.mean(all_token_counts),
                "median": np.median(all_token_counts),
                "std_dev": np.std(all_token_counts),
                "min": min(all_token_counts),
                "max": max(all_token_counts)
            }
            
            # Token distribution histogram (run in thread pool to avoid blocking)
            token_distribution = await loop.run_in_executor(
                None, self._create_distribution_histogram, all_token_counts
            )
            
            return {
                "length_statistics": length_stats,
                "token_statistics": token_stats,
                "token_distribution": token_distribution,
                "sample_size_analyzed": len(all_text_lengths)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing dataset content: {e}")
            return {"error": str(e)}
    
    def _create_distribution_histogram(self, values: List[int], bins: int = 10) -> List[Dict]:
        """Create histogram data for token distribution visualization"""
        if not values:
            return []
        
        try:
            min_val, max_val = min(values), max(values)
            bin_width = (max_val - min_val) / bins if max_val > min_val else 1
            
            histogram = []
            for i in range(bins):
                bin_start = min_val + i * bin_width
                bin_end = min_val + (i + 1) * bin_width
                
                count = sum(1 for v in values if bin_start <= v < bin_end)
                if i == bins - 1:  # Include max value in last bin
                    count = sum(1 for v in values if bin_start <= v <= bin_end)
                
                histogram.append({
                    "range": f"{int(bin_start)}-{int(bin_end)}",
                    "count": count,
                    "percentage": (count / len(values)) * 100
                })
            
            return histogram
            
        except Exception as e:
            self.logger.error(f"Error creating histogram: {e}")
            return []
    
    async def _analyze_template_formats(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze template format distribution in dataset"""
        try:
            format_counts = {}
            sample_examples = {}
            
            # Sample analysis
            sample_size = min(100, len(dataset))
            
            for i in range(sample_size):
                row = dataset[i]
                
                # Identify template format based on available fields
                template_format = "unknown"
                
                if 'instruction' in row and 'output' in row:
                    template_format = "alpaca"
                elif 'messages' in row:
                    template_format = "chatml"
                elif 'conversations' in row:
                    template_format = "sharegpt"
                elif 'query' in row and 'response' in row:
                    template_format = "qwen"
                elif 'text' in row:
                    template_format = "text"
                
                # Count formats
                format_counts[template_format] = format_counts.get(template_format, 0) + 1
                
                # Store examples
                if template_format not in sample_examples:
                    sample_examples[template_format] = {
                        "instruction": row.get('instruction', row.get('query', row.get('text', '')))[:200],
                        "response": row.get('output', row.get('response', ''))[:200],
                        "template_format": template_format
                    }
            
            # Convert to percentage and add colors
            total_samples = sum(format_counts.values())
            template_formats = []
            colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1']
            
            for i, (format_name, count) in enumerate(format_counts.items()):
                template_formats.append({
                    "name": format_name,
                    "count": count,
                    "percentage": (count / total_samples) * 100,
                    "color": colors[i % len(colors)]
                })
            
            return {
                "template_formats": template_formats,
                "sample_examples": list(sample_examples.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing template formats: {e}")
            return {"template_formats": [], "sample_examples": []}
    
    async def _analyze_dataset_quality(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze dataset quality metrics"""
        try:
            # Sample analysis for performance
            sample_size = min(500, len(dataset))
            
            instructions = set()
            responses = []
            empty_count = 0
            
            for i in range(sample_size):
                row = dataset[i]
                
                # Extract instruction and response
                instruction = row.get('instruction', row.get('query', ''))
                response = row.get('output', row.get('response', ''))
                
                if instruction:
                    instructions.add(instruction.lower().strip())
                if response:
                    responses.append(response)
                else:
                    empty_count += 1
            
            # Calculate quality metrics
            unique_instructions = len(instructions)
            duplicate_ratio = 1 - (unique_instructions / max(sample_size, 1))
            avg_response_length = np.mean([len(r.split()) for r in responses]) if responses else 0
            
            return {
                "unique_instructions": unique_instructions,
                "duplicate_ratio": duplicate_ratio,
                "avg_response_length": avg_response_length,
                "empty_responses": empty_count,
                "quality_score": (1 - duplicate_ratio) * (1 - empty_count / max(sample_size, 1))
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing dataset quality: {e}")
            return {}
    
    async def get_comparison_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive comparison analytics for visualization"""
        try:
            # Get comparison statistics from research logger
            comparison_stats = research_logger.get_model_comparison_stats()
            
            # Filter recent comparisons if needed
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            
            # Get detailed comparison metrics (this would need to be implemented based on your data structure)
            comparison_metrics = await self._get_detailed_comparison_metrics(cutoff_time)
            
            return {
                "comparison_stats": comparison_stats,
                "detailed_metrics": comparison_metrics,
                "time_range_hours": time_range_hours,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get comparison analytics: {e}")
            return {"error": str(e)}
    
    async def _get_detailed_comparison_metrics(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Get detailed comparison metrics for visualization"""
        try:
            # This would parse through comparison log files
            comparison_files = list(research_logger.base_log_dir.glob("comparisons/*.json"))
            metrics = []
            
            for file_path in comparison_files[-50:]:  # Get last 50 comparisons
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    
                    # Extract metrics for visualization
                    metric = {
                        "session_id": log_data.get("session_id"),
                        "base_time_ms": log_data.get("base_generation_time_ms", 0),
                        "finetuned_time_ms": log_data.get("finetuned_generation_time_ms", 0),
                        "base_response_length": len(log_data.get("base_response", "")),
                        "finetuned_response_length": len(log_data.get("finetuned_response", "")),
                        "similarity_score": log_data.get("response_similarity", 0),
                        "timestamp": log_data.get("timestamp"),
                        "user_message_length": len(log_data.get("user_message", ""))
                    }
                    
                    metrics.append(metric)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing comparison file {file_path}: {e}")
                    continue
            
            return sorted(metrics, key=lambda x: x.get("timestamp", ""))
            
        except Exception as e:
            self.logger.error(f"Error getting detailed comparison metrics: {e}")
            return []
    
    async def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get current system health metrics for dashboard"""
        try:
            resource_info = get_system_resource_info()
            
            return {
                "system_resources": resource_info,
                "timestamp": datetime.now().isoformat(),
                "health_score": self._calculate_system_health_score(resource_info)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system health metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_system_health_score(self, resource_info: Dict) -> float:
        """Calculate overall system health score (0-1)"""
        try:
            scores = []
            
            # CPU health (prefer < 80% usage)
            cpu_percent = resource_info.get("cpu", {}).get("percent", 0)
            cpu_score = max(0, min(1, (100 - cpu_percent) / 100))
            scores.append(cpu_score)
            
            # Memory health (prefer < 85% usage)
            memory_percent = resource_info.get("memory", {}).get("percent", 0)
            memory_score = max(0, min(1, (100 - memory_percent) / 100))
            scores.append(memory_score)
            
            # GPU health
            gpu_info = resource_info.get("gpu", {})
            if gpu_info.get("gpu_available"):
                gpu_usage_percent = (gpu_info.get("allocated_mb", 0) / max(gpu_info.get("total_mb", 1), 1)) * 100
                gpu_score = max(0, min(1, (100 - gpu_usage_percent) / 100))
                scores.append(gpu_score)
            
            return np.mean(scores) if scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 0.5
    
    def list_dashboards(self) -> List[str]:
        """List available dashboard types for health checks"""
        return [
            "training",
            "comparison", 
            "system",
            "dataset_analytics",
            "resource_monitoring"
        ]
    
    def get_cache_size(self) -> int:
        """Get current cache size for health checks"""
        return len(self._cache)
    
    async def get_dashboard(self, dashboard_type: str) -> Dict[str, Any]:
        """Get dashboard configuration and data"""
        try:
            if dashboard_type == "training":
                return await self._get_training_dashboard()
            elif dashboard_type == "comparison":
                return await self._get_comparison_dashboard()
            elif dashboard_type == "system":
                return await self._get_system_dashboard()
            elif dashboard_type == "dataset_analytics":
                return await self._get_dataset_dashboard()
            elif dashboard_type == "resource_monitoring":
                return await self._get_resource_dashboard()
            else:
                raise ValueError(f"Unknown dashboard type: {dashboard_type}")
        except Exception as e:
            self.logger.error(f"Error getting dashboard {dashboard_type}: {e}")
            return {"error": str(e), "dashboard_type": dashboard_type}
    
    async def create_dashboard(self, dashboard_type: str, config: Dict[str, Any]) -> bool:
        """Create or update dashboard configuration"""
        try:
            # For now, just validate and return success
            valid_types = self.list_dashboards()
            if dashboard_type not in valid_types:
                return False
            
            # Cache the dashboard config
            cache_key = f"dashboard_config_{dashboard_type}"
            self._set_cached_data(cache_key, config)
            
            self.logger.info(f"Dashboard {dashboard_type} created/updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error creating dashboard {dashboard_type}: {e}")
            return False
    
    async def generate_chart(self, chart_type: str, start_time: str = None, 
                           end_time: str = None, filters: Dict = None) -> Dict[str, Any]:
        """Generate chart data"""
        try:
            filters = filters or {}
            
            if chart_type == "training_loss":
                return await self._generate_training_loss_chart(start_time, end_time, filters)
            elif chart_type == "resource_usage":
                return await self._generate_resource_usage_chart(start_time, end_time, filters)
            elif chart_type == "comparison_metrics":
                return await self._generate_comparison_metrics_chart(start_time, end_time, filters)
            else:
                return {"error": f"Unknown chart type: {chart_type}"}
        except Exception as e:
            self.logger.error(f"Error generating chart {chart_type}: {e}")
            return {"error": str(e)}
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        return await self.get_system_health_metrics()
    
    async def get_training_metrics(self, start_time: datetime = None, 
                                 end_time: datetime = None) -> Dict[str, Any]:
        """Get training-related metrics"""
        try:
            # Get recent training data
            training_data = research_logger.get_recent_training_logs(hours=24)
            return {
                "training_data": training_data,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting training metrics: {e}")
            return {"error": str(e)}
    
    async def get_comparison_metrics(self, start_time: datetime = None,
                                   end_time: datetime = None) -> Dict[str, Any]:
        """Get comparison-related metrics"""
        try:
            hours_back = 24
            if start_time:
                hours_back = max(1, int((datetime.now() - start_time).total_seconds() / 3600))
            
            return await self.get_comparison_analytics(hours_back)
        except Exception as e:
            self.logger.error(f"Error getting comparison metrics: {e}")
            return {"error": str(e)}
    
    async def export_data(self, export_type: str, format: str = "json",
                         start_time: datetime = None, end_time: datetime = None) -> str:
        """Export data in various formats"""
        try:
            if export_type == "training":
                data = await self.get_training_metrics(start_time, end_time)
            elif export_type == "comparison":
                data = await self.get_comparison_metrics(start_time, end_time)
            elif export_type == "system":
                data = await self.get_system_metrics()
            else:
                data = {"error": f"Unknown export type: {export_type}"}
            
            if format == "json":
                return json.dumps(data, indent=2, default=str)
            elif format == "csv":
                # Simple CSV export (would need proper CSV formatting)
                return f"Export type: {export_type}\nData: {json.dumps(data)}"
            elif format == "pdf":
                # Mock PDF export
                return f"PDF export of {export_type} data - {len(str(data))} bytes"
            else:
                return json.dumps({"error": f"Unknown format: {format}"})
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return json.dumps({"error": str(e)})
    
    async def stream_training_metrics(self, job_id: str):
        """Stream real-time training metrics"""
        try:
            # Mock streaming for now
            for i in range(5):
                yield json.dumps({
                    "job_id": job_id,
                    "step": i,
                    "loss": 1.0 - (i * 0.1),
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(1)
        except Exception as e:
            self.logger.error(f"Error streaming training metrics: {e}")
            yield json.dumps({"error": str(e)})
    
    async def stream_comparison_metrics(self, session_id: str):
        """Stream real-time comparison metrics"""
        try:
            # Mock streaming for now
            for i in range(3):
                yield json.dumps({
                    "session_id": session_id,
                    "active_comparisons": i + 1,
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(2)
        except Exception as e:
            self.logger.error(f"Error streaming comparison metrics: {e}")
            yield json.dumps({"error": str(e)})
    
    async def generate_training_report(self, job_id: str) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        try:
            analytics = await self.get_training_analytics(job_id)
            return {
                "job_id": job_id,
                "report_type": "training",
                "analytics": analytics,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error generating training report: {e}")
            return {"error": str(e)}
    
    async def generate_comparison_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        try:
            analytics = await self.get_comparison_analytics()
            return {
                "session_id": session_id,
                "report_type": "comparison", 
                "analytics": analytics,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error generating comparison report: {e}")
            return {"error": str(e)}
    
    async def clear_cache(self):
        """Clear visualization cache"""
        try:
            self._cache.clear()
            self.logger.info("Visualization cache cleared")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    async def generate_tensor_visualization(self, dataset_path: str, embeddings_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive tensor visualizations"""
        try:
            dataset_path_obj = Path(dataset_path)
            if not dataset_path_obj.exists():
                return {"error": f"Dataset path not found: {dataset_path}"}
                
            # Load embeddings if available
            embeddings = None
            if embeddings_path and Path(embeddings_path).exists():
                try:
                    embeddings = np.load(embeddings_path)
                    self.logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
                except Exception as e:
                    self.logger.warning(f"Failed to load embeddings: {e}")
            
            # Load preprocessing analytics if available
            analytics_path = dataset_path_obj / "preprocessing_analytics.json"
            preprocessing_analytics = {}
            if analytics_path.exists():
                with open(analytics_path, 'r') as f:
                    preprocessing_analytics = json.load(f)
            
            visualization_data = {}
            
            # Generate tensor visualizations if embeddings are available
            if embeddings is not None and px is not None:
                # PCA visualization
                pca_vis = await self._generate_pca_visualization(embeddings)
                visualization_data["pca_analysis"] = pca_vis
                
                # t-SNE visualization (for smaller datasets)
                if embeddings.shape[0] <= 5000:
                    tsne_vis = await self._generate_tsne_visualization(embeddings)
                    visualization_data["tsne_analysis"] = tsne_vis
                
                # Correlation heatmap
                correlation_vis = await self._generate_correlation_heatmap(embeddings)
                visualization_data["correlation_heatmap"] = correlation_vis
                
                # Embedding distribution
                distribution_vis = await self._generate_embedding_distribution(embeddings)
                visualization_data["embedding_distribution"] = distribution_vis
            
            # Add preprocessing analytics if available
            if preprocessing_analytics:
                visualization_data["preprocessing_summary"] = preprocessing_analytics
            
            return {
                "status": "success",
                "dataset_path": dataset_path,
                "has_embeddings": embeddings is not None,
                "embedding_shape": list(embeddings.shape) if embeddings is not None else None,
                "visualization_data": visualization_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating tensor visualization: {e}")
            return {"error": str(e), "status": "error"}
    
    async def _generate_pca_visualization(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Generate PCA visualization of embeddings"""
        if PCA is None or px is None:
            return {"error": "PCA or plotly not available"}
        
        try:
            # Apply PCA
            pca = PCA(n_components=min(3, embeddings.shape[1]))
            pca_embeddings = pca.fit_transform(embeddings)
            
            # Create 2D scatter plot
            if pca_embeddings.shape[1] >= 2:
                fig_2d = px.scatter(
                    x=pca_embeddings[:, 0],
                    y=pca_embeddings[:, 1],
                    title="PCA Visualization (2D)",
                    labels={"x": "PC1", "y": "PC2"},
                    opacity=0.6
                )
                fig_2d_json = fig_2d.to_json()
            else:
                fig_2d_json = None
            
            # Create 3D scatter plot if possible
            if pca_embeddings.shape[1] >= 3:
                fig_3d = px.scatter_3d(
                    x=pca_embeddings[:, 0],
                    y=pca_embeddings[:, 1],
                    z=pca_embeddings[:, 2],
                    title="PCA Visualization (3D)",
                    labels={"x": "PC1", "y": "PC2", "z": "PC3"},
                    opacity=0.6
                )
                fig_3d_json = fig_3d.to_json()
            else:
                fig_3d_json = None
            
            return {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
                "pca_components": pca_embeddings.tolist()[:100],  # Limit for JSON size
                "plot_2d": fig_2d_json,
                "plot_3d": fig_3d_json,
                "n_components": pca.n_components_
            }
            
        except Exception as e:
            self.logger.error(f"Error in PCA visualization: {e}")
            return {"error": str(e)}
    
    async def _generate_tsne_visualization(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Generate t-SNE visualization of embeddings"""
        if TSNE is None or px is None:
            return {"error": "t-SNE or plotly not available"}
        
        try:
            # Apply t-SNE (limit to 2D for performance)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0] - 1))
            tsne_embeddings = tsne.fit_transform(embeddings)
            
            # Create scatter plot
            fig = px.scatter(
                x=tsne_embeddings[:, 0],
                y=tsne_embeddings[:, 1],
                title="t-SNE Visualization",
                labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                opacity=0.6
            )
            
            return {
                "tsne_components": tsne_embeddings.tolist(),
                "plot": fig.to_json(),
                "perplexity": tsne.perplexity
            }
            
        except Exception as e:
            self.logger.error(f"Error in t-SNE visualization: {e}")
            return {"error": str(e)}
    
    async def _generate_correlation_heatmap(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Generate correlation heatmap for embeddings"""
        if px is None:
            return {"error": "plotly not available"}
        
        try:
            # Calculate correlation matrix (limit size for performance)
            sample_size = min(1000, embeddings.shape[0])
            if sample_size < embeddings.shape[0]:
                # Random sample
                indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
                sample_embeddings = embeddings[indices]
            else:
                sample_embeddings = embeddings
            
            correlation_matrix = np.corrcoef(sample_embeddings)
            
            # Create heatmap
            fig = px.imshow(
                correlation_matrix,
                title=f"Embedding Correlation Heatmap (Sample: {sample_size})",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            
            # Calculate statistics
            correlation_stats = {
                "mean_correlation": float(np.mean(correlation_matrix[correlation_matrix != 1.0])),
                "std_correlation": float(np.std(correlation_matrix[correlation_matrix != 1.0])),
                "max_correlation": float(np.max(correlation_matrix[correlation_matrix != 1.0])),
                "min_correlation": float(np.min(correlation_matrix)),
                "sample_size": sample_size
            }
            
            return {
                "correlation_matrix": correlation_matrix.tolist() if sample_size <= 100 else None,
                "correlation_stats": correlation_stats,
                "heatmap_plot": fig.to_json()
            }
            
        except Exception as e:
            self.logger.error(f"Error in correlation heatmap: {e}")
            return {"error": str(e)}
    
    async def _generate_embedding_distribution(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Generate embedding distribution visualizations"""
        if px is None:
            return {"error": "plotly not available"}
        
        try:
            # Calculate embedding norms
            embedding_norms = np.linalg.norm(embeddings, axis=1)
            
            # Create histogram of embedding norms
            fig_hist = px.histogram(
                x=embedding_norms,
                title="Distribution of Embedding Norms",
                labels={"x": "Embedding Norm", "y": "Frequency"}
            )
            
            # Calculate statistics
            distribution_stats = {
                "mean_norm": float(np.mean(embedding_norms)),
                "std_norm": float(np.std(embedding_norms)),
                "min_norm": float(np.min(embedding_norms)),
                "max_norm": float(np.max(embedding_norms)),
                "median_norm": float(np.median(embedding_norms))
            }
            
            # Dimension-wise statistics
            dim_means = np.mean(embeddings, axis=0)
            dim_stds = np.std(embeddings, axis=0)
            
            # Create dimension statistics plot
            fig_dims = px.line(
                y=dim_means,
                title="Mean Values Across Embedding Dimensions",
                labels={"x": "Dimension", "y": "Mean Value"}
            )
            
            return {
                "distribution_stats": distribution_stats,
                "norm_histogram": fig_hist.to_json(),
                "dimension_means": dim_means.tolist(),
                "dimension_stds": dim_stds.tolist(),
                "dimension_plot": fig_dims.to_json()
            }
            
        except Exception as e:
            self.logger.error(f"Error in embedding distribution: {e}")
            return {"error": str(e)}
    
    async def generate_correlation_matrix_visualization(self, dataset_path: str) -> Dict[str, Any]:
        """Generate correlation matrix visualization for dataset features"""
        try:
            dataset_path_obj = Path(dataset_path)
            
            # Load dataset
            dataset = load_from_disk(str(dataset_path_obj))
            
            # Handle DatasetDict (with train/val splits) vs single Dataset
            if hasattr(dataset, 'keys'):
                dataset = dataset['train']
            
            # Convert to pandas for correlation analysis
            df = dataset.to_pandas()
            
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found for correlation analysis"}
            
            # Calculate correlation matrix
            correlation_matrix = df[numeric_cols].corr()
            
            # Create heatmap
            if px is not None:
                fig = px.imshow(
                    correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                plot_json = fig.to_json()
            else:
                plot_json = None
            
            return {
                "status": "success",
                "correlation_matrix": correlation_matrix.to_dict(),
                "feature_names": list(correlation_matrix.columns),
                "plot": plot_json,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating correlation matrix: {e}")
            return {"error": str(e), "status": "error"}
    
    async def generate_token_distribution_visualization(self, dataset_path: str) -> Dict[str, Any]:
        """Generate token distribution visualizations"""
        try:
            # Load preprocessing analytics
            analytics_path = Path(dataset_path) / "preprocessing_analytics.json"
            if not analytics_path.exists():
                return {"error": "No preprocessing analytics found"}
            
            with open(analytics_path, 'r') as f:
                analytics = json.load(f)
            
            token_distribution = analytics.get("token_distribution", {})
            if not token_distribution:
                return {"error": "No token distribution data found"}
            
            visualization_data = {}
            
            # Token histogram
            if "token_distribution_histogram" in token_distribution and px is not None:
                hist_data = token_distribution["token_distribution_histogram"]
                
                # Extract data for plotting
                ranges = [item["range"] for item in hist_data]
                counts = [item["count"] for item in hist_data]
                
                fig = px.bar(
                    x=ranges,
                    y=counts,
                    title="Token Distribution Histogram",
                    labels={"x": "Token Range", "y": "Count"}
                )
                visualization_data["histogram_plot"] = fig.to_json()
            
            # Token statistics
            token_stats = token_distribution.get("token_statistics", {})
            visualization_data["token_statistics"] = token_stats
            
            # Vocabulary statistics
            vocab_stats = token_distribution.get("vocabulary_stats", {})
            visualization_data["vocabulary_statistics"] = vocab_stats
            
            return {
                "status": "success",
                "visualization_data": visualization_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating token distribution visualization: {e}")
            return {"error": str(e), "status": "error"}

    async def cleanup(self):
        """Cleanup visualization service resources"""
        try:
            await self.clear_cache()
            self.logger.info("VisualizationService cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    # Helper methods for dashboard generation
    async def _get_training_dashboard(self) -> Dict[str, Any]:
        """Get training dashboard configuration"""
        return {
            "type": "training",
            "widgets": [
                {"type": "loss_chart", "title": "Training Loss"},
                {"type": "metrics_table", "title": "Training Metrics"},
                {"type": "resource_usage", "title": "Resource Usage"}
            ],
            "refresh_interval": 10
        }
    
    async def _get_comparison_dashboard(self) -> Dict[str, Any]:
        """Get comparison dashboard configuration"""
        return {
            "type": "comparison",
            "widgets": [
                {"type": "response_comparison", "title": "Model Responses"},
                {"type": "performance_metrics", "title": "Performance Comparison"},
                {"type": "session_history", "title": "Session History"}
            ],
            "refresh_interval": 5
        }
    
    async def _get_system_dashboard(self) -> Dict[str, Any]:
        """Get system dashboard configuration"""
        return {
            "type": "system",
            "widgets": [
                {"type": "system_health", "title": "System Health"},
                {"type": "resource_monitoring", "title": "Resource Monitoring"},
                {"type": "service_status", "title": "Service Status"}
            ],
            "refresh_interval": 15
        }
    
    async def _get_dataset_dashboard(self) -> Dict[str, Any]:
        """Get dataset analytics dashboard configuration"""
        return {
            "type": "dataset_analytics",
            "widgets": [
                {"type": "dataset_overview", "title": "Dataset Overview"},
                {"type": "token_distribution", "title": "Token Distribution"},
                {"type": "quality_metrics", "title": "Quality Metrics"}
            ],
            "refresh_interval": 60
        }
    
    async def _get_resource_dashboard(self) -> Dict[str, Any]:
        """Get resource monitoring dashboard configuration"""
        return {
            "type": "resource_monitoring",
            "widgets": [
                {"type": "cpu_usage", "title": "CPU Usage"},
                {"type": "memory_usage", "title": "Memory Usage"},
                {"type": "gpu_usage", "title": "GPU Usage"}
            ],
            "refresh_interval": 5
        }
    
    # Helper methods for chart generation
    async def _generate_training_loss_chart(self, start_time: str, end_time: str, filters: Dict) -> Dict[str, Any]:
        """Generate training loss chart data"""
        return {
            "chart_type": "training_loss",
            "data": [
                {"step": i, "loss": 1.0 - (i * 0.1)} for i in range(10)
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_resource_usage_chart(self, start_time: str, end_time: str, filters: Dict) -> Dict[str, Any]:
        """Generate resource usage chart data"""
        return {
            "chart_type": "resource_usage",
            "data": [
                {"time": i, "cpu": 50 + (i * 5), "memory": 60 + (i * 3)} for i in range(10)
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_comparison_metrics_chart(self, start_time: str, end_time: str, filters: Dict) -> Dict[str, Any]:
        """Generate comparison metrics chart data"""
        return {
            "chart_type": "comparison_metrics",
            "data": [
                {"comparison": i, "base_time": 100 + (i * 10), "finetuned_time": 120 + (i * 8)} for i in range(10)
            ],
            "timestamp": datetime.now().isoformat()
        }