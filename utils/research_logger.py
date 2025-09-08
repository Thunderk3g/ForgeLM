"""
Research-focused logging utilities for model comparison analysis
"""

import json
import os
import csv
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib

from models.schemas import ResearchLogEntry, ComparisonResponse, ChatMessage
from utils.logging import get_logger


class ResearchLogger:
    """Enhanced logger for research purposes with structured data collection"""
    
    def __init__(self, base_log_dir: str = "logs/research"):
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(__name__)
        
        # Create subdirectories for different types of research data
        (self.base_log_dir / "comparisons").mkdir(exist_ok=True)
        (self.base_log_dir / "sessions").mkdir(exist_ok=True)
        (self.base_log_dir / "metrics").mkdir(exist_ok=True)
        (self.base_log_dir / "exports").mkdir(exist_ok=True)
        (self.base_log_dir / "visualizations").mkdir(exist_ok=True)
        (self.base_log_dir / "training_analytics").mkdir(exist_ok=True)
        (self.base_log_dir / "dataset_analytics").mkdir(exist_ok=True)
        
        self.logger.info(f"Research logger initialized with base directory: {self.base_log_dir}")
    
    def log_comparison(self, 
                      session_id: str,
                      user_message: str,
                      comparison_response: ComparisonResponse,
                      model_pair_info: Dict[str, str],
                      generation_params: Dict[str, Any]) -> str:
        """Log a comparison interaction for research analysis"""
        
        try:
            # Create research log entry
            log_entry = ResearchLogEntry(
                session_id=session_id,
                model_pair=model_pair_info,
                user_message=user_message,
                base_response=comparison_response.base_response,
                finetuned_response=comparison_response.finetuned_response,
                base_generation_time_ms=self._extract_base_time(comparison_response.generation_time_ms),
                finetuned_generation_time_ms=self._extract_finetuned_time(comparison_response.generation_time_ms),
                base_token_count=len(comparison_response.base_response.split()),  # Rough approximation
                finetuned_token_count=len(comparison_response.finetuned_response.split()),
                generation_params=generation_params,
                model_versions={
                    "base_model": model_pair_info.get("base_model", "unknown"),
                    "finetuned_model": model_pair_info.get("adapter_name", "unknown"),
                    "pair_id": model_pair_info.get("pair_id", "unknown")
                }
            )
            
            # Calculate analysis metrics
            log_entry.response_similarity = self._calculate_similarity(
                comparison_response.base_response, 
                comparison_response.finetuned_response
            )
            log_entry.length_ratio = len(comparison_response.finetuned_response) / max(1, len(comparison_response.base_response))
            
            # Save to JSON file
            log_file = self.base_log_dir / "comparisons" / f"{log_entry.log_id}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_entry.dict(), f, indent=2, ensure_ascii=False)
            
            # Append to daily CSV for easy analysis
            self._append_to_daily_csv(log_entry)
            
            self.logger.info(f"Logged comparison for research: {log_entry.log_id}")
            return log_entry.log_id
            
        except Exception as e:
            self.logger.error(f"Failed to log comparison for research: {e}")
            return ""
    
    def _extract_base_time(self, generation_time: Any) -> float:
        """Extract base model generation time from response"""
        if isinstance(generation_time, dict):
            return generation_time.get("base", 0.0)
        elif isinstance(generation_time, (int, float)):
            return float(generation_time) / 2  # Assume split time
        return 0.0
    
    def _extract_finetuned_time(self, generation_time: Any) -> float:
        """Extract finetuned model generation time from response"""
        if isinstance(generation_time, dict):
            return generation_time.get("finetuned", 0.0)
        elif isinstance(generation_time, (int, float)):
            return float(generation_time) / 2  # Assume split time
        return 0.0
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic similarity between two responses"""
        try:
            # Simple word-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
    
    def _append_to_daily_csv(self, log_entry: ResearchLogEntry):
        """Append log entry to daily CSV file for easy analysis"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            csv_file = self.base_log_dir / "exports" / f"comparisons_{today}.csv"
            
            # Check if file exists to write header
            write_header = not csv_file.exists()
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                if write_header:
                    writer.writerow([
                        'timestamp', 'session_id', 'base_model', 'finetuned_model',
                        'user_message_length', 'base_response_length', 'finetuned_response_length',
                        'base_time_ms', 'finetuned_time_ms', 'similarity', 'length_ratio',
                        'temperature', 'max_tokens', 'user_message_hash'
                    ])
                
                # Hash user message for privacy if needed
                user_hash = hashlib.md5(log_entry.user_message.encode()).hexdigest()[:8]
                
                writer.writerow([
                    log_entry.timestamp,
                    log_entry.session_id,
                    log_entry.model_versions.get("base_model", ""),
                    log_entry.model_versions.get("finetuned_model", ""),
                    len(log_entry.user_message),
                    len(log_entry.base_response),
                    len(log_entry.finetuned_response),
                    log_entry.base_generation_time_ms,
                    log_entry.finetuned_generation_time_ms,
                    log_entry.response_similarity or 0.0,
                    log_entry.length_ratio or 0.0,
                    log_entry.generation_params.get("temperature", 0.0),
                    log_entry.generation_params.get("max_new_tokens", 0),
                    user_hash
                ])
                
        except Exception as e:
            self.logger.error(f"Failed to append to daily CSV: {e}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get research summary for a specific session"""
        try:
            comparison_files = list((self.base_log_dir / "comparisons").glob("*.json"))
            session_logs = []
            
            for file_path in comparison_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                        if log_data.get('session_id') == session_id:
                            session_logs.append(log_data)
                except Exception:
                    continue
            
            if not session_logs:
                return {"session_id": session_id, "total_comparisons": 0}
            
            # Calculate summary statistics
            total_comparisons = len(session_logs)
            avg_similarity = sum(log.get('response_similarity', 0) for log in session_logs) / total_comparisons
            avg_length_ratio = sum(log.get('length_ratio', 1) for log in session_logs) / total_comparisons
            avg_base_time = sum(log.get('base_generation_time_ms', 0) for log in session_logs) / total_comparisons
            avg_finetuned_time = sum(log.get('finetuned_generation_time_ms', 0) for log in session_logs) / total_comparisons
            
            return {
                "session_id": session_id,
                "total_comparisons": total_comparisons,
                "average_similarity": round(avg_similarity, 3),
                "average_length_ratio": round(avg_length_ratio, 3),
                "average_base_time_ms": round(avg_base_time, 1),
                "average_finetuned_time_ms": round(avg_finetuned_time, 1),
                "model_pair": session_logs[0].get('model_pair', {}),
                "first_interaction": session_logs[0].get('timestamp'),
                "last_interaction": session_logs[-1].get('timestamp')
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get session summary: {e}")
            return {"session_id": session_id, "error": str(e)}
    
    def export_research_data(self, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           format: str = "json") -> str:
        """Export research data for analysis"""
        try:
            comparison_files = list((self.base_log_dir / "comparisons").glob("*.json"))
            export_data = []
            
            for file_path in comparison_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                        
                        # Filter by date if specified
                        if start_date or end_date:
                            log_timestamp = log_data.get('timestamp', '')
                            if start_date and log_timestamp < start_date:
                                continue
                            if end_date and log_timestamp > end_date:
                                continue
                        
                        export_data.append(log_data)
                except Exception:
                    continue
            
            # Generate export file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "json":
                export_file = self.base_log_dir / "exports" / f"research_export_{timestamp}.json"
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == "csv":
                export_file = self.base_log_dir / "exports" / f"research_export_{timestamp}.csv"
                
                if export_data:
                    with open(export_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=export_data[0].keys())
                        writer.writeheader()
                        writer.writerows(export_data)
            
            self.logger.info(f"Exported {len(export_data)} research entries to: {export_file}")
            return str(export_file)
            
        except Exception as e:
            self.logger.error(f"Failed to export research data: {e}")
            return ""
    
    def log_dataset_generation(self, 
                                 task_id: str, 
                                 dataset_path: str, 
                                 sample_count: int, 
                                 config: Dict[str, Any]) -> None:
        """Log dataset generation for research tracking"""
        try:
            generation_entry = {
                "log_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "dataset_path": dataset_path,
                "sample_count": sample_count,
                "config": config,
                "generation_type": "automated_dataset"
            }
            
            # Save to JSON file
            log_file = self.base_log_dir / "generations" / f"{generation_entry['log_id']}.json"
            log_file.parent.mkdir(exist_ok=True)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(generation_entry, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Logged dataset generation: {task_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log dataset generation: {e}")

    def get_model_comparison_stats(self) -> Dict[str, Any]:
        """Get overall statistics about model comparisons"""
        try:
            comparison_files = list((self.base_log_dir / "comparisons").glob("*.json"))
            
            total_comparisons = len(comparison_files)
            model_pairs = set()
            similarities = []
            length_ratios = []
            performance_data = []
            
            for file_path in comparison_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                        
                        model_pair = log_data.get('model_pair', {})
                        pair_id = f"{model_pair.get('base_model', 'unknown')}_{model_pair.get('adapter_name', 'unknown')}"
                        model_pairs.add(pair_id)
                        
                        if log_data.get('response_similarity') is not None:
                            similarities.append(log_data['response_similarity'])
                        
                        if log_data.get('length_ratio') is not None:
                            length_ratios.append(log_data['length_ratio'])
                        
                        performance_data.append({
                            'base_time': log_data.get('base_generation_time_ms', 0),
                            'finetuned_time': log_data.get('finetuned_generation_time_ms', 0)
                        })
                        
                except Exception:
                    continue
            
            # Calculate statistics
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            avg_length_ratio = sum(length_ratios) / len(length_ratios) if length_ratios else 0
            avg_base_time = sum(p['base_time'] for p in performance_data) / len(performance_data) if performance_data else 0
            avg_finetuned_time = sum(p['finetuned_time'] for p in performance_data) / len(performance_data) if performance_data else 0
            
            return {
                "total_comparisons": total_comparisons,
                "unique_model_pairs": len(model_pairs),
                "model_pairs": list(model_pairs),
                "average_similarity": round(avg_similarity, 3),
                "average_length_ratio": round(avg_length_ratio, 3),
                "average_base_time_ms": round(avg_base_time, 1),
                "average_finetuned_time_ms": round(avg_finetuned_time, 1),
                "performance_advantage": round((avg_base_time - avg_finetuned_time), 1),
                "analysis_period": {
                    "start": min(comparison_files, key=lambda x: x.stat().st_mtime).name if comparison_files else None,
                    "end": max(comparison_files, key=lambda x: x.stat().st_mtime).name if comparison_files else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get comparison stats: {e}")
            return {"error": str(e)}
    
    def log_training_visualization_data(self, task_id: str, step: int, metrics: Dict[str, Any]) -> str:
        """Log training visualization data for real-time charting"""
        try:
            visualization_entry = {
                "log_id": f"viz_{task_id}_{step}_{int(datetime.now().timestamp() * 1000)}",
                "task_id": task_id,
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "type": "training_visualization"
            }
            
            # Save to JSON file for real-time access
            viz_file = self.base_log_dir / "training_analytics" / f"{task_id}_step_{step}.json"
            with open(viz_file, 'w', encoding='utf-8') as f:
                json.dump(visualization_entry, f, indent=2, ensure_ascii=False)
            
            # Also append to aggregated file for the task
            aggregated_file = self.base_log_dir / "training_analytics" / f"{task_id}_metrics.jsonl"
            with open(aggregated_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(visualization_entry) + '\n')
            
            return visualization_entry["log_id"]
            
        except Exception as e:
            self.logger.error(f"Failed to log training visualization data: {e}")
            return ""
    
    def log_dataset_analytics(self, dataset_path: str, analytics_data: Dict[str, Any]) -> str:
        """Log dataset analytics for visualization"""
        try:
            analytics_entry = {
                "log_id": f"dataset_{hashlib.md5(dataset_path.encode()).hexdigest()[:8]}_{int(datetime.now().timestamp())}",
                "dataset_path": dataset_path,
                "timestamp": datetime.now().isoformat(),
                "analytics": analytics_data,
                "type": "dataset_analytics"
            }
            
            # Save to JSON file
            analytics_file = self.base_log_dir / "dataset_analytics" / f"{analytics_entry['log_id']}.json"
            with open(analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics_entry, f, indent=2, ensure_ascii=False)
            
            return analytics_entry["log_id"]
            
        except Exception as e:
            self.logger.error(f"Failed to log dataset analytics: {e}")
            return ""
    
    def get_training_visualization_data(self, task_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get training visualization data for a specific task"""
        try:
            aggregated_file = self.base_log_dir / "training_analytics" / f"{task_id}_metrics.jsonl"
            if not aggregated_file.exists():
                return []
            
            visualization_data = []
            with open(aggregated_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Get the last 'limit' entries
                for line in lines[-limit:]:
                    try:
                        data = json.loads(line.strip())
                        visualization_data.append(data)
                    except json.JSONDecodeError:
                        continue
            
            return visualization_data
            
        except Exception as e:
            self.logger.error(f"Failed to get training visualization data: {e}")
            return []
    
    def log_resource_metrics(self, task_id: str, resource_data: Dict[str, Any]) -> str:
        """Log system resource metrics for visualization"""
        try:
            resource_entry = {
                "log_id": f"resource_{task_id}_{int(datetime.now().timestamp() * 1000)}",
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
                "resources": resource_data,
                "type": "resource_metrics"
            }
            
            # Save to timestamped file
            resource_file = self.base_log_dir / "visualizations" / f"resources_{task_id}.jsonl"
            with open(resource_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(resource_entry) + '\n')
            
            return resource_entry["log_id"]
            
        except Exception as e:
            self.logger.error(f"Failed to log resource metrics: {e}")
            return ""
    
    def get_recent_resource_metrics(self, task_id: str, minutes: int = 10) -> List[Dict[str, Any]]:
        """Get recent resource metrics for a specific task"""
        try:
            resource_file = self.base_log_dir / "visualizations" / f"resources_{task_id}.jsonl"
            if not resource_file.exists():
                return []
            
            cutoff_time = datetime.now().timestamp() - (minutes * 60)
            recent_metrics = []
            
            with open(resource_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(data['timestamp']).timestamp()
                        if entry_time >= cutoff_time:
                            recent_metrics.append(data)
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
            
            return recent_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get recent resource metrics: {e}")
            return []
    
    def cleanup_old_visualization_data(self, days_to_keep: int = 7):
        """Clean up old visualization data to manage storage"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            
            for viz_dir in ["training_analytics", "visualizations", "dataset_analytics"]:
                viz_path = self.base_log_dir / viz_dir
                if viz_path.exists():
                    for file_path in viz_path.glob("*"):
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            self.logger.info(f"Cleaned up old visualization file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old visualization data: {e}")


# Global research logger instance
research_logger = ResearchLogger()

