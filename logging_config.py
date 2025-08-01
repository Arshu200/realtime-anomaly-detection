"""
Logging configuration module for CPU anomaly detection system.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import json
from typing import Dict, Any, Optional


class LogConfig:
    """Centralized logging configuration for the anomaly detection system."""
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs"):
        """
        Initialize logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
        """
        self.log_level = log_level.upper()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Setup structured logging
        self._setup_console_logging()
        self._setup_file_logging()
        self._setup_anomaly_logging()
        
        logger.info(f"Logging system initialized with level: {self.log_level}")
    
    def _setup_console_logging(self):
        """Setup console logging with colored output."""
        logger.add(
            sys.stdout,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    def _setup_file_logging(self):
        """Setup file logging with rotation."""
        log_file = self.log_dir / "anomaly_detection_{time}.log"
        
        logger.add(
            log_file,
            level=self.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    def _setup_anomaly_logging(self):
        """Setup dedicated anomaly logging."""
        anomaly_log_file = self.log_dir / "anomalies_{time}.log"
        
        # Filter function to only log anomaly-related messages
        def anomaly_filter(record):
            return "ANOMALY" in record["message"].upper() or record.get("extra", {}).get("anomaly", False)
        
        logger.add(
            anomaly_log_file,
            level="WARNING",
            format="{time:YYYY-MM-DD HH:mm:ss} | ANOMALY | {message}",
            filter=anomaly_filter,
            rotation="50 MB",
            retention="90 days",
            compression="zip"
        )
    
    def get_logger(self, name: str = None):
        """Get a logger instance with optional name binding."""
        if name:
            return logger.bind(name=name)
        return logger


class StructuredLogger:
    """Structured logger for CPU usage and anomaly detection events."""
    
    def __init__(self, log_config: LogConfig):
        """Initialize with log configuration."""
        self.logger = log_config.get_logger("AnomalyDetection")
        self.metrics_file = log_config.log_dir / "metrics.jsonl"
        self.anomalies_file = log_config.log_dir / "anomalies.jsonl"
    
    def log_cpu_usage(self, timestamp: datetime, cpu_value: float, 
                     instance: str = "unknown", source: str = "prometheus"):
        """
        Log CPU usage measurement.
        
        Args:
            timestamp: Measurement timestamp
            cpu_value: CPU usage value (0-1)
            instance: Instance identifier
            source: Data source (prometheus, simulation, etc.)
        """
        event = {
            "event_type": "cpu_measurement",
            "timestamp": timestamp.isoformat(),
            "cpu_usage": cpu_value,
            "cpu_usage_percent": cpu_value * 100,
            "instance": instance,
            "source": source
        }
        
        self._log_metric(event)
        self.logger.info(f"CPU Usage: {cpu_value:.3f} ({cpu_value*100:.1f}%) from {instance}")
    
    def log_prediction(self, timestamp: datetime, actual_value: float, 
                      predicted_value: float, prediction_interval: tuple,
                      model_type: str = "prophet"):
        """
        Log model prediction.
        
        Args:
            timestamp: Prediction timestamp
            actual_value: Actual CPU value
            predicted_value: Predicted CPU value
            prediction_interval: (lower, upper) prediction interval
            model_type: Type of model used
        """
        event = {
            "event_type": "prediction",
            "timestamp": timestamp.isoformat(),
            "actual_value": actual_value,
            "predicted_value": predicted_value,
            "prediction_lower": prediction_interval[0],
            "prediction_upper": prediction_interval[1],
            "prediction_error": abs(actual_value - predicted_value),
            "model_type": model_type
        }
        
        self._log_metric(event)
        self.logger.debug(f"Prediction: actual={actual_value:.3f}, predicted={predicted_value:.3f}")
    
    def log_anomaly(self, timestamp: datetime, cpu_value: float, 
                   anomaly_score: float, predicted_value: float,
                   threshold_info: Dict[str, float], instance: str = "unknown",
                   anomaly_type: str = "statistical"):
        """
        Log anomaly detection.
        
        Args:
            timestamp: Anomaly timestamp
            cpu_value: Actual CPU value
            anomaly_score: Anomaly score
            predicted_value: Model prediction
            threshold_info: Dictionary with threshold information
            instance: Instance identifier
            anomaly_type: Type of anomaly detection used
        """
        event = {
            "event_type": "anomaly_detected",
            "timestamp": timestamp.isoformat(),
            "cpu_value": cpu_value,
            "predicted_value": predicted_value,
            "anomaly_score": anomaly_score,
            "threshold_upper": threshold_info.get("upper", None),
            "threshold_lower": threshold_info.get("lower", None),
            "instance": instance,
            "anomaly_type": anomaly_type,
            "severity": self._calculate_severity(anomaly_score)
        }
        
        self._log_anomaly(event)
        
        severity = event["severity"]
        self.logger.warning(
            f"ANOMALY DETECTED [{severity}] - CPU: {cpu_value:.3f}, "
            f"Expected: {predicted_value:.3f}, Score: {anomaly_score:.3f}",
            anomaly=True
        )
    
    def log_model_training(self, dataset_size: int, train_size: int, 
                          model_type: str, training_time: float,
                          model_parameters: Dict[str, Any]):
        """
        Log model training event.
        
        Args:
            dataset_size: Total dataset size
            train_size: Training dataset size
            model_type: Type of model
            training_time: Training time in seconds
            model_parameters: Model configuration parameters
        """
        event = {
            "event_type": "model_training",
            "timestamp": datetime.now().isoformat(),
            "dataset_size": dataset_size,
            "train_size": train_size,
            "test_size": dataset_size - train_size,
            "model_type": model_type,
            "training_time_seconds": training_time,
            "model_parameters": model_parameters
        }
        
        self._log_metric(event)
        self.logger.info(f"Model training completed - {model_type}, "
                        f"Train size: {train_size}, Time: {training_time:.2f}s")
    
    def log_model_evaluation(self, metrics: Dict[str, float], 
                           model_type: str = "prophet"):
        """
        Log model evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
            model_type: Type of model evaluated
        """
        event = {
            "event_type": "model_evaluation",
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            **metrics
        }
        
        self._log_metric(event)
        self.logger.info(f"Model evaluation - Precision: {metrics.get('precision', 0):.3f}, "
                        f"Recall: {metrics.get('recall', 0):.3f}, "
                        f"F1: {metrics.get('f1_score', 0):.3f}")
    
    def log_system_status(self, status: str, component: str, 
                         details: Optional[Dict[str, Any]] = None):
        """
        Log system status change.
        
        Args:
            status: Status (started, stopped, error, healthy)
            component: System component (monitor, prometheus, model)
            details: Additional status details
        """
        event = {
            "event_type": "system_status",
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "component": component,
            "details": details or {}
        }
        
        self._log_metric(event)
        
        if status.lower() == "error":
            self.logger.error(f"System error in {component}: {details}")
        else:
            self.logger.info(f"System status - {component}: {status}")
    
    def log_prometheus_query(self, query: str, success: bool, 
                           response_time: float, data_points: int = 0):
        """
        Log Prometheus query execution.
        
        Args:
            query: PromQL query
            success: Whether query was successful
            response_time: Query response time in seconds
            data_points: Number of data points returned
        """
        event = {
            "event_type": "prometheus_query",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "success": success,
            "response_time_seconds": response_time,
            "data_points": data_points
        }
        
        self._log_metric(event)
        
        if success:
            self.logger.debug(f"Prometheus query successful - {data_points} points in {response_time:.3f}s")
        else:
            self.logger.warning(f"Prometheus query failed: {query}")
    
    def _log_metric(self, event: Dict[str, Any]):
        """Log event to metrics file in JSONL format."""
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            converted_event = convert_numpy_types(event)
            
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(converted_event) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write metric log: {e}")
    
    def _log_anomaly(self, event: Dict[str, Any]):
        """Log anomaly to dedicated anomaly file in JSONL format."""
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            converted_event = convert_numpy_types(event)
            
            with open(self.anomalies_file, 'a') as f:
                f.write(json.dumps(converted_event) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write anomaly log: {e}")
    
    def _calculate_severity(self, anomaly_score: float) -> str:
        """Calculate anomaly severity based on score."""
        if anomaly_score >= 3.0:
            return "CRITICAL"
        elif anomaly_score >= 2.0:
            return "HIGH"
        elif anomaly_score >= 1.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_recent_metrics(self, event_type: str = None, 
                          hours: int = 24) -> list:
        """
        Get recent metrics from the log file.
        
        Args:
            event_type: Filter by event type (optional)
            hours: Number of hours to look back
            
        Returns:
            List of metric events
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        metrics = []
        
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            event_time = datetime.fromisoformat(event['timestamp'])
                            
                            if event_time >= cutoff_time:
                                if event_type is None or event['event_type'] == event_type:
                                    metrics.append(event)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
        except Exception as e:
            self.logger.error(f"Failed to read metrics: {e}")
        
        return metrics
    
    def get_recent_anomalies(self, hours: int = 24) -> list:
        """
        Get recent anomalies from the log file.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of anomaly events
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        anomalies = []
        
        try:
            if self.anomalies_file.exists():
                with open(self.anomalies_file, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            event_time = datetime.fromisoformat(event['timestamp'])
                            
                            if event_time >= cutoff_time:
                                anomalies.append(event)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
        except Exception as e:
            self.logger.error(f"Failed to read anomalies: {e}")
        
        return anomalies
    
    def generate_summary_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate a summary report of recent activity.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        metrics = self.get_recent_metrics(hours=hours)
        anomalies = self.get_recent_anomalies(hours=hours)
        
        # Count events by type
        event_counts = {}
        for metric in metrics:
            event_type = metric['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Anomaly severity counts
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'UNKNOWN')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # CPU usage statistics
        cpu_measurements = [m for m in metrics if m['event_type'] == 'cpu_measurement']
        cpu_stats = {}
        if cpu_measurements:
            cpu_values = [m['cpu_usage'] for m in cpu_measurements]
            cpu_stats = {
                'count': len(cpu_values),
                'mean': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            }
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'analysis_period_hours': hours,
            'total_events': len(metrics),
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(cpu_measurements) * 100 if cpu_measurements else 0,
            'event_counts': event_counts,
            'severity_counts': severity_counts,
            'cpu_statistics': cpu_stats
        }
        
        return report


# Global logger instance
_log_config = None
_structured_logger = None


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup global logging configuration."""
    global _log_config, _structured_logger
    
    _log_config = LogConfig(log_level, log_dir)
    _structured_logger = StructuredLogger(_log_config)
    
    return _structured_logger


def get_logger(name: str = None):
    """Get a logger instance."""
    if _log_config is None:
        setup_logging()
    
    return _log_config.get_logger(name)


def get_structured_logger():
    """Get the structured logger instance."""
    if _structured_logger is None:
        setup_logging()
    
    return _structured_logger


if __name__ == "__main__":
    # Example usage
    structured_logger = setup_logging("DEBUG")
    
    # Test different log types
    structured_logger.log_cpu_usage(datetime.now(), 0.75, "test-instance", "simulation")
    
    structured_logger.log_prediction(
        datetime.now(), 0.75, 0.73, (0.65, 0.81)
    )
    
    structured_logger.log_anomaly(
        datetime.now(), 0.95, 2.5, 0.73,
        {"upper": 0.85, "lower": 0.61}, "test-instance"
    )
    
    structured_logger.log_system_status("started", "monitor")
    
    # Generate summary
    summary = structured_logger.generate_summary_report(hours=1)
    print("Summary:", json.dumps(summary, indent=2))