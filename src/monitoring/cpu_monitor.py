"""
CPU monitoring module for real-time CPU usage monitoring and anomaly detection.
"""

import time
from datetime import datetime
from typing import Dict, Optional, Callable
from loguru import logger

from .metrics_collector import PrometheusClient


class CPUMonitor:
    """Specialized CPU monitoring with anomaly detection integration."""
    
    def __init__(self, prometheus_client: PrometheusClient):
        """
        Initialize CPU monitor.
        
        Args:
            prometheus_client: Configured PrometheusClient instance
        """
        self.prometheus_client = prometheus_client
        self.monitoring_active = False
        self.stats = {
            'readings_count': 0,
            'anomalies_detected': 0,
            'start_time': None,
            'last_reading': None
        }
        
        logger.info("CPU monitor initialized")
    
    def start_monitoring(self, detector, storage=None, interval_seconds: int = 30):
        """
        Start continuous CPU monitoring with anomaly detection.
        
        Args:
            detector: Trained anomaly detection model
            storage: Optional storage backend for results
            interval_seconds: Monitoring interval in seconds
        """
        self.monitoring_active = True
        self.stats['start_time'] = datetime.now()
        
        logger.info(f"Starting CPU monitoring (interval: {interval_seconds}s)")
        
        def monitoring_callback(cpu_data):
            """Process each CPU reading."""
            self.stats['readings_count'] += 1
            self.stats['last_reading'] = cpu_data['timestamp']
            
            # Detect anomalies
            result = detector.predict_single_point(
                cpu_data['timestamp'], 
                cpu_data['cpu_usage']
            )
            
            if result['is_anomaly']:
                self.stats['anomalies_detected'] += 1
                logger.warning(f"CPU anomaly detected: {result['actual_value']:.2f}%")
            
            # Store results if storage is available
            if storage:
                try:
                    confidence_score = detector.calculate_confidence_score(
                        result['actual_value'],
                        result['predicted_value'],
                        result.get('prediction_lower', result['predicted_value'] - 10),
                        result.get('prediction_upper', result['predicted_value'] + 10)
                    )
                    
                    storage.store_anomaly_data(
                        timestamp=result['timestamp'],
                        actual_cpu=result['actual_value'],
                        forecasted_cpu=result['predicted_value'],
                        is_anomaly=result.get('is_anomaly', False),
                        confidence_score=confidence_score
                    )
                except Exception as e:
                    logger.debug(f"Storage error: {e}")
        
        try:
            self.prometheus_client.monitor_cpu_continuous(
                interval_seconds=interval_seconds,
                callback=monitoring_callback
            )
        except KeyboardInterrupt:
            logger.info("CPU monitoring stopped by user")
        finally:
            self.monitoring_active = False
    
    def get_monitoring_stats(self) -> Dict:
        """Get current monitoring statistics."""
        if self.stats['start_time']:
            duration = datetime.now() - self.stats['start_time']
            self.stats['duration_minutes'] = duration.total_seconds() / 60
            
            if self.stats['readings_count'] > 0:
                self.stats['anomaly_rate'] = (
                    self.stats['anomalies_detected'] / self.stats['readings_count'] * 100
                )
        
        return self.stats.copy()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        logger.info("CPU monitoring stopped")


class RealTimeAnomalyMonitor:
    """Real-time monitoring system combining Prometheus and anomaly detection."""
    
    def __init__(self, prometheus_client: PrometheusClient, anomaly_detector):
        """
        Initialize the real-time monitor.
        
        Args:
            prometheus_client: Configured PrometheusClient instance
            anomaly_detector: Trained anomaly detection model
        """
        self.prometheus_client = prometheus_client
        self.anomaly_detector = anomaly_detector
        self.monitoring_active = False
        self.anomaly_buffer = []
        self.normal_buffer = []
        
        logger.info("Real-time anomaly monitor initialized")
    
    def process_cpu_reading(self, cpu_data: Dict) -> Dict:
        """
        Process a single CPU reading for anomaly detection.
        
        Args:
            cpu_data: CPU usage data from Prometheus
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Use the anomaly detector to check for anomalies
            result = self.anomaly_detector.predict_single_point(
                cpu_data['timestamp'], 
                cpu_data['cpu_usage']
            )
            
            # Enhance result with Prometheus metadata
            result.update({
                'instance': cpu_data.get('instance', 'unknown'),
                'query_used': cpu_data.get('query_used', 'unknown')
            })
            
            # Log the result
            if result['is_anomaly']:
                logger.warning(f"ANOMALY DETECTED! CPU: {result['actual_value']:.3f}, "
                             f"Expected: {result['predicted_value']:.3f}, "
                             f"Score: {result['anomaly_score']:.3f}")
                self.anomaly_buffer.append(result)
            else:
                self.normal_buffer.append(result)
            
            # Limit buffer sizes to prevent memory issues
            if len(self.anomaly_buffer) > 1000:
                self.anomaly_buffer = self.anomaly_buffer[-500:]
            if len(self.normal_buffer) > 1000:
                self.normal_buffer = self.normal_buffer[-500:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing CPU reading: {e}")
            return {
                'timestamp': cpu_data['timestamp'],
                'error': str(e),
                'is_anomaly': False
            }
    
    def get_monitoring_stats(self) -> Dict:
        """Get current monitoring statistics."""
        return {
            'total_anomalies': len(self.anomaly_buffer),
            'total_normal': len(self.normal_buffer),
            'anomaly_rate': len(self.anomaly_buffer) / (len(self.anomaly_buffer) + len(self.normal_buffer)) * 100 if (len(self.anomaly_buffer) + len(self.normal_buffer)) > 0 else 0,
            'monitoring_active': self.monitoring_active
        }
    
    def clear_buffers(self):
        """Clear monitoring buffers."""
        self.anomaly_buffer.clear()
        self.normal_buffer.clear()
        logger.info("Monitoring buffers cleared")