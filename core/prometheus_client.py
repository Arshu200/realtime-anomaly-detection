"""
Prometheus client for real-time CPU usage monitoring and anomaly detection.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from loguru import logger
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class PrometheusClient:
    """Client for querying Prometheus metrics and feeding data to anomaly detection."""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090", 
                 timeout: int = 30, retry_attempts: int = 3):
        """
        Initialize Prometheus client.
        
        Args:
            prometheus_url: Base URL of Prometheus server
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
        """
        self.prometheus_url = prometheus_url.rstrip('/')
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.session = requests.Session()
        
        # Common headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logger.info(f"Initialized Prometheus client for {self.prometheus_url}")
    
    def check_connection(self) -> bool:
        """Check if Prometheus server is accessible."""
        try:
            response = self.session.get(
                f"{self.prometheus_url}/api/v1/status/config",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info("Successfully connected to Prometheus server")
                return True
            else:
                logger.error(f"Prometheus server returned status code: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Prometheus server: {e}")
            return False
    
    def query_instant(self, query: str, timestamp: Optional[datetime] = None) -> Dict:
        """
        Execute an instant query against Prometheus.
        
        Args:
            query: PromQL query string
            timestamp: Optional timestamp for the query (default: current time)
            
        Returns:
            Dictionary with query results
        """
        params = {'query': query}
        
        if timestamp:
            params['time'] = timestamp.timestamp()
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params=params,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                if result['status'] == 'success':
                    logger.debug(f"Query successful: {query}")
                    return result
                else:
                    logger.error(f"Query failed: {result.get('error', 'Unknown error')}")
                    return {'status': 'error', 'error': result.get('error', 'Unknown error')}
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All query attempts failed for: {query}")
                    return {'status': 'error', 'error': str(e)}
    
    def query_range(self, query: str, start_time: datetime, end_time: datetime, 
                    step: str = '1m') -> Dict:
        """
        Execute a range query against Prometheus.
        
        Args:
            query: PromQL query string
            start_time: Start time for the query
            end_time: End time for the query
            step: Query resolution step (e.g., '1m', '5m', '1h')
            
        Returns:
            Dictionary with query results
        """
        params = {
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': step
        }
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(
                    f"{self.prometheus_url}/api/v1/query_range",
                    params=params,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                if result['status'] == 'success':
                    logger.debug(f"Range query successful: {query}")
                    return result
                else:
                    logger.error(f"Range query failed: {result.get('error', 'Unknown error')}")
                    return {'status': 'error', 'error': result.get('error', 'Unknown error')}
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Range query attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"All range query attempts failed for: {query}")
                    return {'status': 'error', 'error': str(e)}
    
    def get_cpu_usage(self, instance: Optional[str] = None) -> Optional[Dict]:
        """
        Get current CPU usage from Prometheus.
        
        Args:
            instance: Specific instance to query (optional)
            
        Returns:
            Dictionary with CPU usage data or None if failed
        """
        # Common CPU usage queries for different metric formats
        queries = [
            "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "avg(100 - (avg by (instance) (rate(cpu_usage_idle[5m])) * 100))",
            "cpu_usage_percent",
            "system_cpu_usage",
            "container_cpu_usage_seconds_total"
        ]
        
        if instance:
            queries = [f'{query}{{instance="{instance}"}}' for query in queries]
        
        for query in queries:
            logger.debug(f"Trying CPU query: {query}")
            result = self.query_instant(query)
            
            if result['status'] == 'success' and result['data']['result']:
                data = result['data']['result'][0]
                timestamp = datetime.fromtimestamp(float(data['value'][0]))
                cpu_value = float(data['value'][1])
                
                # Convert to 0-1 range if it's in percentage (0-100)
                if cpu_value > 1.0:
                    cpu_value = cpu_value / 100.0
                
                return {
                    'timestamp': timestamp,
                    'cpu_usage': cpu_value,
                    'instance': data.get('metric', {}).get('instance', 'unknown'),
                    'query_used': query
                }
        
        logger.warning("No CPU usage data found with any of the standard queries")
        return None
    
    def get_historical_cpu_data(self, hours_back: int = 24, step: str = '1m') -> Optional[pd.DataFrame]:
        """
        Get historical CPU usage data from Prometheus.
        
        Args:
            hours_back: Number of hours to look back
            step: Query resolution step
            
        Returns:
            DataFrame with historical CPU usage data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Try different CPU queries
        queries = [
            "100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "avg(cpu_usage_percent)",
            "avg(system_cpu_usage)"
        ]
        
        for query in queries:
            logger.info(f"Querying historical data with: {query}")
            result = self.query_range(query, start_time, end_time, step)
            
            if result['status'] == 'success' and result['data']['result']:
                # Parse the time series data
                time_series = result['data']['result'][0]['values']
                
                data = []
                for timestamp_str, value_str in time_series:
                    timestamp = datetime.fromtimestamp(float(timestamp_str))
                    cpu_value = float(value_str)
                    
                    # Convert to 0-1 range if needed
                    if cpu_value > 1.0:
                        cpu_value = cpu_value / 100.0
                    
                    data.append({
                        'timestamp': timestamp,
                        'cpu_usage': cpu_value
                    })
                
                df = pd.DataFrame(data)
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Retrieved {len(df)} historical data points")
                return df
        
        logger.warning("No historical CPU data found")
        return None
    
    def simulate_cpu_data(self) -> Dict:
        """
        Simulate CPU usage data when Prometheus is not available.
        Used for testing and development.
        
        Returns:
            Dictionary with simulated CPU usage data
        """
        # Generate realistic CPU usage pattern
        base_usage = 0.3 + 0.4 * np.sin(time.time() / 3600)  # Hourly pattern
        noise = np.random.normal(0, 0.05)  # Random noise
        
        cpu_usage = np.clip(base_usage + noise, 0.0, 1.0)
        
        # Occasionally add spikes for testing
        if np.random.random() < 0.02:  # 2% chance of spike
            cpu_usage = min(1.0, cpu_usage + np.random.uniform(0.2, 0.4))
        
        return {
            'timestamp': datetime.now(),
            'cpu_usage': cpu_usage,
            'instance': 'simulated',
            'query_used': 'simulation'
        }
    
    def monitor_cpu_continuous(self, interval_seconds: int = 30, 
                             callback=None, max_iterations: Optional[int] = None):
        """
        Continuously monitor CPU usage.
        
        Args:
            interval_seconds: Interval between queries in seconds
            callback: Function to call with each CPU reading
            max_iterations: Maximum number of iterations (None for infinite)
        """
        logger.info(f"Starting continuous CPU monitoring (interval: {interval_seconds}s)")
        
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                # Get CPU data
                cpu_data = self.get_cpu_usage()
                
                if cpu_data is None:
                    logger.warning("Failed to get CPU data, using simulation")
                    cpu_data = self.simulate_cpu_data()
                
                # Log the data
                logger.info(f"CPU Usage: {cpu_data['cpu_usage']:.3f} at {cpu_data['timestamp']}")
                
                # Call callback if provided
                if callback:
                    try:
                        callback(cpu_data)
                    except Exception as e:
                        logger.error(f"Callback function failed: {e}")
                
                # Wait for next iteration
                time.sleep(interval_seconds)
                iteration += 1
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            raise
    
    def get_multiple_metrics(self, queries: List[str]) -> Dict[str, Dict]:
        """
        Query multiple metrics at once.
        
        Args:
            queries: List of PromQL queries
            
        Returns:
            Dictionary mapping query to result
        """
        results = {}
        
        for query in queries:
            result = self.query_instant(query)
            results[query] = result
        
        return results
    
    def export_historical_data(self, filename: str = "prometheus_cpu_data.csv", 
                              hours_back: int = 24):
        """
        Export historical CPU data to CSV file.
        
        Args:
            filename: Output filename
            hours_back: Hours of data to export
        """
        logger.info(f"Exporting {hours_back} hours of historical data to {filename}")
        
        df = self.get_historical_cpu_data(hours_back)
        
        if df is not None:
            df.to_csv(filename, index=False)
            logger.info(f"Exported {len(df)} data points to {filename}")
        else:
            logger.error("Failed to export data - no data available")


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
                logger.info(f"Normal CPU usage: {result['actual_value']:.3f} "
                           f"(predicted: {result['predicted_value']:.3f})")
                self.normal_buffer.append(result)
            
            # Keep buffers from growing too large
            if len(self.anomaly_buffer) > 100:
                self.anomaly_buffer = self.anomaly_buffer[-50:]
            if len(self.normal_buffer) > 1000:
                self.normal_buffer = self.normal_buffer[-500:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing CPU reading: {e}")
            return {
                'timestamp': cpu_data['timestamp'],
                'actual_value': cpu_data['cpu_usage'],
                'is_anomaly': False,
                'error': str(e)
            }
    
    def start_monitoring(self, interval_seconds: int = 30):
        """
        Start real-time monitoring.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        logger.info("Starting real-time anomaly monitoring")
        self.monitoring_active = True
        
        def monitoring_callback(cpu_data):
            if self.monitoring_active:
                self.process_cpu_reading(cpu_data)
        
        self.prometheus_client.monitor_cpu_continuous(
            interval_seconds=interval_seconds,
            callback=monitoring_callback
        )
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        logger.info("Stopping real-time monitoring")
        self.monitoring_active = False
    
    def get_recent_anomalies(self, count: int = 10) -> List[Dict]:
        """Get recent anomalies from the buffer."""
        return self.anomaly_buffer[-count:] if self.anomaly_buffer else []
    
    def get_monitoring_stats(self) -> Dict:
        """Get monitoring statistics."""
        total_readings = len(self.anomaly_buffer) + len(self.normal_buffer)
        
        return {
            'total_readings': total_readings,
            'anomalies_detected': len(self.anomaly_buffer),
            'normal_readings': len(self.normal_buffer),
            'anomaly_rate': len(self.anomaly_buffer) / total_readings * 100 if total_readings > 0 else 0,
            'monitoring_active': self.monitoring_active
        }


if __name__ == "__main__":
    # Example usage
    client = PrometheusClient("http://localhost:9090")
    
    # Test connection
    if client.check_connection():
        print("Connected to Prometheus!")
        
        # Get current CPU usage
        cpu_data = client.get_cpu_usage()
        if cpu_data:
            print(f"Current CPU: {cpu_data['cpu_usage']:.2%}")
        
        # Get historical data
        historical = client.get_historical_cpu_data(hours_back=1)
        if historical is not None:
            print(f"Retrieved {len(historical)} historical points")
    else:
        print("Could not connect to Prometheus - using simulation mode")
        cpu_data = client.simulate_cpu_data()
        print(f"Simulated CPU: {cpu_data['cpu_usage']:.2%}")