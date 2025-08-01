"""
InfluxDB storage integration for anomaly detection system.
"""

import time
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from loguru import logger

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
    from influxdb_client.client.exceptions import InfluxDBError
    INFLUXDB_AVAILABLE = True
except ImportError:
    logger.warning("InfluxDB client not available. Install with: pip install influxdb-client")
    INFLUXDB_AVAILABLE = False


@dataclass
class InfluxDBConfig:
    """Configuration for InfluxDB connection."""
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "test_anamoly"
    bucket: str = "anomaly_detection"
    batch_size: int = 100
    flush_interval: int = 1000  # milliseconds
    timeout: int = 10000  # milliseconds
    retry_attempts: int = 3
    retry_interval: int = 1000  # milliseconds


class InfluxDBAnomalyStorage:
    """InfluxDB storage for anomaly detection results."""
    
    def __init__(self, config: InfluxDBConfig):
        """
        Initialize InfluxDB storage client.
        
        Args:
            config: InfluxDB configuration object
        """
        if not INFLUXDB_AVAILABLE:
            raise ImportError("InfluxDB client not available. Install with: pip install influxdb-client")
            
        self.config = config
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.batch_data: List[Point] = []
        self.batch_lock = threading.Lock()
        self.last_flush = time.time() * 1000  # milliseconds
        self.connected = False
        
        logger.info(f"Initializing InfluxDB storage: {config.url}")
        
        # Initialize connection
        self._connect()
        
    def _connect(self) -> bool:
        """
        Establish connection to InfluxDB with retry logic.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        for attempt in range(self.config.retry_attempts):
            try:
                self.client = InfluxDBClient(
                    url=self.config.url,
                    token=self.config.token,
                    org=self.config.org,
                    timeout=self.config.timeout
                )
                
                # Test connection
                health = self.client.health()
                if health.status == "pass":
                    self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                    self.connected = True
                    logger.info(f"Connected to InfluxDB successfully (attempt {attempt + 1})")
                    return True
                else:
                    logger.warning(f"InfluxDB health check failed: {health.message}")
                    
            except Exception as e:
                logger.warning(f"InfluxDB connection attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_interval / 1000.0)
                    
        self.connected = False
        logger.error("Failed to connect to InfluxDB after all attempts")
        return False
        
    def _reconnect_if_needed(self) -> bool:
        """
        Reconnect to InfluxDB if connection is lost.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.connected:
            logger.info("Attempting to reconnect to InfluxDB...")
            return self._connect()
        return True
        
    def store_anomaly_data(self, 
                          timestamp: datetime, 
                          actual_cpu: float, 
                          forecasted_cpu: float, 
                          is_anomaly: bool, 
                          confidence_score: float,
                          host: str = "localhost") -> bool:
        """
        Store anomaly detection data to InfluxDB.
        
        Args:
            timestamp: Timestamp of the measurement
            actual_cpu: Actual CPU usage value
            forecasted_cpu: Forecasted CPU usage value
            is_anomaly: Whether an anomaly was detected
            confidence_score: Confidence score of the detection
            host: Host identifier
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        if not self._reconnect_if_needed():
            logger.error("Cannot store data - InfluxDB connection failed")
            return False
            
        try:
            # Ensure timestamp is in UTC
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            elif timestamp.tzinfo != timezone.utc:
                timestamp = timestamp.astimezone(timezone.utc)
                
            # Create data points following the schema requirements
            points = []
            
            # CPU metrics - actual
            actual_point = Point("cpu_metrics") \
                .tag("metric_type", "actual") \
                .tag("host", host) \
                .field("cpu_usage", float(actual_cpu)) \
                .time(timestamp)
            points.append(actual_point)
            
            # CPU metrics - forecasted
            forecasted_point = Point("cpu_metrics") \
                .tag("metric_type", "forecasted") \
                .tag("host", host) \
                .field("cpu_usage", float(forecasted_cpu)) \
                .time(timestamp)
            points.append(forecasted_point)
            
            # Anomaly detection results
            anomaly_point = Point("anomaly_detection") \
                .tag("host", host) \
                .field("is_anomaly", bool(is_anomaly)) \
                .field("anomaly_score", 1.0 if is_anomaly else 0.0) \
                .field("confidence", float(confidence_score)) \
                .time(timestamp)
            points.append(anomaly_point)
            
            # Add to batch
            with self.batch_lock:
                self.batch_data.extend(points)
                
                # Check if we should flush the batch
                current_time = time.time() * 1000
                should_flush = (
                    len(self.batch_data) >= self.config.batch_size or
                    (current_time - self.last_flush) >= self.config.flush_interval
                )
                
                if should_flush:
                    return self._flush_batch()
                    
            return True
            
        except Exception as e:
            logger.error(f"Error storing anomaly data: {e}")
            return False
            
    def _flush_batch(self) -> bool:
        """
        Flush batched data to InfluxDB.
        
        Returns:
            bool: True if flushed successfully, False otherwise
        """
        if not self.batch_data:
            return True
            
        try:
            logger.debug(f"Flushing {len(self.batch_data)} points to InfluxDB")
            
            # Write data with retry logic
            for attempt in range(self.config.retry_attempts):
                try:
                    self.write_api.write(
                        bucket=self.config.bucket,
                        org=self.config.org,
                        record=self.batch_data
                    )
                    
                    # Clear batch and update flush time
                    self.batch_data.clear()
                    self.last_flush = time.time() * 1000
                    
                    logger.debug("Batch flushed successfully")
                    return True
                    
                except InfluxDBError as e:
                    logger.warning(f"InfluxDB write attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.retry_attempts - 1:
                        # Exponential backoff
                        delay = self.config.retry_interval * (2 ** attempt) / 1000.0
                        time.sleep(delay)
                        
                        # Try to reconnect
                        if not self._reconnect_if_needed():
                            break
                    else:
                        logger.error("Failed to write batch after all attempts")
                        return False
                        
        except Exception as e:
            logger.error(f"Error flushing batch: {e}")
            return False
            
        return False
        
    def force_flush(self) -> bool:
        """
        Force flush any pending data.
        
        Returns:
            bool: True if flushed successfully, False otherwise
        """
        with self.batch_lock:
            return self._flush_batch()
            
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status and statistics.
        
        Returns:
            Dict with connection information
        """
        status = {
            "connected": self.connected,
            "url": self.config.url,
            "org": self.config.org,
            "bucket": self.config.bucket,
            "batch_size": len(self.batch_data),
            "last_flush": self.last_flush
        }
        
        if self.client:
            try:
                health = self.client.health()
                status["health"] = health.status
                status["health_message"] = health.message
            except Exception as e:
                status["health"] = "error"
                status["health_message"] = str(e)
                
        return status
        
    def test_connection(self) -> bool:
        """
        Test InfluxDB connection.
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        return self._reconnect_if_needed()
        
    def close(self):
        """Close InfluxDB connection and flush any pending data."""
        logger.info("Closing InfluxDB connection...")
        
        # Flush any pending data
        if self.batch_data:
            self.force_flush()
            
        # Close client
        if self.client:
            try:
                self.client.close()
                logger.info("InfluxDB connection closed")
            except Exception as e:
                logger.warning(f"Error closing InfluxDB connection: {e}")
                
        self.connected = False
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_influxdb_storage(config_dict: Dict[str, Any]) -> Optional[InfluxDBAnomalyStorage]:
    """
    Create InfluxDB storage from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        InfluxDBAnomalyStorage instance or None if creation failed
    """
    if not INFLUXDB_AVAILABLE:
        logger.warning("InfluxDB client not available, storage disabled")
        return None
        
    try:
        config = InfluxDBConfig(
            url=config_dict.get('url', 'http://localhost:8086'),
            token=config_dict.get('token', ''),
            org=config_dict.get('org', 'test_anamoly'),
            bucket=config_dict.get('bucket', 'anomaly_detection'),
            batch_size=config_dict.get('batch_size', 100),
            flush_interval=config_dict.get('flush_interval', 1000),
            timeout=config_dict.get('timeout', 10000),
            retry_attempts=config_dict.get('retry_attempts', 3),
            retry_interval=config_dict.get('retry_interval', 1000)
        )
        
        storage = InfluxDBAnomalyStorage(config)
        
        if storage.test_connection():
            logger.info("InfluxDB storage created successfully")
            return storage
        else:
            logger.warning("InfluxDB storage created but connection failed")
            return storage  # Return anyway for offline usage
            
    except Exception as e:
        logger.error(f"Failed to create InfluxDB storage: {e}")
        return None


if __name__ == "__main__":
    # Test example
    import yaml
    
    # Test configuration
    test_config = InfluxDBConfig(
        url="http://localhost:8086",
        token="PCIUyihftjXbIAQjR8g-xgEas7H5ssaEQG6ppz-V_w-bxnNeC3FM941sgC8Sclmjhs2VmD7PyGV1oOepRw8xrA==",
        org="test_anamoly",
        bucket="anomaly_detection"
    )
    
    with InfluxDBAnomalyStorage(test_config) as storage:
        # Test storing data
        result = storage.store_anomaly_data(
            timestamp=datetime.now(timezone.utc),
            actual_cpu=85.5,
            forecasted_cpu=75.0,
            is_anomaly=True,
            confidence_score=0.92,
            host="test-host"
        )
        
        print(f"Storage test result: {result}")
        print(f"Connection status: {storage.get_connection_status()}")