"""
Base storage interface for anomaly detection data persistence.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class AnomalyData:
    """Data structure for anomaly detection results."""
    timestamp: datetime
    actual_cpu: float
    forecasted_cpu: float
    is_anomaly: bool
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None


class StorageInterface(ABC):
    """Abstract interface for anomaly detection data storage."""
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to storage backend.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection to storage backend.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def store_anomaly_data(self, timestamp: datetime, actual_cpu: float, 
                          forecasted_cpu: float, is_anomaly: bool, 
                          confidence_score: float, **kwargs) -> bool:
        """
        Store anomaly detection results.
        
        Args:
            timestamp: Data timestamp
            actual_cpu: Actual CPU usage value
            forecasted_cpu: Predicted CPU usage value
            is_anomaly: Whether anomaly was detected
            confidence_score: Confidence score of detection
            **kwargs: Additional metadata
            
        Returns:
            True if storage successful, False otherwise
        """
        pass
    
    @abstractmethod
    def store_cpu_metrics(self, timestamp: datetime, cpu_usage: float,
                         metric_type: str = "actual", **kwargs) -> bool:
        """
        Store CPU metrics data.
        
        Args:
            timestamp: Data timestamp
            cpu_usage: CPU usage percentage
            metric_type: Type of metric (actual, forecasted, etc.)
            **kwargs: Additional metadata
            
        Returns:
            True if storage successful, False otherwise
        """
        pass
    
    @abstractmethod
    def query_anomalies(self, start_time: datetime, end_time: datetime) -> List[AnomalyData]:
        """
        Query anomaly data for a time range.
        
        Args:
            start_time: Query start time
            end_time: Query end time
            
        Returns:
            List of anomaly data points
        """
        pass
    
    @abstractmethod
    def get_anomaly_count(self, start_time: datetime, end_time: datetime) -> int:
        """
        Get count of anomalies in a time range.
        
        Args:
            start_time: Query start time
            end_time: Query end time
            
        Returns:
            Number of anomalies detected
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check storage backend health status.
        
        Returns:
            Dictionary with health status information
        """
        pass


class BaseStorage(StorageInterface):
    """Base storage implementation with common functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base storage.
        
        Args:
            config: Storage configuration dictionary
        """
        self.config = config
        self.connected = False
        self.connection = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def is_connected(self) -> bool:
        """Check if storage is connected."""
        return self.connected
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def validate_config(self) -> bool:
        """
        Validate storage configuration.
        
        Returns:
            True if configuration is valid
        """
        # Override in subclasses for specific validation
        return True