"""
Utility functions for anomaly detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Union, Tuple, Dict, Any
from loguru import logger


def prepare_data_for_prophet(df: pd.DataFrame, 
                           timestamp_col: str = 'timestamp', 
                           value_col: str = 'cpu_usage') -> pd.DataFrame:
    """
    Prepare data for Prophet model training/prediction.
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        value_col: Name of value column
        
    Returns:
        DataFrame with 'ds' and 'y' columns for Prophet
    """
    if timestamp_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{timestamp_col}' and '{value_col}' columns")
    
    # Create Prophet format DataFrame
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df[timestamp_col]),
        'y': df[value_col].astype(float)
    })
    
    # Remove any rows with NaN values
    prophet_df = prophet_df.dropna()
    
    # Sort by timestamp
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    logger.debug(f"Prepared {len(prophet_df)} data points for Prophet")
    return prophet_df


def calculate_confidence_score(actual: float, predicted: float, 
                             lower_bound: float, upper_bound: float) -> float:
    """
    Calculate confidence score for anomaly detection.
    
    Args:
        actual: Actual observed value
        predicted: Predicted value
        lower_bound: Lower prediction bound
        upper_bound: Upper prediction bound
        
    Returns:
        Confidence score between 0 and 1
    """
    if upper_bound <= lower_bound:
        logger.warning("Invalid bounds: upper_bound <= lower_bound")
        return 0.5
    
    # Calculate normalized distance from prediction
    prediction_range = upper_bound - lower_bound
    distance_from_prediction = abs(actual - predicted)
    
    # Normalize distance by prediction range
    normalized_distance = distance_from_prediction / (prediction_range + 1e-8)
    
    # Convert to confidence score (1 = high confidence, 0 = low confidence)
    confidence = max(0.0, min(1.0, 1.0 - normalized_distance))
    
    return confidence


def validate_cpu_data(cpu_usage: float) -> Tuple[bool, str]:
    """
    Validate CPU usage data.
    
    Args:
        cpu_usage: CPU usage value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(cpu_usage, (int, float)):
        return False, "CPU usage must be a number"
    
    if np.isnan(cpu_usage) or np.isinf(cpu_usage):
        return False, "CPU usage cannot be NaN or infinite"
    
    if cpu_usage < 0:
        return False, "CPU usage cannot be negative"
    
    if cpu_usage > 100:
        return False, "CPU usage cannot exceed 100%"
    
    return True, ""


def normalize_timestamp(timestamp: Union[str, datetime, pd.Timestamp]) -> datetime:
    """
    Normalize timestamp to datetime object with timezone.
    
    Args:
        timestamp: Timestamp in various formats
        
    Returns:
        Normalized datetime object
    """
    if isinstance(timestamp, str):
        try:
            dt = pd.to_datetime(timestamp)
        except Exception as e:
            logger.error(f"Failed to parse timestamp '{timestamp}': {e}")
            raise ValueError(f"Invalid timestamp format: {timestamp}")
    elif isinstance(timestamp, pd.Timestamp):
        dt = timestamp.to_pydatetime()
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    # Add timezone if naive
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt


def calculate_anomaly_statistics(anomalies_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics for detected anomalies.
    
    Args:
        anomalies_df: DataFrame with anomaly detection results
        
    Returns:
        Dictionary with anomaly statistics
    """
    if 'is_anomaly' not in anomalies_df.columns:
        raise ValueError("DataFrame must contain 'is_anomaly' column")
    
    total_points = len(anomalies_df)
    anomaly_count = anomalies_df['is_anomaly'].sum()
    normal_count = total_points - anomaly_count
    
    stats = {
        'total_points': total_points,
        'anomaly_count': int(anomaly_count),
        'normal_count': int(normal_count),
        'anomaly_rate': float(anomaly_count / total_points * 100) if total_points > 0 else 0.0
    }
    
    # Calculate additional statistics if available
    if 'anomaly_score' in anomalies_df.columns:
        anomaly_data = anomalies_df[anomalies_df['is_anomaly']]
        if len(anomaly_data) > 0:
            stats.update({
                'avg_anomaly_score': float(anomaly_data['anomaly_score'].mean()),
                'max_anomaly_score': float(anomaly_data['anomaly_score'].max()),
                'min_anomaly_score': float(anomaly_data['anomaly_score'].min())
            })
    
    if 'confidence' in anomalies_df.columns:
        stats['avg_confidence'] = float(anomalies_df['confidence'].mean())
    
    if 'y' in anomalies_df.columns:
        anomaly_data = anomalies_df[anomalies_df['is_anomaly']]
        normal_data = anomalies_df[~anomalies_df['is_anomaly']]
        
        if len(anomaly_data) > 0:
            stats.update({
                'avg_anomaly_value': float(anomaly_data['y'].mean()),
                'max_anomaly_value': float(anomaly_data['y'].max()),
                'min_anomaly_value': float(anomaly_data['y'].min())
            })
        
        if len(normal_data) > 0:
            stats['avg_normal_value'] = float(normal_data['y'].mean())
    
    return stats


def detect_data_quality_issues(df: pd.DataFrame, 
                              timestamp_col: str = 'ds', 
                              value_col: str = 'y') -> Dict[str, Any]:
    """
    Detect data quality issues in the dataset.
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        value_col: Name of value column
        
    Returns:
        Dictionary with data quality information
    """
    issues = {
        'total_rows': len(df),
        'missing_timestamps': 0,
        'missing_values': 0,
        'invalid_values': 0,
        'duplicate_timestamps': 0,
        'time_gaps': [],
        'outliers': []
    }
    
    if timestamp_col in df.columns:
        issues['missing_timestamps'] = df[timestamp_col].isna().sum()
        issues['duplicate_timestamps'] = df[timestamp_col].duplicated().sum()
        
        # Check for time gaps
        if len(df) > 1:
            df_sorted = df.sort_values(timestamp_col)
            time_diffs = df_sorted[timestamp_col].diff()
            median_diff = time_diffs.median()
            
            # Gaps larger than 2x median interval
            large_gaps = time_diffs > (median_diff * 2)
            if large_gaps.any():
                gap_indices = df_sorted[large_gaps].index.tolist()
                issues['time_gaps'] = gap_indices[:10]  # Limit to first 10
    
    if value_col in df.columns:
        issues['missing_values'] = df[value_col].isna().sum()
        
        # Check for invalid values (negative or > 100 for CPU)
        invalid_mask = (df[value_col] < 0) | (df[value_col] > 100)
        issues['invalid_values'] = invalid_mask.sum()
        
        # Detect outliers using IQR method
        Q1 = df[value_col].quantile(0.25)
        Q3 = df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)
        outlier_indices = df[outlier_mask].index.tolist()
        issues['outliers'] = outlier_indices[:20]  # Limit to first 20
    
    return issues


def resample_time_series(df: pd.DataFrame, 
                        frequency: str = '30T',
                        timestamp_col: str = 'ds',
                        value_col: str = 'y',
                        agg_method: str = 'mean') -> pd.DataFrame:
    """
    Resample time series data to a regular frequency.
    
    Args:
        df: Input DataFrame
        frequency: Target frequency (e.g., '30T' for 30 minutes)
        timestamp_col: Name of timestamp column
        value_col: Name of value column
        agg_method: Aggregation method ('mean', 'median', 'max', 'min')
        
    Returns:
        Resampled DataFrame
    """
    if timestamp_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{timestamp_col}' and '{value_col}' columns")
    
    # Set timestamp as index
    df_copy = df.copy()
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
    df_copy = df_copy.set_index(timestamp_col)
    
    # Resample based on aggregation method
    if agg_method == 'mean':
        resampled = df_copy[value_col].resample(frequency).mean()
    elif agg_method == 'median':
        resampled = df_copy[value_col].resample(frequency).median()
    elif agg_method == 'max':
        resampled = df_copy[value_col].resample(frequency).max()
    elif agg_method == 'min':
        resampled = df_copy[value_col].resample(frequency).min()
    else:
        raise ValueError(f"Unsupported aggregation method: {agg_method}")
    
    # Convert back to DataFrame
    result = resampled.reset_index()
    result.columns = [timestamp_col, value_col]
    
    # Remove NaN values
    result = result.dropna()
    
    logger.info(f"Resampled data from {len(df)} to {len(result)} points at {frequency} frequency")
    return result