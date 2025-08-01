"""
Machine learning models for anomaly detection.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, Any, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class AnomalyModel:
    """Base class for anomaly detection models."""
    
    def __init__(self):
        """Initialize base anomaly model."""
        self.model = None
        self.is_trained = False
        self.performance_metrics = {}
    
    def train(self, data: pd.DataFrame) -> None:
        """Train the model."""
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions."""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def detect_anomalies(self, data: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """Detect anomalies in the data."""
        raise NotImplementedError("Subclasses must implement detect_anomalies method")


class ProphetModel(AnomalyModel):
    """Facebook Prophet model for time-series anomaly detection."""
    
    def __init__(self, 
                 daily_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 yearly_seasonality: bool = False,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 interval_width: float = 0.95,
                 uncertainty_samples: int = 1000):
        """
        Initialize Prophet model.
        
        Args:
            daily_seasonality: Enable daily seasonality
            weekly_seasonality: Enable weekly seasonality  
            yearly_seasonality: Enable yearly seasonality
            changepoint_prior_scale: Prior scale for changepoints
            seasonality_prior_scale: Prior scale for seasonality
            interval_width: Prediction interval width
            uncertainty_samples: Number of uncertainty samples
        """
        super().__init__()
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Prophet model with configured parameters."""
        self.model = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width,
            uncertainty_samples=self.uncertainty_samples
        )
        
        logger.debug("Prophet model initialized with configured parameters")
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the Prophet model.
        
        Args:
            data: DataFrame with 'ds' (datetime) and 'y' (value) columns
        """
        if not {'ds', 'y'}.issubset(data.columns):
            raise ValueError("Data must contain 'ds' and 'y' columns")
        
        logger.info(f"Training Prophet model with {len(data)} data points")
        
        try:
            self.model.fit(data)
            self.is_trained = True
            logger.info("Prophet model training completed successfully")
            
        except Exception as e:
            logger.error(f"Prophet model training failed: {e}")
            raise
    
    def predict(self, periods: int = None, future_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Args:
            periods: Number of future periods to predict
            future_df: Pre-made future dataframe
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if future_df is None:
            if periods is None:
                raise ValueError("Either periods or future_df must be provided")
            future_df = self.model.make_future_dataframe(periods=periods, freq='30T')
        
        try:
            forecast = self.model.predict(future_df)
            logger.debug(f"Generated predictions for {len(forecast)} time points")
            return forecast
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def detect_anomalies(self, data: pd.DataFrame, forecast: pd.DataFrame, 
                        threshold_multiplier: float = 2.0) -> pd.DataFrame:
        """
        Detect anomalies by comparing actual values with predictions.
        
        Args:
            data: Actual data with 'ds' and 'y' columns
            forecast: Forecast data from predict method
            threshold_multiplier: Multiplier for prediction interval
            
        Returns:
            DataFrame with anomaly detection results
        """
        # Merge actual data with forecast
        merged = pd.merge(data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                         on='ds', how='inner')
        
        # Calculate dynamic thresholds
        merged['threshold_upper'] = merged['yhat_upper'] + \
                                   (merged['yhat_upper'] - merged['yhat']) * (threshold_multiplier - 1)
        merged['threshold_lower'] = merged['yhat_lower'] - \
                                   (merged['yhat'] - merged['yhat_lower']) * (threshold_multiplier - 1)
        
        # Detect anomalies
        merged['is_anomaly'] = (
            (merged['y'] > merged['threshold_upper']) | 
            (merged['y'] < merged['threshold_lower'])
        )
        
        # Calculate anomaly score
        merged['anomaly_score'] = np.abs(merged['y'] - merged['yhat']) / \
                                 (merged['yhat_upper'] - merged['yhat_lower'] + 1e-8)
        
        # Calculate confidence score
        merged['confidence'] = np.clip(1 - merged['anomaly_score'], 0, 1)
        
        anomaly_count = merged['is_anomaly'].sum()
        anomaly_rate = anomaly_count / len(merged) * 100
        
        logger.info(f"Detected {anomaly_count} anomalies ({anomaly_rate:.2f}% of data)")
        
        return merged
    
    def predict_single_point(self, timestamp: str, value: float) -> Dict[str, Any]:
        """
        Predict anomaly for a single data point.
        
        Args:
            timestamp: Timestamp string
            value: CPU usage value
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create future dataframe for single point
        future_df = pd.DataFrame({'ds': [pd.to_datetime(timestamp)]})
        
        try:
            forecast = self.model.predict(future_df)
            
            if len(forecast) == 0:
                return {
                    'timestamp': timestamp,
                    'actual_value': value,
                    'error': 'No forecast generated',
                    'is_anomaly': False
                }
            
            pred = forecast.iloc[0]
            
            # Calculate if it's an anomaly
            threshold_upper = pred['yhat_upper'] + (pred['yhat_upper'] - pred['yhat'])
            threshold_lower = pred['yhat_lower'] - (pred['yhat'] - pred['yhat_lower'])
            
            is_anomaly = (value > threshold_upper) or (value < threshold_lower)
            
            # Calculate anomaly score
            pred_range = pred['yhat_upper'] - pred['yhat_lower']
            anomaly_score = abs(value - pred['yhat']) / (pred_range + 1e-8)
            
            return {
                'timestamp': timestamp,
                'actual_value': value,
                'predicted_value': pred['yhat'],
                'prediction_lower': pred['yhat_lower'],
                'prediction_upper': pred['yhat_upper'],
                'threshold_upper': threshold_upper,
                'threshold_lower': threshold_lower,
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'confidence': max(0, 1 - anomaly_score)
            }
            
        except Exception as e:
            logger.error(f"Single point prediction failed: {e}")
            return {
                'timestamp': timestamp,
                'actual_value': value,
                'error': str(e),
                'is_anomaly': False
            }
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'daily_seasonality': self.daily_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'yearly_seasonality': self.yearly_seasonality,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'interval_width': self.interval_width,
            'uncertainty_samples': self.uncertainty_samples,
            'is_trained': self.is_trained
        }