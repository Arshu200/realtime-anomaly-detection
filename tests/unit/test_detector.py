"""
Unit tests for the CPUAnomalyDetector class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from anomaly_detector.detector import CPUAnomalyDetector
from anomaly_detector.models import ProphetModel


class TestCPUAnomalyDetector:
    """Test cases for CPUAnomalyDetector."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample CPU usage data for testing."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='30T'
        )
        
        # Generate realistic CPU usage pattern
        np.random.seed(42)
        base_usage = 30 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 48)
        noise = np.random.normal(0, 5, len(dates))
        cpu_usage = np.clip(base_usage + noise, 0, 100)
        
        return pd.DataFrame({
            'ds': dates,
            'y': cpu_usage
        })
    
    @pytest.fixture
    def detector(self):
        """Create a CPUAnomalyDetector instance."""
        return CPUAnomalyDetector(
            threshold_multiplier=2.0,
            interval_width=0.95
        )
    
    def test_detector_initialization(self, detector):
        """Test that detector initializes correctly."""
        assert detector.threshold_multiplier == 2.0
        assert detector.interval_width == 0.95
        assert detector.model is None
        assert not detector.is_trained
    
    def test_prepare_data(self, detector):
        """Test data preparation for Prophet."""
        raw_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
            'cpu_usage': np.random.uniform(20, 80, 10)
        })
        
        prepared_data = detector.prepare_data(raw_data)
        
        assert 'ds' in prepared_data.columns
        assert 'y' in prepared_data.columns
        assert len(prepared_data) == 10
        assert prepared_data['ds'].dtype.kind == 'M'  # datetime type
    
    def test_train_model(self, detector, sample_data):
        """Test model training."""
        detector.train_model(sample_data, train_ratio=0.8)
        
        assert detector.is_trained
        assert detector.model is not None
        assert detector.train_data is not None
        assert len(detector.train_data) < len(sample_data)
    
    def test_predict(self, detector, sample_data):
        """Test prediction generation."""
        detector.train_model(sample_data, train_ratio=0.8)
        detector.predict()
        
        assert detector.forecast is not None
        assert 'yhat' in detector.forecast.columns
        assert 'yhat_lower' in detector.forecast.columns
        assert 'yhat_upper' in detector.forecast.columns
    
    def test_detect_anomalies(self, detector, sample_data):
        """Test anomaly detection."""
        detector.train_model(sample_data, train_ratio=0.8)
        detector.predict()
        detector.detect_anomalies()
        
        assert detector.anomalies is not None
        assert 'is_anomaly' in detector.anomalies.columns
        assert 'anomaly_score' in detector.anomalies.columns
        assert detector.anomalies['is_anomaly'].dtype == bool
    
    def test_predict_single_point(self, detector, sample_data):
        """Test single point prediction."""
        detector.train_model(sample_data, train_ratio=0.8)
        
        timestamp = datetime.now().isoformat()
        cpu_value = 75.5
        
        result = detector.predict_single_point(timestamp, cpu_value)
        
        assert 'timestamp' in result
        assert 'actual_value' in result
        assert 'predicted_value' in result
        assert 'is_anomaly' in result
        assert 'confidence' in result
        assert result['actual_value'] == cpu_value
    
    def test_calculate_confidence_score(self, detector):
        """Test confidence score calculation."""
        score = detector.calculate_confidence_score(50.0, 52.0, 45.0, 55.0)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_model_save_load(self, detector, sample_data, tmp_path):
        """Test model saving and loading."""
        # Train model
        detector.train_model(sample_data, train_ratio=0.8)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        detector.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Create new detector and load model
        new_detector = CPUAnomalyDetector()
        new_detector.load_model(str(model_path))
        
        assert new_detector.is_trained
        assert new_detector.model is not None
    
    def test_evaluate_model(self, detector, sample_data):
        """Test model evaluation."""
        detector.train_model(sample_data, train_ratio=0.8)
        detector.predict()
        detector.detect_anomalies()
        
        performance = detector.evaluate_model()
        
        assert isinstance(performance, dict)
        assert 'mape' in performance
        assert 'mae' in performance
        assert 'rmse' in performance


class TestProphetModel:
    """Test cases for ProphetModel."""
    
    @pytest.fixture
    def prophet_model(self):
        """Create a ProphetModel instance."""
        return ProphetModel(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
    
    @pytest.fixture
    def sample_prophet_data(self):
        """Create sample data in Prophet format."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        values = 50 + 10 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 5, 100)
        
        return pd.DataFrame({
            'ds': dates,
            'y': np.clip(values, 0, 100)
        })
    
    def test_prophet_initialization(self, prophet_model):
        """Test ProphetModel initialization."""
        assert prophet_model.daily_seasonality
        assert prophet_model.weekly_seasonality
        assert not prophet_model.yearly_seasonality
        assert not prophet_model.is_trained
    
    def test_prophet_train(self, prophet_model, sample_prophet_data):
        """Test Prophet model training."""
        prophet_model.train(sample_prophet_data)
        
        assert prophet_model.is_trained
        assert prophet_model.model is not None
    
    def test_prophet_predict(self, prophet_model, sample_prophet_data):
        """Test Prophet prediction."""
        prophet_model.train(sample_prophet_data)
        forecast = prophet_model.predict(periods=24)
        
        assert len(forecast) == len(sample_prophet_data) + 24
        assert 'yhat' in forecast.columns
        assert 'yhat_lower' in forecast.columns
        assert 'yhat_upper' in forecast.columns
    
    def test_prophet_detect_anomalies(self, prophet_model, sample_prophet_data):
        """Test Prophet anomaly detection."""
        prophet_model.train(sample_prophet_data)
        forecast = prophet_model.predict(periods=0)
        
        anomalies = prophet_model.detect_anomalies(
            sample_prophet_data, forecast, threshold_multiplier=2.0
        )
        
        assert 'is_anomaly' in anomalies.columns
        assert 'anomaly_score' in anomalies.columns
        assert 'confidence' in anomalies.columns


if __name__ == "__main__":
    pytest.main([__file__])