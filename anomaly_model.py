"""
Anomaly detection model using Facebook Prophet for CPU usage time series.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
from loguru import logger
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CPUAnomalyDetector:
    """Anomaly detection for CPU usage using Facebook Prophet."""
    
    def __init__(self, threshold_multiplier=2.0, interval_width=0.95):
        """
        Initialize the anomaly detector.
        
        Args:
            threshold_multiplier: Multiplier for prediction interval to determine anomalies
            interval_width: Prophet prediction interval width (0.95 = 95% confidence)
        """
        self.model = None
        self.threshold_multiplier = threshold_multiplier
        self.interval_width = interval_width
        self.train_data = None
        self.forecast = None
        self.anomalies = None
        self.model_performance = {}
        
    def prepare_data(self, df):
        """
        Prepare data for Prophet model.
        
        Args:
            df: DataFrame with 'timestamp' and 'cpu_usage' columns
            
        Returns:
            DataFrame in Prophet format (ds, y columns)
        """
        logger.info("Preparing data for Prophet model...")
        
        # Prophet expects 'ds' (datestamp) and 'y' (target) columns
        prophet_df = pd.DataFrame({
            'ds': df['timestamp'],
            'y': df['cpu_usage']
        })
        
        # Sort by timestamp
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        logger.info(f"Data prepared. Shape: {prophet_df.shape}")
        logger.info(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
        
        return prophet_df
    
    def train_model(self, df, train_ratio=0.8):
        """
        Train the Prophet model on historical data.
        
        Args:
            df: DataFrame with timestamp and cpu_usage columns
            train_ratio: Ratio of data to use for training
        """
        logger.info(f"Training Prophet model with {train_ratio*100}% of data...")
        
        # Prepare data
        prophet_data = self.prepare_data(df)
        
        # Split into train/test
        split_idx = int(len(prophet_data) * train_ratio)
        self.train_data = prophet_data[:split_idx].copy()
        self.test_data = prophet_data[split_idx:].copy()
        
        logger.info(f"Training data: {len(self.train_data)} points")
        logger.info(f"Test data: {len(self.test_data)} points")
        
        # Initialize Prophet model
        self.model = Prophet(
            interval_width=self.interval_width,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,  # Not enough data for yearly patterns
            changepoint_prior_scale=0.05,  # Control flexibility of trend changes
            seasonality_prior_scale=10.0,  # Control flexibility of seasonality
            uncertainty_samples=1000
        )
        
        # Add custom seasonalities if needed
        # self.model.add_seasonality(name='hourly', period=1, fourier_order=8)
        
        # Fit the model
        logger.info("Fitting Prophet model...")
        self.model.fit(self.train_data)
        
        logger.info("Model training completed successfully!")
        
    def predict(self, periods=None, future_df=None):
        """
        Generate predictions using the trained model.
        
        Args:
            periods: Number of future periods to predict
            future_df: Custom future dataframe (if None, will create one)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        logger.info("Generating predictions...")
        
        if future_df is None:
            if periods is None:
                # Create future dataframe for the entire dataset
                future_df = self.model.make_future_dataframe(
                    periods=len(self.test_data), 
                    freq='30T'  # 30-minute intervals
                )
            else:
                future_df = self.model.make_future_dataframe(periods=periods, freq='30T')
        
        # Generate forecast
        self.forecast = self.model.predict(future_df)
        
        logger.info(f"Predictions generated for {len(self.forecast)} time points")
        
        return self.forecast
    
    def detect_anomalies(self, actual_df=None):
        """
        Detect anomalies by comparing actual values with prediction intervals.
        
        Args:
            actual_df: DataFrame with actual values (if None, uses test data)
        """
        if self.forecast is None:
            raise ValueError("Must generate predictions before detecting anomalies")
            
        logger.info("Detecting anomalies...")
        
        if actual_df is None:
            actual_df = self.test_data
        else:
            actual_df = self.prepare_data(actual_df)
        
        # Merge actual values with forecast
        merged = pd.merge(actual_df, self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                         on='ds', how='inner')
        
        # Calculate dynamic thresholds based on prediction intervals
        merged['threshold_upper'] = merged['yhat_upper'] + \
                                   (merged['yhat_upper'] - merged['yhat']) * (self.threshold_multiplier - 1)
        merged['threshold_lower'] = merged['yhat_lower'] - \
                                   (merged['yhat'] - merged['yhat_lower']) * (self.threshold_multiplier - 1)
        
        # Detect anomalies
        merged['is_anomaly'] = (
            (merged['y'] > merged['threshold_upper']) | 
            (merged['y'] < merged['threshold_lower'])
        )
        
        # Calculate anomaly scores (distance from prediction)
        merged['anomaly_score'] = np.abs(merged['y'] - merged['yhat']) / \
                                 (merged['yhat_upper'] - merged['yhat_lower'] + 1e-8)
        
        self.anomalies = merged
        
        num_anomalies = merged['is_anomaly'].sum()
        anomaly_rate = num_anomalies / len(merged) * 100
        
        logger.info(f"Detected {num_anomalies} anomalies out of {len(merged)} points ({anomaly_rate:.2f}%)")
        
        return self.anomalies
    
    def evaluate_model(self):
        """Evaluate model performance on test data."""
        if self.anomalies is None:
            logger.warning("No anomalies detected yet. Run detect_anomalies() first.")
            return None
            
        logger.info("Evaluating model performance...")
        
        # Calculate metrics on non-anomalous points (normal behavior)
        normal_points = self.anomalies[~self.anomalies['is_anomaly']]
        
        if len(normal_points) > 0:
            mae = mean_absolute_error(normal_points['y'], normal_points['yhat'])
            mse = mean_squared_error(normal_points['y'], normal_points['yhat'])
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((normal_points['y'] - normal_points['yhat']) / 
                                normal_points['y'])) * 100
            
            self.model_performance = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'num_normal_points': len(normal_points),
                'num_anomalies': self.anomalies['is_anomaly'].sum(),
                'anomaly_rate': self.anomalies['is_anomaly'].sum() / len(self.anomalies) * 100
            }
            
            logger.info("=== Model Performance ===")
            logger.info(f"MAE: {mae:.6f}")
            logger.info(f"RMSE: {rmse:.6f}")
            logger.info(f"MAPE: {mape:.2f}%")
            logger.info(f"Anomaly Rate: {self.model_performance['anomaly_rate']:.2f}%")
        
        return self.model_performance
    
    def plot_results(self, save_path='anomaly_detection_results.png'):
        """Plot anomaly detection results."""
        if self.anomalies is None:
            logger.warning("No anomalies to plot. Run detect_anomalies() first.")
            return
            
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Time series with predictions and anomalies
        axes[0].plot(self.train_data['ds'], self.train_data['y'], 
                    label='Training Data', color='blue', alpha=0.7)
        
        axes[0].plot(self.anomalies['ds'], self.anomalies['y'], 
                    label='Test Data', color='green', alpha=0.7)
        
        axes[0].plot(self.anomalies['ds'], self.anomalies['yhat'], 
                    label='Predictions', color='orange', linewidth=2)
        
        # Fill prediction intervals
        axes[0].fill_between(self.anomalies['ds'], 
                           self.anomalies['yhat_lower'], 
                           self.anomalies['yhat_upper'], 
                           alpha=0.2, color='orange', label='Prediction Interval')
        
        # Highlight anomalies
        anomaly_points = self.anomalies[self.anomalies['is_anomaly']]
        if len(anomaly_points) > 0:
            axes[0].scatter(anomaly_points['ds'], anomaly_points['y'], 
                          color='red', s=50, label='Anomalies', zorder=5)
        
        axes[0].set_title('CPU Usage Anomaly Detection', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Timestamp')
        axes[0].set_ylabel('CPU Usage')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores
        axes[1].plot(self.anomalies['ds'], self.anomalies['anomaly_score'], 
                    color='purple', linewidth=1, alpha=0.8)
        axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                       label='Anomaly Threshold')
        
        # Highlight anomalies
        if len(anomaly_points) > 0:
            axes[1].scatter(anomaly_points['ds'], anomaly_points['anomaly_score'], 
                          color='red', s=30, zorder=5)
        
        axes[1].set_title('Anomaly Scores Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Timestamp')
        axes[1].set_ylabel('Anomaly Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Anomaly detection plot saved to {save_path}")
    
    def plot_interactive_results(self, save_path='interactive_anomaly_results.html'):
        """Create interactive plotly visualization of results."""
        if self.anomalies is None:
            logger.warning("No anomalies to plot. Run detect_anomalies() first.")
            return
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU Usage with Anomaly Detection', 'Anomaly Scores'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Training data
        fig.add_trace(
            go.Scatter(
                x=self.train_data['ds'],
                y=self.train_data['y'],
                mode='lines',
                name='Training Data',
                line=dict(color='blue', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Test data
        fig.add_trace(
            go.Scatter(
                x=self.anomalies['ds'],
                y=self.anomalies['y'],
                mode='lines',
                name='Test Data',
                line=dict(color='green', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=self.anomalies['ds'],
                y=self.anomalies['yhat'],
                mode='lines',
                name='Predictions',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
        
        # Prediction intervals
        fig.add_trace(
            go.Scatter(
                x=self.anomalies['ds'],
                y=self.anomalies['yhat_upper'],
                mode='lines',
                line=dict(color='orange', width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.anomalies['ds'],
                y=self.anomalies['yhat_lower'],
                mode='lines',
                fill='tonexty',
                line=dict(color='orange', width=0),
                name='Prediction Interval',
                opacity=0.2
            ),
            row=1, col=1
        )
        
        # Anomalies
        anomaly_points = self.anomalies[self.anomalies['is_anomaly']]
        if len(anomaly_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_points['ds'],
                    y=anomaly_points['y'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=8)
                ),
                row=1, col=1
            )
        
        # Anomaly scores
        fig.add_trace(
            go.Scatter(
                x=self.anomalies['ds'],
                y=self.anomalies['anomaly_score'],
                mode='lines',
                name='Anomaly Score',
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
        
        # Anomaly threshold
        fig.add_hline(
            y=1.0, line_dash="dash", line_color="red",
            annotation_text="Anomaly Threshold",
            row=2, col=1
        )
        
        fig.update_layout(
            title='Interactive CPU Usage Anomaly Detection',
            height=800,
            showlegend=True
        )
        
        fig.write_html(save_path)
        logger.info(f"Interactive anomaly results saved to {save_path}")
    
    def predict_single_point(self, timestamp, cpu_value):
        """
        Predict if a single CPU usage point is anomalous.
        
        Args:
            timestamp: Timestamp of the measurement
            cpu_value: CPU usage value
            
        Returns:
            Dict with prediction results
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Create future dataframe for the timestamp
        future_df = pd.DataFrame({'ds': [pd.to_datetime(timestamp)]})
        
        # Get prediction
        prediction = self.model.predict(future_df)
        
        # Calculate if it's an anomaly
        pred_value = prediction['yhat'].iloc[0]
        pred_upper = prediction['yhat_upper'].iloc[0]
        pred_lower = prediction['yhat_lower'].iloc[0]
        
        threshold_upper = pred_upper + (pred_upper - pred_value) * (self.threshold_multiplier - 1)
        threshold_lower = pred_lower - (pred_value - pred_lower) * (self.threshold_multiplier - 1)
        
        is_anomaly = (cpu_value > threshold_upper) or (cpu_value < threshold_lower)
        anomaly_score = abs(cpu_value - pred_value) / (pred_upper - pred_lower + 1e-8)
        
        # Calculate confidence score
        confidence_score = self.calculate_confidence_score(cpu_value, pred_value, pred_lower, pred_upper)
        
        result = {
            'timestamp': timestamp,
            'actual_value': cpu_value,
            'predicted_value': pred_value,
            'prediction_lower': pred_lower,
            'prediction_upper': pred_upper,
            'threshold_lower': threshold_lower,
            'threshold_upper': threshold_upper,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'confidence_score': confidence_score
        }
        
        return result
    
    def calculate_confidence_score(self, actual_value, predicted_value, pred_lower, pred_upper):
        """
        Calculate confidence score for anomaly detection.
        
        This score represents the confidence in the anomaly detection result,
        taking into account the prediction interval width and the distance from prediction.
        
        Args:
            actual_value: Actual CPU usage value
            predicted_value: Predicted CPU usage value  
            pred_lower: Lower bound of prediction interval
            pred_upper: Upper bound of prediction interval
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Calculate prediction interval width (uncertainty)
        interval_width = pred_upper - pred_lower
        
        # Calculate normalized distance from prediction
        distance_from_prediction = abs(actual_value - predicted_value)
        
        # Avoid division by zero
        if interval_width <= 1e-8:
            return 0.5  # Neutral confidence when no uncertainty info
            
        # Calculate raw confidence based on distance relative to interval width
        # Values far from prediction (relative to interval) have higher confidence
        # Values close to prediction boundaries have lower confidence
        raw_confidence = distance_from_prediction / interval_width
        
        # Apply sigmoid transformation to get score between 0 and 1
        # This gives smooth transition and reasonable confidence scores
        confidence = 1 / (1 + np.exp(-2 * (raw_confidence - 0.5)))
        
        # Additional confidence boost for values clearly outside prediction interval
        if actual_value > pred_upper or actual_value < pred_lower:
            # Outside interval - boost confidence
            outside_distance = max(
                actual_value - pred_upper if actual_value > pred_upper else 0,
                pred_lower - actual_value if actual_value < pred_lower else 0
            )
            outside_boost = min(0.3, outside_distance / interval_width * 0.2)
            confidence = min(1.0, confidence + outside_boost)
        
        # Ensure confidence is within valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def save_model(self, model_path='cpu_anomaly_model.pkl', metadata_path='model_metadata.json'):
        """Save the trained model and metadata."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        # Save the Prophet model
        joblib.dump(self.model, model_path)
        
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
        
        # Save metadata
        metadata = {
            'threshold_multiplier': self.threshold_multiplier,
            'interval_width': self.interval_width,
            'model_performance': convert_numpy_types(self.model_performance),
            'train_data_size': len(self.train_data) if self.train_data is not None else 0,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path='cpu_anomaly_model.pkl', metadata_path='model_metadata.json'):
        """Load a saved model and metadata."""
        # Load the Prophet model
        self.model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.threshold_multiplier = metadata['threshold_multiplier']
        self.interval_width = metadata['interval_width']
        self.model_performance = metadata.get('model_performance', {})
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model was trained at: {metadata.get('trained_at', 'Unknown')}")
        
    def get_anomaly_summary(self):
        """Get summary of detected anomalies."""
        if self.anomalies is None:
            return None
            
        anomaly_points = self.anomalies[self.anomalies['is_anomaly']].copy()
        
        if len(anomaly_points) == 0:
            return {"num_anomalies": 0, "anomalies": []}
        
        # Sort by anomaly score (descending)
        anomaly_points = anomaly_points.sort_values('anomaly_score', ascending=False)
        
        summary = {
            "num_anomalies": len(anomaly_points),
            "anomaly_rate": len(anomaly_points) / len(self.anomalies) * 100,
            "avg_anomaly_score": anomaly_points['anomaly_score'].mean(),
            "max_anomaly_score": anomaly_points['anomaly_score'].max(),
            "anomalies": []
        }
        
        for _, row in anomaly_points.iterrows():
            summary["anomalies"].append({
                "timestamp": row['ds'].isoformat(),
                "actual_value": float(row['y']),
                "predicted_value": float(row['yhat']),
                "anomaly_score": float(row['anomaly_score'])
            })
        
        return summary


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import CPUUsageProcessor
    
    # Load and preprocess data
    processor = CPUUsageProcessor("dataset.csv")
    data = processor.run_full_eda()
    
    # Train anomaly detection model
    detector = CPUAnomalyDetector()
    detector.train_model(data)
    detector.predict()
    detector.detect_anomalies()
    detector.evaluate_model()
    
    # Generate visualizations
    detector.plot_results()
    detector.plot_interactive_results()
    
    # Save model
    detector.save_model()
    
    # Get anomaly summary
    summary = detector.get_anomaly_summary()
    print(f"Detected {summary['num_anomalies']} anomalies")