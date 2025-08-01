"""
Main monitoring script for real-time CPU anomaly detection.
"""

import time
import argparse
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path

from data_preprocessing import CPUUsageProcessor
from anomaly_model import CPUAnomalyDetector
from model_evaluation import ModelEvaluator
from prometheus_client import PrometheusClient, RealTimeAnomalyMonitor
from logging_config import setup_logging, get_structured_logger
from influxdb_storage import create_influxdb_storage, InfluxDBAnomalyStorage
from loguru import logger


class AnomalyDetectionSystem:
    """Main system orchestrating all components."""
    
    def __init__(self, config: dict):
        """Initialize the system with configuration."""
        self.config = config
        
        # Setup logging
        self.structured_logger = setup_logging(
            log_level=config.get('log_level', 'INFO'),
            log_dir=config.get('log_dir', 'logs')
        )
        
        # Initialize components
        self.processor = None
        self.detector = None
        self.evaluator = None
        self.prometheus_client = None
        self.monitor = None
        self.influxdb_storage = None
        
        # Initialize InfluxDB storage if enabled
        self._setup_influxdb_storage()
        
        logger.info("Anomaly Detection System initialized")
    
    def _setup_influxdb_storage(self):
        """Setup InfluxDB storage if enabled in configuration."""
        influxdb_config = self.config.get('influxdb', {})
        
        if not influxdb_config.get('enabled', False):
            logger.info("InfluxDB storage disabled in configuration")
            return
            
        logger.info("Setting up InfluxDB storage...")
        self.influxdb_storage = create_influxdb_storage(influxdb_config)
        
        if self.influxdb_storage:
            logger.info("InfluxDB storage initialized successfully")
        else:
            logger.warning("InfluxDB storage initialization failed - continuing without storage")
    
    def _store_anomaly_data_safely(self, timestamp, actual_cpu, forecasted_cpu, is_anomaly, confidence_score):
        """
        Safely store anomaly data to InfluxDB with error handling.
        
        This method ensures that storage failures don't impact the detection performance.
        """
        if not self.influxdb_storage:
            return False
            
        try:
            return self.influxdb_storage.store_anomaly_data(
                timestamp=timestamp,
                actual_cpu=actual_cpu,
                forecasted_cpu=forecasted_cpu,
                is_anomaly=is_anomaly,
                confidence_score=confidence_score
            )
        except Exception as e:
            logger.error(f"Failed to store anomaly data: {e}")
            return False
    
    def prepare_data(self, dataset_path: str):
        """Prepare training data."""
        logger.info(f"Preparing data from {dataset_path}")
        
        self.processor = CPUUsageProcessor(dataset_path)
        
        # Run EDA if requested
        if self.config.get('run_eda', True):
            logger.info("Running EDA pipeline...")
            self.processor.run_full_eda()
        else:
            # Just load and resample data
            self.processor.load_data()
            self.processor.resample_data(frequency='30T')
        
        return self.processor.get_processed_data()
    
    def train_model(self, data):
        """Train the anomaly detection model."""
        logger.info("Training anomaly detection model...")
        
        start_time = time.time()
        
        self.detector = CPUAnomalyDetector(
            threshold_multiplier=self.config.get('threshold_multiplier', 2.0),
            interval_width=self.config.get('interval_width', 0.95)
        )
        
        # Train model
        train_ratio = self.config.get('train_ratio', 0.8)
        self.detector.train_model(data, train_ratio=train_ratio)
        
        # Generate predictions
        self.detector.predict()
        
        # Detect anomalies
        self.detector.detect_anomalies()
        
        # Evaluate model
        performance = self.detector.evaluate_model()
        
        training_time = time.time() - start_time
        
        # Log training event
        self.structured_logger.log_model_training(
            dataset_size=len(data),
            train_size=len(self.detector.train_data),
            model_type="prophet",
            training_time=training_time,
            model_parameters={
                'threshold_multiplier': self.detector.threshold_multiplier,
                'interval_width': self.detector.interval_width,
                'train_ratio': train_ratio
            }
        )
        
        # Log evaluation metrics
        if performance:
            self.structured_logger.log_model_evaluation(performance, "prophet")
        
        # Save model
        model_path = self.config.get('model_path', 'cpu_anomaly_model.pkl')
        self.detector.save_model(model_path)
        
        # Generate visualizations
        if self.config.get('generate_plots', True):
            self.detector.plot_results()
            self.detector.plot_interactive_results()
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        return performance
    
    def evaluate_with_simulation(self, data):
        """Evaluate model with simulated anomalies."""
        if self.detector is None:
            logger.error("Model must be trained before evaluation")
            return None
            
        logger.info("Running evaluation with simulated anomalies...")
        
        self.evaluator = ModelEvaluator()
        
        # Create simulated dataset
        simulated_data = self.evaluator.simulate_anomalies(
            data, 
            anomaly_ratio=self.config.get('simulation_anomaly_ratio', 0.05)
        )
        
        # Prepare simulated data for model
        sim_prophet_data = self.detector.prepare_data(simulated_data)
        
        # Get predictions for simulated data
        sim_forecast = self.detector.model.predict(
            self.detector.model.make_future_dataframe(
                periods=len(sim_prophet_data), freq='30T'
            )
        )
        
        # Merge with actual data for anomaly detection
        merged = pd.merge(
            sim_prophet_data, 
            sim_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
            on='ds', how='inner'
        )
        
        # Calculate anomaly scores
        merged['threshold_upper'] = merged['yhat_upper'] + \
                                   (merged['yhat_upper'] - merged['yhat']) * (self.detector.threshold_multiplier - 1)
        merged['threshold_lower'] = merged['yhat_lower'] - \
                                   (merged['yhat'] - merged['yhat_lower']) * (self.detector.threshold_multiplier - 1)
        
        merged['is_detected_anomaly'] = (
            (merged['y'] > merged['threshold_upper']) | 
            (merged['y'] < merged['threshold_lower'])
        )
        
        merged['anomaly_score'] = np.abs(merged['y'] - merged['yhat']) / \
                                 (merged['yhat_upper'] - merged['yhat_lower'] + 1e-8)
        
        # Add true labels from simulation
        merged = pd.merge(
            merged, 
            simulated_data[['timestamp', 'is_true_anomaly']].rename(columns={'timestamp': 'ds'}),
            on='ds', how='left'
        )
        merged['is_true_anomaly'] = merged['is_true_anomaly'].fillna(False)
        
        # Evaluate performance
        evaluation_results = self.evaluator.evaluate_detection_performance(
            merged['is_true_anomaly'].astype(int),
            merged['is_detected_anomaly'].astype(int),
            merged['anomaly_score']
        )
        
        # Generate evaluation plots
        if self.config.get('generate_evaluation_plots', True):
            merged['timestamp'] = merged['ds']  # For plotting compatibility
            merged['cpu_usage'] = merged['y']
            
            self.evaluator.plot_confusion_matrix(
                merged['is_true_anomaly'].astype(int),
                merged['is_detected_anomaly'].astype(int)
            )
            
            self.evaluator.plot_roc_curve(
                merged['is_true_anomaly'].astype(int),
                merged['anomaly_score']
            )
            
            self.evaluator.plot_detection_overview(merged)
            
            # Threshold analysis
            threshold_results, optimal_threshold = self.evaluator.evaluate_detection_rate_by_threshold(
                merged['is_true_anomaly'].astype(int),
                merged['anomaly_score']
            )
            
            if threshold_results is not None:
                self.evaluator.plot_threshold_analysis(threshold_results)
                logger.info(f"Optimal threshold found: {optimal_threshold:.3f}")
        
        # Generate evaluation report
        self.evaluator.generate_evaluation_report()
        
        logger.info("Evaluation with simulated anomalies completed")
        return evaluation_results
    
    def setup_prometheus_client(self):
        """Setup Prometheus client for real-time monitoring."""
        prometheus_url = self.config.get('prometheus_url', 'http://localhost:9090')
        
        self.prometheus_client = PrometheusClient(
            prometheus_url=prometheus_url,
            timeout=self.config.get('prometheus_timeout', 30),
            retry_attempts=self.config.get('prometheus_retries', 3)
        )
        
        # Test connection
        if self.prometheus_client.check_connection():
            logger.info("Connected to Prometheus server")
            self.structured_logger.log_system_status("connected", "prometheus")
        else:
            logger.warning("Could not connect to Prometheus - will use simulation mode")
            self.structured_logger.log_system_status("simulation_mode", "prometheus")
        
        return self.prometheus_client
    
    def start_real_time_monitoring(self):
        """Start real-time monitoring."""
        if self.detector is None:
            logger.error("Model must be trained before starting monitoring")
            return
            
        if self.prometheus_client is None:
            self.setup_prometheus_client()
        
        # Create real-time monitor
        self.monitor = RealTimeAnomalyMonitor(self.prometheus_client, self.detector)
        
        # Enhanced monitoring callback with structured logging and InfluxDB storage
        def enhanced_callback(cpu_data):
            # Log CPU usage
            self.structured_logger.log_cpu_usage(
                cpu_data['timestamp'],
                cpu_data['cpu_usage'],
                cpu_data.get('instance', 'unknown'),
                'prometheus' if cpu_data.get('query_used') != 'simulation' else 'simulation'
            )
            
            # Process with anomaly detection using the original method
            result = self.monitor.process_cpu_reading(cpu_data)
            
            # Log prediction
            if 'predicted_value' in result:
                self.structured_logger.log_prediction(
                    result['timestamp'],
                    result['actual_value'],
                    result['predicted_value'],
                    (result.get('prediction_lower', 0), result.get('prediction_upper', 1))
                )
                
                # Store data in InfluxDB if available
                try:
                    # Calculate confidence score using the detector's method
                    confidence_score = self.detector.calculate_confidence_score(
                        result['actual_value'],
                        result['predicted_value'],
                        result.get('prediction_lower', result['predicted_value'] - 10),
                        result.get('prediction_upper', result['predicted_value'] + 10)
                    )
                    
                    # Store the data
                    self._store_anomaly_data_safely(
                        timestamp=result['timestamp'],
                        actual_cpu=result['actual_value'],
                        forecasted_cpu=result['predicted_value'],
                        is_anomaly=result.get('is_anomaly', False),
                        confidence_score=confidence_score
                    )
                except Exception as e:
                    logger.debug(f"Error storing data to InfluxDB: {e}")
            
            # Log anomaly if detected
            if result.get('is_anomaly', False):
                self.structured_logger.log_anomaly(
                    result['timestamp'],
                    result['actual_value'],
                    result.get('anomaly_score', 0),
                    result.get('predicted_value', 0),
                    {
                        'upper': result.get('threshold_upper', 1),
                        'lower': result.get('threshold_lower', 0)
                    },
                    result.get('instance', 'unknown')
                )
        
        # Start monitoring
        monitoring_interval = self.config.get('monitoring_interval', 30)
        
        logger.info(f"Starting real-time monitoring (interval: {monitoring_interval}s)")
        self.structured_logger.log_system_status("started", "monitor")
        
        try:
            self.prometheus_client.monitor_cpu_continuous(
                interval_seconds=monitoring_interval,
                callback=enhanced_callback
            )
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            self.structured_logger.log_system_status("stopped", "monitor")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            self.structured_logger.log_system_status("error", "monitor", {"error": str(e)})
        finally:
            # Cleanup InfluxDB storage
            if self.influxdb_storage:
                logger.info("Cleaning up InfluxDB storage...")
                self.influxdb_storage.close()
    
    def load_saved_model(self, model_path: str):
        """Load a previously saved model."""
        logger.info(f"Loading saved model from {model_path}")
        
        self.detector = CPUAnomalyDetector()
        self.detector.load_model(model_path)
        
        logger.info("Model loaded successfully")
        return self.detector
    
    def generate_system_report(self):
        """Generate a comprehensive system report."""
        logger.info("Generating system report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'system_status': {}
        }
        
        # Model performance
        if self.detector and hasattr(self.detector, 'model_performance'):
            report['model_performance'] = self.detector.model_performance
        
        # Recent metrics from structured logger
        if self.structured_logger:
            recent_metrics = self.structured_logger.generate_summary_report(hours=24)
            report['recent_activity'] = recent_metrics
        
        # Monitoring stats
        if self.monitor:
            monitoring_stats = self.monitor.get_monitoring_stats()
            report['monitoring_stats'] = monitoring_stats
        
        # Save report
        report_path = Path(self.config.get('log_dir', 'logs')) / 'system_report.json'
        
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
        
        converted_report = convert_numpy_types(report)
        
        with open(report_path, 'w') as f:
            import json
            json.dump(converted_report, f, indent=2)
        
        logger.info(f"System report saved to {report_path}")
        return report


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CPU Anomaly Detection System')
    
    parser.add_argument('--mode', choices=['train', 'monitor', 'evaluate'], default='train',
                       help='Operation mode')
    parser.add_argument('--dataset', default='dataset.csv',
                       help='Path to training dataset')
    parser.add_argument('--model-path', default='cpu_anomaly_model.pkl',
                       help='Path to save/load model')
    parser.add_argument('--prometheus-url', default='http://localhost:9090',
                       help='Prometheus server URL')
    parser.add_argument('--monitoring-interval', type=int, default=30,
                       help='Monitoring interval in seconds')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--threshold-multiplier', type=float, default=2.0,
                       help='Anomaly threshold multiplier')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--no-eda', action='store_true',
                       help='Skip EDA during training')
    
    return parser.parse_args()


def load_config_from_file(config_file: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_file)
    
    if config_path.exists():
        logger.info(f"Loading configuration from {config_file}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Configuration file {config_file} not found, using defaults")
        return {}


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load configuration from file first
    file_config = load_config_from_file('config.yaml')
    
    # Configuration - merge file config with command line arguments
    config = {
        'log_level': args.log_level,
        'log_dir': 'logs',
        'prometheus_url': args.prometheus_url,
        'monitoring_interval': args.monitoring_interval,
        'threshold_multiplier': args.threshold_multiplier,
        'model_path': args.model_path,
        'generate_plots': not args.no_plots,
        'run_eda': not args.no_eda,
        'train_ratio': 0.8,
        'interval_width': 0.95,
        'simulation_anomaly_ratio': 0.05,
        'generate_evaluation_plots': True
    }
    
    # Merge with file configuration (file config takes precedence for missing CLI args)
    for key, value in file_config.items():
        if key not in config or key in ['influxdb']:  # Always include influxdb config from file
            config[key] = value
    
    # Initialize system
    system = AnomalyDetectionSystem(config)
    
    try:
        if args.mode == 'train':
            logger.info("Starting training mode...")
            
            # Prepare data
            data = system.prepare_data(args.dataset)
            
            # Train model
            performance = system.train_model(data)
            
            # Evaluate with simulated anomalies
            eval_results = system.evaluate_with_simulation(data)
            
            logger.info("Training completed successfully!")
            
        elif args.mode == 'monitor':
            logger.info("Starting monitoring mode...")
            
            # Try to load existing model
            if Path(args.model_path).exists():
                system.load_saved_model(args.model_path)
            else:
                logger.error(f"Model file not found: {args.model_path}")
                logger.info("Please train a model first using --mode train")
                return
            
            # Start real-time monitoring
            system.start_real_time_monitoring()
            
        elif args.mode == 'evaluate':
            logger.info("Starting evaluation mode...")
            
            # Load model
            if Path(args.model_path).exists():
                system.load_saved_model(args.model_path)
                
                # Load data for evaluation
                data = system.prepare_data(args.dataset)
                
                # Run evaluation
                eval_results = system.evaluate_with_simulation(data)
                
                logger.info("Evaluation completed!")
            else:
                logger.error(f"Model file not found: {args.model_path}")
                return
        
        # Generate final system report
        system.generate_system_report()
        
    except Exception as e:
        logger.error(f"System error: {e}")
        raise


if __name__ == "__main__":
    main()