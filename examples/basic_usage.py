#!/usr/bin/env python3
"""
Basic usage example for the Real-Time Anomaly Detection System.

This example demonstrates the simplest way to use the anomaly detection system
for training a model and monitoring CPU usage.
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from anomaly_detector import CPUAnomalyDetector
from storage import InfluxDBAnomalyStorage, InfluxDBConfig
from monitoring import CPUMonitor, PrometheusClient
from utils import load_config, setup_logging


def main():
    """Basic usage example."""
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    print("🔍 Real-Time Anomaly Detection System - Basic Usage Example")
    print("=" * 60)
    
    # Load configuration
    config = load_config("../config/config.yaml")
    
    # 1. Initialize and train anomaly detector
    print("\n1. Training Anomaly Detection Model...")
    
    detector = CPUAnomalyDetector(
        threshold_multiplier=2.0,
        interval_width=0.95
    )
    
    # Example: Load sample data (replace with your actual data file)
    sample_data_path = "../dataset.csv"
    if os.path.exists(sample_data_path):
        import pandas as pd
        
        # Load sample data
        data = pd.read_csv(sample_data_path)
        print(f"   Loaded {len(data)} data points from {sample_data_path}")
        
        # Prepare data for Prophet
        from anomaly_detector.utils import prepare_data_for_prophet
        prophet_data = prepare_data_for_prophet(data)
        
        # Train model
        detector.train_model(prophet_data, train_ratio=0.8)
        detector.predict()
        detector.detect_anomalies()
        
        # Save model
        model_path = "../trained_model.pkl"
        detector.save_model(model_path)
        print(f"   ✓ Model trained and saved to {model_path}")
        
        # Show training results
        performance = detector.evaluate_model()
        if performance:
            print(f"   Model Performance:")
            print(f"     - Precision: {performance.get('precision', 0):.3f}")
            print(f"     - Recall: {performance.get('recall', 0):.3f}")
            print(f"     - F1-Score: {performance.get('f1', 0):.3f}")
    else:
        print(f"   ⚠️  Sample data file not found: {sample_data_path}")
        print("   Creating a simple trained detector for demonstration...")
        
        # Create minimal sample data for demonstration
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate sample CPU data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='30T'
        )
        
        # Generate realistic CPU usage pattern
        np.random.seed(42)
        base_usage = 30 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 48)  # Daily pattern
        noise = np.random.normal(0, 5, len(dates))
        cpu_usage = np.clip(base_usage + noise, 0, 100)
        
        sample_data = pd.DataFrame({
            'ds': dates,
            'y': cpu_usage
        })
        
        # Train with sample data
        detector.train_model(sample_data, train_ratio=0.8)
        detector.predict()
        detector.detect_anomalies()
        
        print("   ✓ Model trained with generated sample data")
    
    # 2. Setup storage (optional)
    print("\n2. Setting up Storage...")
    
    influxdb_config = InfluxDBConfig(
        url=config.get('influxdb', {}).get('url', 'http://localhost:8086'),
        token=config.get('influxdb', {}).get('token', ''),
        org=config.get('influxdb', {}).get('org', 'test_anamoly'),
        bucket=config.get('influxdb', {}).get('bucket', 'anomaly_detection')
    )
    
    try:
        storage = InfluxDBAnomalyStorage(influxdb_config)
        if storage.connect():
            print("   ✓ InfluxDB storage connected")
        else:
            print("   ⚠️  InfluxDB storage not available - continuing without storage")
            storage = None
    except Exception as e:
        print(f"   ⚠️  InfluxDB storage error: {e}")
        storage = None
    
    # 3. Setup monitoring
    print("\n3. Setting up CPU Monitoring...")
    
    prometheus_client = PrometheusClient(
        prometheus_url=config.get('prometheus_url', 'http://localhost:9090')
    )
    
    if prometheus_client.check_connection():
        print("   ✓ Prometheus client connected")
    else:
        print("   ⚠️  Prometheus not available - will use simulation mode")
    
    cpu_monitor = CPUMonitor(prometheus_client)
    
    # 4. Test single prediction
    print("\n4. Testing Single Prediction...")
    
    from datetime import datetime
    test_timestamp = datetime.now().isoformat()
    test_cpu_value = 75.5
    
    result = detector.predict_single_point(test_timestamp, test_cpu_value)
    
    print(f"   Test Input: {test_cpu_value}% CPU at {test_timestamp}")
    print(f"   Prediction: {result.get('predicted_value', 0):.2f}%")
    print(f"   Is Anomaly: {'Yes' if result.get('is_anomaly', False) else 'No'}")
    print(f"   Confidence: {result.get('confidence', 0):.3f}")
    
    # 5. Store test result (if storage available)
    if storage:
        try:
            success = storage.store_anomaly_data(
                timestamp=datetime.now(),
                actual_cpu=test_cpu_value,
                forecasted_cpu=result.get('predicted_value', 0),
                is_anomaly=result.get('is_anomaly', False),
                confidence_score=result.get('confidence', 0)
            )
            if success:
                print("   ✓ Test result stored in InfluxDB")
        except Exception as e:
            print(f"   ⚠️  Storage error: {e}")
    
    # 6. Demonstration of real-time monitoring (short duration)
    print("\n5. Real-Time Monitoring Demo (30 seconds)...")
    print("   Press Ctrl+C to stop monitoring early")
    
    try:
        import time
        import threading
        
        monitoring_active = True
        
        def monitoring_demo():
            """Run monitoring for a short demonstration."""
            count = 0
            while monitoring_active and count < 3:  # Run for ~30 seconds (3 x 10s intervals)
                try:
                    # Get CPU data (will use simulation if Prometheus unavailable)
                    cpu_data = prometheus_client.get_current_cpu_usage()
                    
                    if cpu_data:
                        # Process with anomaly detection
                        result = detector.predict_single_point(
                            cpu_data['timestamp'],
                            cpu_data['cpu_usage']
                        )
                        
                        status = "🚨 ANOMALY" if result.get('is_anomaly', False) else "✓ Normal"
                        print(f"   {status}: {cpu_data['cpu_usage']:.1f}% CPU "
                              f"(predicted: {result.get('predicted_value', 0):.1f}%, "
                              f"confidence: {result.get('confidence', 0):.2f})")
                        
                        # Store if available
                        if storage:
                            try:
                                storage.store_anomaly_data(
                                    timestamp=datetime.now(),
                                    actual_cpu=cpu_data['cpu_usage'],
                                    forecasted_cpu=result.get('predicted_value', 0),
                                    is_anomaly=result.get('is_anomaly', False),
                                    confidence_score=result.get('confidence', 0)
                                )
                            except:
                                pass  # Silent fail for demo
                    
                    count += 1
                    time.sleep(10)  # 10 second intervals for demo
                    
                except Exception as e:
                    print(f"   Monitoring error: {e}")
                    break
        
        # Run monitoring in thread for demo
        monitor_thread = threading.Thread(target=monitoring_demo)
        monitor_thread.start()
        monitor_thread.join()
        
        monitoring_active = False
        
    except KeyboardInterrupt:
        print("\n   Monitoring stopped by user")
        monitoring_active = False
    except Exception as e:
        print(f"\n   Monitoring demo error: {e}")
    
    # Cleanup
    if storage:
        storage.close()
    
    print("\n" + "=" * 60)
    print("✓ Basic usage example completed!")
    print("\nNext steps:")
    print("1. Review the generated model and results")
    print("2. Check Grafana dashboard at http://localhost:3000")
    print("3. Run full monitoring: python src/main.py --mode monitor")
    print("4. See docs/api_reference.md for advanced usage")


if __name__ == "__main__":
    main()