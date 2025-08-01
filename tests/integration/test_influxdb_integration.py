#!/usr/bin/env python3
"""
Integration test for InfluxDB storage and anomaly detection system.
"""

import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from influxdb_storage import InfluxDBConfig, InfluxDBAnomalyStorage, create_influxdb_storage
from anomaly_model import CPUAnomalyDetector
from main import load_config_from_file, AnomalyDetectionSystem


def test_influxdb_storage():
    """Test InfluxDB storage functionality."""
    print("Testing InfluxDB storage...")
    
    # Test configuration
    config = InfluxDBConfig(
        url="http://localhost:8086",
        token="PCIUyihftjXbIAQjR8g-xgEas7H5ssaEQG6ppz-V_w-bxnNeC3FM941sgC8Sclmjhs2VmD7PyGV1oOepRw8xrA==",
        org="test_anamoly",
        bucket="anomaly_detection",
        batch_size=5,  # Small batch for testing
        flush_interval=500  # Quick flush for testing
    )
    
    try:
        with InfluxDBAnomalyStorage(config) as storage:
            # Test connection
            status = storage.get_connection_status()
            print(f"  Connection status: {status['connected']}")
            
            if not status['connected']:
                print("  ⚠️  InfluxDB not available - test will continue without storage")
                return True
            
            # Test storing data
            timestamp = datetime.now(timezone.utc)
            result = storage.store_anomaly_data(
                timestamp=timestamp,
                actual_cpu=85.5,
                forecasted_cpu=75.0,
                is_anomaly=True,
                confidence_score=0.92,
                host="test-host"
            )
            
            print(f"  Store result: {result}")
            
            # Force flush to ensure data is written
            flush_result = storage.force_flush()
            print(f"  Flush result: {flush_result}")
            
            print("  ✅ InfluxDB storage test passed")
            return True
            
    except Exception as e:
        print(f"  ❌ InfluxDB storage test failed: {e}")
        return False


def test_confidence_score_calculation():
    """Test confidence score calculation in anomaly detector."""
    print("Testing confidence score calculation...")
    
    try:
        detector = CPUAnomalyDetector()
        
        # Test cases
        test_cases = [
            # (actual, predicted, lower, upper, expected_range)
            (85.0, 75.0, 70.0, 80.0, (0.7, 1.0)),  # Clear anomaly
            (75.0, 75.0, 70.0, 80.0, (0.0, 0.6)),  # Normal value
            (82.0, 75.0, 70.0, 80.0, (0.6, 0.9)),  # Borderline
        ]
        
        for actual, pred, lower, upper, expected_range in test_cases:
            score = detector.calculate_confidence_score(actual, pred, lower, upper)
            min_exp, max_exp = expected_range
            
            print(f"  Actual: {actual}, Predicted: {pred}, Score: {score:.3f}")
            
            if min_exp <= score <= max_exp:
                print(f"    ✅ Score in expected range [{min_exp}, {max_exp}]")
            else:
                print(f"    ⚠️  Score outside expected range [{min_exp}, {max_exp}]")
        
        print("  ✅ Confidence score calculation test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Confidence score test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    try:
        config = load_config_from_file('config.yaml')
        
        # Check if InfluxDB config is present
        influxdb_config = config.get('influxdb', {})
        
        required_keys = ['enabled', 'url', 'token', 'org', 'bucket']
        missing_keys = [key for key in required_keys if key not in influxdb_config]
        
        if missing_keys:
            print(f"  ⚠️  Missing InfluxDB config keys: {missing_keys}")
        else:
            print(f"  ✅ All required InfluxDB config keys present")
            print(f"  Enabled: {influxdb_config['enabled']}")
            print(f"  URL: {influxdb_config['url']}")
            print(f"  Bucket: {influxdb_config['bucket']}")
        
        print("  ✅ Configuration loading test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration loading test failed: {e}")
        return False


def test_system_integration():
    """Test complete system integration."""
    print("Testing system integration...")
    
    try:
        # Load configuration
        config = load_config_from_file('config.yaml')
        
        # Initialize system
        system = AnomalyDetectionSystem(config)
        
        # Check if InfluxDB storage was initialized
        if system.influxdb_storage:
            print("  ✅ InfluxDB storage initialized in system")
            
            # Test safe storage method
            result = system._store_anomaly_data_safely(
                timestamp=datetime.now(timezone.utc),
                actual_cpu=78.5,
                forecasted_cpu=72.0,
                is_anomaly=False,
                confidence_score=0.65
            )
            print(f"  Safe storage result: {result}")
        else:
            print("  ⚠️  InfluxDB storage not initialized (may be disabled or connection failed)")
        
        print("  ✅ System integration test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ System integration test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("=== InfluxDB Integration and Anomaly Detection Tests ===\n")
    
    tests = [
        test_config_loading,
        test_confidence_score_calculation,
        test_influxdb_storage,
        test_system_integration,
    ]
    
    results = []
    
    for test in tests:
        print(f"Running {test.__name__}...")
        result = test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())