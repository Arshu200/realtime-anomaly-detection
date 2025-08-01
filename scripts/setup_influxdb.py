#!/usr/bin/env python3
"""
InfluxDB setup script for the Real-Time Anomaly Detection System.

This script helps set up and configure InfluxDB for the anomaly detection system.
"""

import sys
import requests
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path


class InfluxDBSetup:
    """Helper class for InfluxDB setup and configuration."""
    
    def __init__(self, url: str = "http://localhost:8086"):
        """Initialize InfluxDB setup."""
        self.url = url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def check_health(self) -> bool:
        """Check if InfluxDB is running and healthy."""
        try:
            response = self.session.get(f"{self.url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✓ InfluxDB is running: {health_data.get('status', 'unknown')}")
                return True
            else:
                print(f"✗ InfluxDB health check failed: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to InfluxDB at {self.url}: {e}")
            return False
    
    def setup_initial_user(self, username: str, password: str, org: str, bucket: str) -> Optional[str]:
        """Set up initial user and organization."""
        setup_data = {
            "username": username,
            "password": password,
            "org": org,
            "bucket": bucket,
            "retentionPeriodHrs": 720  # 30 days
        }
        
        try:
            response = self.session.post(f"{self.url}/api/v2/setup", json=setup_data, timeout=30)
            
            if response.status_code == 201:
                result = response.json()
                token = result.get('auth', {}).get('token')
                print(f"✓ InfluxDB setup completed successfully")
                print(f"  Organization: {org}")
                print(f"  Bucket: {bucket}")
                print(f"  Token: {token}")
                return token
            elif response.status_code == 422:
                print("ℹ️  InfluxDB is already set up")
                return None
            else:
                print(f"✗ Setup failed: HTTP {response.status_code}")
                print(f"  Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Setup request failed: {e}")
            return None
    
    def create_bucket(self, token: str, org: str, bucket_name: str, retention_days: int = 30) -> bool:
        """Create a new bucket."""
        headers = {'Authorization': f'Token {token}'}
        
        bucket_data = {
            "name": bucket_name,
            "orgID": org,
            "retentionRules": [
                {
                    "type": "expire",
                    "everySeconds": retention_days * 24 * 60 * 60
                }
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.url}/api/v2/buckets",
                json=bucket_data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 201:
                print(f"✓ Bucket '{bucket_name}' created successfully")
                return True
            elif response.status_code == 422:
                error_data = response.json()
                if "already exists" in error_data.get('message', ''):
                    print(f"ℹ️  Bucket '{bucket_name}' already exists")
                    return True
                else:
                    print(f"✗ Bucket creation failed: {error_data.get('message', 'Unknown error')}")
                    return False
            else:
                print(f"✗ Bucket creation failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Bucket creation request failed: {e}")
            return False
    
    def test_write_read(self, token: str, org: str, bucket: str) -> bool:
        """Test writing and reading data."""
        headers = {'Authorization': f'Token {token}'}
        
        # Test write
        write_data = f"test_measurement,host=test_host value=42.0 {int(time.time() * 1000000000)}"
        
        try:
            response = self.session.post(
                f"{self.url}/api/v2/write",
                params={'org': org, 'bucket': bucket, 'precision': 'ns'},
                data=write_data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 204:
                print("✓ Test write successful")
            else:
                print(f"✗ Test write failed: HTTP {response.status_code}")
                return False
            
            # Test read
            query = f'''
                from(bucket: "{bucket}")
                |> range(start: -1h)
                |> filter(fn: (r) => r["_measurement"] == "test_measurement")
                |> filter(fn: (r) => r["host"] == "test_host")
                |> last()
            '''
            
            response = self.session.post(
                f"{self.url}/api/v2/query",
                params={'org': org},
                headers={**headers, 'Content-Type': 'application/vnd.flux'},
                data=query,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✓ Test read successful")
                return True
            else:
                print(f"✗ Test read failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Test write/read failed: {e}")
            return False


def main():
    """Main setup function."""
    print("🔧 InfluxDB Setup for Real-Time Anomaly Detection")
    print("=" * 50)
    
    # Configuration
    influxdb_url = "http://localhost:8086"
    username = "admin"
    password = "admin123" 
    org = "test_anamoly"
    bucket = "anomaly_detection"
    
    print(f"Target InfluxDB: {influxdb_url}")
    print(f"Organization: {org}")
    print(f"Bucket: {bucket}")
    print()
    
    # Initialize setup
    setup = InfluxDBSetup(influxdb_url)
    
    # Step 1: Health check
    print("1. Checking InfluxDB health...")
    if not setup.check_health():
        print("\n❌ InfluxDB is not running or not accessible.")
        print("Please ensure InfluxDB is installed and running:")
        print("  Docker: docker run -p 8086:8086 influxdb:2.7")
        print("  Native: sudo systemctl start influxdb")
        sys.exit(1)
    
    # Step 2: Initial setup
    print("\n2. Setting up initial user and organization...")
    token = setup.setup_initial_user(username, password, org, bucket)
    
    if token is None:
        print("Please provide an existing admin token:")
        token = input("Token: ").strip()
        if not token:
            print("❌ Token is required for setup")
            sys.exit(1)
    
    # Step 3: Create additional buckets if needed
    print("\n3. Creating additional buckets...")
    additional_buckets = [
        ("anomaly_detection_test", 7),  # Test bucket with 7-day retention
        ("system_metrics", 90),         # System metrics with 90-day retention
    ]
    
    for bucket_name, retention_days in additional_buckets:
        setup.create_bucket(token, org, bucket_name, retention_days)
    
    # Step 4: Test write/read
    print("\n4. Testing write and read operations...")
    if setup.test_write_read(token, org, bucket):
        print("✓ InfluxDB is ready for anomaly detection system")
    else:
        print("✗ Write/read test failed - please check configuration")
        sys.exit(1)
    
    # Step 5: Generate configuration
    print("\n5. Generating configuration...")
    
    config_template = f"""# InfluxDB Configuration for Anomaly Detection System
influxdb:
  enabled: true
  url: "{influxdb_url}"
  token: "{token}"
  org: "{org}"
  bucket: "{bucket}"
  batch_size: 100
  flush_interval: 1000
  timeout: 10000
  retry_attempts: 3
  retry_interval: 1000

# Update your config/influxdb_config.yaml with these values
"""
    
    # Save to file
    config_file = Path("../config/influxdb_config_generated.yaml")
    try:
        with open(config_file, 'w') as f:
            f.write(config_template)
        print(f"✓ Configuration saved to {config_file}")
    except Exception as e:
        print(f"⚠️  Could not save config file: {e}")
    
    print("\n" + config_template)
    
    print("=" * 50)
    print("✅ InfluxDB setup completed successfully!")
    print("\nNext steps:")
    print("1. Update your config/config.yaml with the token above")
    print("2. Test the connection: python -c \"from src.storage.influxdb_storage import *\"")
    print("3. Start monitoring: python src/main.py --mode monitor")
    print("4. Check Grafana dashboard at http://localhost:3000")


if __name__ == "__main__":
    main()