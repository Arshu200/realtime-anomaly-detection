# InfluxDB Integration and Grafana Visualization Setup Guide

This guide provides step-by-step instructions for setting up InfluxDB storage and Grafana visualization for the real-time CPU anomaly detection system.

## Prerequisites

- Docker and Docker Compose (recommended for InfluxDB and Grafana)
- Python 3.8+ with the enhanced anomaly detection system
- Network access to InfluxDB and Grafana instances

## 1. InfluxDB Setup

### Option A: Using Docker (Recommended)

1. **Create InfluxDB Docker container:**

```bash
docker run -d \
  --name influxdb \
  -p 8086:8086 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=admin123 \
  -e DOCKER_INFLUXDB_INIT_ORG=test_anamoly \
  -e DOCKER_INFLUXDB_INIT_BUCKET=anomaly_detection \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=PCIUyihftjXbIAQjR8g-xgEas7H5ssaEQG6ppz-V_w-bxnNeC3FM941sgC8Sclmjhs2VmD7PyGV1oOepRw8xrA== \
  influxdb:2.7
```

2. **Verify InfluxDB is running:**

```bash
curl -i http://localhost:8086/health
```

You should see: `HTTP/1.1 200 OK` and `{"name":"influxdb","message":"ready for queries and writes","status":"pass"}`

### Option B: Manual Installation

1. **Download and install InfluxDB 2.x:**
   - Visit [InfluxDB Downloads](https://www.influxdata.com/downloads/)
   - Follow installation instructions for your OS

2. **Initialize InfluxDB:**

```bash
influx setup \
  --username admin \
  --password admin123 \
  --org test_anamoly \
  --bucket anomaly_detection \
  --token PCIUyihftjXbIAQjR8g-xgEas7H5ssaEQG6ppz-V_w-bxnNeC3FM941sgC8Sclmjhs2VmD7PyGV1oOepRw8xrA== \
  --force
```

### InfluxDB Configuration Verification

1. **Access InfluxDB UI:** http://localhost:8086
2. **Login with credentials:**
   - Username: `admin`
   - Password: `admin123`
3. **Verify organization and bucket exist:**
   - Organization: `test_anamoly`
   - Bucket: `anomaly_detection`

## 2. Grafana Setup

### Option A: Using Docker (Recommended)

1. **Create Grafana Docker container:**

```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana:latest
```

2. **Access Grafana:** http://localhost:3000
   - Username: `admin`
   - Password: `admin`

### Option B: Manual Installation

1. **Download and install Grafana:**
   - Visit [Grafana Downloads](https://grafana.com/grafana/download)
   - Follow installation instructions for your OS

2. **Start Grafana service:**

```bash
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

## 3. Grafana Data Source Configuration

1. **Add InfluxDB Data Source:**
   - Go to Configuration → Data Sources
   - Click "Add data source"
   - Select "InfluxDB"

2. **Configure InfluxDB connection:**
   ```
   Name: InfluxDB
   URL: http://localhost:8086
   Query Language: Flux
   Organization: test_anamoly
   Token: PCIUyihftjXbIAQjR8g-xgEas7H5ssaEQG6ppz-V_w-bxnNeC3FM941sgC8Sclmjhs2VmD7PyGV1oOepRw8xrA==
   Default Bucket: anomaly_detection
   ```

3. **Test connection:**
   - Click "Save & Test"
   - Should show "Connection successful"

## 4. Import Grafana Dashboard

1. **Import the provided dashboard:**
   - Go to Create → Import
   - Upload `grafana_dashboard.json` file
   - Or copy-paste the JSON content
   - Select the InfluxDB data source
   - Click "Import"

2. **Dashboard Features:**
   - **CPU Usage Comparison**: Actual vs Forecasted values
   - **Anomaly Detection Points**: Visual markers for detected anomalies
   - **Current Status**: Real-time anomaly status indicator
   - **Confidence Score**: Detection confidence over time
   - **Anomaly Count**: Number of anomalies in the last hour
   - **Anomaly Rate**: Percentage of anomalies over 24 hours
   - **Anomaly Timeline**: State timeline showing anomaly periods

## 5. Anomaly Detection System Configuration

1. **Update configuration file (`config.yaml`):**

```yaml
# InfluxDB Configuration
influxdb:
  enabled: true
  url: "http://localhost:8086"
  token: "PCIUyihftjXbIAQjR8g-xgEas7H5ssaEQG6ppz-V_w-bxnNeC3FM941sgC8Sclmjhs2VmD7PyGV1oOepRw8xrA=="
  org: "test_anamoly"
  bucket: "anomaly_detection"
  batch_size: 100
  flush_interval: 1000  # milliseconds
  timeout: 10000  # milliseconds
  retry_attempts: 3
  retry_interval: 1000  # milliseconds
```

2. **Install required dependencies:**

```bash
pip install -r requirements.txt
```

3. **Test InfluxDB connection:**

```python
python -c "
from influxdb_storage import InfluxDBConfig, InfluxDBAnomalyStorage
from datetime import datetime, timezone

config = InfluxDBConfig(
    url='http://localhost:8086',
    token='PCIUyihftjXbIAQjR8g-xgEas7H5ssaEQG6ppz-V_w-bxnNeC3FM941sgC8Sclmjhs2VmD7PyGV1oOepRw8xrA==',
    org='test_anamoly',
    bucket='anomaly_detection'
)

with InfluxDBAnomalyStorage(config) as storage:
    result = storage.store_anomaly_data(
        timestamp=datetime.now(timezone.utc),
        actual_cpu=75.0,
        forecasted_cpu=70.0,
        is_anomaly=False,
        confidence_score=0.85
    )
    print(f'Test storage result: {result}')
"
```

## 6. Running the Enhanced System

1. **Train the model (if needed):**

```bash
python main.py --mode train --dataset dataset.csv
```

2. **Start real-time monitoring with InfluxDB storage:**

```bash
python main.py --mode monitor --model-path cpu_anomaly_model.pkl
```

3. **Monitor logs for InfluxDB integration:**

```bash
tail -f logs/application.log | grep -i influx
```

## 7. Data Schema Validation

The system stores data in two measurements:

### CPU Metrics
```
Measurement: cpu_metrics
Tags: 
  - metric_type: "actual" | "forecasted"
  - host: hostname
Fields:
  - cpu_usage: float (CPU percentage)
Timestamp: UTC
```

### Anomaly Detection
```
Measurement: anomaly_detection
Tags:
  - host: hostname
Fields:
  - is_anomaly: boolean
  - anomaly_score: float (1.0 for anomaly, 0.0 for normal)
  - confidence: float (0.0 to 1.0)
Timestamp: UTC
```

## 8. Grafana Alerting Setup

1. **Configure Notification Channels:**
   - Go to Alerting → Notification channels
   - Add email, Slack, or webhook notifications

2. **Set up Alert Rules:**

   **High Anomaly Rate Alert:**
   ```flux
   from(bucket: "anomaly_detection")
     |> range(start: -1h)
     |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
     |> filter(fn: (r) => r["_field"] == "is_anomaly")
     |> filter(fn: (r) => r["_value"] == true)
     |> count()
   ```
   - Threshold: > 10 anomalies in 1 hour

   **No Data Alert:**
   ```flux
   from(bucket: "anomaly_detection")
     |> range(start: -5m)
     |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
     |> count()
   ```
   - Threshold: = 0 (no data received)

## 9. Performance Optimization

1. **InfluxDB Retention Policies:**

```bash
# Set retention policy for automatic data cleanup
influx bucket update \
  --name anomaly_detection \
  --retention 30d \
  --org test_anamoly
```

2. **Grafana Performance:**
   - Use appropriate time ranges
   - Configure auto-refresh intervals (30s-5m)
   - Use downsampling for long time ranges

3. **System Monitoring:**
   - Monitor InfluxDB disk usage
   - Check Grafana query performance
   - Monitor anomaly detection latency

## 10. Troubleshooting

### Common Issues

1. **InfluxDB Connection Failed:**
   - Check if InfluxDB is running: `docker ps` or `systemctl status influxdb`
   - Verify network connectivity: `telnet localhost 8086`
   - Check token and organization configuration

2. **Grafana No Data:**
   - Verify data source configuration
   - Check bucket name and organization
   - Test Flux queries in InfluxDB UI first

3. **Performance Issues:**
   - Check InfluxDB batch settings in config
   - Monitor system resources (CPU, memory, disk)
   - Verify network latency between components

### Useful Commands

```bash
# Check InfluxDB health
curl http://localhost:8086/health

# Check InfluxDB buckets
influx bucket list --org test_anamoly

# Test Grafana API
curl -u admin:admin http://localhost:3000/api/health

# View system logs
tail -f logs/application.log

# Monitor Docker containers
docker logs influxdb
docker logs grafana
```

## 11. Production Considerations

1. **Security:**
   - Use strong passwords and tokens
   - Configure HTTPS/TLS
   - Implement proper firewall rules
   - Regular security updates

2. **High Availability:**
   - Set up InfluxDB clustering
   - Configure Grafana load balancing
   - Implement backup strategies

3. **Monitoring:**
   - Monitor InfluxDB performance metrics
   - Set up Grafana health checks
   - Configure comprehensive alerting

4. **Backup and Recovery:**
   - Regular InfluxDB backups
   - Grafana dashboard exports
   - Configuration backups

This setup provides a complete monitoring solution for real-time CPU anomaly detection with persistent storage and comprehensive visualization capabilities.