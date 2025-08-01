# Installation Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM
- **Storage**: 2GB available disk space
- **Network**: Access to InfluxDB and Grafana instances

### Recommended Requirements
- **CPU**: 4+ cores, 2.5+ GHz
- **Memory**: 8GB RAM (16GB for large datasets)
- **Storage**: 10GB SSD
- **Network**: 1 Gbps for high-frequency monitoring

### Production Requirements
- **CPU**: 8+ cores, 3.0+ GHz
- **Memory**: 16GB+
- **Storage**: 50GB+ SSD with RAID
- **Network**: Dedicated network for monitoring stack

## Quick Installation

### 1. Clone Repository
```bash
git clone https://github.com/Arshu200/realtime-anomaly-detection
cd realtime-anomaly-detection
```

### 2. Install Python Dependencies
```bash
# Using pip
pip install -r requirements.txt

# Using conda (recommended)
conda create -n anomaly-detection python=3.8
conda activate anomaly-detection
pip install -r requirements.txt
```

### 3. Install Package (Optional)
```bash
# Development installation
pip install -e .

# Production installation
pip install .
```

## Infrastructure Setup

### InfluxDB Installation

#### Option A: Docker (Recommended)
```bash
docker run -d \
  --name influxdb \
  -p 8086:8086 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=admin123 \
  -e DOCKER_INFLUXDB_INIT_ORG=test_anamoly \
  -e DOCKER_INFLUXDB_INIT_BUCKET=anomaly_detection \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=your-secure-token-here \
  influxdb:2.7
```

#### Option B: Native Installation

**Ubuntu/Debian:**
```bash
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list
sudo apt-get update && sudo apt-get install influxdb2
sudo systemctl enable influxdb
sudo systemctl start influxdb
```

**CentOS/RHEL:**
```bash
cat <<EOF | sudo tee /etc/yum.repos.d/influxdata.repo
[influxdata]
name = InfluxData Repository - Stable
baseurl = https://repos.influxdata.com/rhel/\$releasever/\$basearch/stable
enabled = 1
gpgcheck = 1
gpgkey = https://repos.influxdata.com/influxdata-archive_compat.key
EOF
sudo yum install influxdb2
sudo systemctl enable influxdb
sudo systemctl start influxdb
```

**macOS:**
```bash
brew install influxdb
brew services start influxdb
```

#### Initialize InfluxDB
```bash
influx setup \
  --username admin \
  --password admin123 \
  --org test_anamoly \
  --bucket anomaly_detection \
  --token your-secure-token-here \
  --force
```

### Grafana Installation

#### Option A: Docker (Recommended)
```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  -v grafana-storage:/var/lib/grafana \
  grafana/grafana:latest
```

#### Option B: Native Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install -y apt-transport-https
sudo apt-get install -y software-properties-common wget
sudo wget -q -O /usr/share/keyrings/grafana.key https://apt.grafana.com/gpg.key
echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://apt.grafana.com stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

**CentOS/RHEL:**
```bash
cat <<EOF | sudo tee /etc/yum.repos.d/grafana.repo
[grafana]
name=grafana
baseurl=https://rpm.grafana.com
repo_gpgcheck=1
enabled=1
gpgcheck=1
gpgkey=https://rpm.grafana.com/gpg.key
sslverify=1
sslcacert=/etc/pki/tls/certs/ca-bundle.crt
EOF
sudo yum install grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

**macOS:**
```bash
brew install grafana
brew services start grafana
```

## Configuration Setup

### 1. Update Configuration Files

#### Main Configuration (`config/config.yaml`)
```yaml
# Update with your specific settings
influxdb:
  enabled: true
  url: "http://localhost:8086"
  token: "your-actual-token-here"
  org: "test_anamoly"
  bucket: "anomaly_detection"

prometheus_url: "http://localhost:9090"  # Update if different

monitoring_interval: 30
log_level: "INFO"
```

#### InfluxDB Configuration (`config/influxdb_config.yaml`)
```yaml
influxdb:
  url: "http://localhost:8086"
  token: "your-actual-token-here"
  org: "test_anamoly"
  bucket: "anomaly_detection"
  # Adjust performance settings as needed
  batch_size: 100
  flush_interval: 1000
```

### 2. Environment Variables (Optional)
```bash
export INFLUXDB_URL="http://localhost:8086"
export INFLUXDB_TOKEN="your-actual-token-here"
export INFLUXDB_ORG="test_anamoly"
export PROMETHEUS_URL="http://localhost:9090"
```

### 3. Create Directories
```bash
mkdir -p logs
chmod 755 logs
```

## Verification

### 1. Test Python Environment
```bash
python -c "
import pandas as pd
import numpy as np
import prophet
import influxdb_client
print('✓ All required packages installed successfully')
"
```

### 2. Test InfluxDB Connection
```bash
curl -i http://localhost:8086/health
# Expected: HTTP/1.1 200 OK
```

### 3. Test Grafana Access
```bash
curl -i http://localhost:3000/api/health
# Expected: HTTP/1.1 200 OK
```

### 4. Test Application
```bash
cd src
python main.py --help
# Should display help information
```

## Grafana Setup

### 1. Access Grafana
- URL: http://localhost:3000
- Username: admin
- Password: admin (or what you set in installation)

### 2. Add InfluxDB Data Source
1. Navigate to Configuration → Data Sources
2. Click "Add data source"
3. Select "InfluxDB"
4. Configure:
   - **Name**: InfluxDB
   - **URL**: http://localhost:8086
   - **Query Language**: Flux
   - **Organization**: test_anamoly
   - **Token**: your-secure-token-here
   - **Default Bucket**: anomaly_detection
5. Click "Save & Test"

### 3. Import Dashboard
1. Navigate to Create → Import
2. Upload `grafana/dashboards/anomaly_detection_dashboard.json`
3. Select the InfluxDB data source
4. Click "Import"

## Performance Optimization

### For High-Frequency Monitoring (≤10s intervals)
```yaml
# config/config.yaml
monitoring_interval: 10
influxdb:
  batch_size: 50
  flush_interval: 500
log_level: "WARNING"
```

### For Resource-Constrained Environments
```yaml
# config/config.yaml
monitoring_interval: 60
influxdb:
  batch_size: 200
  flush_interval: 5000
prophet_uncertainty_samples: 100
```

### For High-Accuracy Detection
```yaml
# config/config.yaml
threshold_multiplier: 1.5
interval_width: 0.99
prophet_uncertainty_samples: 2000
```

## Troubleshooting Installation

### Common Issues

#### Python Package Conflicts
```bash
# Create clean virtual environment
python -m venv venv_clean
source venv_clean/bin/activate  # On Windows: venv_clean\Scripts\activate
pip install -r requirements.txt
```

#### InfluxDB Connection Issues
```bash
# Check if InfluxDB is running
sudo systemctl status influxdb  # Linux
brew services list | grep influx  # macOS

# Check network connectivity
telnet localhost 8086

# Verify token and organization
influx auth list --org test_anamoly
```

#### Prophet Installation Issues
```bash
# Install Prophet dependencies
# Ubuntu/Debian:
sudo apt-get install python3-dev build-essential

# CentOS/RHEL:
sudo yum install python3-devel gcc gcc-c++

# macOS:
brew install gcc

# Then reinstall Prophet
pip uninstall prophet
pip install prophet
```

#### Permission Issues
```bash
# Fix log directory permissions
sudo chown -R $USER:$USER logs/
chmod 755 logs/

# Fix Python package permissions
pip install --user -r requirements.txt
```

### Getting Additional Help

1. **Check system status**: Run `scripts/validate_system.py`
2. **Enable debug logging**: Set `log_level: "DEBUG"` in config
3. **Check log files**: Review `logs/application.log`
4. **Test components individually**: Use Python REPL to test imports

## Next Steps

After successful installation:

1. **Train a model**: See [API Reference](api_reference.md) for training
2. **Start monitoring**: Begin real-time CPU anomaly detection
3. **Configure alerts**: Set up Grafana alerting rules
4. **Monitor performance**: Use system monitoring and logs

## Production Deployment Considerations

### Security
- Change default passwords
- Use secure tokens for InfluxDB
- Configure TLS/SSL for production
- Set up proper firewall rules

### Monitoring
- Set up log rotation
- Configure disk space monitoring
- Set up system health checks
- Plan for backup and recovery

### Scaling
- Consider container orchestration (Kubernetes)
- Plan for high availability setup
- Configure load balancing if needed
- Set up monitoring for multiple systems