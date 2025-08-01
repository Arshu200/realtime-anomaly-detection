# System Architecture

## Overview

The Real-Time Anomaly Detection System is designed with a modular, scalable architecture that separates concerns and provides clear interfaces between components.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Real-Time Anomaly Detection System              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   Data Sources  │    │   Processing    │    │     Storage     │  │
│  │                 │    │                 │    │                 │  │
│  │  • Prometheus   │───▶│  • CPU Monitor  │───▶│  • InfluxDB     │  │
│  │  • Simulation   │    │  • Data Proc.   │    │  • File Storage │  │
│  │  • Manual Input │    │  • Validation   │    │  • Logs         │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                                  │                                  │
│  ┌─────────────────┐             ▼               ┌─────────────────┐  │
│  │  Visualization  │    ┌─────────────────┐     │ Anomaly Engine  │  │
│  │                 │◀───│ Anomaly Results │◀────│                 │  │
│  │  • Grafana      │    │                 │     │ • Prophet Model │  │
│  │  • Dashboards   │    │ • Timestamps    │     │ • Threshold     │  │
│  │  • Alerts       │    │ • Scores        │     │ • Confidence    │  │
│  └─────────────────┘    │ • Confidence    │     └─────────────────┘  │
│                         └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Collection Layer (`src/monitoring/`)

#### PrometheusClient (`metrics_collector.py`)
- **Purpose**: Collect CPU metrics from Prometheus server
- **Features**:
  - Multiple query support
  - Automatic fallback to simulation mode
  - Connection health monitoring
  - Historical data export
- **Interface**: REST API queries to Prometheus

#### CPUMonitor (`cpu_monitor.py`)
- **Purpose**: Specialized CPU monitoring with real-time processing
- **Features**:
  - Continuous monitoring loops
  - Statistical tracking
  - Integration with anomaly detection
  - Storage backend integration

### 2. Anomaly Detection Engine (`src/anomaly_detector/`)

#### CPUAnomalyDetector (`detector.py`)
- **Purpose**: Main anomaly detection orchestrator
- **Features**:
  - Prophet model integration
  - Model training and evaluation
  - Single-point and batch prediction
  - Performance metrics calculation

#### ProphetModel (`models.py`)
- **Purpose**: Facebook Prophet time-series forecasting
- **Features**:
  - Seasonal decomposition (daily, weekly)
  - Trend analysis and changepoint detection
  - Uncertainty quantification
  - Configurable parameters

#### Utility Functions (`utils.py`)
- **Purpose**: Common anomaly detection utilities
- **Features**:
  - Data validation and normalization
  - Confidence score calculation
  - Statistics computation
  - Data quality assessment

### 3. Storage Layer (`src/storage/`)

#### Base Storage Interface (`base_storage.py`)
- **Purpose**: Abstract interface for storage backends
- **Features**:
  - Standardized API for data persistence
  - Connection management
  - Health checking
  - Context manager support

#### InfluxDB Storage (`influxdb_storage.py`)
- **Purpose**: Time-series database integration
- **Features**:
  - Batch processing for performance
  - Automatic retry mechanisms
  - Connection pooling
  - Schema management

### 4. Configuration Management (`src/utils/`)

#### Configuration Loader (`config_loader.py`)
- **Purpose**: Centralized configuration management
- **Features**:
  - YAML configuration loading
  - Environment variable substitution
  - Configuration merging
  - Validation support

#### Logging System (`logger.py`)
- **Purpose**: Structured logging and monitoring
- **Features**:
  - Multiple log levels and formats
  - JSON structured logging
  - File rotation and management
  - Performance metrics logging

## Data Flow

### Training Mode Flow
1. **Data Loading**: Load historical CPU usage data from CSV
2. **Data Preprocessing**: Clean, validate, and resample data
3. **Model Training**: Train Prophet model with seasonal patterns
4. **Evaluation**: Test model with simulated anomalies
5. **Model Persistence**: Save trained model and metadata

### Monitoring Mode Flow
1. **Data Collection**: Continuous CPU metrics from Prometheus
2. **Real-time Processing**: Process each data point through anomaly detection
3. **Anomaly Detection**: Compare actual vs predicted values
4. **Result Storage**: Store results in InfluxDB
5. **Alerting**: Log anomalies and trigger notifications

### Evaluation Mode Flow
1. **Model Loading**: Load existing trained model
2. **Data Simulation**: Generate synthetic anomalies
3. **Performance Testing**: Calculate precision, recall, F1-score
4. **Visualization**: Generate performance plots and reports

## Configuration Architecture

### Hierarchical Configuration
```
config/
├── config.yaml           # Main application configuration
├── influxdb_config.yaml  # Database-specific settings
└── grafana_config.yaml   # Visualization configuration
```

### Environment Integration
- Environment variable substitution: `${VAR_NAME}`
- Default value support: `${VAR_NAME:default_value}`
- Configuration merging and inheritance

### Configuration Validation
- Type checking and validation
- Required parameter verification
- Range and format validation

## Scalability Considerations

### Horizontal Scaling
- **Multiple Instances**: Deploy multiple detection instances
- **Load Distribution**: Distribute monitoring across systems
- **Data Partitioning**: Use InfluxDB retention policies

### Vertical Scaling
- **Memory Optimization**: Configurable batch sizes
- **CPU Optimization**: Parallel processing support
- **Storage Optimization**: Compression and downsampling

### Performance Tuning
- **High-Frequency Mode**: Optimized for ≤10s intervals
- **Resource-Constrained Mode**: Reduced memory footprint
- **High-Accuracy Mode**: Enhanced prediction parameters

## Security Architecture

### Data Protection
- **Secure Connections**: TLS/SSL support for InfluxDB
- **Token Management**: Secure token storage and rotation
- **Data Encryption**: Optional encryption for sensitive data

### Access Control
- **Authentication**: InfluxDB and Grafana authentication
- **Authorization**: Role-based access control
- **Audit Logging**: Security event logging

## Error Handling and Resilience

### Fault Tolerance
- **Connection Failures**: Automatic retry with exponential backoff
- **Data Validation**: Input validation and sanitization
- **Graceful Degradation**: Fallback to simulation mode

### Monitoring and Observability
- **Health Checks**: Component health monitoring
- **Performance Metrics**: System performance tracking
- **Error Tracking**: Comprehensive error logging

## Extension Points

### Adding New Data Sources
1. Implement metrics collector interface
2. Add configuration support
3. Integrate with monitoring pipeline

### Adding New Storage Backends
1. Implement `StorageInterface` from `base_storage.py`
2. Add configuration schema
3. Register in storage factory

### Adding New Detection Algorithms
1. Extend `AnomalyModel` base class
2. Implement required methods
3. Add to model factory

### Adding New Visualizations
1. Create Grafana dashboard JSON
2. Add query templates
3. Configure alert rules

## Dependencies and Technologies

### Core Technologies
- **Python 3.8+**: Main programming language
- **Facebook Prophet**: Time-series forecasting
- **InfluxDB 2.x**: Time-series database
- **Grafana 8.0+**: Visualization and alerting
- **Prometheus**: Metrics collection

### Python Libraries
- **pandas/numpy**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **matplotlib/plotly**: Visualization
- **loguru**: Advanced logging
- **pyyaml**: Configuration management