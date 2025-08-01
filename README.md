# Real-Time CPU Usage Anomaly Detection System

A comprehensive Python-based real-time anomaly detection pipeline for CPU usage monitoring using time-series analysis with Facebook Prophet and Prometheus integration.

## Features

- **Data Preprocessing & EDA**: Automated data loading, resampling, statistical analysis, and visualization
- **Time Series Anomaly Detection**: Facebook Prophet model for capturing temporal patterns and seasonality
- **Model Evaluation**: Comprehensive evaluation with simulated anomalies, ROC curves, and performance metrics
- **Real-Time Monitoring**: Prometheus integration for live CPU metrics polling
- **Structured Logging**: Advanced logging with loguru for anomalies, predictions, and system events
- **Visualization**: Interactive plots and comprehensive evaluation charts
- **Model Persistence**: Save and load trained models for production use

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │────│  Data Ingestion │────│   Prophet Model │
│    Metrics      │    │    & Polling    │    │  (Anomaly Det.) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Historical    │────│  Data Preprocessing│────│   Model Training│
│    Dataset      │    │     & EDA       │    │   & Evaluation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Structured    │────│  Real-time      │────│   Anomaly       │
│    Logging      │    │  Monitoring     │    │   Detection     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd realtime-anomaly-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python main.py --help
   ```

## Quick Start

### 1. Train the Model

Train an anomaly detection model using the provided dataset:

```bash
python main.py --mode train --dataset dataset.csv
```

This will:
- Load and preprocess the dataset
- Train a Facebook Prophet model
- Evaluate performance with simulated anomalies
- Generate visualizations and evaluation reports
- Save the trained model

### 2. Real-Time Monitoring

Start real-time monitoring (requires a trained model):

```bash
python main.py --mode monitor --monitoring-interval 30
```

This will:
- Connect to Prometheus (or use simulation mode if unavailable)
- Load the trained model
- Continuously monitor CPU usage
- Detect and log anomalies in real-time

### 3. Model Evaluation

Evaluate a trained model with detailed analysis:

```bash
python main.py --mode evaluate
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `train` | Operation mode: `train`, `monitor`, or `evaluate` |
| `--dataset` | `dataset.csv` | Path to training dataset |
| `--model-path` | `cpu_anomaly_model.pkl` | Path to save/load model |
| `--prometheus-url` | `http://localhost:9090` | Prometheus server URL |
| `--monitoring-interval` | `30` | Monitoring interval in seconds |
| `--log-level` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `--threshold-multiplier` | `2.0` | Anomaly threshold multiplier |
| `--no-plots` | `False` | Disable plot generation |
| `--no-eda` | `False` | Skip EDA during training |

## Dataset Format

The system expects a CSV file with the following format:

```csv
timestamp,cpu_usage
2025-06-01 00:00:00,0.534
2025-06-01 00:40:00,0.592
2025-06-01 01:20:00,0.629
...
```

- **timestamp**: ISO format timestamp
- **cpu_usage**: CPU usage as a decimal (0.0 to 1.0) or percentage (0-100)

## Model Performance

The Prophet-based model achieves:
- **Precision**: ~46.7% (configurable via threshold tuning)
- **Recall**: ~65.0% (good at catching real anomalies)
- **F1-Score**: ~54.3% (balanced performance)
- **MAPE**: ~3.0% (low prediction error on normal data)
- **Anomaly Rate**: ~6.4% (reasonable anomaly detection rate)

## Output Files

### Training Mode
- `cpu_anomaly_model.pkl` - Trained Prophet model
- `model_metadata.json` - Model configuration and performance metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve analysis
- `detection_overview.png` - Comprehensive detection visualization
- `threshold_analysis.png` - Threshold optimization analysis
- `evaluation_report.txt` - Detailed evaluation report

### Monitoring Mode
- `logs/metrics.jsonl` - Structured metrics log (JSONL format)
- `logs/anomalies.jsonl` - Dedicated anomaly log
- `logs/anomaly_detection_*.log` - General system logs
- `logs/system_report.json` - System status and statistics

## Prometheus Integration

### Supported Metrics

The system automatically tries multiple Prometheus queries:
- `node_cpu_seconds_total{mode="idle"}` (Node Exporter)
- `cpu_usage_idle` (Custom metrics)
- `cpu_usage_percent` (Direct percentage)
- `system_cpu_usage` (System metrics)
- `container_cpu_usage_seconds_total` (Container metrics)

### Simulation Mode

When Prometheus is unavailable, the system automatically switches to simulation mode, generating realistic CPU usage patterns for testing and development.

## Logging

### Structured Logging Features
- **CPU Usage Logging**: All measurements with timestamps and sources
- **Prediction Logging**: Model predictions and intervals
- **Anomaly Logging**: Detailed anomaly events with severity levels
- **System Status**: Component health and error tracking

### Log Formats
- **Console**: Colored, human-readable format
- **Files**: JSON Lines (JSONL) for structured analysis
- **Rotation**: Automatic log rotation and compression

### Anomaly Severity Levels
- **CRITICAL**: Score ≥ 3.0
- **HIGH**: Score ≥ 2.0
- **MEDIUM**: Score ≥ 1.5
- **LOW**: Score < 1.5

## Advanced Usage

### Custom Threshold Tuning

```bash
python main.py --mode train --threshold-multiplier 1.5
```

Lower values increase sensitivity (more anomalies detected).

### High-Frequency Monitoring

```bash
python main.py --mode monitor --monitoring-interval 10
```

### Debug Mode

```bash
python main.py --mode monitor --log-level DEBUG
```

## API Integration

The system provides a modular architecture for integration:

```python
from anomaly_model import CPUAnomalyDetector
from prometheus_client import PrometheusClient

# Load trained model
detector = CPUAnomalyDetector()
detector.load_model('cpu_anomaly_model.pkl')

# Check single point
result = detector.predict_single_point(timestamp, cpu_value)
if result['is_anomaly']:
    print(f"Anomaly detected: {result['anomaly_score']}")
```

## Troubleshooting

### Common Issues

1. **Prometheus Connection Failed**
   - System automatically switches to simulation mode
   - Verify Prometheus URL and accessibility

2. **Model Loading Error**
   - Ensure model file exists: `cpu_anomaly_model.pkl`
   - Train a new model if needed

3. **Memory Issues**
   - Reduce dataset size or increase system memory
   - Use data sampling for large datasets

### Performance Optimization

- **CPU Usage**: Use longer monitoring intervals (60+ seconds)
- **Memory**: Enable log rotation and cleanup old files
- **Accuracy**: Tune threshold multiplier based on evaluation results

## Development

### Project Structure

```
├── main.py                 # Main application entry point
├── data_preprocessing.py   # Data loading and EDA
├── anomaly_model.py       # Prophet-based anomaly detection
├── model_evaluation.py    # Model evaluation and metrics
├── prometheus_client.py   # Prometheus integration
├── logging_config.py      # Logging configuration
├── requirements.txt       # Python dependencies
├── dataset.csv           # Sample dataset
└── logs/                 # Log files and reports
```

### Adding Custom Metrics

Extend the `PrometheusClient` class to support additional metrics:

```python
def get_custom_metric(self):
    query = "your_custom_metric"
    result = self.query_instant(query)
    # Process result...
    return processed_data
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Specify your license here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review log files in the `logs/` directory
- Create an issue in the repository

---

**Note**: This system is designed for production monitoring environments. Ensure proper security configurations when deploying with real Prometheus instances.