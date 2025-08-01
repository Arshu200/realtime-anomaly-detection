# Grafana Dashboard Configuration and Flux Queries

## InfluxDB Data Source Configuration

Before importing the dashboard, configure InfluxDB as a data source in Grafana:

1. **URL**: `http://localhost:8086`
2. **Organization**: `test_anamoly`
3. **Token**: `PCIUyihftjXbIAQjR8g-xgEas7H5ssaEQG6ppz-V_w-bxnNeC3FM941sgC8Sclmjhs2VmD7PyGV1oOepRw8xrA==`
4. **Default Bucket**: `anomaly_detection`

## Flux Queries for Dashboard Panels

### 1. Real-Time CPU Usage (Actual vs Forecasted)

This query retrieves both actual and forecasted CPU usage values for comparison:

```flux
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual" or r["metric_type"] == "forecasted")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "cpu_usage")
```

### 2. Anomaly Detection Points

This query identifies and displays anomaly points:

```flux
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "anomaly_score")
  |> filter(fn: (r) => r["_value"] == 1.0)
  |> yield(name: "anomalies")
```

### 3. Anomaly Count (Last 1 Hour)

This query counts anomalies in the last hour:

```flux
from(bucket: "anomaly_detection")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> count()
  |> yield(name: "anomaly_count")
```

### 4. Latest Anomaly Status

This query shows the most recent anomaly status:

```flux
from(bucket: "anomaly_detection")
  |> range(start: -5m)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> last()
  |> yield(name: "latest_status")
```

### 5. Confidence Score Over Time

This query displays confidence scores:

```flux
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "confidence")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "confidence")
```

### 6. Anomaly Rate (Last 24 Hours)

This query calculates the anomaly rate over the last 24 hours:

```flux
anomalies = from(bucket: "anomaly_detection")
  |> range(start: -24h)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> count()

total = from(bucket: "anomaly_detection")
  |> range(start: -24h)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> count()

join(
  tables: {anomalies: anomalies, total: total},
  on: ["host"]
)
|> map(fn: (r) => ({
  _time: now(),
  _value: float(v: r._value_anomalies) / float(v: r._value_total) * 100.0
}))
|> yield(name: "anomaly_rate")
```

## Panel Configuration Tips

1. **Time Series Panel**: Use for CPU usage comparison and confidence scores
2. **Stat Panel**: Use for anomaly count and latest status
3. **Gauge Panel**: Use for anomaly rate percentage
4. **State Timeline**: Use for anomaly status over time
5. **Table Panel**: Use for detailed anomaly information

## Alerting Configuration

Set up alerts for:
- High anomaly rate (>10% in last hour)
- Sustained anomalies (>5 consecutive anomalies)
- No data received (missing metrics for >5 minutes)

## Dashboard Variables

Configure these variables for dynamic filtering:
- `$host`: Host selection
- `$time_range`: Time range selection
- `$refresh_rate`: Auto-refresh interval