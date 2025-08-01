// Anomaly Detection Queries for Grafana Dashboards
// Use these Flux queries for anomaly visualization and analysis

// 1. Anomaly Detection Points
// Show all detected anomalies as points on timeline
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> yield(name: "anomaly_points")

// 2. Anomaly Count - Real-time
// Count of anomalies detected in real-time
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> aggregateWindow(every: v.windowPeriod, fn: count, createEmpty: false)
  |> yield(name: "anomaly_count_timeseries")

// 3. Total Anomaly Count
// Total number of anomalies in time range
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> count()
  |> yield(name: "total_anomalies")

// 4. Anomaly Rate Percentage
// Calculate anomaly rate as percentage of total data points
anomalies = from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> count()
  |> set(key: "type", value: "anomalies")

total = from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> count()
  |> set(key: "type", value: "total")

union(tables: [anomalies, total])
  |> pivot(rowKey: ["_time"], columnKey: ["type"], valueColumn: "_value")
  |> map(fn: (r) => ({
      _time: r._time,
      _value: if exists r.total and r.total > 0 then float(v: r.anomalies) / float(v: r.total) * 100.0 else 0.0
  }))
  |> yield(name: "anomaly_rate_percentage")

// 5. Current Anomaly Status
// Shows current anomaly detection status (latest reading)
from(bucket: "anomaly_detection")
  |> range(start: -5m)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> last()
  |> map(fn: (r) => ({
      _time: r._time,
      _value: if r._value == true then 1.0 else 0.0,
      status: if r._value == true then "ANOMALY" else "NORMAL"
  }))
  |> yield(name: "current_status")

// 6. Anomaly Confidence Scores
// Show confidence scores for all predictions
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "confidence")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "confidence_scores")

// 7. Latest Confidence Score
// Show the most recent confidence score
from(bucket: "anomaly_detection")
  |> range(start: -5m)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "confidence")
  |> last()
  |> yield(name: "latest_confidence")

// 8. Anomaly Severity Distribution
// Group anomalies by severity level (if available)
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> group(columns: ["severity"])
  |> count()
  |> yield(name: "anomaly_by_severity")

// 9. Anomaly Duration Analysis
// Calculate time between anomalies
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> elapsed(unit: 1m)
  |> yield(name: "anomaly_intervals")

// 10. Anomaly Score Distribution
// Show distribution of anomaly scores
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "anomaly_score")
  |> filter(fn: (r) => r["_value"] > 0.0)
  |> histogram(buckets: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
  |> yield(name: "anomaly_score_distribution")

// 11. Anomaly Frequency by Hour
// Show when anomalies occur most frequently (daily patterns)
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> map(fn: (r) => ({
      _time: r._time,
      _value: r._value,
      hour: date.hour(t: r._time)
  }))
  |> group(columns: ["hour"])
  |> count()
  |> yield(name: "anomaly_by_hour")

// 12. Recent Anomalies List
// List of recent anomalies with details
from(bucket: "anomaly_detection")
  |> range(start: -24h)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 50)
  |> yield(name: "recent_anomalies")

// 13. Anomaly Detection Performance
// Show prediction accuracy metrics over time
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "confidence")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> yield(name: "detection_performance")

// 14. Sustained Anomalies
// Detect periods of sustained anomalous behavior
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> map(fn: (r) => ({
      _time: r._time,
      _value: if r._value == true then 1.0 else 0.0
  }))
  |> timedMovingAverage(every: 1m, period: 10m)
  |> filter(fn: (r) => r._value > 0.5)  // More than 50% anomalies in 10min window
  |> yield(name: "sustained_anomalies")

// 15. Anomaly Alert Status
// Generate alert levels based on recent anomaly activity
from(bucket: "anomaly_detection")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
  |> filter(fn: (r) => r["_field"] == "is_anomaly")
  |> filter(fn: (r) => r["_value"] == true)
  |> count()
  |> map(fn: (r) => ({
      _time: now(),
      _value: r._value,
      alert_level: if r._value > 10 then "CRITICAL" 
                  else if r._value > 5 then "WARNING"
                  else if r._value > 0 then "INFO"
                  else "OK"
  }))
  |> yield(name: "alert_status")