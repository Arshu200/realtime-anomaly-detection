// CPU Usage Queries for Grafana Dashboards
// Use these Flux queries in your Grafana panels

// 1. CPU Usage Comparison - Actual vs Forecasted
// Shows both actual and predicted CPU usage over time
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual" or r["metric_type"] == "forecasted")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "cpu_usage_comparison")

// 2. Current CPU Usage - Latest Value
// Shows the most recent CPU usage value
from(bucket: "anomaly_detection")
  |> range(start: -5m)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual")
  |> last()
  |> yield(name: "current_cpu")

// 3. CPU Usage Statistics - Min, Max, Average
// Calculate statistical metrics for CPU usage
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> duplicate(column: "_value", as: "mean_value")
  |> group()
  |> reduce(
      fn: (r, accumulator) => ({
          min: if r._value < accumulator.min then r._value else accumulator.min,
          max: if r._value > accumulator.max then r._value else accumulator.max,
          mean: (accumulator.mean * accumulator.count + r._value) / (accumulator.count + 1.0),
          count: accumulator.count + 1.0
      }),
      identity: {min: 100.0, max: 0.0, mean: 0.0, count: 0.0}
  )
  |> yield(name: "cpu_stats")

// 4. CPU Usage Percentiles
// Calculate percentile values for CPU usage distribution
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> quantile(q: 0.95, method: "estimate_tdigest")
  |> yield(name: "cpu_p95")

// 5. CPU Usage Rate of Change
// Calculate the rate of change in CPU usage
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> derivative(unit: 1m, nonNegative: false)
  |> yield(name: "cpu_rate_of_change")

// 6. CPU Usage Moving Average
// Calculate moving average for trend analysis
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> timedMovingAverage(every: 10m, period: 30m)
  |> yield(name: "cpu_moving_average")

// 7. CPU Usage Hourly Average
// Group CPU usage by hour for daily pattern analysis
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> yield(name: "cpu_hourly")

// 8. CPU Usage Distribution Histogram
// Create histogram data for CPU usage distribution
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual")
  |> histogram(buckets: [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
  |> yield(name: "cpu_histogram")

// 9. CPU Usage by Host (if multiple hosts)
// Group CPU usage by host for multi-system monitoring
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual")
  |> group(columns: ["host"])
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "cpu_by_host")

// 10. CPU Usage Peak Detection
// Identify peak CPU usage periods
from(bucket: "anomaly_detection")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "cpu_metrics")
  |> filter(fn: (r) => r["_field"] == "cpu_usage")
  |> filter(fn: (r) => r["metric_type"] == "actual")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> filter(fn: (r) => r._value > 80.0)  // Adjust threshold as needed
  |> yield(name: "cpu_peaks")