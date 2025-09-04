# Smart Traffic Analyzer – Real-Time Computer Vision for Urban Congestion Monitoring

## Features

- Real-time vehicle detection & tracking
- Congestion scoring
- REST API (traffic data, alerts, system health)
- Monitoring dashboards (Grafana)
- CI/CD tests on video pipeline

## API Reference

This API provides real-time traffic insights, congestion monitoring, system health, and model management.  
All responses are in JSON format. 

### Traffic Data API

#### `GET /traffic/status`  
Returns congestion snapshot for a specific road.

**Query Parameters**  
- `road_id` (string, required) – Identifier of the monitored road.  
- `time_range` (string, optional) – ISO8601 start & end time.  

**Response Example**  
```json
{
  "road_id": "LAG-EXP-001",
  "timestamp": "2025-09-04T06:00:00Z",
  "avg_speed_kph": 23.5,
  "vehicle_count": 124,
  "congestion_score": 0.78
}
```

#### `GET /traffic/roads`
List all monitored roads.

#### `POST /traffic/roads`
Register a new monitored road/stream.

**Response Example**  
```json
{
  "road_id": "LAG-EXP-002",
  "stream_url": "rtsp://camera-url",
  "location": "Ikeja"
}
```

#### `PUT /traffic/roads/{road_id}`: 
Update metadata of an existing road (rename, update stream URL).

#### `DELETE /traffic/roads/{road_id}`
Remove a monitored road (stop ingestion).

### Alert Management API

#### POST /traffic/alerts`
Create a new congestion alert subscription.

#### `GET /traffic/alerts`:
List all active subscriptions.

#### `PUT /traffic/alerts/{alert_id}`:
Update threshold conditions for an alert

### `DELETE /traffic/alerts/{alert_id}`:
Delete an alert subscription.

### System Health API

#### `GET /system/health`
Returns uptime, model load time, GPU/CPU utilization.

#### `GET /system/metrics`
Returns Prometheus-compatible metrics (latency, FPS, inference errors).

### Model Management API

#### `GET /model/version`
Returns model name, version, and date deployed.

#### `POST /model/reload`
Hot reload updated weights.

#### `POST /model/upload`
Upload a new model version.

#### `DELETE /model/version/{version_id}`
Delete an old model version (if multiple are stored).