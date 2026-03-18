# Plastic-Ledger Server API

The backend for the Plastic-Ledger project uses Django and Django REST Framework to expose REST endpoints for running and tracking the Machine Learning pipeline.

## Endpoints

### 1. Start a Pipeline Run
**POST** `/api/pipeline/runs/`

Initiates a new background pipeline execution.

#### Request Body (JSON)
| Field | Type | Description | Required | Default |
|-------|------|-------------|----------|---------|
| `bbox` | string | Comma-separated bounding box `lon_min,lat_min,lon_max,lat_max` | Yes | - |
| `target_date` | string (YYYY-MM-DD) | The target date for the run | Yes | - |
| `cloud_cover` | integer | Max cloud cover percentage allowed | No | 20 |
| `backtrack_days` | integer | Number of days to back-track particles | No | 30 |

#### Example Request
```bash
curl -X POST http://localhost:8000/api/pipeline/runs/ \
     -H "Content-Type: application/json" \
     -d '{
           "bbox": "80.0,8.0,82.0,10.0",
           "target_date": "2024-01-31",
           "cloud_cover": 20,
           "backtrack_days": 30
         }'
```

#### Example Response (201 Created)
```json
{
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "PENDING",
    "bbox": "80.0,8.0,82.0,10.0",
    "target_date": "2024-01-31",
    "cloud_cover": 20,
    "backtrack_days": 30,
    "output_dir": null,
    "created_at": "2024-03-18T12:00:00Z",
    "completed_at": null,
    "summary": null,
    "error_message": null
}
```

### 2. List Pipeline Runs
**GET** `/api/pipeline/runs/`

Returns a paginated list of all pipeline runs, ordered by newest first.

#### Example Request
```bash
curl -X GET http://localhost:8000/api/pipeline/runs/
```

### 3. Get Pipeline Run Details
**GET** `/api/pipeline/runs/{id}/`

Retrieves the status and outcome of a specific pipeline run.

#### Example Request
```bash
curl -X GET http://localhost:8000/api/pipeline/runs/123e4567-e89b-12d3-a456-426614174000/
```

#### Example Response (200 OK)
```json
{
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "COMPLETED",
    "bbox": "80.0,8.0,82.0,10.0",
    "target_date": "2024-01-31",
    "cloud_cover": 20,
    "backtrack_days": 30,
    "output_dir": "D:/Plastic-Ledger/data/runs/123e4567-e89b-12d3-a456-426614174000",
    "created_at": "2024-03-18T12:00:00Z",
    "completed_at": "2024-03-18T12:05:00Z",
    "summary": { "stages_completed": [1, 2, 3, 4, 5, 6, 7], "stages_failed": [], "outputs": {} },
    "error_message": null
}
```

## Running the Server

Make sure your MySQL server is running, the database `plastic_ledger` exists, and run:
```bash
python manage.py makemigrations api
python manage.py migrate
python manage.py runserver
```
