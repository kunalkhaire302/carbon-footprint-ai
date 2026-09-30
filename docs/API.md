# API v1

Base path: `/api/v1`. JSON responses include `X-Request-ID`; clients may supply 1–64 letters, digits, `.`, `_`, `:`, or `-`.

## Endpoints

| Method | Path | Purpose | Limit |
|---|---|---|---|
| GET | `/health` | Process availability | platform/edge |
| GET | `/ready` | Model readiness/version | platform/edge |
| GET | `/version` | App/API/model versions | metadata limit |
| GET | `/metadata` | Model metadata and data policy | metadata limit |
| POST | `/predict` | Validated footprint estimate | prediction limit |

Legacy `/predict`, `/health`, `/ready`, and `/version` remain compatible. `/history` and `/retrain` were intentionally removed.

## Prediction request

```json
{
  "electricity_usage_kwh": 250,
  "vehicle_type": "petrol",
  "vehicle_km": 800,
  "flights_short_haul": 1,
  "flights_long_haul": 0,
  "diet_type": "non-vegetarian",
  "waste_kg_weekly": 15,
  "household_size": 2,
  "grocery_spend_monthly": 300,
  "heating_source": "none",
  "internet_usage_hours": 6
}
```

All fields are required; unknown fields are rejected. Numeric bounds and categorical values are defined in `backend/app/schemas/prediction.py`.

## Prediction response

The canonical contract contains `prediction`, `grade`, `breakdown`, `benchmark`, `recommendations`, `model`, and `metadata`. Compatibility keys (`total_footprint_tco2e`, `category_breakdown`, `comparison`, `suggestions`) remain temporarily.

`confidence` is `null`: this model does not produce a calibrated interval. Recommendation savings are approximate and marked `is_estimate`.

## Errors

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request payload",
    "fields": {"internet_usage_hours": "Must be between 0 and 24"}
  },
  "request_id": "..."
}
```

400 malformed JSON; 413 oversized body; 415 wrong content type; 422 validation; 429 rate limit; 500 unexpected; 503 model unavailable. Production errors never contain tracebacks, paths, or environment data. Full machine-readable contract: [`openapi.yaml`](openapi.yaml).
