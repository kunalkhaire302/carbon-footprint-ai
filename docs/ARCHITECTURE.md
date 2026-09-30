# Architecture

## System

```mermaid
flowchart LR
  U[Browser] -->|HTTPS JSON| F[Flask + static frontend]
  F --> V[Strict input schema]
  V --> S[Prediction service]
  S --> M[Versioned ML pipeline]
  S --> C[Pure carbon calculations]
  S --> R[Recommendation ranking]
  F --> L[Structured privacy-safe logs]
```

The product is intentionally one deployable web service. Flask serves static files and `/api/v1`; Gunicorn supplies concurrent workers. The API is stateless and does not persist lifestyle inputs. A database, queue, and microservice split are omitted because no current requirement needs them.

## Boundaries

- `api`: HTTP methods, status codes, and versioned routes.
- `schemas`: untrusted input validation and API contracts.
- `services`: use-case orchestration.
- `domain`: pure factors, calculations, grading, and benchmarks.
- `ml`: validated, thread-safe model lifecycle.
- `middleware`: bounded per-process rate limiting and request policy.
- `frontend`: browser-local state, API client, accessible presentation.

## Inference flow

```mermaid
sequenceDiagram
  Browser->>API: POST /api/v1/predict + X-Request-ID
  API->>Schema: validate exact fields/ranges/categories
  Schema-->>API: typed PredictionInput
  API->>Model: pipeline.predict
  API->>Domain: category breakdown + grade
  API->>Service: ranked approximate actions
  API-->>Browser: v1 response + X-Request-ID
```

## Deployment

Render installs locked runtime dependencies and starts two Gunicorn workers with two threads each. `/ready` is the deployment health check. Startup loads and validates the artifact and metadata; readiness stays false if either fails.
