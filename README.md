# Carbon Footprint AI

A privacy-conscious climate analytics application that estimates an annual personal carbon footprint from lifestyle inputs, explains the result with deterministic categories, compares configured benchmarks, and ranks approximate reduction actions.

> **Important:** The model is trained entirely on synthetic, formula-generated data. This is an educational estimate, not a certified carbon audit, regulatory calculation, or validated measure of real-world accuracy.

[Live application](https://carbon-footprint-ai.onrender.com) · [API documentation](docs/API.md) · [Architecture](docs/ARCHITECTURE.md) · [Security](docs/SECURITY.md)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [ML and Data](#ml-and-data)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Local Setup](#local-setup)
- [Configuration](#configuration)
- [API](#api)
- [Development and Testing](#development-and-testing)
- [Deployment](#deployment)
- [Security and Privacy](#security-and-privacy)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Author](#author)

---

## Overview

Carbon Footprint AI takes a validated set of lifestyle inputs and returns an estimated annual footprint in tCO₂e/year. The total comes from an XGBoost regression pipeline. A separate, deterministic calculation provides the per-category breakdown, grading, benchmark comparison, and ranked recommendations, so the explanation is not produced by the ML model itself.

The application is a single stateless web service: a static frontend served alongside a Flask API.

## Key Features

- Strictly validated calculator covering home energy, transport, flights, diet, goods, waste, and digital use.
- Versioned Flask API with stable error responses, request IDs, rate limiting, health/readiness endpoints, and model metadata.
- XGBoost pipeline with deterministic data generation, train/validation/test separation, cross-validation, versioned artifacts, and recorded hashes.
- Explainable deterministic category breakdown, kept distinct from the ML total.
- Ranked recommendations with clearly labeled approximate savings.
- Responsive dark/light UI with keyboard focus, accessible errors and status messages, chart text alternatives, and reduced-motion support.
- No server-side storage or logging of lifestyle inputs. Optional recent summaries remain in the browser only.

## Architecture

```text
Browser (HTML/CSS/JS + Chart.js)
        │ HTTPS /api/v1
        ▼
Flask / Gunicorn
  ├─ exact request schema + request ID + safe errors
  ├─ versioned XGBoost preprocessing pipeline
  ├─ pure carbon calculations and grading
  └─ recommendation ranking + structured logs
```

The system is intentionally one stateless web service. No database, queue, authentication, or microservice split is included because the current product does not require them. See the [architecture document](docs/ARCHITECTURE.md).

## ML and Data

The committed dataset has 5,000 synthetic records generated with seed 42. The v1 model registry contains one complete preprocessing + model pipeline and a metadata file recording dependency versions, feature schema, dataset/artifact hashes, methodology, and metrics.

Held-out synthetic test metrics for v1:

| Metric | Result            |
|--------|------------------:|
| MAE    | 0.2129 tCO₂e/year |
| RMSE   | 0.2771 tCO₂e/year |
| R²     | 0.9866            |
| MAPE   | 3.00%             |

These results measure how well the model recovers the synthetic generator only. They do not measure accuracy against real-world footprints. Read the [model card](docs/ML_MODEL_CARD.md), [ML pipeline](docs/ML_PIPELINE.md), and [data methodology](docs/DATA_METHODOLOGY.md) before interpreting output.

## Technology Stack

| Area        | Technologies                                               |
|-------------|------------------------------------------------------------|
| Backend     | Python 3.13, Flask, Flask-CORS, Gunicorn                   |
| ML / data   | scikit-learn, XGBoost, pandas, NumPy, joblib               |
| Frontend    | Vanilla HTML, CSS, JavaScript, Chart.js                    |
| Quality     | pytest, Ruff, Black, Bandit, pip-audit, GitHub Actions     |
| Hosting     | Render                                                     |

Runtime dependencies are pinned in `requirements.txt` (Flask 3.1.3, flask-cors 6.0.0, scikit-learn 1.7.2, xgboost 3.2.0, pandas 2.2.3, numpy 2.2.2, joblib 1.5.2, gunicorn 23.0.0). Development tools are listed in `requirements-dev.txt`.

## Repository Structure

Top-level layout:

```text
.
├── .github/workflows/   # CI workflows
├── backend/             # Flask application and backend code
├── data/                # Synthetic dataset
├── docs/                # Project documentation
├── frontend/            # Browser UI
├── model/               # Model registry (e.g. model/registry/v1/)
├── scripts/             # Training, evaluation, and smoke-test scripts
├── tests/               # Test suite
├── .env.example         # Example environment configuration
├── CHANGELOG.md
├── CONTRIBUTING.md
├── metrics.json
├── pyproject.toml
├── render.yaml          # Render deployment configuration
├── requirements.txt
└── requirements-dev.txt
```

## Local Setup

```bash
git clone https://github.com/kunalkhaire302/carbon-footprint-ai.git
cd carbon-footprint-ai
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell/CMD)

python -m pip install -r requirements-dev.txt
python -m backend.app
```

Open `http://localhost:5000`. Copy values from `.env.example` into your environment when customization is needed.

## Configuration

Settings are read from environment variables. `.env.example` provides these local defaults:

| Variable                           | Example value                                  |
|------------------------------------|------------------------------------------------|
| `APP_ENV`                          | `development`                                  |
| `API_HOST`                         | `127.0.0.1`                                    |
| `API_PORT`                         | `5000`                                         |
| `FRONTEND_URL`                     | `http://localhost:5000`                        |
| `ALLOWED_CORS_ORIGINS`             | `http://localhost:5000,http://127.0.0.1:5000`  |
| `LOG_LEVEL`                        | `INFO`                                         |
| `MODEL_PATH`                       | `model/registry/v1/pipeline.joblib`            |
| `MODEL_METADATA_PATH`              | `model/registry/v1/metadata.json`              |
| `MODEL_VERSION`                    | `v1`                                           |
| `MAX_CONTENT_LENGTH`               | `32768`                                        |
| `PREDICTION_RATE_LIMIT_PER_MINUTE` | `30`                                           |
| `METADATA_RATE_LIMIT_PER_MINUTE`   | `120`                                          |
| `RATE_LIMIT_WINDOW_SECONDS`        | `60`                                           |

None of these values are secrets. Never commit real credentials. In production, set `ALLOWED_CORS_ORIGINS` to your real frontend origin(s) only.

## API

Canonical endpoint: `POST /api/v1/predict`. The legacy `POST /predict` route remains compatible.

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"household_size":2,"electricity_usage_kwh":250,"heating_source":"none","vehicle_type":"petrol","vehicle_km":800,"flights_short_haul":1,"flights_long_haul":0,"diet_type":"non-vegetarian","grocery_spend_monthly":300,"waste_kg_weekly":15,"internet_usage_hours":6}'
```

Operational routes: `/health`, `/ready`, `/version`, `/api/v1/metadata`.

Full schemas and error formats: [API.md](docs/API.md) and [OpenAPI specification](docs/openapi.yaml).

## Development and Testing

```bash
pytest
ruff check backend scripts tests
black --check backend scripts tests
bandit -q -r backend scripts -x tests
pip-audit -r requirements.txt
python -m scripts.evaluate_model
python -m scripts.smoke_test --local
```

Regenerate the dataset and train a new v1 artifact:

```bash
python backend/dataset.py
python -m scripts.train_model
```

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Deployment

The project is configured for [Render](https://render.com) via `render.yaml`:

| Setting      | Value                                                                                                                                      |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Build        | `pip install --requirement requirements.txt`                                                                                               |
| Start        | `gunicorn --workers 2 --threads 2 --timeout 30 --max-requests 1000 --max-requests-jitter 100 --access-logfile - backend.app:app`          |
| Health check | `/ready`                                                                                                                                   |
| Python       | 3.13.9                                                                                                                                     |
| Auto-deploy  | Enabled                                                                                                                                    |

`render.yaml` also sets `APP_ENV=production`, `LOG_LEVEL=INFO`, the model paths, `PREDICTION_RATE_LIMIT_PER_MINUTE=30`, and `ALLOWED_CORS_ORIGINS=https://carbon-footprint-ai.onrender.com`. Review the [production checklist](docs/PRODUCTION_CHECKLIST.md) before deploying.

## Security and Privacy

- Rejects unknown or nonsensical input and limits request body size and request rate.
- Uses an origin allowlist and browser security headers.
- Returns stable errors without stack traces.
- Never stores raw lifestyle profiles server-side.
- CI scans code and dependencies.

See [SECURITY.md](docs/SECURITY.md) and the [production checklist](docs/PRODUCTION_CHECKLIST.md).

## Limitations

- Emission factors and benchmarks preserve legacy assumptions and still require authoritative source review and dated regional calibration.
- No observational validation or calibrated prediction interval exists. Reported metrics reflect the synthetic data only.
- In-memory rate limits are per worker. Add an edge or distributed limiter when scaling beyond one instance.
- Formal assistive-technology and staging load audits remain deployment gates.

## Roadmap

Proposed, not implemented:

- Authoritative, dated, region-specific emission factors and benchmarks.
- Validation against observational data and calibrated prediction intervals.
- A distributed rate limiter for multi-instance deployments.
- Persistence only with explicit accounts, retention, export, and deletion design.

## Documentation

- [Production audit](docs/PRODUCTION_AUDIT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API](docs/API.md) · [OpenAPI](docs/openapi.yaml)
- [Security](docs/SECURITY.md)
- [Data methodology](docs/DATA_METHODOLOGY.md)
- [Model card](docs/ML_MODEL_CARD.md)
- [ML pipeline](docs/ML_PIPELINE.md)
- [Development](docs/DEVELOPMENT.md)
- [Production checklist](docs/PRODUCTION_CHECKLIST.md)
- [Changelog](CHANGELOG.md)

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and run the checks in [Development and Testing](#development-and-testing) before opening a pull request.

## Author

Made and developed by [Kunal Khaire](https://github.com/kunalkhaire302).
