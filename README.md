# Carbon Footprint AI

A privacy-conscious climate analytics application that estimates an annual personal footprint from lifestyle inputs, explains the result with deterministic categories, compares configured benchmarks, and ranks approximate reduction actions.

> The model is trained entirely on synthetic, formula-generated data. This is an educational estimate—not a certified carbon audit, regulatory calculation, or validated measure of real-world accuracy.

[Live application](https://carbon-footprint-ai.onrender.com) · [API documentation](docs/API.md) · [Architecture](docs/ARCHITECTURE.md) · [Security](docs/SECURITY.md)

## Product

- Strictly validated calculator for home energy, transport, flights, diet, goods, waste, and digital use.
- Versioned Flask API with stable errors, request IDs, rate limiting, health/readiness, and model metadata.
- XGBoost pipeline with deterministic generation, train/validation/test separation, cross-validation, versioned artifacts, and recorded hashes.
- Explainable deterministic category breakdown kept distinct from the ML total.
- Ranked recommendations with clearly labeled approximate savings.
- Responsive dark/light UI with keyboard focus, accessible errors/status, chart text alternatives, and reduced-motion support.
- No server-side storage or logging of lifestyle inputs; optional recent summaries remain in the browser.

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

The system is intentionally one stateless web service. No database, queue, authentication, or microservice split is included because the current product does not require them. See [the architecture document](docs/ARCHITECTURE.md).

## ML and data

The committed dataset has 5,000 synthetic records generated with seed 42. The v1 registry contains one complete preprocessing+model pipeline and metadata with dependency versions, feature schema, dataset/artifact hashes, methodology, and actual metrics.

Held-out synthetic test metrics for v1:

| Metric | Result |
|---|---:|
| MAE | 0.2129 tCO₂e/year |
| RMSE | 0.2771 tCO₂e/year |
| R² | 0.9866 |
| MAPE | 3.00% |

These results measure recovery of the synthetic generator only. Read the [model card](docs/ML_MODEL_CARD.md), [pipeline](docs/ML_PIPELINE.md), and [data methodology](docs/DATA_METHODOLOGY.md) before interpreting output.

## Tech stack

- Python 3.13, Flask, Gunicorn
- scikit-learn, XGBoost, pandas, NumPy, joblib
- Vanilla HTML, CSS, JavaScript, Chart.js
- pytest, Ruff, Black, Bandit, pip-audit, GitHub Actions
- Render

## Local setup

```bash
git clone https://github.com/kunalkhaire302/carbon-footprint-ai.git
cd carbon-footprint-ai
python -m venv .venv
python -m pip install -r requirements-dev.txt
python -m backend.app
```

Open `http://localhost:5000`. Copy `.env.example` values into your environment when customization is needed.

## API

Canonical endpoint: `POST /api/v1/predict`. Legacy `POST /predict` remains compatible.

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"household_size":2,"electricity_usage_kwh":250,"heating_source":"none","vehicle_type":"petrol","vehicle_km":800,"flights_short_haul":1,"flights_long_haul":0,"diet_type":"non-vegetarian","grocery_spend_monthly":300,"waste_kg_weekly":15,"internet_usage_hours":6}'
```

Operational routes: `/health`, `/ready`, `/version`, `/api/v1/metadata`. Full schemas and errors are in [API.md](docs/API.md) and [OpenAPI](docs/openapi.yaml).

## Development and testing

```bash
pytest
ruff check backend scripts tests
black --check backend scripts tests
bandit -q -r backend scripts -x tests
pip-audit -r requirements.txt
python -m scripts.evaluate_model
python -m scripts.smoke_test --local
```

Regenerate data and train a new v1 artifact with:

```bash
python backend/dataset.py
python -m scripts.train_model
```

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Deployment

Render uses:

- Build: `pip install --requirement requirements.txt`
- Start: `gunicorn --workers 2 --threads 2 --timeout 30 --max-requests 1000 --max-requests-jitter 100 --access-logfile - backend.app:app`
- Health check: `/ready`

Required production configuration is represented in `render.yaml`; local options are in `.env.example`. Restrict `ALLOWED_CORS_ORIGINS` to real frontend origins.

## Security and privacy

The service rejects unknown or nonsensical input, limits body size and request rate, uses an origin allowlist and browser security headers, returns stable errors without stack traces, and never stores raw lifestyle profiles. CI scans code and dependencies. Review [SECURITY.md](docs/SECURITY.md) and the [production checklist](docs/PRODUCTION_CHECKLIST.md).

## Limitations and roadmap

- Emission factors and benchmarks preserve legacy assumptions and still require authoritative source review and dated regional calibration.
- No observational validation or calibrated prediction interval exists.
- In-memory rate limits are per worker; add an edge/distributed limiter when scaling beyond one instance.
- Formal assistive-technology and staging load audits remain deployment gates.
- Persistence should only be added with explicit accounts, retention, export, and deletion design.

## Documentation

- [Production audit](docs/PRODUCTION_AUDIT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API](docs/API.md)
- [Security](docs/SECURITY.md)
- [Data methodology](docs/DATA_METHODOLOGY.md)
- [Model card](docs/ML_MODEL_CARD.md)
- [ML pipeline](docs/ML_PIPELINE.md)
- [Development](docs/DEVELOPMENT.md)
- [Production checklist](docs/PRODUCTION_CHECKLIST.md)
- [Changelog](CHANGELOG.md)

## Author

Made and developed by [Kunal Khaire](https://github.com/kunalkhaire302).

<!-- Featured-repo -->
