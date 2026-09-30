# Development

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` for local values; the application reads real environment variables and never requires committed secrets.

## Run

```bash
python -m backend.app
```

Open `http://localhost:5000`. The Flask service serves both frontend and API.

## Data and model

```bash
python backend/dataset.py
python -m scripts.train_model
python -m scripts.evaluate_model
```

## Quality

```bash
pytest
ruff check backend scripts tests
black --check backend scripts tests
bandit -q -r backend scripts -x tests
pip-audit -r requirements.txt
python -m scripts.smoke_test --local
```

Run a small local load check only against an approved non-production target:

```bash
python -m scripts.load_test --base-url http://localhost:5000 --requests 20 --concurrency 4
```

## Troubleshooting

- `/ready` 503: verify registry artifact and metadata paths, feature list, and dependency versions.
- Browser CORS error: include the exact frontend origin in `ALLOWED_CORS_ORIGINS`.
- 422: inspect `error.fields`; inputs are exact and unknown keys are rejected.
- 429: wait for the configured window or adjust the development-only limit.
