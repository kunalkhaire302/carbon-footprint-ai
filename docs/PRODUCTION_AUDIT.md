# Production Audit

Date: 2026-09-30  
Repository: `kunalkhaire302/carbon-footprint-ai`  
Baseline commit: `23695e7`

## Executive summary

The repository is a functional single-process Flask demonstration with a polished static UI and a small synthetic-data ML workflow. It is not yet production-grade. The highest risks are an unauthenticated remote retraining endpoint, unrestricted CORS, raw exception disclosure, missing request validation, persistence and exposure of lifestyle inputs, incompatible serialized-model/runtime versions, and unsupported scientific claims in documentation. The smallest suitable target remains a stateless Flask service plus static frontend; no database or microservices are justified.

## 1. Current architecture

- Flask serves both the static frontend and JSON endpoints from `backend/app.py`.
- `backend/utils.py` loads separate joblib model/preprocessor artifacts at import time and contains inference, carbon calculations, grading, benchmarks, and recommendations.
- `backend/dataset.py` creates 5,000 deterministic synthetic rows.
- `backend/model.py` trains three regressors and selects by held-out R².
- The frontend is vanilla HTML/CSS/JavaScript with Chart.js and Font Awesome from public CDNs.
- Render runs a single Gunicorn web service.

## 2. Current data flow

Browser form → `POST /predict` → pandas DataFrame → preprocessor → XGBoost → deterministic breakdown → grade/benchmark → recommendations → response. The request and full result are then written to `history.json`; `/history` returns all stored lifestyle records.

Baseline request produced `7.13 tCO₂e/year`, grade `F`, percentile `1`, with category values Transport 2.07, Electricity 0.60, Diet 2.30, Goods 1.80, Waste 0.04, Digital 0.18.

## 3. Current ML pipeline

- Features: 7 numeric and 4 categorical inputs; target is a formula-generated value plus Gaussian noise.
- Split: one seeded 80/20 split; five-fold R² on the training partition.
- Candidate models: linear regression, random forest, XGBoost.
- Stored artifacts: independent preprocessor and estimator pickles.
- Recorded XGBoost metrics: R² 0.9845, MAE 0.2283, RMSE 0.3025, CV R² 0.9764.
- These metrics measure recovery of the synthetic generator, not real-world carbon-accounting accuracy.
- Local inspection emitted scikit-learn version mismatch warnings (artifact 1.4.1.post1 vs runtime 1.7.2) and an XGBoost pickle compatibility warning.

## 4. Current API behavior

Endpoints are `/`, `/predict`, `/history`, and `/retrain`. There is no API versioning, health/readiness/version/metadata route, schema, request ID, consistent error contract, request-size limit, or rate limit. `/predict` accepts arbitrary JSON keys and values. `/retrain` runs a subprocess and returns stderr details.

## 5. Current frontend architecture

The page has a semantic form shell and responsive grid, but submission uses scattered DOM access and a direct `fetch('/predict')`. Errors use `alert()`, no timeout or cancellation exists, server validation errors cannot be rendered per field, chart canvases lack text alternatives, theme preference is not persisted, and dynamic HTML is created with `innerHTML`.

## 6. Security vulnerabilities

| Priority | Finding |
|---|---|
| P0 | Unauthenticated `/retrain` permits remote CPU exhaustion and artifact replacement. |
| P0 | Raw exception strings and subprocess stderr expose internals. |
| P0 | Lifestyle inputs are written to `history.json` and exposed through `/history` without consent or access control. |
| P1 | CORS allows every origin. |
| P1 | No request-size limit, rate limit, strict content-type handling, or input schema. |
| P1 | No CSP, clickjacking, MIME-sniffing, referrer, or permissions headers. |
| P1 | `innerHTML` renders response fields and would become an XSS path if response content were influenced externally. |
| P2 | Third-party CDN assets are unpinned and lack integrity metadata. |
| P2 | No automated dependency or static security scanning. |

## 7. Performance bottlenecks

- pandas DataFrame creation is acceptable at current scale but prediction timing is unmeasured.
- Gunicorn uses default worker settings and no timeout/max-request controls.
- Chart.js, Font Awesome, and fonts are render-blocking third-party assets.
- Full glass blur and hover transforms are expensive on low-end/mobile devices.
- History file read/write occurs synchronously on every prediction and is unsafe across workers.

## 8. Reliability issues

- Relative artifact paths depend on process working directory.
- Model load failure is reduced to `None` and discovered only during a request.
- Joblib artifacts are sensitive to dependency version changes and are already warning locally.
- JSON history writes are non-atomic and race across workers.
- No startup validation, readiness signal, timeouts, graceful degradation, or stable error codes.

## 9. ML risks

- The model approximates a known synthetic formula, so its high score is expected and does not establish external validity.
- Model selection uses the test split, biasing the final reported result.
- No independent final test set, CV dispersion, MAPE, residual analysis, leakage check, OOD detection, or feature-schema checksum.
- Separate preprocessor/model artifacts can be mismatched.
- Pickle/joblib loading is unsafe for untrusted artifacts.
- Feature importance and per-prediction drivers are absent.

## 10. Data-quality issues

- All records are synthetic and generated from assumptions, not observed footprints.
- Factors have no machine-readable source, geography, effective date, or uncertainty.
- Dataset values are constrained by generator distributions and may not represent real users.
- README statements attribute factors to standards without verifiable citations and report metric values that differ from `metrics.json`.
- Currency is not geographically defined; goods spend is therefore ambiguous.

## 11. Deployment risks

- Render pins Python 3.10.0 while local runtime and artifact metadata differ.
- No health check path, deployment validation, pre-deploy tests, or configured environment.
- Model files are copied from Git without integrity/version metadata.
- No explicit Gunicorn worker, timeout, access-log, or forwarded-header configuration.
- The service writes to an ephemeral filesystem.

## 12. Testing gaps

There are no automated tests. Validation, calculations, threshold boundaries, artifact loading, endpoints, response compatibility, frontend states, accessibility, deployment configuration, and smoke behavior are all uncovered.

## 13. Accessibility gaps

- No skip link, status/error live region, explicit form error association, or reduced-motion mode.
- Chart meaning is only visual.
- Theme button state is not exposed with `aria-pressed`.
- Focus styling is incomplete; hover motion is excessive.
- History table intentionally overflows on mobile.

## 14. Observability gaps

No structured logging, request correlation, latency measurement, error code, prediction count, validation count, model status metric, or deployment health signal exists. Raw inputs should not be logged when these are added.

## 15. Documentation gaps

- README claims “production-ready,” “microservices-style,” strict validation, statelessness, sourced emission factors, and fused pipeline serialization although the code does not support those claims.
- README request examples do not match actual field names.
- No API, architecture, security, development, data methodology, model card, pipeline, or production checklist documents exist.
- No license file exists despite the README claiming MIT.

## 16. Recommended architecture

Keep one deployable service and a static frontend:

```text
backend/app/
  api/              versioned Flask blueprints
  domain/           pure calculations, factors, grading, benchmarks
  schemas/          explicit request/response contracts
  services/         prediction and recommendation orchestration
  ml/               validated, thread-safe model lifecycle
  middleware/       request IDs, errors, security, rate limiting
scripts/            deterministic dataset/training/evaluation/smoke commands
tests/              unit and API tests
frontend/           static accessible UI and centralized API client
model/registry/v1/  pipeline, metadata, metrics
```

Use an application factory, environment-backed typed configuration, one versioned pipeline artifact, pure domain functions, JSON production logs, in-process bounded IP rate limiting, and a stateless API. A database, authentication, Prometheus server, task queue, or microservice split is not currently justified.

## 17. Prioritized implementation roadmap

### P0 — Critical

1. Remove public retraining and server-side lifestyle history persistence/exposure.
2. Add strict JSON schema validation, bounded values, safe centralized errors, and request IDs.
3. Add application factory, absolute model paths, startup validation, and readiness failure.
4. Align runtime dependencies with artifacts, then produce one reproducible versioned pipeline artifact.
5. Add regression/API/security tests before changing response behavior.

### P1 — High

1. Add `/api/v1/predict`, compatibility `/predict`, health/readiness/version/metadata endpoints.
2. Restrict CORS, add security headers, request-size limit, configurable rate limiting, and no-store API caching.
3. Centralize emission factors, benchmarks, grading, calculations, and recommendations as typed pure modules.
4. Add structured privacy-preserving logs and latency counters.
5. Add CI for tests, Ruff, Black, Bandit, pip-audit, and deployment validation.
6. Improve frontend error/loading/validation/accessibility and centralized API handling.

### P2 — Medium

1. Add model registry metadata, model card, data methodology, residual/CV reporting, and feature importance.
2. Add OpenAPI contract, architecture/security/development docs, smoke/load scripts, and production checklist.
3. Optimize external frontend assets and responsive chart behavior.
4. Add measured performance baselines.

### P3 — Low

1. Consider regional benchmark/factor packs after authoritative sources and product geography are selected.
2. Consider persistence only with an explicit user-account, retention, deletion, and privacy design.
3. Consider richer explainability only after validating the model against observational data.

## Baseline preservation policy

The legacy `/predict` route and legacy response keys will remain during migration. The v1 response will add explicit model and request metadata without fabricating confidence. The existing 7.13 reference payload is the regression baseline. Intentional removals are `/retrain` and server history persistence because they are active security/privacy defects; browser-local history may remain as an opt-in convenience.
