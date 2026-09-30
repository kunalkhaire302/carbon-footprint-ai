# Changelog

All notable changes follow Keep a Changelog; versions follow Semantic Versioning.

## [1.0.0] - 2026-09-30

### Added

- Application factory, environment configuration, `/api/v1`, health/readiness/version/metadata routes.
- Strict schemas, structured errors, request IDs, CORS allowlist, security headers, body limits, and rate limiting.
- Versioned reproducible model pipeline, metadata, expanded evaluation, model card, and data methodology.
- Pure carbon calculation, grading, benchmark, and recommendation modules.
- Browser timeout/offline/validation UX, accessible chart summaries, reduced motion, and local-only history.
- Backend/API/model tests, CI quality and security checks, smoke/load scripts, and production documentation.

### Removed

- Unauthenticated remote retraining.
- Server-side storage and public exposure of lifestyle history.

### Changed

- Render now validates `/ready` and runs bounded Gunicorn workers.
- Legacy `/predict` remains compatible while canonical clients use `/api/v1/predict`.
