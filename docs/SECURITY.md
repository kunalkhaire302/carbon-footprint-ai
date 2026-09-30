# Security

## Threat model

Public assets and unauthenticated calculation endpoints are exposed to malformed input, resource exhaustion, cross-origin abuse, dependency compromise, artifact tampering, and accidental personal-data retention. The service has no accounts, secrets, payments, or privileged user operations.

## Controls

- Exact schema, types, finite-number checks, ranges, categories, and unknown-field rejection.
- 32 KiB request limit and configurable per-IP prediction rate limit.
- Origin allowlist; no wildcard production CORS.
- CSP, MIME-sniffing, frame, referrer, and permissions headers.
- Central errors with stable codes and request IDs; no traceback or path disclosure.
- No remote model training, history endpoint, or server-side lifestyle storage.
- Production debug disabled; secrets and `.env` ignored.
- CI runs Bandit and pip-audit.

The in-memory rate limiter is intentionally process-local. It reduces accidental/low-volume abuse, not distributed attacks. Add an edge or Redis-backed limiter when multiple instances or adversarial traffic justify it.

## Artifact policy

Joblib artifacts are executable Python serialization and must only come from the trusted training pipeline. Metadata records an artifact SHA-256, dependency versions, dataset hash, and source commit. Do not load user-supplied models.

## Privacy and logging

Lifestyle inputs can be sensitive. The API processes them in memory, does not persist them, and logs only endpoint, status, duration, model/error code, and request ID. Browser history is device-local and capped at ten summaries.

## Incident basics

Disable auto-deploy, roll back to the prior trusted commit/artifact, rotate any exposed platform secrets, preserve structured logs, identify affected requests by request ID, patch and test, then document the event. Report vulnerabilities privately to the repository owner; do not include sensitive exploit data in a public issue.
