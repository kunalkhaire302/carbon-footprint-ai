# Production Checklist

## Security

- [x] Secrets excluded and environment template provided
- [x] CORS allowlist
- [x] Rate limit enabled
- [x] Debug disabled in production
- [x] Security headers and request limit
- [x] No raw input persistence or logging

## API

- [x] Strict validation and stable errors
- [x] Request IDs
- [x] Health, readiness, version, metadata
- [x] `/api/v1` with `/predict` compatibility

## ML

- [x] Versioned pipeline artifact and metadata
- [x] Deterministic training and untouched test set
- [x] Evaluation and model card
- [x] Startup model validation
- [ ] External observational validation

## Frontend

- [x] Responsive layout and reduced motion
- [x] Accessible labels, errors, status, focus, chart text
- [x] Loading, timeout, offline, API error states
- [x] Device-local limited history
- [ ] Formal WCAG audit with assistive technologies

## DevOps and documentation

- [x] CI quality/security/test/model/smoke stages
- [x] Render build/start/health configuration
- [x] API, architecture, security, ML, data, development docs
- [ ] Validate a staging deployment before production promotion
- [ ] Configure platform alerts and external uptime check
