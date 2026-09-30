from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections import Counter
from typing import cast

from flask import Flask, Response, g, jsonify, request
from flask.typing import ResponseReturnValue
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, HTTPException, RequestEntityTooLarge

from backend.app.api.routes import (
    api,
)
from backend.app.api.routes import (
    health as api_health,
)
from backend.app.api.routes import (
    predict as api_predict,
)
from backend.app.api.routes import (
    ready as api_ready,
)
from backend.app.api.routes import (
    version as api_version,
)
from backend.app.config import ROOT, Settings
from backend.app.errors import ApiError, ValidationError
from backend.app.middleware.rate_limit import RateLimiter
from backend.app.ml.model_loader import ModelManager

APP_VERSION = "1.0.0"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                "level": record.levelname,
                "message": record.getMessage(),
                "request_id": getattr(record, "request_id", None),
                "endpoint": getattr(record, "endpoint", None),
                "status_code": getattr(record, "status_code", None),
                "duration_ms": getattr(record, "duration_ms", None),
                "error_code": getattr(record, "error_code", None),
                "model_version": getattr(record, "model_version", None),
            },
            separators=(",", ":"),
        )


def _configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if settings.environment == "production" else logging.Formatter("%(levelname)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level)


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings.from_env()
    _configure_logging(settings)
    app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
    app.config.update(
        SETTINGS=settings, APP_VERSION=APP_VERSION, MAX_CONTENT_LENGTH=settings.max_content_length, JSON_SORT_KEYS=False
    )
    CORS(
        app,
        resources={
            r"/api/*": {"origins": settings.allowed_origins},
            r"/predict": {"origins": settings.allowed_origins},
        },
    )

    model = ModelManager(settings.model_path, settings.model_metadata_path, settings.model_version)
    model.load()
    app.extensions["model_manager"] = model
    app.extensions["rate_limiter"] = RateLimiter(settings.rate_limit_window_seconds)
    app.extensions["metrics"] = Counter()
    app.register_blueprint(api, url_prefix="/api/v1")

    @app.get("/")
    def index() -> Response:
        return app.send_static_file("index.html")

    @app.post("/predict")
    def legacy_predict() -> ResponseReturnValue:
        return api_predict()

    @app.get("/health")
    def legacy_health() -> ResponseReturnValue:
        return api_health()

    @app.get("/ready")
    def legacy_ready() -> ResponseReturnValue:
        return api_ready()

    @app.get("/version")
    def legacy_version() -> ResponseReturnValue:
        return api_version()

    @app.before_request
    def begin_request() -> None:
        provided = request.headers.get("X-Request-ID", "")
        g.request_id = provided if REQUEST_ID_PATTERN.fullmatch(provided) else uuid.uuid4().hex
        g.request_started = time.perf_counter()

    @app.after_request
    def secure_response(response: Response) -> Response:
        response.headers["X-Request-ID"] = g.get("request_id", "")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' "
            "https://fonts.googleapis.com https://cdnjs.cloudflare.com; font-src 'self' https://fonts.gstatic.com "
            "https://cdnjs.cloudflare.com; img-src 'self' data:; connect-src 'self'"
        )
        if request.path.startswith("/api/") or request.path in {"/predict", "/health", "/ready", "/version"}:
            response.headers["Cache-Control"] = "no-store"
        duration = round((time.perf_counter() - g.get("request_started", time.perf_counter())) * 1000, 3)
        metrics = cast(dict[str, float], app.extensions["metrics"])
        metrics["request_count"] += 1
        metrics["request_latency_ms_total"] += duration
        if response.status_code >= 400:
            metrics["error_count"] += 1
        logging.getLogger("request").info(
            "request_complete",
            extra={
                "request_id": g.get("request_id"),
                "endpoint": request.endpoint,
                "status_code": response.status_code,
                "duration_ms": duration,
                "model_version": model.version,
            },
        )
        return response

    @app.errorhandler(ValidationError)
    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError) -> ResponseReturnValue:
        metrics = cast(dict[str, float], app.extensions["metrics"])
        if error.code == "VALIDATION_ERROR":
            metrics["validation_failure_count"] += 1
        if error.code == "RATE_LIMITED":
            metrics["rate_limit_count"] += 1
        error_payload: dict[str, object] = {"code": error.code, "message": error.message}
        body: dict[str, object] = {
            "error": error_payload,
            "request_id": g.get("request_id"),
        }
        if error.fields:
            error_payload["fields"] = error.fields
        return jsonify(body), error.status_code

    @app.errorhandler(BadRequest)
    def malformed_json(_: BadRequest) -> ResponseReturnValue:
        return (
            jsonify(
                error={"code": "MALFORMED_JSON", "message": "Malformed JSON request"}, request_id=g.get("request_id")
            ),
            400,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_: RequestEntityTooLarge) -> ResponseReturnValue:
        return (
            jsonify(
                error={"code": "REQUEST_TOO_LARGE", "message": "Request body exceeds the configured limit"},
                request_id=g.get("request_id"),
            ),
            413,
        )

    @app.errorhandler(Exception)
    def unexpected(error: Exception) -> ResponseReturnValue:
        if isinstance(error, HTTPException):
            return (
                jsonify(error={"code": "HTTP_ERROR", "message": error.name}, request_id=g.get("request_id")),
                error.code or 500,
            )
        logging.getLogger(__name__).exception(
            "unhandled_exception", extra={"request_id": g.get("request_id"), "error_code": "INTERNAL_ERROR"}
        )
        return (
            jsonify(
                error={"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
                request_id=g.get("request_id"),
            ),
            500,
        )

    return app


app = create_app()
