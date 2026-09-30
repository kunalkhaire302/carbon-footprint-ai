from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar, cast

from flask import Blueprint, current_app, g, jsonify, request
from flask.typing import ResponseReturnValue

from backend.app.config import Settings
from backend.app.errors import ApiError
from backend.app.middleware.rate_limit import RateLimiter
from backend.app.ml.model_loader import ModelManager
from backend.app.schemas.prediction import PredictionInput
from backend.app.services.prediction_service import create_prediction

F = TypeVar("F", bound=Callable[..., object])
api = Blueprint("api", __name__)


def limited(bucket: str, setting: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapped(*args: object, **kwargs: object) -> object:
            limiter = cast(RateLimiter, current_app.extensions["rate_limiter"])
            settings = cast(Settings, current_app.config["SETTINGS"])
            client = request.remote_addr or "unknown"
            limiter.check(f"{bucket}:{client}", getattr(settings, setting))
            return func(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator


@api.get("/health")
def health() -> ResponseReturnValue:
    return jsonify(status="ok", service="carbon-footprint-api"), 200


@api.get("/ready")
def ready() -> ResponseReturnValue:
    model = cast(ModelManager, current_app.extensions["model_manager"])
    status = 200 if model.ready else 503
    return (
        jsonify(status="ready" if model.ready else "not_ready", model_loaded=model.ready, model_version=model.version),
        status,
    )


@api.get("/version")
@limited("metadata", "metadata_rate_limit")
def version() -> ResponseReturnValue:
    model = cast(ModelManager, current_app.extensions["model_manager"])
    return (
        jsonify(application_version=current_app.config["APP_VERSION"], api_version="v1", model_version=model.version),
        200,
    )


@api.get("/metadata")
@limited("metadata", "metadata_rate_limit")
def metadata() -> ResponseReturnValue:
    model = cast(ModelManager, current_app.extensions["model_manager"])
    return (
        jsonify(
            service="carbon-footprint-api",
            model={"version": model.version, "loaded": model.ready, "metadata": model.metadata if model.ready else {}},
            data_policy="Prediction inputs are processed in memory and are not persisted by the API.",
        ),
        200,
    )


@api.get("/metrics")
@limited("metadata", "metadata_rate_limit")
def metrics() -> ResponseReturnValue:
    model = cast(ModelManager, current_app.extensions["model_manager"])
    counters = cast(dict[str, float], current_app.extensions["metrics"])
    requests = counters["request_count"]
    return (
        jsonify(
            request_count=requests,
            error_count=counters["error_count"],
            prediction_count=counters["prediction_count"],
            validation_failure_count=counters["validation_failure_count"],
            rate_limit_count=counters["rate_limit_count"],
            average_request_latency_ms=round(counters["request_latency_ms_total"] / requests, 3) if requests else 0,
            model_loaded=model.ready,
            model_version=model.version,
            model_load_time_ms=model.load_time_ms,
        ),
        200,
    )


@api.post("/predict")
@limited("prediction", "prediction_rate_limit")
def predict() -> ResponseReturnValue:
    if not request.is_json:
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/json", 415)
    payload = request.get_json(silent=False)
    data = PredictionInput.parse(payload)
    started = time.perf_counter()
    model = cast(ModelManager, current_app.extensions["model_manager"])
    result = create_prediction(data, model, g.request_id)
    result_metadata = result["metadata"]
    if isinstance(result_metadata, dict):
        result_metadata["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 3)
    counters = cast(dict[str, float], current_app.extensions["metrics"])
    counters["prediction_count"] += 1
    return jsonify(result), 200
