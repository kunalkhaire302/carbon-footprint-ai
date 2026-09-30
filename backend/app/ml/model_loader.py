from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
import xgboost

from backend.app.errors import ModelUnavailableError
from backend.app.schemas.prediction import PredictionInput

LOGGER = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model_path: Path, metadata_path: Path, version: str) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.version = version
        self._pipeline: Any = None
        self._lock = threading.RLock()
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self.load_time_ms: float | None = None

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        started = time.perf_counter()
        try:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Model artifact not found: {self.model_path.name}")
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            actual_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
            if metadata.get("artifact_sha256") != actual_hash:
                raise ValueError("Model artifact integrity check failed")
            if (
                metadata.get("scikit_learn_version") != sklearn.__version__
                or metadata.get("xgboost_version") != xgboost.__version__
            ):
                raise ValueError("Model runtime versions do not match training metadata")
            pipeline = joblib.load(self.model_path)
            if not callable(getattr(pipeline, "predict", None)):
                raise TypeError("Model artifact does not provide predict()")
            expected = set(PredictionInput.__dataclass_fields__)
            if set(metadata.get("features", [])) != expected:
                raise ValueError("Model feature schema does not match API schema")
            self._pipeline = pipeline
            self.metadata = metadata
            self.version = str(metadata.get("model_version", self.version))
            self.load_error = None
        except Exception as exc:
            self._pipeline = None
            self.load_error = type(exc).__name__
            LOGGER.error("model_load_failed", extra={"error_code": "MODEL_LOAD_FAILED"})
        finally:
            self.load_time_ms = round((time.perf_counter() - started) * 1000, 3)

    def predict(self, data: PredictionInput) -> tuple[float, float]:
        if not self.ready:
            raise ModelUnavailableError()
        started = time.perf_counter()
        with self._lock:
            value = float(self._pipeline.predict(pd.DataFrame([data.to_dict()]))[0])
        return round(value, 2), round((time.perf_counter() - started) * 1000, 3)
