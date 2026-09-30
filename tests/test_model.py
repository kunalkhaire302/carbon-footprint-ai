from __future__ import annotations

from pathlib import Path

from backend.app.ml.model_loader import ModelManager
from backend.app.schemas.prediction import PredictionInput


def test_model_loads_and_predicts(valid_payload):
    root = Path(__file__).resolve().parents[1]
    manager = ModelManager(root / "model/registry/v1/pipeline.joblib", root / "model/registry/v1/metadata.json", "v1")
    manager.load()
    assert manager.ready
    value, duration = manager.predict(PredictionInput.parse(valid_payload))
    assert isinstance(value, float)
    assert value >= 0
    assert duration >= 0


def test_invalid_artifact_is_not_ready():
    root = Path(__file__).resolve().parents[1]
    manager = ModelManager(root / "tests/missing.joblib", root / "tests/missing-metadata.json", "test")
    manager.load()
    assert not manager.ready
    assert manager.load_error == "FileNotFoundError"
