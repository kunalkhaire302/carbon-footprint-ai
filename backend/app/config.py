from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in os.getenv(name, default).split(",") if value.strip())


@dataclass(frozen=True)
class Settings:
    environment: str
    host: str
    port: int
    log_level: str
    frontend_url: str
    allowed_origins: tuple[str, ...]
    model_path: Path
    model_metadata_path: Path
    model_version: str
    max_content_length: int
    prediction_rate_limit: int
    metadata_rate_limit: int
    rate_limit_window_seconds: int

    @classmethod
    def from_env(cls) -> Settings:
        environment = os.getenv("APP_ENV", "development").lower()
        return cls(
            environment=environment,
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", os.getenv("API_PORT", "5000"))),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            frontend_url=os.getenv("FRONTEND_URL", "http://localhost:5000"),
            allowed_origins=_csv("ALLOWED_CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"),
            model_path=Path(os.getenv("MODEL_PATH", ROOT / "model" / "registry" / "v1" / "pipeline.joblib")),
            model_metadata_path=Path(
                os.getenv("MODEL_METADATA_PATH", ROOT / "model" / "registry" / "v1" / "metadata.json")
            ),
            model_version=os.getenv("MODEL_VERSION", "v1"),
            max_content_length=int(os.getenv("MAX_CONTENT_LENGTH", str(32 * 1024))),
            prediction_rate_limit=int(os.getenv("PREDICTION_RATE_LIMIT_PER_MINUTE", "30")),
            metadata_rate_limit=int(os.getenv("METADATA_RATE_LIMIT_PER_MINUTE", "120")),
            rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
        )


class TestSettings(Settings):
    @classmethod
    def from_env(cls) -> TestSettings:
        base = Settings.from_env()
        return cls(**{**base.__dict__, "environment": "testing", "prediction_rate_limit": 10_000})
