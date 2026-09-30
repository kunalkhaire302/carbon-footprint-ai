from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ApiError(Exception):
    code: str
    message: str
    status_code: int
    fields: dict[str, str] = field(default_factory=dict)


class ValidationError(ApiError):
    def __init__(self, fields: dict[str, str]) -> None:
        super().__init__("VALIDATION_ERROR", "Invalid request payload", 422, fields)


class ModelUnavailableError(ApiError):
    def __init__(self) -> None:
        super().__init__("MODEL_UNAVAILABLE", "Prediction model is unavailable", 503)
