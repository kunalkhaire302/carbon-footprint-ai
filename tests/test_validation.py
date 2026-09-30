from __future__ import annotations

import math

import pytest

from backend.app.errors import ValidationError
from backend.app.schemas.prediction import PredictionInput


def test_valid_payload(valid_payload):
    assert PredictionInput.parse(valid_payload).household_size == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("internet_usage_hours", 25),
        ("electricity_usage_kwh", -1),
        ("vehicle_km", math.inf),
        ("waste_kg_weekly", None),
        ("flights_short_haul", 1.5),
        ("vehicle_type", "hydrogen"),
    ],
)
def test_invalid_values(valid_payload, field, value):
    valid_payload[field] = value
    with pytest.raises(ValidationError) as raised:
        PredictionInput.parse(valid_payload)
    assert field in raised.value.fields


def test_missing_and_unknown_fields(valid_payload):
    valid_payload.pop("diet_type")
    valid_payload["unknown"] = 1
    with pytest.raises(ValidationError) as raised:
        PredictionInput.parse(valid_payload)
    assert raised.value.fields["diet_type"] == "Field is required"
    assert "unknown" in raised.value.fields["_schema"]
