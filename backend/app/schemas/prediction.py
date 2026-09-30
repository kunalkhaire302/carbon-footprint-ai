from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from backend.app.errors import ValidationError

NUMERIC_RULES: dict[str, tuple[type, float, float]] = {
    "household_size": (int, 1, 20),
    "electricity_usage_kwh": (float, 0, 10_000),
    "vehicle_km": (float, 0, 20_000),
    "flights_short_haul": (int, 0, 100),
    "flights_long_haul": (int, 0, 100),
    "waste_kg_weekly": (float, 0, 500),
    "grocery_spend_monthly": (float, 0, 100_000),
    "internet_usage_hours": (float, 0, 24),
}

CATEGORY_RULES: dict[str, frozenset[str]] = {
    "vehicle_type": frozenset({"petrol", "diesel", "hybrid", "electric", "none"}),
    "diet_type": frozenset({"vegan", "vegetarian", "pescatarian", "non-vegetarian"}),
    "heating_source": frozenset({"natural gas", "oil", "electricity", "none"}),
}


@dataclass(frozen=True)
class PredictionInput:
    electricity_usage_kwh: float
    vehicle_type: str
    vehicle_km: float
    flights_short_haul: int
    flights_long_haul: int
    diet_type: str
    waste_kg_weekly: float
    household_size: int
    grocery_spend_monthly: float
    heating_source: str
    internet_usage_hours: float

    @classmethod
    def parse(cls, payload: Any) -> PredictionInput:
        if not isinstance(payload, dict):
            raise ValidationError({"body": "Must be a JSON object"})

        allowed = set(NUMERIC_RULES) | set(CATEGORY_RULES)
        errors: dict[str, str] = {}
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            errors["_schema"] = f"Unexpected fields: {', '.join(unknown)}"
        for name in missing:
            errors[name] = "Field is required"

        values: dict[str, int | float | str] = {}
        for name, (target_type, minimum, maximum) in NUMERIC_RULES.items():
            if name not in payload:
                continue
            raw = payload[name]
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                errors[name] = "Must be a number"
                continue
            value = float(raw)
            if not math.isfinite(value):
                errors[name] = "Must be finite"
            elif value < minimum or value > maximum:
                errors[name] = f"Must be between {minimum:g} and {maximum:g}"
            elif target_type is int and not value.is_integer():
                errors[name] = "Must be a whole number"
            else:
                values[name] = int(value) if target_type is int else value

        for name, choices in CATEGORY_RULES.items():
            if name not in payload:
                continue
            value = payload[name]
            if not isinstance(value, str) or value not in choices:
                errors[name] = f"Must be one of: {', '.join(sorted(choices))}"
            else:
                values[name] = value

        if errors:
            raise ValidationError(errors)
        return cls(
            electricity_usage_kwh=float(values["electricity_usage_kwh"]),
            vehicle_type=str(values["vehicle_type"]),
            vehicle_km=float(values["vehicle_km"]),
            flights_short_haul=int(values["flights_short_haul"]),
            flights_long_haul=int(values["flights_long_haul"]),
            diet_type=str(values["diet_type"]),
            waste_kg_weekly=float(values["waste_kg_weekly"]),
            household_size=int(values["household_size"]),
            grocery_spend_monthly=float(values["grocery_spend_monthly"]),
            heating_source=str(values["heating_source"]),
            internet_usage_hours=float(values["internet_usage_hours"]),
        )

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)
