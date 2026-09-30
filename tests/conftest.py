from __future__ import annotations

import pytest

from backend.app import create_app
from backend.app.config import TestSettings


@pytest.fixture()
def app():
    application = create_app(TestSettings.from_env())
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def valid_payload() -> dict[str, object]:
    return {
        "household_size": 2,
        "electricity_usage_kwh": 250,
        "heating_source": "none",
        "vehicle_type": "petrol",
        "vehicle_km": 800,
        "flights_short_haul": 1,
        "flights_long_haul": 0,
        "diet_type": "non-vegetarian",
        "grocery_spend_monthly": 300,
        "waste_kg_weekly": 15,
        "internet_usage_hours": 6,
    }
