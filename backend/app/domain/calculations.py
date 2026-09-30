from __future__ import annotations

from backend.app.domain.emission_factors import (
    DIET_TCO2E_PER_YEAR,
    DIGITAL,
    ELECTRICITY,
    FLIGHT_TCO2E_PER_TRIP,
    GOODS,
    HEATING_TCO2E_PER_YEAR,
    VEHICLE_KG_PER_KM,
    WASTE,
)
from backend.app.schemas.prediction import PredictionInput


def calculate_breakdown(data: PredictionInput) -> dict[str, float]:
    household = data.household_size
    electricity = data.electricity_usage_kwh * 12 * ELECTRICITY.value / 1000 / household
    vehicle = VEHICLE_KG_PER_KM[data.vehicle_type] * data.vehicle_km * 12 / 1000
    flights = (
        data.flights_short_haul * FLIGHT_TCO2E_PER_TRIP["short"]
        + data.flights_long_haul * FLIGHT_TCO2E_PER_TRIP["long"]
    )
    heating = HEATING_TCO2E_PER_YEAR[data.heating_source] / household
    return {
        "Transport": round(vehicle + flights, 2),
        "Electricity": round(electricity + heating, 2),
        "Diet": round(DIET_TCO2E_PER_YEAR[data.diet_type], 2),
        "Goods": round(data.grocery_spend_monthly * 12 * GOODS.value / household, 2),
        "Waste": round(data.waste_kg_weekly * 52 * WASTE.value / 1000, 2),
        "Digital": round(data.internet_usage_hours * DIGITAL.value, 2),
    }
