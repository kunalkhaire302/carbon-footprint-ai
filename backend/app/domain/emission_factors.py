from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmissionFactor:
    value: float
    unit: str
    geography: str
    effective_date: str
    source: str
    notes: str = ""


# These preserve the legacy calculator. Sources require independent scientific review.
ELECTRICITY = EmissionFactor(0.4, "kgCO2e/kWh", "India proxy", "2026-04-12", "legacy application assumption")
WASTE = EmissionFactor(0.05, "kgCO2e/kg waste", "unspecified", "2026-04-12", "legacy application assumption")
GOODS = EmissionFactor(0.001, "tCO2e/USD", "unspecified", "2026-04-12", "legacy spend proxy")
DIGITAL = EmissionFactor(0.03, "tCO2e/year per daily hour", "unspecified", "2026-04-12", "legacy usage proxy")
VEHICLE_KG_PER_KM = {"petrol": 0.2, "diesel": 0.23, "hybrid": 0.12, "electric": 0.05, "none": 0.0}
FLIGHT_TCO2E_PER_TRIP = {"short": 0.15, "long": 0.8}
DIET_TCO2E_PER_YEAR = {"vegan": 1.0, "vegetarian": 1.3, "pescatarian": 1.6, "non-vegetarian": 2.3}
HEATING_TCO2E_PER_YEAR = {"natural gas": 1.2, "oil": 1.8, "electricity": 0.8, "none": 0.0}
