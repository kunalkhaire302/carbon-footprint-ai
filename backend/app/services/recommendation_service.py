from __future__ import annotations

from typing import cast

RECOMMENDATIONS = {
    "Transport": [
        ("Switch to public transport or an EV", 0.4, "Hard"),
        ("Carpool twice a week", 0.15, "Medium"),
        ("Avoid one long-haul flight", 0.3, "Medium"),
    ],
    "Electricity": [
        ("Switch to lower-carbon electricity where available", 0.5, "Hard"),
        ("Replace remaining bulbs with LEDs", 0.05, "Easy"),
        ("Reduce heating demand with thermostat scheduling", 0.1, "Medium"),
    ],
    "Diet": [
        ("Shift toward a plant-rich diet", 0.4, "Medium"),
        ("Choose three meat-free days each week", 0.15, "Easy"),
        ("Prefer seasonal produce", 0.05, "Easy"),
    ],
    "Goods": [("Buy durable goods second-hand", 0.2, "Easy"), ("Reduce fast-fashion purchases", 0.15, "Easy")],
    "Waste": [("Compost suitable organic waste", 0.3, "Medium"), ("Separate recyclable materials", 0.2, "Easy")],
    "Digital": [
        ("Remove unused cloud files and subscriptions", 0.2, "Easy"),
        ("Use HD instead of 4K when suitable", 0.4, "Easy"),
    ],
}


def generate_recommendations(breakdown: dict[str, float]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for category, amount in sorted(breakdown.items(), key=lambda item: item[1], reverse=True)[:3]:
        if amount < 0.2:
            continue
        for action, ratio, difficulty in RECOMMENDATIONS.get(category, []):
            saving = round(amount * ratio, 2)
            if saving > 0.05:
                ranked.append(
                    {
                        "category": category,
                        "action": action,
                        "estimated_saving_tco2e": saving,
                        "co2_saved_tyr": saving,
                        "difficulty": difficulty,
                        "impact": "High" if ratio >= 0.3 else "Medium" if ratio >= 0.15 else "Low",
                        "reason": f"{category} is one of this estimate's largest categories.",
                        "is_estimate": True,
                    }
                )
    ranked.sort(key=lambda item: cast(float, item["estimated_saving_tco2e"]), reverse=True)
    for index, item in enumerate(ranked[:5], 1):
        item["rank"] = index
    return ranked[:5]
