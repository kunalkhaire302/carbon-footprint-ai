from __future__ import annotations

from backend.app.domain.benchmarks import INDIA_AVG, WORLD_AVG
from backend.app.domain.calculations import calculate_breakdown
from backend.app.domain.grading import calculate_grade
from backend.app.ml.model_loader import ModelManager
from backend.app.schemas.prediction import PredictionInput
from backend.app.services.recommendation_service import generate_recommendations


def create_prediction(data: PredictionInput, model: ModelManager, request_id: str) -> dict[str, object]:
    total, prediction_ms = model.predict(data)
    breakdown = calculate_breakdown(data)
    percentile, letter = calculate_grade(total)
    recommendations = generate_recommendations(breakdown)
    india = INDIA_AVG
    world = WORLD_AVG
    return {
        "prediction": {"total_tco2e_per_year": total, "confidence": None},
        "grade": {"letter": letter, "label": "Relative grade based on configured benchmark thresholds"},
        "breakdown": [{"category": key, "tco2e_per_year": value} for key, value in breakdown.items()],
        "benchmark": {"india": india, "global": world, "percentile_proxy": percentile},
        "recommendations": recommendations,
        "model": {"version": model.version, "prediction_type": "synthetic-data regression estimate"},
        "metadata": {"request_id": request_id, "prediction_time_ms": prediction_ms, "api_version": "v1"},
        # Compatibility contract for the existing frontend and /predict clients.
        "total_footprint_tco2e": total,
        "category_breakdown": breakdown,
        "comparison": {
            "india_avg": india,
            "world_avg": world,
            "your_value": total,
            "percentile": percentile,
            "grade": letter,
        },
        "suggestions": recommendations,
    }
