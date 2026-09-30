from __future__ import annotations

import pytest

from backend.app.domain.calculations import calculate_breakdown
from backend.app.domain.grading import calculate_grade
from backend.app.schemas.prediction import PredictionInput


def test_reference_breakdown(valid_payload):
    breakdown = calculate_breakdown(PredictionInput.parse(valid_payload))
    assert breakdown == {
        "Transport": 2.07,
        "Electricity": 0.6,
        "Diet": 2.3,
        "Goods": 1.8,
        "Waste": 0.04,
        "Digital": 0.18,
    }


@pytest.mark.parametrize(
    ("value", "grade"),
    [(0, "A"), (1.9, "B"), (2.8499, "B"), (2.85, "C"), (4.7, "D"), (7.0499, "D"), (7.05, "F"), (100, "F")],
)
def test_grade_boundaries(value, grade):
    assert calculate_grade(value)[1] == grade
