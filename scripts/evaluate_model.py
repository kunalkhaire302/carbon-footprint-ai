from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "carbon_data.csv")
    pipeline = joblib.load(ROOT / "model" / "registry" / "v1" / "pipeline.joblib")
    target = data.pop("total_footprint_tco2e")
    predicted = pipeline.predict(data)
    print(
        json.dumps(
            {
                "dataset_scope": "full synthetic dataset; diagnostic only, not held-out performance",
                "mae": float(mean_absolute_error(target, predicted)),
                "rmse": float(mean_squared_error(target, predicted) ** 0.5),
                "r2": float(r2_score(target, predicted)),
                "mape": float(mean_absolute_percentage_error(target, predicted)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
