from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "carbon_data.csv"
TARGET = "total_footprint_tco2e"
SEED = 42
MODEL_VERSION = "v1"
REGISTRY = ROOT / "model" / "registry" / MODEL_VERSION


def build_pipeline(model: object, numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def git_sha() -> str | None:
    return os.getenv("GITHUB_SHA") or os.getenv("RENDER_GIT_COMMIT")


def main() -> None:
    frame = pd.read_csv(DATASET)
    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.50, random_state=SEED
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=1),
        "xgboost": XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=1,
        ),
    }
    results: dict[str, object] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator, numeric, categorical)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        validation = metrics(y_validation, pipeline.predict(X_validation))
        results[name] = {
            "validation": validation,
            "cross_validation": {
                "folds": 5,
                "r2_mean": float(scores["test_r2"].mean()),
                "r2_std": float(scores["test_r2"].std()),
                "mae_mean": float(-scores["test_neg_mean_absolute_error"].mean()),
                "rmse_mean": float(-scores["test_neg_root_mean_squared_error"].mean()),
            },
        }
        fitted[name] = pipeline

    winner = max(results, key=lambda name: results[name]["validation"]["r2"])
    final = fitted[winner]
    final.fit(pd.concat([X_train, X_validation]), pd.concat([y_train, y_validation]))
    results[winner]["test"] = metrics(y_test, final.predict(X_test))

    REGISTRY.mkdir(parents=True, exist_ok=True)
    artifact = REGISTRY / "pipeline.joblib"
    joblib.dump(final, artifact)
    dataset_hash = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    metadata = {
        "model_version": MODEL_VERSION,
        "model_name": winner,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "dataset_version": dataset_hash[:12],
        "dataset_sha256": dataset_hash,
        "features": list(X.columns),
        "training_samples": len(X_train) + len(X_validation),
        "test_samples": len(X_test),
        "validation_methodology": (
            "70/15/15 seeded split; model selection on validation; final report on untouched test; "
            "5-fold CV on training split"
        ),
        "metrics": results[winner],
        "git_commit_sha": git_sha(),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "limitations": (
            "Trained entirely on formula-generated synthetic data; metrics do not establish real-world accuracy."
        ),
    }
    (REGISTRY / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REGISTRY / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"model": winner, "version": MODEL_VERSION, "test": results[winner]["test"]}, indent=2))


if __name__ == "__main__":
    main()
