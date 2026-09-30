# ML Pipeline

1. `python backend/dataset.py` regenerates the seeded synthetic dataset.
2. `python -m scripts.train_model` creates a 70/15/15 split and compares linear regression, random forest, and XGBoost.
3. Five-fold cross-validation records R² dispersion, MAE, and RMSE on the training split.
4. Validation R² selects the candidate; the winner is refit on train+validation.
5. MAE, RMSE, R², and MAPE are calculated once on the untouched test set.
6. One complete preprocessing+model pipeline is saved at `model/registry/v1/pipeline.joblib`.
7. Metadata records versions, features, hashes, metrics, sample counts, methodology, and commit SHA.
8. `python -m scripts.evaluate_model` performs a diagnostic full-dataset check; it is not a held-out metric.

Training never runs in the API process. Create a new registry version for material data, factor, feature, or estimator changes; do not overwrite a deployed version silently.
