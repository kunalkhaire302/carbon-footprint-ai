# Model Card — Carbon Footprint Estimator v1

## Intended use

Educational estimation and product demonstration from eleven lifestyle fields. The output can help organize approximate categories and actions. It is not a certified carbon inventory, regulatory disclosure, financial basis, or proof of environmental performance.

## Model and data

- XGBoost regression pipeline with imputation, scaling, and one-hot encoding.
- Trained on 5,000 formula-generated synthetic samples.
- 4,250 train+validation samples and 750 untouched test samples.
- Dataset and artifact hashes are in `model/registry/v1/metadata.json`.

## Evaluation

Selection uses a seeded 70/15/15 split: validation chooses the candidate, the selected estimator is refit on train+validation, and the untouched test set is reported. Five-fold CV runs on the training split.

Actual v1 held-out test results: MAE 0.2129, RMSE 0.2771, R² 0.9866, MAPE 0.0315. These describe synthetic-target recovery only and must not be interpreted as real-world accuracy.

## Explainability

The API returns a deterministic category breakdown and reasons for recommendations. These are separate from the ML total. No statistical confidence interval is claimed; `confidence` is `null`.

## Limitations and ethics

The dataset lacks observational validation and may encode geographic, socioeconomic, and lifestyle assumptions. Inputs may be sensitive. The service does not retain them. Out-of-distribution inputs are bounded by schema but still may be poorly represented. Use neutral guidance; do not shame users or make consequential decisions from grades.
