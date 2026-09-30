# Data Methodology

## Scope

`data/carbon_data.csv` contains 5,000 fully synthetic records generated with seed 42. It is not survey, utility, transport, purchase, or observational data. The target is a deterministic sum of configured factors plus Gaussian noise (`σ=0.15 tCO₂e/year`, floor `0.5`). Model metrics therefore measure how well an estimator recovers this generator.

## Inputs

Monthly electricity and driving, annual flights, weekly waste, monthly goods spend, daily internet use, household size, diet, vehicle, and heating categories are sampled from the distributions in `backend/dataset.py`.

## Factors and units

Runtime factors are centralized in `backend/app/domain/emission_factors.py`. Each scalar factor identifies its unit, geography, effective date, source label, and notes. Current values preserve legacy behavior and are explicitly labeled application assumptions pending authoritative source review. They must not be presented as certified or current regional factors.

## Calibration and bias

The generator is calibrated only by its author-defined distributions. It excludes housing type, renewable tariffs, vehicle efficiency, trip distance, radiative-forcing choices, supply-chain geography, food quantities, income, and uncertainty. Its category and geographic distribution can systematically misrepresent real people, especially outside the assumed context.

## Reproducibility

Run `python backend/dataset.py`; the committed dataset SHA-256 is stored in model metadata. Changing factors or distributions requires a new dataset version, model version, model card update, and regression review.
