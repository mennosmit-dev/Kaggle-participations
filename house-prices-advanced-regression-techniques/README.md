# House Prices – Advanced Regression Techniques

This repository contains a cleaned-up version of my work for the Kaggle competition **House Prices: Advanced Regression Techniques**.

The goal is to predict house sale prices from roughly 150 mixed-type features. I experimented with several regression pipelines, starting with a strong linear baseline and then extending it with neural-network models.

## Competition
Kaggle competition: House Prices – Advanced Regression Techniques

## Experiments

### Version 2 — LASSO regression
A regularized linear baseline with:
- median imputation for numeric features
- constant-value imputation for categorical features
- one-hot encoding
- feature scaling for numeric columns
- cross-validated `alpha` tuning

This is a strong and interpretable baseline because LASSO performs embedded feature selection.

### Version 3 — Two-step LASSO + Neural Network
A staged approach:
1. fit a LASSO-based feature selector on the training data
2. keep only the selected transformed features
3. train an `MLPRegressor` on that reduced feature space

This version makes the feature-selection stage explicit and separate from the neural network.

### Version 4 — Single-flow LASSO + Neural Network pipeline
An end-to-end scikit-learn pipeline that combines:
- preprocessing
- LASSO-based feature selection via `SelectFromModel`
- neural-network regression

This is cleaner to tune because the whole workflow can be optimized in one search object.

## Reported notebook results
These are the scores reported in the original notebook draft:

- Gradient Boosting: **0.138**
- Pure LASSO regression: **0.134**
- LASSO + Neural Network (two-step): **0.858**
- LASSO + Neural Network (single-flow): **0.128**

## Project structure

```text
house_prices_repo/
├── README.md
├── requirements.txt
└── src/
    ├── common.py
    ├── version2_lasso.py
    ├── version3_lasso_mlp_two_step.py
    └── version4_lasso_mlp_pipeline.py
```

## Usage

Place `train.csv` and `test.csv` from the Kaggle competition in a local data directory, then run one of the scripts below.

### Version 2
```bash
python src/version2_lasso.py --train data/train.csv --test data/test.csv --output submissions/submission_lasso.csv
```

### Version 3
```bash
python src/version3_lasso_mlp_two_step.py --train data/train.csv --test data/test.csv --output submissions/submission_lasso_mlp_two_step.csv
```

### Version 4
```bash
python src/version4_lasso_mlp_pipeline.py --train data/train.csv --test data/test.csv --output submissions/submission_lasso_mlp_pipeline.csv
```

## Notes
- The scripts train on `log1p(SalePrice)` and convert predictions back with `expm1`.
- Sparse columns from the notebook are dropped consistently to mirror the original experiments.
- Some incomplete notebook fragments were turned into clean, runnable code where needed.
