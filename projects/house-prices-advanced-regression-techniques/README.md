# House Prices - Advanced Regression Techniques

Examined two strategies to predict the price of a house based on approx 150 features.
(1) Gradient Boosting via TensorFlowDecisionForest library which is very practical since automatically deals with the categorical and missing data.
(2) LASSO regression, which required more preprocessing (like one-hot encoding), but is nice as it selects features, which could be combined with other methods as second step with only those features.

Kaggle competition:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Result
Best public score: **0.13899**

## What I did

- Loaded train.csv and dropped Id column
- Basic EDA and SalePrice distribution
- Random train/validation split
- Converted pandas dataframe → TF datasets
- Trained tfdf.keras.GradientBoostedTreesModel
- Evaluated with validation dataset
- Checked feature importance
- Generated submission.csv

## Model

TensorFlow Decision Forests:
- GradientBoostedTreesModel
- Regression task
- Minimal preprocessing required

## Repo

https://github.com/mennosmit-dev/Kaggle-participations/tree/main/projects/house-prices-advanced-regression-techniques
