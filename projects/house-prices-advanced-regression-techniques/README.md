# House Prices - Advanced Regression Techniques

Kaggle competition:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Result
Best public score: **0.13899**

NOTE:
The notebook text says Random Forest, but I actually trained a
**Gradient Boosted Trees model** using TensorFlow Decision Forests.

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
