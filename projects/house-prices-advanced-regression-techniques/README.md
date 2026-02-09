# House Prices - Advanced Regression Techniques

Examined two strategies to predict the price of a house based on approx 150 features:
- Gradient Boosting via TensorFlowDecisionForest library which is very practical since automatically deals with categorical and missing data, which occurs often in this dataset.
- LASSO regression, which required more preprocessing (like one-hot encoding), but is nice as it selects features, which could be combined with other methods as second step with only those features.
- Other intersting ideas? Perhaps PCA/kPCA first to extract majority information similar as applied in macro-economic modelling?


## Result
- Gradient Boosting: **0.13899**
- Lasso regression: ...

## Steps I undertook for Gradient Boosting
- Extracted, loaded and transformed data (very minimal transformation)
- Basic EDA and SalePrice (dep) distribution
- Train/validation split
- Converted pandas dataframe to TF datasets
- Trained tfdf.keras.GradientBoostedTreesModel
- Evaluated with validation dataset
- Checked feature importance
- Generated submission.csv

## What I did for LASSO




## The competition
Link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
