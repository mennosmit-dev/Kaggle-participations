# IN PROGRESS (UNDER CONSTRUCTION)


# House Prices - Advanced Regression Techniques

Examined two strategies to predict the price of a house based on approx 150 features:
- Gradient Boosting via TensorFlowDecisionForest library which is very practical since automatically deals with categorical and missing data, which occurs often in this dataset.
- LASSO regression, which required more preprocessing (like one-hot encoding), but is nice as it selects features, which could be combined with other methods as second step with only those features.
- (working on it)
- 
- Other intersting ideas? Perhaps PCA/kPCA first to extract majority information similar as applied in macro-economic modelling?


## Result (RMSE)
- Gradient Boosting: **0.138**
- Pure LASSO regression: **0.134**
- LASSO + Neural Network (2-step-approach): **0.858**
- LASSO-Neural Network (Single-flow apprach): **0.128**

-  
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
- preproces.....
-Run LASSO for variety of hyperparams, choose best model, apply on test data

## What I did for LASSO + Neural Network (2-step-approach):
- Run static LASSO model from above
- With the non-zero features, use a neural network with 2 hidden layers, try several hyperparams

## What I did for LASSO-Neural Network (Single-flow apprach):
- In a single pipeline first use lasso and then neural network (one flow combination)



## The competition
Link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
