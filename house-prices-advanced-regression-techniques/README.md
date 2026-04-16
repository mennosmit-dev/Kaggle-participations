# House Prices – Advanced Regression Techniques

This project explores multiple machine learning approaches to predict house prices using the Kaggle dataset:

👉 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The dataset contains ~150 features, including numerical and categorical variables with missing values. Different modeling strategies were tested, ranging from tree-based methods to linear models and neural networks.

---

## 📊 Results (RMSE)

| Model                                      | RMSE  |
|-------------------------------------------|------:|
| Gradient Boosting (TF-DF)                 | 0.138 |
| LASSO Regression                         | 0.134 |
| LASSO + Neural Network (2-step)           | 0.858 |
| LASSO + Neural Network (Single pipeline)  | 0.128 |

---

## 🚀 Approaches

### 1. Gradient Boosting (TensorFlow Decision Forests)

File: `src/version1_tfdf_gradient_boosting.py`

This approach uses **TensorFlow Decision Forests (TF-DF)** to train a Gradient Boosted Trees model.

**Why this approach:**
- Handles categorical variables automatically
- No need for extensive preprocessing
- Robust to missing values
- Strong baseline with minimal effort

**Steps:**
- Load and inspect dataset
- Minimal preprocessing (drop `Id`)
- Basic EDA (target distribution + numeric features)
- Train/validation split (random mask)
- Convert pandas → TF dataset
- Train `GradientBoostedTreesModel`
- Evaluate on validation set
- Analyze feature importance
- Generate Kaggle submission

Code reference: :contentReference[oaicite:0]{index=0}

---

### 2. LASSO Regression

File: `src/version2_lasso.py`

A linear model with L1 regularization used for **feature selection and prediction**.

**Key steps:**
- Data preprocessing:
  - Handle missing values
  - One-hot encode categorical variables
- Feature scaling
- Hyperparameter tuning (`alpha`)
- Model training and validation
- Prediction on test set

**Why LASSO:**
- Automatically selects relevant features
- Reduces overfitting
- Good interpretability

---

### 3. LASSO + Neural Network (2-Step Approach)

File: `src/version3_lasso_mlp_two_step.py`

A sequential modeling approach:

**Pipeline:**
1. Train LASSO → select non-zero features  
2. Train a neural network (MLP) on selected features  

**Neural network:**
- 2 hidden layers
- Tuned hyperparameters (neurons, learning rate, etc.)

**Observation:**
- Did not perform well in this setup (likely due to feature compression + NN mismatch)

---

### 4. LASSO + Neural Network (Single Pipeline)

File: `src/version4_lasso_mlp_pipeline.py`

A unified pipeline combining:
- Feature selection (LASSO-like behavior)
- Neural network training in a single flow

**Why this works better:**
- End-to-end optimization
- Avoids information loss from strict feature filtering
- Better synergy between linear and non-linear modeling


---

## 🧠 Future Ideas

Some potential improvements and experiments:

- PCA / Kernel PCA for dimensionality reduction
- Ensemble methods (blend TF-DF + LASSO + NN)
- Stacking models
- Feature engineering (domain-driven)
- LightGBM / XGBoost comparison
- Cross-validation instead of single split

---
