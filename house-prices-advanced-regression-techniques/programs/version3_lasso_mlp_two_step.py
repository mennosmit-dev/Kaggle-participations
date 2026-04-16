"""
Version 3: explicit two-step LASSO feature selection followed by MLP regression.
"""

from __future__ import annotations

import argparse

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor

from common import load_csv, make_preprocessor, prepare_test_features, save_submission, split_features_target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a two-step LASSO + MLP model for Kaggle House Prices."
    )
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--output", required=True, help="Path to output submission CSV")
    args = parser.parse_args()

    train_df = load_csv(args.train)
    test_df = load_csv(args.test)

    X, y = split_features_target(train_df)
    test_ids, X_test = prepare_test_features(test_df)

    preprocess = make_preprocessor(X)
    X_prepared = preprocess.fit_transform(X)
    X_test_prepared = preprocess.transform(X_test)

    lasso_grid = GridSearchCV(
        estimator=Lasso(max_iter=50000, tol=1e-3),
        param_grid={"alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]},
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1,
    )
    lasso_grid.fit(X_prepared, y)

    best_alpha = lasso_grid.best_params_["alpha"]
    print(f"Best LASSO alpha: {best_alpha}")
    print(f"Best LASSO CV RMSE (log scale): {-lasso_grid.best_score_:.6f}")

    selector = SelectFromModel(
        estimator=Lasso(alpha=best_alpha, max_iter=50000, tol=1e-3),
        threshold=1e-8,
    )
    selector.fit(X_prepared, y)

    X_selected = selector.transform(X_prepared)
    X_test_selected = selector.transform(X_test_prepared)

    print(f"Selected features: {X_selected.shape[1]}")

    mlp = MLPRegressor(
        random_state=42,
        max_iter=1200,
        early_stopping=True,
        n_iter_no_change=15,
    )

    mlp_grid = GridSearchCV(
        estimator=mlp,
        param_grid={
            "hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [1e-3, 3e-4],
            "activation": ["relu", "tanh"],
        },
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1,
    )
    mlp_grid.fit(X_selected, y)

    print(f"Best MLP params: {mlp_grid.best_params_}")
    print(f"Best MLP CV RMSE (log scale): {-mlp_grid.best_score_:.6f}")

    pred_log = mlp_grid.best_estimator_.predict(X_test_selected)
    save_submission(test_ids, pred_log, args.output)
    print(f"Submission written to: {args.output}")


if __name__ == "__main__":
    main()
