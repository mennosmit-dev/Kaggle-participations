"""
Version 2: pure LASSO regression baseline.
"""

from __future__ import annotations

import argparse

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from common import load_csv, make_preprocessor, prepare_test_features, save_submission, split_features_target


def build_pipeline(X):
    return Pipeline(
        steps=[
            ("prep", make_preprocessor(X)),
            ("lasso", Lasso(max_iter=50000, tol=1e-3)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LASSO model for Kaggle House Prices.")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--output", required=True, help="Path to output submission CSV")
    args = parser.parse_args()

    train_df = load_csv(args.train)
    test_df = load_csv(args.test)

    X, y = split_features_target(train_df)
    test_ids, X_test = prepare_test_features(test_df)

    pipe = build_pipeline(X)

    param_grid = {
        "lasso__alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X, y)

    print(f"Best alpha: {grid.best_params_['lasso__alpha']}")
    print(f"Best CV RMSE (log scale): {-grid.best_score_:.6f}")

    pred_log = grid.best_estimator_.predict(X_test)
    save_submission(test_ids, pred_log, args.output)
    print(f"Submission written to: {args.output}")


if __name__ == "__main__":
    main()
