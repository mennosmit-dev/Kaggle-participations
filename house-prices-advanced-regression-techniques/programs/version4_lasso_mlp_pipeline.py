"""
Version 4: single-flow preprocessing + LASSO selection + MLP regression pipeline.
"""

from __future__ import annotations

import argparse

from scipy.stats import loguniform
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from common import load_csv, make_preprocessor, prepare_test_features, save_submission, split_features_target


def build_pipeline(X):
    return Pipeline(
        steps=[
            ("prep", make_preprocessor(X)),
            (
                "select",
                SelectFromModel(
                    estimator=Lasso(max_iter=50000, tol=1e-3),
                    threshold=1e-8,
                ),
            ),
            (
                "mlp",
                MLPRegressor(
                    random_state=42,
                    max_iter=1200,
                    early_stopping=True,
                    n_iter_no_change=10,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a single-flow LASSO + MLP pipeline for Kaggle House Prices."
    )
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--output", required=True, help="Path to output submission CSV")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of parameter settings sampled by RandomizedSearchCV",
    )
    args = parser.parse_args()

    train_df = load_csv(args.train)
    test_df = load_csv(args.test)

    X, y = split_features_target(train_df)
    test_ids, X_test = prepare_test_features(test_df)

    pipe = build_pipeline(X)

    param_dist = {
        "select__estimator__alpha": loguniform(1e-5, 3e-1),
        "mlp__alpha": loguniform(1e-7, 1e-2),
        "mlp__learning_rate_init": loguniform(1e-4, 3e-2),
        "mlp__hidden_layer_sizes": [(32,), (64,), (128,), (256,), (128, 64), (256, 128)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__solver": ["adam"],
        "mlp__learning_rate": ["constant", "adaptive"],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    search.fit(X, y)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV RMSE (log scale): {-search.best_score_:.6f}")

    pred_log = search.best_estimator_.predict(X_test)
    save_submission(test_ids, pred_log, args.output)
    print(f"Submission written to: {args.output}")


if __name__ == "__main__":
    main()
