"""
Gradient Boosted Trees to predict housing prices - this code was inspired by the idea from (Gusthema, Kin) who used Random Forests instead which scored RMSE = 0.14665; and mine 0.13899 ;)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_decision_forests as tfdf


KAGGLE_INPUT_DIR = Path("../input/house-prices-advanced-regression-techniques")
TRAIN_CSV = KAGGLE_INPUT_DIR / "train.csv"
TEST_CSV = KAGGLE_INPUT_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = KAGGLE_INPUT_DIR / "sample_submission.csv"
KAGGLE_WORKING_DIR = Path("/kaggle/working")
SUBMISSION_OUT = KAGGLE_WORKING_DIR / "submission.csv"

LABEL_COL = "SalePrice"
ID_COL = "Id"


def print_versions() -> None:
    """Print TensorFlow and TF-DF versions."""
    print(f"TensorFlow v{tf.__version__}")
    print(f"TensorFlow Decision Forests v{tfdf.__version__}")


def load_train_dataframe(path: Path) -> pd.DataFrame:
    """Load training data and print shape."""
    df = pd.read_csv(path)
    print(f"Full train dataset shape is {df.shape}")
    return df


def split_dataset(
    dataset: pd.DataFrame,
    test_ratio: float = 0.30,
    seed: int | None = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random split of a dataframe into train/validation.

    Uses a random mask (similar to your notebook). Seed is optional.
    """
    rng = np.random.default_rng(seed)
    test_indices = rng.random(len(dataset)) < test_ratio
    train_df = dataset.loc[~test_indices].copy()
    valid_df = dataset.loc[test_indices].copy()
    return train_df, valid_df


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Plot SalePrice distribution."""
    print(df[LABEL_COL].describe())
    plt.figure(figsize=(9, 8))
    # distplot is deprecated; histplot is the modern alternative.
    sns.histplot(df[LABEL_COL], bins=100, kde=True)
    plt.title("SalePrice distribution")
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Plot histograms for numeric features."""
    print(list(set(df.dtypes.tolist())))
    df_num = df.select_dtypes(include=["float64", "int64"])
    _ = df_num.head()
    df_num.hist(figsize=(16, 20), bins=50)
    plt.suptitle("Numeric feature distributions")
    plt.tight_layout()
    plt.show()


def to_tfdf_dataset(
    df: pd.DataFrame,
    label: str | None = None,
) -> tf.data.Dataset:
    """Convert pandas dataframe to TF-DF dataset."""
    if label is None:
        return tfdf.keras.pd_dataframe_to_tf_dataset(
            df,
            task=tfdf.keras.Task.REGRESSION,
        )
    return tfdf.keras.pd_dataframe_to_tf_dataset(
        df,
        label=label,
        task=tfdf.keras.Task.REGRESSION,
    )


def plot_training_logs_rmse(model: tfdf.keras.CoreModel) -> None:
    """Plot RMSE from TF-DF training logs vs number of trees."""
    logs = model.make_inspector().training_logs()
    plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("RMSE (training logs)")
    plt.title("RMSE vs number of trees")
    plt.tight_layout()
    plt.show()


def print_variable_importances(inspector: tfdf.inspector.AbstractInspector) -> None:
    """Print available variable importance keys."""
    print("Available variable importances:")
    for importance in inspector.variable_importances().keys():
        print("\t", importance)


def plot_variable_importance_num_as_root(
    inspector: tfdf.inspector.AbstractInspector,
    top_n: int = 30,
) -> None:
    """
    Plot variable importance for NUM_AS_ROOT.

    If the list is long, plotting top_n keeps the chart readable.
    """
    metric = "NUM_AS_ROOT"
    if metric not in inspector.variable_importances():
        print(f'Variable importance "{metric}" not available for this model/run.')
        return

    variable_importances = inspector.variable_importances()[metric]
    variable_importances = variable_importances[:top_n]

    feature_names = [vi[0].name for vi in variable_importances]
    importance_values = [vi[1] for vi in variable_importances]
    feature_ranks = range(len(feature_names))

    plt.figure(figsize=(12, 6))
    bars = plt.barh(feature_ranks, importance_values)
    plt.yticks(feature_ranks, feature_names)
    plt.gca().invert_yaxis()

    for value, patch in zip(importance_values, bars.patches):
        plt.text(
            patch.get_x() + patch.get_width(),
            patch.get_y(),
            f"{value:.4f}",
            va="top",
        )

    plt.xlabel(metric)
    plt.title("Variable importance: NUM_AS_ROOT")
    plt.tight_layout()
    plt.show()


def write_submission(
    model: tfdf.keras.CoreModel,
    test_csv: Path,
    out_path: Path,
    sample_submission_csv: Path | None = None,
) -> None:
    """
    Create and write Kaggle submission.csv.

    - Loads test.csv
    - Pops Id
    - Predicts SalePrice
    - Writes /kaggle/working/submission.csv
    """
    test_data = pd.read_csv(test_csv)
    ids = test_data.pop(ID_COL)

    test_ds = to_tfdf_dataset(test_data, label=None)
    preds = model.predict(test_ds).squeeze()

    if sample_submission_csv is not None and sample_submission_csv.exists():
        sub = pd.read_csv(sample_submission_csv)
        sub[LABEL_COL] = preds
    else:
        sub = pd.DataFrame({ID_COL: ids, LABEL_COL: preds})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to: {out_path}")


def main() -> None:
    print_versions()

    # Load training data
    dataset_df = load_train_dataframe(TRAIN_CSV)

    # Preview + basic info
    print(dataset_df.head(3))
    print(
        "There are 79 feature columns. Using these features your model has to "
        f"predict the house sale price indicated by the label column named {LABEL_COL}."
    )

    # Drop Id
    dataset_df = dataset_df.drop(ID_COL, axis=1)
    print(dataset_df.head(3))

    # Inspect dtypes
    dataset_df.info()

    # EDA
    plot_target_distribution(dataset_df)
    plot_numeric_distributions(dataset_df)

    # Split train/valid
    train_df, valid_df = split_dataset(dataset_df, test_ratio=0.30, seed=42)
    print(f"{len(train_df)} examples in training, {len(valid_df)} examples in testing.")

    # Convert to TF-DF datasets
    train_ds = to_tfdf_dataset(train_df, label=LABEL_COL)
    valid_ds = to_tfdf_dataset(valid_df, label=LABEL_COL)

    # List models (optional)
    print("Available TF-DF models:")
    print(tfdf.keras.get_all_models())

    # Model (note: despite the notebook text, we use Gradient Boosted Trees)
    model = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION)
    model.compile(metrics=["mse"])  # optional
    model.fit(x=train_ds)

    # Visualize one tree (works in notebooks; may not render in plain scripts)
    try:
        tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0, max_depth=3)
    except Exception as exc:  # noqa: BLE001
        print(f"Model plot skipped (notebook-only rendering). Reason: {exc}")

    # Training logs plot
    plot_training_logs_rmse(model)

    # Inspector eval + validation eval
    inspector = model.make_inspector()
    print("Inspector evaluation:")
    print(inspector.evaluation())

    evaluation = model.evaluate(x=valid_ds, return_dict=True)
    print("Validation evaluation:")
    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")

    # Variable importances
    print_variable_importances(inspector)
    if "NUM_AS_ROOT" in inspector.variable_importances():
        print('Top variable importances for "NUM_AS_ROOT":')
        print(inspector.variable_importances()["NUM_AS_ROOT"][:20])
    plot_variable_importance_num_as_root(inspector, top_n=30)

    # Submission
    write_submission(
        model=model,
        test_csv=TEST_CSV,
        out_path=SUBMISSION_OUT,
        sample_submission_csv=SAMPLE_SUBMISSION_CSV,
    )


if __name__ == "__main__":
    main()
