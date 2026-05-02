from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .experiment_config import DEFAULT_EXPERIMENT, ExperimentConfig
from .paths import OUTPUT_DIR, SAMPLE_PATH


BASELINE_METRICS_PATH = OUTPUT_DIR / "baseline_metrics_by_condition.csv"
BASELINE_PREDICTIONS_PATH = OUTPUT_DIR / "baseline_predictions.csv"
BASELINE_PLOT_PATH = OUTPUT_DIR / "baseline_metrics_by_condition.png"


def _metrics_row(method: str, condition: str, y_true: pd.Series, y_pred, config: ExperimentConfig) -> dict:
    labels = list(config.labels)
    true_prepared = (y_true == labels[0]).mean()
    pred_prepared = (pd.Series(y_pred) == labels[0]).mean()
    return {
        "method": method,
        "condition": condition,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "true_prepared_rate": true_prepared,
        "pred_prepared_rate": pred_prepared,
        "prepared_rate_error": pred_prepared - true_prepared,
        "abs_prepared_rate_error": abs(pred_prepared - true_prepared),
    }


def _categorical_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                lambda df: list(df.columns),
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def _fold_count(y: pd.Series, requested_folds: int) -> int:
    min_class = y.value_counts().min()
    return max(2, min(requested_folds, int(min_class)))


def _cross_val_predictions(model, x: pd.DataFrame, y: pd.Series, folds: int, seed: int):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(model, x, y, cv=cv, method="predict")


def run_baselines(
    sample_path: Path = SAMPLE_PATH,
    output_dir: Path = OUTPUT_DIR,
    folds: int = 5,
    seed: int = 42,
    config: ExperimentConfig = DEFAULT_EXPERIMENT,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "baseline_metrics_by_condition.csv"
    predictions_path = output_dir / "baseline_predictions.csv"
    plot_path = output_dir / "baseline_metrics_by_condition.png"

    sample = pd.read_csv(sample_path, dtype=str)
    sample = sample[sample[config.target].isin(config.labels)].copy()
    y = sample[config.target]
    folds = _fold_count(y, folds)

    metrics_rows = []
    prediction_rows = []

    for condition, fields in config.disclosure_fields.items():
        x = sample[fields].fillna("Unknown").astype(str) if fields else pd.DataFrame(index=sample.index)

        models = {
            "majority": DummyClassifier(strategy="most_frequent"),
        }
        if fields:
            models.update(
                {
                    "logistic_regression": _categorical_pipeline(
                        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
                    ),
                    "random_forest": _categorical_pipeline(
                        RandomForestClassifier(
                            n_estimators=500,
                            random_state=seed,
                            class_weight="balanced",
                            min_samples_leaf=2,
                            n_jobs=-1,
                        )
                    ),
                }
            )

        for method, model in models.items():
            if method == "majority":
                y_pred = _cross_val_predictions(model, pd.DataFrame(index=sample.index), y, folds, seed)
            else:
                y_pred = _cross_val_predictions(model, x, y, folds, seed)

            metrics_rows.append(_metrics_row(method, condition, y, y_pred, config))
            for respondent_id, target, prediction in zip(sample["id"], y, y_pred):
                prediction_rows.append(
                    {
                        "id": respondent_id,
                        "method": method,
                        "condition": condition,
                        "target": target,
                        "prediction": prediction,
                        "folds": folds,
                        "seed": seed,
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    plot_baseline_metrics(metrics, plot_path)
    return metrics


def plot_baseline_metrics(metrics: pd.DataFrame, plot_path: Path = BASELINE_PLOT_PATH) -> None:
    if metrics.empty:
        return

    plt.figure(figsize=(8.5, 5))
    for method, subset in metrics.groupby("method"):
        ordered = subset.sort_values("condition")
        plt.plot(ordered["condition"], ordered["accuracy"], marker="o", label=f"{method} accuracy")
    plt.ylim(0, 1)
    plt.xlabel("Disclosure condition")
    plt.ylabel("Accuracy")
    plt.title("Baseline Accuracy by Disclosure Condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
