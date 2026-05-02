import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .experiment_config import DEFAULT_EXPERIMENT, ExperimentConfig
from .paths import DEFAULT_PREDICTIONS_PATH, OUTPUT_DIR, SAMPLE_PATH


GROUP_FIELDS = ["age", "education", "income_agg", "disability"]


def output_paths(predictions_path: Path, output_prefix: str | None) -> tuple[Path, Path, Path, Path]:
    if output_prefix:
        prefix = output_prefix
    elif predictions_path.name == DEFAULT_PREDICTIONS_PATH.name:
        prefix = ""
    else:
        prefix = predictions_path.stem.replace("predictions_", "")

    suffix = f"_{prefix}" if prefix else ""
    return (
        OUTPUT_DIR / f"metrics_by_condition{suffix}.csv",
        OUTPUT_DIR / f"metrics_by_group{suffix}.csv",
        OUTPUT_DIR / f"classification_report_by_condition{suffix}.json",
        OUTPUT_DIR / f"metrics_by_condition{suffix}.png",
    )


def evaluate_predictions(
    predictions_path: Path,
    output_prefix: str | None = None,
    config: ExperimentConfig = DEFAULT_EXPERIMENT,
) -> pd.DataFrame:
    if not predictions_path.is_absolute():
        predictions_path = OUTPUT_DIR / predictions_path
    if not predictions_path.exists():
        raise SystemExit(f"No predictions found at {predictions_path}. Run run_llm.py first.")

    summary_path, group_summary_path, report_path, plot_path = output_paths(predictions_path, output_prefix)

    labels = list(config.labels)
    df = pd.read_json(predictions_path, lines=True)
    df = df[df["target"].isin(labels)]
    df = df[df["prediction"].isin(labels)]

    rows = []
    reports = {}
    for condition in config.disclosure_fields:
        subset = df[df["condition"] == condition].copy()
        if subset.empty:
            continue

        y_true = subset["target"]
        y_pred = subset["prediction"]
        true_first_label = (y_true == labels[0]).mean()
        pred_first_label = (y_pred == labels[0]).mean()
        rows.append(
            {
                "condition": condition,
                "n": len(subset),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                "true_prepared_rate": true_first_label,
                "pred_prepared_rate": pred_first_label,
                "prepared_rate_error": pred_first_label - true_first_label,
                "abs_prepared_rate_error": abs(pred_first_label - true_first_label),
            }
        )
        reports[condition] = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(summary_path, index=False)
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    if SAMPLE_PATH.exists():
        sample = pd.read_csv(SAMPLE_PATH, dtype={"id": str})
        df["id"] = df["id"].astype(str)
        group_columns = ["id", *[g for g in GROUP_FIELDS if g in sample.columns]]
        merged = df.merge(sample[group_columns], on="id", how="left")
        group_rows = []
        for condition in config.disclosure_fields:
            condition_df = merged[merged["condition"] == condition]
            for group_field in GROUP_FIELDS:
                if group_field not in condition_df.columns:
                    continue
                for group_value, subset in condition_df.groupby(group_field, dropna=False):
                    if len(subset) < 5:
                        continue
                    y_true = subset["target"]
                    y_pred = subset["prediction"]
                    true_first_label = (y_true == labels[0]).mean()
                    pred_first_label = (y_pred == labels[0]).mean()
                    group_rows.append(
                        {
                            "condition": condition,
                            "group_field": group_field,
                            "group_value": group_value,
                            "n": len(subset),
                            "accuracy": accuracy_score(y_true, y_pred),
                            "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                            "prepared_rate_error": pred_first_label - true_first_label,
                            "abs_prepared_rate_error": abs(pred_first_label - true_first_label),
                        }
                    )
        pd.DataFrame(group_rows).to_csv(group_summary_path, index=False)

    if not metrics.empty:
        plt.figure(figsize=(8, 4.5))
        plt.plot(metrics["condition"], metrics["accuracy"], marker="o", label="Accuracy")
        plt.plot(metrics["condition"], metrics["macro_f1"], marker="o", label="Macro-F1")
        plt.plot(
            metrics["condition"],
            metrics["abs_prepared_rate_error"],
            marker="o",
            label="Abs prepared-rate error",
        )
        plt.ylim(0, 1)
        plt.xlabel("Disclosure condition")
        plt.ylabel("Metric")
        plt.title("LLM Simulation Performance by Disclosure Condition")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

    print(metrics.to_string(index=False))
    print(f"Wrote metrics: {summary_path}")
    if SAMPLE_PATH.exists():
        print(f"Wrote group metrics: {group_summary_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote plot: {plot_path}")
    return metrics

