import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "work" / "outputs"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"
SAMPLE_PATH = OUTPUT_DIR / "sample.csv"
SUMMARY_PATH = OUTPUT_DIR / "metrics_by_condition.csv"
GROUP_SUMMARY_PATH = OUTPUT_DIR / "metrics_by_group.csv"
REPORT_PATH = OUTPUT_DIR / "classification_report_by_condition.json"
PLOT_PATH = OUTPUT_DIR / "metrics_by_condition.png"

VALID_LABELS = ["Prepared Individuals", "Unprepared Individuals"]
CONDITION_ORDER = ["C0", "C1", "C2", "C3"]
GROUP_FIELDS = ["age", "education", "income_agg", "disability"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(PREDICTIONS_PATH), help="Prediction JSONL path.")
    parser.add_argument("--output-prefix", default=None, help="Output prefix for metrics files.")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    if not predictions_path.is_absolute():
        predictions_path = OUTPUT_DIR / predictions_path
    if not predictions_path.exists():
        raise SystemExit(f"No predictions found at {predictions_path}. Run run_llm.py first.")

    if args.output_prefix:
        prefix = args.output_prefix
    elif predictions_path.name == PREDICTIONS_PATH.name:
        prefix = ""
    else:
        prefix = predictions_path.stem.replace("predictions_", "")

    summary_path = OUTPUT_DIR / f"metrics_by_condition{('_' + prefix) if prefix else ''}.csv"
    group_summary_path = OUTPUT_DIR / f"metrics_by_group{('_' + prefix) if prefix else ''}.csv"
    report_path = OUTPUT_DIR / f"classification_report_by_condition{('_' + prefix) if prefix else ''}.json"
    plot_path = OUTPUT_DIR / f"metrics_by_condition{('_' + prefix) if prefix else ''}.png"

    df = pd.read_json(predictions_path, lines=True)
    df = df[df["target"].isin(VALID_LABELS)]
    df = df[df["prediction"].isin(VALID_LABELS)]

    rows = []
    reports = {}
    for condition in CONDITION_ORDER:
        subset = df[df["condition"] == condition].copy()
        if subset.empty:
            continue

        y_true = subset["target"]
        y_pred = subset["prediction"]
        true_prepared = (y_true == "Prepared Individuals").mean()
        pred_prepared = (y_pred == "Prepared Individuals").mean()
        rows.append(
            {
                "condition": condition,
                "n": len(subset),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, labels=VALID_LABELS, average="macro"),
                "true_prepared_rate": true_prepared,
                "pred_prepared_rate": pred_prepared,
                "prepared_rate_error": pred_prepared - true_prepared,
                "abs_prepared_rate_error": abs(pred_prepared - true_prepared),
            }
        )
        reports[condition] = classification_report(
            y_true,
            y_pred,
            labels=VALID_LABELS,
            output_dict=True,
            zero_division=0,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(summary_path, index=False)
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    if SAMPLE_PATH.exists():
        sample = pd.read_csv(SAMPLE_PATH, dtype={"id": str})
        df["id"] = df["id"].astype(str)
        merged = df.merge(sample[["id", *[g for g in GROUP_FIELDS if g in sample.columns]]], on="id", how="left")
        group_rows = []
        for condition in CONDITION_ORDER:
            condition_df = merged[merged["condition"] == condition]
            for group_field in GROUP_FIELDS:
                if group_field not in condition_df.columns:
                    continue
                for group_value, subset in condition_df.groupby(group_field, dropna=False):
                    if len(subset) < 5:
                        continue
                    y_true = subset["target"]
                    y_pred = subset["prediction"]
                    true_prepared = (y_true == "Prepared Individuals").mean()
                    pred_prepared = (y_pred == "Prepared Individuals").mean()
                    group_rows.append(
                        {
                            "condition": condition,
                            "group_field": group_field,
                            "group_value": group_value,
                            "n": len(subset),
                            "accuracy": accuracy_score(y_true, y_pred),
                            "macro_f1": f1_score(y_true, y_pred, labels=VALID_LABELS, average="macro"),
                            "prepared_rate_error": pred_prepared - true_prepared,
                            "abs_prepared_rate_error": abs(pred_prepared - true_prepared),
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

    print(metrics.to_string(index=False))
    print(f"Wrote metrics: {summary_path}")
    if SAMPLE_PATH.exists():
        print(f"Wrote group metrics: {group_summary_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
