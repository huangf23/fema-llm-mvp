import argparse
from pathlib import Path

from fema_llm_mvp.baselines import (
    run_baselines,
)
from fema_llm_mvp.paths import OUTPUT_DIR, SAMPLE_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", default=str(SAMPLE_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    sample_path = Path(args.sample)
    output_dir = Path(args.output_dir)
    metrics = run_baselines(sample_path=sample_path, output_dir=output_dir, folds=args.folds, seed=args.seed)
    print(metrics.to_string(index=False))
    print(f"Wrote metrics: {output_dir / 'baseline_metrics_by_condition.csv'}")
    print(f"Wrote predictions: {output_dir / 'baseline_predictions.csv'}")
    print(f"Wrote plot: {output_dir / 'baseline_metrics_by_condition.png'}")


if __name__ == "__main__":
    main()
