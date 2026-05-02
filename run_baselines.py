import argparse

from fema_llm_mvp.baselines import (
    BASELINE_METRICS_PATH,
    BASELINE_PLOT_PATH,
    BASELINE_PREDICTIONS_PATH,
    run_baselines,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = run_baselines(folds=args.folds, seed=args.seed)
    print(metrics.to_string(index=False))
    print(f"Wrote metrics: {BASELINE_METRICS_PATH}")
    print(f"Wrote predictions: {BASELINE_PREDICTIONS_PATH}")
    print(f"Wrote plot: {BASELINE_PLOT_PATH}")


if __name__ == "__main__":
    main()
