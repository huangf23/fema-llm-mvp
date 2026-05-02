import argparse
from pathlib import Path

from fema_llm_mvp.evaluation import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="predictions.jsonl", help="Prediction JSONL path.")
    parser.add_argument("--output-prefix", default=None, help="Output prefix for metrics files.")
    parser.add_argument("--sample", default=None, help="Sample CSV path for subgroup metrics.")
    args = parser.parse_args()

    evaluate_predictions(
        Path(args.predictions),
        output_prefix=args.output_prefix,
        sample_path=Path(args.sample) if args.sample else None,
    )


if __name__ == "__main__":
    main()
