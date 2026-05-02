import argparse

from dotenv import load_dotenv

from fema_llm_mvp.inference import prediction_path_for, run_predictions
from fema_llm_mvp.paths import ROOT
from fema_llm_mvp.profiles import resolve_profile


def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None, help="Model profile name from .env, e.g. deepseek_v4_flash.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output", default=None, help="Prediction JSONL path. Defaults to predictions.jsonl or predictions_<profile>.jsonl.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between requests.")
    args = parser.parse_args()

    profile = resolve_profile(
        profile=args.profile,
        model_override=args.model,
        base_url_override=args.base_url,
        temperature_override=args.temperature,
    )
    output_path = prediction_path_for(args.profile, args.output)
    written_path = run_predictions(profile=profile, output_path=output_path, limit=args.limit, sleep=args.sleep)
    print(f"Wrote predictions to {written_path}")


if __name__ == "__main__":
    main()
