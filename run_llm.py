import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "work" / "outputs"
PROMPTS_PATH = OUTPUT_DIR / "prompts.jsonl"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"
VALID_LABELS = {"Prepared Individuals", "Unprepared Individuals"}


def parse_prediction(text: str) -> str:
    text = text.strip()
    try:
        data = json.loads(text)
        pred = data.get("prediction", "")
    except json.JSONDecodeError:
        pred = text

    pred = str(pred).strip()
    if pred in VALID_LABELS:
        return pred

    lower = pred.lower()
    if "unprepared" in lower:
        return "Unprepared Individuals"
    if "prepared" in lower:
        return "Prepared Individuals"
    return "INVALID"


def env_name(profile: str, suffix: str) -> str:
    return f"{profile.upper().replace('-', '_')}_{suffix}"


def profile_value(profile: str | None, suffix: str, generic_name: str | None = None) -> str | None:
    if profile:
        value = os.getenv(env_name(profile, suffix))
        if value:
            return value
    if generic_name:
        return os.getenv(generic_name)
    return None


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    thinking: str | None,
) -> tuple[str, str]:
    extra_body = {}
    thinking = (thinking or "").strip().lower()
    if thinking in {"enabled", "disabled"}:
        extra_body["thinking"] = {"type": thinking}

    kwargs = {}
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful survey response simulator. "
                    "You must return only JSON matching the requested schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=64,
        **kwargs,
    )
    raw = response.choices[0].message.content or ""
    return raw, parse_prediction(raw)


def load_completed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                completed.add((str(row["id"]), row["condition"]))
    return completed


def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None, help="Model profile name from .env, e.g. deepseek_v4_flash.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output", default=None, help="Prediction JSONL path. Defaults to predictions.jsonl or predictions_<profile>.jsonl.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between requests.")
    args = parser.parse_args()

    api_key = (
        profile_value(args.profile, "API_KEY")
        or os.getenv("API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = args.base_url or profile_value(args.profile, "BASE_URL", "OPENAI_BASE_URL") or None
    model = profile_value(args.profile, "MODEL", "OPENAI_MODEL") or args.model
    thinking = profile_value(args.profile, "THINKING", "DEEPSEEK_THINKING")
    if not api_key:
        raise SystemExit("API key is not set. Copy .env.example to .env and add a key.")

    prompts = pd.read_json(PROMPTS_PATH, lines=True)
    if args.limit:
        prompts = prompts.head(args.limit)

    if args.output:
        predictions_path = Path(args.output)
        if not predictions_path.is_absolute():
            predictions_path = OUTPUT_DIR / predictions_path
    elif args.profile:
        predictions_path = OUTPUT_DIR / f"predictions_{safe_name(args.profile)}.jsonl"
    else:
        predictions_path = PREDICTIONS_PATH

    completed = load_completed(predictions_path)
    client = OpenAI(api_key=api_key, base_url=base_url)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    with predictions_path.open("a", encoding="utf-8") as f:
        for row in tqdm(prompts.to_dict("records"), total=len(prompts)):
            key = (str(row["id"]), row["condition"])
            if key in completed:
                continue
            try:
                raw, prediction = call_model(client, model, row["prompt"], args.temperature, thinking)
                out = {
                    "id": str(row["id"]),
                    "condition": row["condition"],
                    "target": row["target"],
                    "prediction": prediction,
                    "raw_output": raw,
                    "profile": args.profile,
                    "model": model,
                    "base_url": base_url,
                    "temperature": args.temperature,
                }
            except Exception as exc:
                out = {
                    "id": str(row["id"]),
                    "condition": row["condition"],
                    "target": row["target"],
                    "prediction": "ERROR",
                    "raw_output": repr(exc),
                    "profile": args.profile,
                    "model": model,
                    "base_url": base_url,
                    "temperature": args.temperature,
                }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Wrote predictions to {predictions_path}")


if __name__ == "__main__":
    main()
