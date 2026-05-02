import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from .experiment_config import DEFAULT_EXPERIMENT, ExperimentConfig
from .paths import DEFAULT_PREDICTIONS_PATH, OUTPUT_DIR, PROMPTS_PATH, resolve_output_path
from .profiles import ModelProfile
from .prompts import SYSTEM_PROMPT
from .utils import safe_name


def parse_prediction(text: str, config: ExperimentConfig = DEFAULT_EXPERIMENT) -> str:
    text = text.strip()
    try:
        data = json.loads(text)
        pred = data.get("prediction", "")
    except json.JSONDecodeError:
        pred = text

    pred = str(pred).strip()
    if pred in config.labels:
        return pred

    lower = pred.lower()
    for label in config.labels:
        if label.lower() in lower:
            return label
    return "INVALID"


def prediction_path_for(profile: str | None, output: str | None) -> Path:
    if output:
        return resolve_output_path(output)
    if profile:
        return OUTPUT_DIR / f"predictions_{safe_name(profile)}.jsonl"
    return DEFAULT_PREDICTIONS_PATH


def load_completed(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                completed.add((str(row["id"]), row["condition"], int(row.get("repeat_index", 0))))
    return completed


def call_model(
    client: OpenAI,
    profile: ModelProfile,
    prompt: str,
    config: ExperimentConfig = DEFAULT_EXPERIMENT,
) -> tuple[str, str]:
    extra_body = {}
    thinking = (profile.thinking or "").strip().lower()
    if thinking in {"enabled", "disabled"}:
        extra_body["thinking"] = {"type": thinking}

    kwargs = {}
    if profile.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if profile.top_p is not None:
        kwargs["top_p"] = profile.top_p
    if profile.seed is not None:
        kwargs["seed"] = profile.seed
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(
        model=profile.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=profile.temperature,
        max_tokens=profile.max_tokens,
        **kwargs,
    )
    raw = response.choices[0].message.content or ""
    return raw, parse_prediction(raw, config)


def run_predictions(
    profile: ModelProfile,
    prompts_path: Path = PROMPTS_PATH,
    output_path: Path | None = None,
    limit: int | None = None,
    sleep: float = 0.0,
    concurrency: int = 1,
    config: ExperimentConfig = DEFAULT_EXPERIMENT,
) -> Path:
    output_path = output_path or prediction_path_for(profile.name, None)
    prompts = pd.read_json(prompts_path, lines=True)
    if limit:
        prompts = prompts.head(limit)

    client = OpenAI(api_key=profile.api_key, base_url=profile.base_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)

    tasks = []
    for row in prompts.to_dict("records"):
        for repeat_index in range(profile.repeat):
            key = (str(row["id"]), row["condition"], repeat_index)
            if key not in completed:
                tasks.append((row, repeat_index))

    def run_one(row: dict, repeat_index: int) -> dict:
        try:
            raw, prediction = call_model(client, profile, row["prompt"], config)
            return {
                "id": str(row["id"]),
                "condition": row["condition"],
                "target": row["target"],
                "prediction": prediction,
                "raw_output": raw,
                "repeat_index": repeat_index,
                "profile": profile.name,
                "model": profile.model,
                "base_url": profile.base_url,
                "temperature": profile.temperature,
                "top_p": profile.top_p,
                "max_tokens": profile.max_tokens,
                "seed": profile.seed,
                "json_mode": profile.json_mode,
                "thinking": profile.thinking,
            }
        except Exception as exc:
            return {
                "id": str(row["id"]),
                "condition": row["condition"],
                "target": row["target"],
                "prediction": "ERROR",
                "raw_output": repr(exc),
                "repeat_index": repeat_index,
                "profile": profile.name,
                "model": profile.model,
                "base_url": profile.base_url,
                "temperature": profile.temperature,
                "top_p": profile.top_p,
                "max_tokens": profile.max_tokens,
                "seed": profile.seed,
                "json_mode": profile.json_mode,
                "thinking": profile.thinking,
            }

    with output_path.open("a", encoding="utf-8") as f:
        if concurrency <= 1:
            for row, repeat_index in tqdm(tasks, total=len(tasks)):
                out = run_one(row, repeat_index)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                if sleep:
                    time.sleep(sleep)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(run_one, row, repeat_index) for row, repeat_index in tasks]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    out = future.result()
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    if sleep:
                        time.sleep(sleep)

    return output_path
