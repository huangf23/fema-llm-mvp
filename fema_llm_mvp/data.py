import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .experiment_config import DEFAULT_EXPERIMENT, ExperimentConfig
from .paths import DATA_DIR, OUTPUT_DIR, ZIP_PATH
from .prompts import build_prompt
from .utils import clean_value


def extract_source_file(config: ExperimentConfig = DEFAULT_EXPERIMENT) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extracted_xlsx = DATA_DIR / config.xlsx_name
    if extracted_xlsx.exists():
        return extracted_xlsx

    inner_zip_path = DATA_DIR / config.inner_zip
    if not inner_zip_path.exists():
        with zipfile.ZipFile(ZIP_PATH) as archive:
            with archive.open(config.inner_zip) as source:
                inner_zip_path.write_bytes(source.read())

    with zipfile.ZipFile(inner_zip_path) as archive:
        archive.extract(config.xlsx_name, DATA_DIR)

    return extracted_xlsx


def load_core_survey(xlsx_path: Path, config: ExperimentConfig = DEFAULT_EXPERIMENT) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=config.sheet_name, header=None)
    variable_names = raw.iloc[1].astype(str).tolist()
    df = raw.iloc[2:].copy()
    df.columns = variable_names
    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int, config: ExperimentConfig = DEFAULT_EXPERIMENT) -> pd.DataFrame:
    valid = df[df[config.target].isin(config.labels)].copy()
    if n >= len(valid):
        sample = valid
    else:
        sampled_parts = []
        for _, group in valid.groupby(config.target):
            group_n = max(1, round(n * len(group) / len(valid)))
            sampled_parts.append(group.sample(n=group_n, random_state=seed))
        sample = pd.concat(sampled_parts).sample(frac=1, random_state=seed)
        if len(sample) > n:
            sample = sample.sample(n=n, random_state=seed)
    return sample.reset_index(drop=True)


def prepare_outputs(n: int, seed: int, config: ExperimentConfig = DEFAULT_EXPERIMENT) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = extract_source_file(config)
    df = load_core_survey(xlsx_path, config)
    sample = stratified_sample(df, n, seed, config)

    keep_fields = sorted({config.target, "id", *[f for fs in config.disclosure_fields.values() for f in fs]})
    existing_keep_fields = [f for f in keep_fields if f in sample.columns]
    sample[existing_keep_fields].to_csv(OUTPUT_DIR / "sample.csv", index=False)

    prompt_rows = []
    for _, row in sample.iterrows():
        respondent_id = clean_value(row.get("id"))
        for condition in config.disclosure_fields:
            prompt_rows.append(
                {
                    "id": respondent_id,
                    "condition": condition,
                    "target": clean_value(row[config.target]),
                    "prompt": build_prompt(row, condition, config),
                }
            )

    prompts = pd.DataFrame(prompt_rows)
    prompts.to_json(OUTPUT_DIR / "prompts.jsonl", orient="records", lines=True, force_ascii=False)

    metadata = {
        "source_zip": str(ZIP_PATH),
        "source_xlsx": str(xlsx_path),
        "sample_size": len(sample),
        "prompt_count": len(prompts),
        "config": asdict(config),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

