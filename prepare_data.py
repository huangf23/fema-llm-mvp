import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
FEMA_DIR = ROOT.parent
ZIP_PATH = FEMA_DIR / "218642-V1.zip"
WORK_DIR = ROOT / "work"
DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "outputs"

INNER_ZIP = "fema_national_household_survey_2023.zip"
XLSX_NAME = "fema_national_household_survey_2023_data_and_codebook.xlsx"
SHEET_NAME = "Core Survey"

TARGET = "dis_2_prepstages"
LABELS = ["Prepared Individuals", "Unprepared Individuals"]


QUESTION_TEXT = (
    "Thinking about preparing yourself for a disaster, which of the following "
    "best represents this person's preparedness status?"
)


FIELD_LABELS = {
    "state": "state or territory",
    "geographic_division": "geographic division",
    "census_region": "census region",
    "rurality": "rurality",
    "age": "age group",
    "sex": "sex",
    "education": "education",
    "race_selfid": "race",
    "ethnicity": "Hispanic/Latino origin",
    "income_agg": "annual household income",
    "dis_perception": "perceived likelihood that a disaster would impact them",
    "dis_exp": "whether respondent/family experienced disaster impacts",
    "dis_stepshelp": "belief that preparation would help",
    "dis_confidence": "confidence in taking preparedness steps",
    "disability": "disability or health condition affecting emergency response",
    "care": "responsibility for assisting an elderly person or someone with disability",
    "numadult": "number of adults in household",
    "numchild": "number of children in household",
    "homeownership": "home tenure",
    "finres_insuranceresidence": "homeowners or renters insurance",
}


DISCLOSURE_FIELDS = {
    "C0": [],
    "C1": [
        "age",
        "sex",
        "education",
        "race_selfid",
        "ethnicity",
        "income_agg",
    ],
    "C2": [
        "age",
        "sex",
        "education",
        "race_selfid",
        "ethnicity",
        "income_agg",
        "state",
        "geographic_division",
        "census_region",
        "rurality",
        "dis_perception",
        "dis_exp",
    ],
    "C3": [
        "age",
        "sex",
        "education",
        "race_selfid",
        "ethnicity",
        "income_agg",
        "state",
        "geographic_division",
        "census_region",
        "rurality",
        "dis_perception",
        "dis_exp",
        "dis_stepshelp",
        "dis_confidence",
        "disability",
        "care",
        "numadult",
        "numchild",
        "homeownership",
        "finres_insuranceresidence",
    ],
}


def extract_2023_file() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extracted_xlsx = DATA_DIR / XLSX_NAME
    if extracted_xlsx.exists():
        return extracted_xlsx

    inner_zip_path = DATA_DIR / INNER_ZIP
    if not inner_zip_path.exists():
        with zipfile.ZipFile(ZIP_PATH) as archive:
            with archive.open(INNER_ZIP) as source:
                inner_zip_path.write_bytes(source.read())

    with zipfile.ZipFile(inner_zip_path) as archive:
        archive.extract(XLSX_NAME, DATA_DIR)

    return extracted_xlsx


def load_core_survey(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME, header=None)
    variable_names = raw.iloc[1].astype(str).tolist()
    df = raw.iloc[2:].copy()
    df.columns = variable_names
    return df.reset_index(drop=True)


def clean_value(value) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "Unknown"
    return text


def build_profile(row: pd.Series, fields: list[str]) -> str:
    if not fields:
        return "No respondent-specific information is provided."

    lines = []
    for field in fields:
        label = FIELD_LABELS[field]
        value = clean_value(row.get(field))
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def build_prompt(row: pd.Series, condition: str) -> str:
    profile = build_profile(row, DISCLOSURE_FIELDS[condition])
    return f"""You are simulating one respondent in the FEMA National Household Survey.

A disaster is an event that could threaten lives, disrupt public or emergency services such as water and power, or damage property.

Respondent information:
{profile}

Task:
Predict the respondent's answer to this survey classification:
{QUESTION_TEXT}

Choose exactly one option:
- Prepared Individuals
- Unprepared Individuals

Return only valid JSON with this schema:
{{"prediction": "Prepared Individuals"}}
"""


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    valid = df[df[TARGET].isin(LABELS)].copy()
    if n >= len(valid):
        sample = valid
    else:
        sampled_parts = []
        for _, group in valid.groupby(TARGET):
            group_n = max(1, round(n * len(group) / len(valid)))
            sampled_parts.append(group.sample(n=group_n, random_state=seed))
        sample = pd.concat(sampled_parts).sample(frac=1, random_state=seed)
        if len(sample) > n:
            sample = sample.sample(n=n, random_state=seed)
    return sample.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400, help="Stratified sample size.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = extract_2023_file()
    df = load_core_survey(xlsx_path)
    sample = stratified_sample(df, args.n, args.seed)

    keep_fields = sorted({TARGET, "id", *[f for fs in DISCLOSURE_FIELDS.values() for f in fs]})
    existing_keep_fields = [f for f in keep_fields if f in sample.columns]
    sample[existing_keep_fields].to_csv(OUTPUT_DIR / "sample.csv", index=False)

    prompt_rows = []
    for _, row in sample.iterrows():
        respondent_id = clean_value(row.get("id"))
        for condition in DISCLOSURE_FIELDS:
            prompt_rows.append(
                {
                    "id": respondent_id,
                    "condition": condition,
                    "target": clean_value(row[TARGET]),
                    "prompt": build_prompt(row, condition),
                }
            )

    prompts = pd.DataFrame(prompt_rows)
    prompts.to_json(OUTPUT_DIR / "prompts.jsonl", orient="records", lines=True, force_ascii=False)

    metadata = {
        "source_zip": str(ZIP_PATH),
        "source_xlsx": str(xlsx_path),
        "sheet": SHEET_NAME,
        "target": TARGET,
        "labels": LABELS,
        "sample_size": len(sample),
        "prompt_count": len(prompts),
        "conditions": DISCLOSURE_FIELDS,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote sample: {OUTPUT_DIR / 'sample.csv'} ({len(sample)} rows)")
    print(f"Wrote prompts: {OUTPUT_DIR / 'prompts.jsonl'} ({len(prompts)} rows)")
    print(f"Wrote metadata: {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
