import pandas as pd

from .experiment_config import ExperimentConfig
from .utils import clean_value


SYSTEM_PROMPT = (
    "You are a careful survey response simulator. "
    "You must return only JSON matching the requested schema."
)


def build_profile(row: pd.Series, fields: list[str], config: ExperimentConfig) -> str:
    if not fields:
        return "No respondent-specific information is provided."

    lines = []
    for field in fields:
        label = config.field_labels[field]
        value = clean_value(row.get(field))
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def build_prompt(row: pd.Series, condition: str, config: ExperimentConfig) -> str:
    profile = build_profile(row, config.disclosure_fields[condition], config)
    labels = "\n".join(f"- {label}" for label in config.labels)
    return f"""You are simulating one respondent in the FEMA National Household Survey.

A disaster is an event that could threaten lives, disrupt public or emergency services such as water and power, or damage property.

Respondent information:
{profile}

Task:
Predict the respondent's answer to this survey classification:
{config.question_text}

Choose exactly one option:
{labels}

Return only valid JSON with this schema:
{{"prediction": "{config.labels[0]}"}}
"""

