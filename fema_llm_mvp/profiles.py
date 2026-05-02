import os
from dataclasses import dataclass

from .utils import parse_bool


@dataclass(frozen=True)
class ModelProfile:
    name: str | None
    api_key: str
    base_url: str | None
    model: str
    thinking: str | None = None
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 64
    seed: int | None = None
    json_mode: bool = True
    repeat: int = 1


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


def _float_value(profile: str | None, suffix: str, generic_name: str, default: float) -> float:
    value = profile_value(profile, suffix, generic_name)
    return float(value) if value not in {None, ""} else default


def _optional_float(profile: str | None, suffix: str, generic_name: str) -> float | None:
    value = profile_value(profile, suffix, generic_name)
    return float(value) if value not in {None, ""} else None


def _optional_int(profile: str | None, suffix: str, generic_name: str) -> int | None:
    value = profile_value(profile, suffix, generic_name)
    return int(value) if value not in {None, ""} else None


def _int_value(profile: str | None, suffix: str, generic_name: str, default: int) -> int:
    value = profile_value(profile, suffix, generic_name)
    return int(value) if value not in {None, ""} else default


def resolve_profile(
    profile: str | None,
    model_override: str | None = None,
    base_url_override: str | None = None,
    temperature_override: float | None = None,
) -> ModelProfile:
    api_key = (
        profile_value(profile, "API_KEY")
        or os.getenv("API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit("API key is not set. Copy .env.example to .env and add a key.")

    model = model_override or profile_value(profile, "MODEL", "OPENAI_MODEL")
    if not model:
        raise SystemExit("Model is not set. Set OPENAI_MODEL or <PROFILE>_MODEL.")

    temperature = (
        temperature_override
        if temperature_override is not None
        else _float_value(profile, "TEMPERATURE", "OPENAI_TEMPERATURE", 0.0)
    )
    return ModelProfile(
        name=profile,
        api_key=api_key,
        base_url=base_url_override or profile_value(profile, "BASE_URL", "OPENAI_BASE_URL") or None,
        model=model,
        thinking=profile_value(profile, "THINKING", "DEEPSEEK_THINKING"),
        temperature=temperature,
        top_p=_optional_float(profile, "TOP_P", "OPENAI_TOP_P"),
        max_tokens=_int_value(profile, "MAX_TOKENS", "OPENAI_MAX_TOKENS", 64),
        seed=_optional_int(profile, "SEED", "OPENAI_SEED"),
        json_mode=parse_bool(profile_value(profile, "JSON_MODE", "OPENAI_JSON_MODE"), default=True),
        repeat=_int_value(profile, "REPEAT", "OPENAI_REPEAT", 1),
    )

