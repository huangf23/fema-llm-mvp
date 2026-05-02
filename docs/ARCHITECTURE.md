# Extensible Experiment Architecture

This project is organized around one principle: keep the experimental factors explicit and isolate them from data preparation, model calls, and evaluation.

## Layers

```text
prepare_data.py       Thin CLI for data sampling and prompt generation
run_llm.py            Thin CLI for one model profile
run_profiles.py       Thin CLI for multiple model profiles
evaluate.py           Thin CLI for metrics

fema_llm_mvp/
  experiment_config.py  Target variable, labels, disclosure conditions, field labels
  data.py               FEMA extraction, loading, stratified sampling, prompt row output
  prompts.py            Prompt template and system prompt
  profiles.py           ModelProfile and .env profile resolution
  inference.py          OpenAI-compatible API calls, retries/resume bookkeeping
  evaluation.py         Overall and subgroup metrics
  paths.py              Shared paths
  utils.py              Small helpers
```

## Current Core Factors

The main scientific factor is progressive information disclosure:

- `C0`: no respondent-specific information
- `C1`: demographics
- `C2`: demographics plus geography, perceived risk, and disaster experience
- `C3`: C2 plus efficacy, confidence, household constraints, and resources

These are defined in `fema_llm_mvp/experiment_config.py`.

## Model-Side Factors

Each model is a profile in `.env`. A profile can control:

- `API_KEY`
- `BASE_URL`
- `MODEL`
- `THINKING`
- `TEMPERATURE`
- `TOP_P`
- `MAX_TOKENS`
- `SEED`
- `JSON_MODE`
- `REPEAT`

For example:

```bash
EXPERIMENT_PROFILES=deepseek_v4_flash,deepseek_v4_flash_temp07

DEEPSEEK_V4_FLASH_API_KEY=sk-...
DEEPSEEK_V4_FLASH_BASE_URL=https://api.deepseek.com
DEEPSEEK_V4_FLASH_MODEL=deepseek-v4-flash
DEEPSEEK_V4_FLASH_THINKING=disabled
DEEPSEEK_V4_FLASH_TEMPERATURE=0
DEEPSEEK_V4_FLASH_MAX_TOKENS=64
DEEPSEEK_V4_FLASH_JSON_MODE=true
DEEPSEEK_V4_FLASH_REPEAT=1

DEEPSEEK_V4_FLASH_TEMP07_API_KEY=sk-...
DEEPSEEK_V4_FLASH_TEMP07_BASE_URL=https://api.deepseek.com
DEEPSEEK_V4_FLASH_TEMP07_MODEL=deepseek-v4-flash
DEEPSEEK_V4_FLASH_TEMP07_THINKING=disabled
DEEPSEEK_V4_FLASH_TEMP07_TEMPERATURE=0.7
DEEPSEEK_V4_FLASH_TEMP07_MAX_TOKENS=64
DEEPSEEK_V4_FLASH_TEMP07_JSON_MODE=true
DEEPSEEK_V4_FLASH_TEMP07_REPEAT=3
```

## Adding A New Model

1. Add the profile name to `EXPERIMENT_PROFILES`.
2. Add `<PROFILE>_API_KEY`, `<PROFILE>_BASE_URL`, and `<PROFILE>_MODEL`.
3. Optional: add generation parameters such as `<PROFILE>_TEMPERATURE`.
4. Run:

```bash
python run_profiles.py --limit 40
python run_profiles.py
```

No Python code change is needed for OpenAI-compatible providers.

## Adding A New Disclosure Condition

Edit `disclosure_fields` in `fema_llm_mvp/experiment_config.py`:

```python
"C4": [
    "age",
    "...",
]
```

Then regenerate prompts:

```bash
python prepare_data.py --n 400 --seed 42
```

## Adding A New Target Variable

Edit `target`, `labels`, and `question_text` in `fema_llm_mvp/experiment_config.py`.

Avoid leakage: do not include the target variable or direct derivatives of the target in any disclosure condition.

## Adding Prompt Variants

Prompt wording lives in `fema_llm_mvp/prompts.py`.

For formal experiments, prefer adding a named prompt template variant instead of editing the default template in place. This keeps prompt wording as an explicit experimental factor.

## Output Convention

Each model profile writes separate outputs:

```text
work/outputs/predictions_<profile>.jsonl
work/outputs/metrics_by_condition_<profile>.csv
work/outputs/metrics_by_group_<profile>.csv
work/outputs/classification_report_by_condition_<profile>.json
work/outputs/metrics_by_condition_<profile>.png
```

Rows in `predictions_<profile>.jsonl` record model parameters such as `temperature`, `top_p`, `max_tokens`, `seed`, `json_mode`, `thinking`, and `repeat_index`.

