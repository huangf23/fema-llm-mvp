# FEMA NHS LLM MVP

This MVP tests whether progressive information disclosure improves LLM simulation of disaster preparedness status in the FEMA National Household Survey.

For full instructions on running this on another PC and using multiple OpenAI-compatible models, see [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md).

## Setup

```bash
cd /data2/home/hf01/FEMA/llm_mvp
source .venv/bin/activate
```

Optional, for live LLM calls:

```bash
cp .env.example .env
# edit .env and set API_KEY/provider settings
```

DeepSeek V4 Flash example:

```bash
API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
DEEPSEEK_THINKING=disabled
```

## Pipeline

Prepare a stratified sample and prompts:

```bash
python prepare_data.py --n 400 --seed 42
```

Run LLM predictions:

```bash
python run_profiles.py --limit 40
python run_profiles.py
```

Evaluate predictions:

```bash
python evaluate.py --predictions predictions_deepseek_v4_flash.jsonl --output-prefix deepseek_v4_flash
```

## Disclosure Conditions

- `C0`: disaster definition and target question only
- `C1`: `C0` plus demographics
- `C2`: `C1` plus geography, perceived risk, and disaster experience
- `C3`: `C2` plus efficacy, confidence, household constraints, and resources

The default target is `dis_2_prepstages`, with `Unknown` excluded.
