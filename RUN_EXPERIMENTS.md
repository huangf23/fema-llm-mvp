# FEMA NHS LLM MVP: Runbook

这个文件用于在另一台 PC 上复现实验。所有模型都按 OpenAI-compatible API 处理，模型和厂商信息只写在 `.env` 里。

## 1. 准备项目

把整个目录复制到实验 PC：

```bash
/data2/home/hf01/FEMA/llm_mvp
```

同时确保原始数据 zip 在项目父目录：

```bash
/data2/home/hf01/FEMA/218642-V1.zip
```

如果路径不同，建议保持结构为：

```text
FEMA/
  218642-V1.zip
  llm_mvp/
```

## 2. 创建虚拟环境

```bash
cd /path/to/FEMA/llm_mvp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd C:\path\to\FEMA\llm_mvp
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 配置模型

```bash
cp .env.example .env
```

在 `.env` 里填写要跑的 profile。例如只跑 DeepSeek V4 Flash：

```bash
EXPERIMENT_PROFILES=deepseek_v4_flash

DEEPSEEK_V4_FLASH_API_KEY=sk-...
DEEPSEEK_V4_FLASH_BASE_URL=https://api.deepseek.com
DEEPSEEK_V4_FLASH_MODEL=deepseek-v4-flash
DEEPSEEK_V4_FLASH_THINKING=disabled
DEEPSEEK_V4_FLASH_TEMPERATURE=0
DEEPSEEK_V4_FLASH_MAX_TOKENS=64
DEEPSEEK_V4_FLASH_JSON_MODE=true
DEEPSEEK_V4_FLASH_REPEAT=1
```

如果要跑多个模型，用逗号分隔：

```bash
EXPERIMENT_PROFILES=deepseek_v4_flash,qwen,openrouter
```

每个 profile 需要对应三项：

```bash
PROFILE_API_KEY=
PROFILE_BASE_URL=
PROFILE_MODEL=
```

常用可选项：

```bash
PROFILE_THINKING=disabled
PROFILE_TEMPERATURE=0
PROFILE_TOP_P=
PROFILE_MAX_TOKENS=64
PROFILE_SEED=
PROFILE_JSON_MODE=true
PROFILE_REPEAT=1
```

其中 `PROFILE` 是大写形式。例如 profile 名是 `qwen`，变量就是：

```bash
QWEN_API_KEY=
QWEN_BASE_URL=
QWEN_MODEL=
```

profile 名是 `gemini_openai`，变量就是：

```bash
GEMINI_OPENAI_API_KEY=
GEMINI_OPENAI_BASE_URL=
GEMINI_OPENAI_MODEL=
```

## 4. 生成样本和 prompts

MVP 默认 400 个样本，四个信息披露条件 C0-C3，共 1600 条 prompt：

```bash
python prepare_data.py --n 400 --seed 42
```

输出：

```text
work/outputs/sample.csv
work/outputs/prompts.jsonl
work/outputs/metadata.json
```

## 5. 先做 smoke test

强烈建议先跑 40 条 prompt，确认 key、base URL、模型名和输出格式都没问题：

```bash
python run_profiles.py --limit 40
```

每个模型会输出：

```text
work/outputs/predictions_<profile>.jsonl
work/outputs/metrics_by_condition_<profile>.csv
work/outputs/metrics_by_group_<profile>.csv
work/outputs/metrics_by_condition_<profile>.png
```

## 6. 跑完整 MVP

```bash
python run_profiles.py
```

如果接口限速，可以加等待：

```bash
python run_profiles.py --sleep 0.2
```

如果需要并发调用，例如 10 个请求并行：

```bash
python run_profiles.py --concurrency 10
```

如果只想跑某几个 profile，不改 `.env` 也可以：

```bash
python run_profiles.py --profiles deepseek_v4_flash,qwen
```

## 7. 单独跑某一个模型

```bash
python run_llm.py --profile deepseek_v4_flash
python evaluate.py --predictions predictions_deepseek_v4_flash.jsonl --output-prefix deepseek_v4_flash
```

也可以不用 profile，直接用通用环境变量：

```bash
API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com OPENAI_MODEL=deepseek-v4-flash \
python run_llm.py --model deepseek-v4-flash --output predictions_manual.jsonl
```

## 8. 结果文件解释

- `predictions_<profile>.jsonl`: 每条 prompt 的原始输出和清洗后的预测
- `metrics_by_condition_<profile>.csv`: C0-C3 的总体 accuracy、macro-F1、Prepared 比例误差
- `metrics_by_group_<profile>.csv`: 按年龄、教育、收入、残障分组的误差
- `classification_report_by_condition_<profile>.json`: 每个条件的分类报告
- `metrics_by_condition_<profile>.png`: 指标随披露层级变化图
- `baseline_metrics_by_condition.csv`: majority、logistic regression、random forest 的 C0-C3 对比
- `baseline_predictions.csv`: baseline 的交叉验证预测明细

## 9. 推荐运行顺序

```bash
source .venv/bin/activate
python prepare_data.py --n 400 --seed 42
python run_baselines.py
python run_profiles.py --limit 40
python run_profiles.py
```

如果 smoke test 失败，优先检查 `.env` 中的 `API_KEY`、`BASE_URL`、`MODEL`。
