from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEMA_DIR = ROOT.parent
ZIP_PATH = FEMA_DIR / "218642-V1.zip"
WORK_DIR = ROOT / "work"
DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "outputs"
PROMPTS_PATH = OUTPUT_DIR / "prompts.jsonl"
SAMPLE_PATH = OUTPUT_DIR / "sample.csv"
DEFAULT_PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"

