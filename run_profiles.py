import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "work" / "outputs"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def parse_profiles(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default=os.getenv("EXPERIMENT_PROFILES", ""))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    profiles = parse_profiles(args.profiles)
    if not profiles:
        raise SystemExit("No profiles configured. Set EXPERIMENT_PROFILES in .env or pass --profiles.")

    for profile in profiles:
        output_name = f"predictions_{safe_name(profile)}.jsonl"
        cmd = [
            sys.executable,
            "run_llm.py",
            "--profile",
            profile,
            "--output",
            output_name,
            "--temperature",
            str(args.temperature),
            "--sleep",
            str(args.sleep),
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        run_command(cmd)

        if not args.skip_eval:
            run_command(
                [
                    sys.executable,
                    "evaluate.py",
                    "--predictions",
                    output_name,
                    "--output-prefix",
                    safe_name(profile),
                ]
            )

    print(f"\nDone. Outputs are in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
