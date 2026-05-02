import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv

from fema_llm_mvp.paths import OUTPUT_DIR, ROOT
from fema_llm_mvp.utils import safe_name


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
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
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
            "--sleep",
            str(args.sleep),
            "--concurrency",
            str(args.concurrency),
        ]
        if args.temperature is not None:
            cmd.extend(["--temperature", str(args.temperature)])
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
