import argparse
from pathlib import Path

from fema_llm_mvp.data import prepare_outputs
from fema_llm_mvp.paths import OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400, help="Stratified sample size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for sample, prompts, and metadata.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metadata = prepare_outputs(n=args.n, seed=args.seed, output_dir=output_dir)
    print(f"Wrote sample: {output_dir / 'sample.csv'} ({metadata['sample_size']} rows)")
    print(f"Wrote prompts: {output_dir / 'prompts.jsonl'} ({metadata['prompt_count']} rows)")
    print(f"Wrote metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
