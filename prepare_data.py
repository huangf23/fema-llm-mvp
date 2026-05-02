import argparse

from fema_llm_mvp.data import prepare_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400, help="Stratified sample size.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = prepare_outputs(n=args.n, seed=args.seed)
    print(f"Wrote sample: work/outputs/sample.csv ({metadata['sample_size']} rows)")
    print(f"Wrote prompts: work/outputs/prompts.jsonl ({metadata['prompt_count']} rows)")
    print("Wrote metadata: work/outputs/metadata.json")


if __name__ == "__main__":
    main()
