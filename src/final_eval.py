"""
Evaluate the current judge_prompt.md on the held-out test split.

This script is strictly read-only with respect to the optimization loop:
  - Never modifies prompts/judge_prompt.md
  - Never writes data/failed_cases.jsonl
  - Never calls improve_prompt.py
  - The test split is determined by the same seed used during optimization,
    so it was never seen by the optimizer.

Writes: results/final_test_metrics.json
        results/final_test_predictions.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

JUDGE_PROMPT_PATH = ROOT / "prompts" / "judge_prompt.md"
FINAL_TEST_METRICS_PATH = ROOT / "results" / "final_test_metrics.json"
FINAL_TEST_PREDICTIONS_PATH = ROOT / "results" / "final_test_predictions.jsonl"

DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate judge prompt on the held-out test split."
    )
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Max examples to use from the test split.")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the seed used during optimization.")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    from load_data import load_helpsteer2, sample_split
    from run_judge import run_evaluation
    from metrics import compute_metrics, load_predictions, print_metrics

    print(f"Loading HelpSteer2 (seed={args.seed})...")
    splits = load_helpsteer2(seed=args.seed)
    test_pool = splits["test"]
    examples = sample_split(test_pool, args.sample_size, seed=args.seed)

    print(f"Test split: {len(test_pool)} total examples available.")
    print(f"Evaluating {len(examples)} examples.")
    print("NOTE: This is the held-out test split — never seen by the optimizer.\n")

    _, meta = run_evaluation(
        examples,
        judge_prompt_path=JUDGE_PROMPT_PATH,
        model=args.judge_model,
        output_path=FINAL_TEST_PREDICTIONS_PATH,
        split="test",
        seed=args.seed,
    )

    records = load_predictions(FINAL_TEST_PREDICTIONS_PATH)
    m = compute_metrics(records, metadata=meta)

    FINAL_TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_TEST_METRICS_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 50)
    print_metrics(m)
    print(f"\nFull results written to {FINAL_TEST_METRICS_PATH}")
