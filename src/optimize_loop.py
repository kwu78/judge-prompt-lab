"""
MVP 5: Closed optimization loop.

Runs N iterations of:
  1. Evaluate current judge_prompt.md on dev sample
  2. Compute metrics → results/baseline_metrics.json
  3. Save failure cases → data/failed_cases.jsonl
  4. Generate candidate → prompts/judge_prompt_candidate.md
  5. Evaluate candidate → results/candidate_metrics.json
  6. Accept or reject (with configurable threshold)
  7. Log decision
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def run_loop(
    iterations: int,
    sample_size: int,
    judge_model: str,
    optimizer_model: str,
    seed: int,
    min_mae_delta: float,
) -> None:
    from load_data import load_helpsteer2, sample_split
    from run_judge import run_evaluation
    from metrics import compute_metrics, load_predictions, print_metrics
    from error_analysis import identify_failures, load_predictions as load_preds_ea
    from improve_prompt import propose_revised_prompt
    from compare_prompts import compare_and_decide

    import json

    PREDICTIONS_PATH = ROOT / "results" / "predictions.jsonl"
    BASELINE_METRICS_PATH = ROOT / "results" / "baseline_metrics.json"
    FAILED_CASES_PATH = ROOT / "data" / "failed_cases.jsonl"
    CANDIDATE_PATH = ROOT / "prompts" / "judge_prompt_candidate.md"
    SPLIT = "dev"

    print(f"Loading HelpSteer2 (seed={seed})...")
    splits = load_helpsteer2(seed=seed)
    examples = sample_split(splits[SPLIT], sample_size, seed=seed)
    print(f"Dev sample: {len(examples)} examples")
    print(f"Accept threshold: avg_mae improvement >= {min_mae_delta}\n")

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} / {iterations}")
        print(f"{'='*60}\n")

        # Step 1: Evaluate current judge prompt.
        print("-- Step 1: Evaluate current judge prompt --")
        _, meta = run_evaluation(
            examples,
            model=judge_model,
            output_path=PREDICTIONS_PATH,
            split=SPLIT,
            seed=seed,
        )

        # Step 2: Compute and save metrics with provenance metadata.
        print("\n-- Step 2: Compute metrics --")
        records = load_predictions(PREDICTIONS_PATH)
        m = compute_metrics(records, metadata=meta)
        print_metrics(m)
        BASELINE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_METRICS_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")

        # Step 3: Save failure cases (dev set only — never exposed to test).
        print("\n-- Step 3: Save failure cases --")
        failures = identify_failures(load_preds_ea(PREDICTIONS_PATH), top_n=20)
        FAILED_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FAILED_CASES_PATH, "w", encoding="utf-8") as f:
            for case in failures:
                f.write(json.dumps(case) + "\n")
        print(f"Saved {len(failures)} failure cases.")

        # Step 4: Generate candidate prompt.
        print("\n-- Step 4: Generate candidate judge prompt --")
        propose_revised_prompt(
            model=optimizer_model,
            metrics_path=BASELINE_METRICS_PATH,
            failed_cases_path=FAILED_CASES_PATH,
            output_path=CANDIDATE_PATH,
        )

        # Steps 5 & 6: Evaluate candidate and accept/reject.
        print("\n-- Steps 5 & 6: Evaluate candidate and decide --")
        decision = compare_and_decide(
            examples,
            judge_model=judge_model,
            baseline_metrics_path=BASELINE_METRICS_PATH,
            candidate_prompt_path=CANDIDATE_PATH,
            min_mae_delta=min_mae_delta,
            split=SPLIT,
            seed=seed,
        )

        status = "ACCEPTED" if decision["accepted"] else "REJECTED"
        print(f"\nIteration {iteration} complete — {status}")
        print(f"  baseline={decision['baseline_avg_mae']}  "
              f"candidate={decision['candidate_avg_mae']}  "
              f"improvement={decision['mae_improvement']}")

    print(f"\n{'='*60}")
    print("Optimization loop finished.")
    print(f"Final judge prompt: {ROOT / 'prompts' / 'judge_prompt.md'}")
    print(f"Experiment log:     {ROOT / 'results' / 'experiment_log.jsonl'}")
    print(f"\nTo evaluate on held-out test set:")
    print(f"  python src/final_eval.py --sample-size 100 --seed {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the judge prompt optimization loop.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--judge-model", type=str,
                        default=os.getenv("JUDGE_MODEL", "claude-sonnet-4-6"))
    parser.add_argument("--optimizer-model", type=str,
                        default=os.getenv("OPTIMIZER_MODEL", "claude-opus-4-8"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-mae-delta", type=float, default=0.02,
                        help="Minimum avg_mae improvement required to accept a candidate prompt.")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    run_loop(
        iterations=args.iterations,
        sample_size=args.sample_size,
        judge_model=args.judge_model,
        optimizer_model=args.optimizer_model,
        seed=args.seed,
        min_mae_delta=args.min_mae_delta,
    )
