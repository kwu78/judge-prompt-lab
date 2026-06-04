"""
Closed optimization loop for the SHP pairwise preference judge prompt.

Runs N iterations of:
  1. Evaluate current preference_judge_prompt.md on SHP validation sample
  2. Compute metrics → results/preference_metrics.json
  3. Run error analysis → results/preference_error_summary.json
                          results/preference_failed_cases.jsonl
  4. Generate candidate → prompts/preference_judge_prompt_candidate.md
  5. Evaluate candidate → results/preference_candidate_metrics.json
  6. Accept or reject (with configurable accuracy threshold)
  7. Log decision → results/preference_experiment_log.jsonl

Accepts a candidate only when ALL hold:
  - accuracy improvement >= min_accuracy_delta  (default 0.02)
  - parse_failures non-increasing
  - |position_bias| does not worsen by more than max_position_bias_delta (default 0.02)
  - split / sample_size match between baseline and candidate evals

With --confirm-runs N: candidate is evaluated N times (different seeds) and the
averaged metrics are used for the accept/reject decision.
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


def run_preference_loop(
    iterations: int,
    split: str,
    sample_size: int,
    judge_model: str,
    optimizer_model: str,
    seed: int,
    min_accuracy_delta: float,
    max_position_bias_delta: float = 0.02,
    confirm_runs: int = 1,
) -> None:
    from load_preference_data import sample_shp
    from run_preference_judge import run_preference_evaluation
    from preference_metrics import compute_preference_metrics, print_preference_metrics
    from preference_error_analysis import run_analysis as run_error_analysis
    from improve_preference_prompt import propose_revised_preference_prompt
    from compare_preference_prompts import compare_and_decide_preference

    PREDICTIONS_PATH = ROOT / "results" / "preference_predictions.jsonl"
    BASELINE_METRICS_PATH = ROOT / "results" / "preference_metrics.json"
    ERROR_SUMMARY_PATH = ROOT / "results" / "preference_error_summary.json"
    FAILED_CASES_PATH = ROOT / "results" / "preference_failed_cases.jsonl"
    CANDIDATE_PATH = ROOT / "prompts" / "preference_judge_prompt_candidate.md"

    print(f"Loading SHP (split={split}, sample_size={sample_size}, seed={seed})...")
    examples = sample_shp(split=split, size=sample_size, seed=seed)
    print(f"Sample: {len(examples)} examples")
    print(f"Accept threshold : accuracy improvement >= {min_accuracy_delta}")
    print(f"Bias guard       : |position_bias| worsening <= {max_position_bias_delta}")
    if confirm_runs > 1:
        print(f"Confirm runs     : {confirm_runs} (candidate averaged before decision)")
    print()

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} / {iterations}")
        print(f"{'='*60}\n")

        # Step 1: Evaluate current judge prompt.
        print("-- Step 1: Evaluate current preference judge prompt --")
        results, meta = run_preference_evaluation(
            examples,
            model=judge_model,
            output_path=PREDICTIONS_PATH,
            split=split,
            seed=seed,
        )

        # Step 2: Compute and save metrics (include seed + model for provenance).
        print("\n-- Step 2: Compute metrics --")
        m = compute_preference_metrics(results)
        m["seed"] = seed
        m["judge_model"] = judge_model
        print_preference_metrics(m)
        BASELINE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_METRICS_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"Metrics saved to {BASELINE_METRICS_PATH}")

        # Step 3: Error analysis (writes error summary + failed cases).
        print("\n-- Step 3: Run error analysis --")
        run_error_analysis(
            predictions_path=PREDICTIONS_PATH,
            failed_path=FAILED_CASES_PATH,
            summary_path=ERROR_SUMMARY_PATH,
            top_domains=10,
        )

        # Step 4: Generate candidate prompt.
        print("\n-- Step 4: Generate candidate preference judge prompt --")
        propose_revised_preference_prompt(
            model=optimizer_model,
            metrics_path=BASELINE_METRICS_PATH,
            error_summary_path=ERROR_SUMMARY_PATH,
            failed_cases_path=FAILED_CASES_PATH,
            output_path=CANDIDATE_PATH,
        )

        # Steps 5 & 6: Evaluate candidate and accept/reject.
        print("\n-- Steps 5 & 6: Evaluate candidate and decide --")
        decision = compare_and_decide_preference(
            examples,
            judge_model=judge_model,
            optimizer_model=optimizer_model,
            baseline_metrics_path=BASELINE_METRICS_PATH,
            candidate_prompt_path=CANDIDATE_PATH,
            min_accuracy_delta=min_accuracy_delta,
            max_position_bias_delta=max_position_bias_delta,
            split=split,
            seed=seed,
            confirm_runs=confirm_runs,
        )

        status = "ACCEPTED" if decision["accepted"] else "REJECTED"
        print(f"\nIteration {iteration} complete — {status}")
        print(
            f"  baseline={decision['baseline_accuracy']}  "
            f"candidate={decision['candidate_accuracy']}  "
            f"delta={decision['accuracy_delta']}"
        )

    print(f"\n{'='*60}")
    print("Optimization loop finished.")
    print(f"Final prompt   : {ROOT / 'prompts' / 'preference_judge_prompt.md'}")
    print(f"Experiment log : {ROOT / 'results' / 'preference_experiment_log.jsonl'}")
    print(f"\nTo print a summary:")
    print(f"  python src/summarize_preference_experiment.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the preference judge prompt optimization loop."
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--split", default="validation", choices=["train", "validation", "test"]
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument(
        "--judge-model", default=os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")
    )
    parser.add_argument(
        "--optimizer-model", default=os.getenv("OPTIMIZER_MODEL", "claude-opus-4-8")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-accuracy-delta",
        type=float,
        default=0.02,
        help="Minimum accuracy improvement required to accept a candidate prompt.",
    )
    parser.add_argument(
        "--max-position-bias-delta",
        type=float,
        default=0.02,
        help="Maximum allowed worsening of |position_bias| before rejection (default: 0.02).",
    )
    parser.add_argument(
        "--confirm-runs",
        type=int,
        default=1,
        help="Evaluate candidate this many times and average before deciding (default: 1).",
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    run_preference_loop(
        iterations=args.iterations,
        split=args.split,
        sample_size=args.sample_size,
        judge_model=args.judge_model,
        optimizer_model=args.optimizer_model,
        seed=args.seed,
        min_accuracy_delta=args.min_accuracy_delta,
        max_position_bias_delta=args.max_position_bias_delta,
        confirm_runs=args.confirm_runs,
    )
