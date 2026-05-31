"""
MVP 4: Evaluate the candidate judge prompt on the same dev sample,
compare to baseline, and accept or reject.

Accept only if ALL of the following hold:
  1. candidate avg_mae improves over baseline by at least min_mae_delta
  2. candidate parse_failures <= baseline parse_failures
  3. candidate was evaluated on the same split, sample_size, and seed as baseline

Logs the decision (including rejection reasons) to results/experiment_log.jsonl.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
JUDGE_PROMPT_PATH = ROOT / "prompts" / "judge_prompt.md"
CANDIDATE_PATH = ROOT / "prompts" / "judge_prompt_candidate.md"
BASELINE_METRICS_PATH = ROOT / "results" / "baseline_metrics.json"
CANDIDATE_METRICS_PATH = ROOT / "results" / "candidate_metrics.json"
CANDIDATE_PREDICTIONS_PATH = ROOT / "results" / "candidate_predictions.jsonl"
EXPERIMENT_LOG_PATH = ROOT / "results" / "experiment_log.jsonl"

DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")
DEFAULT_MIN_MAE_DELTA = 0.02


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def log_decision(decision: dict, log_path: Path = EXPERIMENT_LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision) + "\n")


def compare_and_decide(
    examples: list[dict],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    baseline_metrics_path: Path = BASELINE_METRICS_PATH,
    candidate_prompt_path: Path = CANDIDATE_PATH,
    judge_prompt_path: Path = JUDGE_PROMPT_PATH,
    candidate_metrics_path: Path = CANDIDATE_METRICS_PATH,
    candidate_predictions_path: Path = CANDIDATE_PREDICTIONS_PATH,
    min_mae_delta: float = DEFAULT_MIN_MAE_DELTA,
    split: str = "dev",
    seed: int = 42,
) -> dict:
    from run_judge import run_evaluation
    from metrics import compute_metrics, load_predictions

    # Evaluate the candidate prompt on the same examples.
    print(f"Evaluating candidate prompt ({len(examples)} examples)...")
    _, cand_meta = run_evaluation(
        examples,
        judge_prompt_path=candidate_prompt_path,
        model=judge_model,
        output_path=candidate_predictions_path,
        split=split,
        seed=seed,
    )

    cand_records = load_predictions(candidate_predictions_path)
    candidate_metrics = compute_metrics(cand_records, metadata=cand_meta)
    candidate_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_metrics_path.write_text(json.dumps(candidate_metrics, indent=2), encoding="utf-8")
    print(f"Candidate metrics saved to {candidate_metrics_path}")

    baseline_metrics = load_metrics(baseline_metrics_path)
    baseline_avg_mae = baseline_metrics.get("avg_mae")
    candidate_avg_mae = candidate_metrics.get("avg_mae")
    b_parse = baseline_metrics.get("n_parse_failures", 0)
    c_parse = candidate_metrics.get("n_parse_failures", 0)

    # --- Strict accept/reject rules ---
    rejection_reasons: list[str] = []

    if baseline_avg_mae is None or candidate_avg_mae is None:
        rejection_reasons.append("avg_mae unavailable in baseline or candidate")
    else:
        improvement = round(baseline_avg_mae - candidate_avg_mae, 4)
        if improvement < min_mae_delta:
            rejection_reasons.append(
                f"mae improvement {improvement:.4f} < required threshold {min_mae_delta}"
            )

    if c_parse > b_parse:
        rejection_reasons.append(
            f"parse_failures increased: baseline={b_parse} candidate={c_parse}"
        )

    for field in ("split", "sample_size", "seed"):
        bv = baseline_metrics.get(field)
        cv = candidate_metrics.get(field)
        if bv is not None and cv is not None and bv != cv:
            rejection_reasons.append(
                f"eval condition mismatch on '{field}': baseline={bv!r} candidate={cv!r}"
            )

    accepted = len(rejection_reasons) == 0
    improvement = (
        round(baseline_avg_mae - candidate_avg_mae, 4)
        if (baseline_avg_mae is not None and candidate_avg_mae is not None)
        else None
    )

    print(f"\nBaseline avg_mae:  {baseline_avg_mae}  (parse_failures={b_parse})")
    print(f"Candidate avg_mae: {candidate_avg_mae}  (parse_failures={c_parse})")
    print(f"MAE improvement:   {improvement}  (threshold={min_mae_delta})")
    if accepted:
        print("Decision: ACCEPT")
        shutil.copy2(candidate_prompt_path, judge_prompt_path)
        print(f"Candidate copied to {judge_prompt_path}")
    else:
        print("Decision: REJECT")
        for reason in rejection_reasons:
            print(f"  - {reason}")

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "baseline_avg_mae": baseline_avg_mae,
        "candidate_avg_mae": candidate_avg_mae,
        "mae_improvement": improvement,
        "min_mae_delta": min_mae_delta,
        "baseline_parse_failures": b_parse,
        "candidate_parse_failures": c_parse,
        "judge_model": judge_model,
        "split": split,
        "sample_size": len(examples),
        "seed": seed,
    }
    log_decision(decision)
    print(f"Decision logged to {EXPERIMENT_LOG_PATH}")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline and candidate judge prompts.")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "train"])
    parser.add_argument("--min-mae-delta", type=float, default=DEFAULT_MIN_MAE_DELTA,
                        help="Minimum avg_mae improvement required to accept candidate.")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    for path, label in [
        (BASELINE_METRICS_PATH, "baseline_metrics.json"),
        (CANDIDATE_PATH, "judge_prompt_candidate.md"),
    ]:
        if not path.exists():
            print(f"Error: {label} not found. Run the earlier pipeline steps first.")
            sys.exit(1)

    sys.path.insert(0, str(ROOT / "src"))
    from load_data import load_helpsteer2, sample_split

    print(f"Loading HelpSteer2 (seed={args.seed})...")
    splits = load_helpsteer2(seed=args.seed)
    examples = sample_split(splits[args.split], args.sample_size, seed=args.seed)

    compare_and_decide(
        examples,
        judge_model=args.judge_model,
        min_mae_delta=args.min_mae_delta,
        split=args.split,
        seed=args.seed,
    )
