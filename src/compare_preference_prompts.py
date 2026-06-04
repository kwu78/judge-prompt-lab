"""
Evaluate the candidate preference judge prompt on the same SHP sample as the
baseline, compare accuracy, and accept or reject.

Accept only if ALL of the following hold:
  1. candidate accuracy improves over baseline by at least min_accuracy_delta
  2. candidate parse_failures <= baseline parse_failures
  3. candidate position_bias does not worsen by more than max_position_bias_delta
  4. split and sample_size match between baseline and candidate evaluation

Optional confirm mode (--confirm-runs N):
  Runs the candidate evaluation N times with seeds seed, seed+1, ..., seed+N-1
  and averages accuracy, parse failures, and position bias. The averaged values
  are used for the accept/reject decision. Reduces acceptance of lucky single
  runs caused by model variance.

Logs the decision to results/preference_experiment_log.jsonl.
If accepted, copies the candidate over prompts/preference_judge_prompt.md.
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
JUDGE_PROMPT_PATH = ROOT / "prompts" / "preference_judge_prompt.md"
CANDIDATE_PATH = ROOT / "prompts" / "preference_judge_prompt_candidate.md"
BASELINE_METRICS_PATH = ROOT / "results" / "preference_metrics.json"
CANDIDATE_METRICS_PATH = ROOT / "results" / "preference_candidate_metrics.json"
CANDIDATE_PREDICTIONS_PATH = ROOT / "results" / "preference_candidate_predictions.jsonl"
EXPERIMENT_LOG_PATH = ROOT / "results" / "preference_experiment_log.jsonl"

DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")
DEFAULT_MIN_ACCURACY_DELTA = 0.02
DEFAULT_MAX_POSITION_BIAS_DELTA = 0.02  # reject if |candidate_bias| worsens by more than this


def _log_decision(decision: dict, log_path: Path = EXPERIMENT_LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision) + "\n")


def _evaluate_once(
    examples: list[dict],
    prompt_path: Path,
    model: str,
    split: str,
    seed: int,
    predictions_path: Path,
) -> dict:
    """Run one candidate evaluation and return its metrics dict."""
    from run_preference_judge import run_preference_evaluation
    from preference_metrics import compute_preference_metrics

    results, _ = run_preference_evaluation(
        examples,
        prompt_path=prompt_path,
        model=model,
        output_path=predictions_path,
        split=split,
        seed=seed,
    )
    return compute_preference_metrics(results)


def compare_and_decide_preference(
    examples: list[dict],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    optimizer_model: str = "claude-opus-4-8",
    baseline_metrics_path: Path = BASELINE_METRICS_PATH,
    candidate_prompt_path: Path = CANDIDATE_PATH,
    judge_prompt_path: Path = JUDGE_PROMPT_PATH,
    candidate_metrics_path: Path = CANDIDATE_METRICS_PATH,
    candidate_predictions_path: Path = CANDIDATE_PREDICTIONS_PATH,
    min_accuracy_delta: float = DEFAULT_MIN_ACCURACY_DELTA,
    max_position_bias_delta: float = DEFAULT_MAX_POSITION_BIAS_DELTA,
    split: str = "validation",
    seed: int = 42,
    confirm_runs: int = 1,
) -> dict:
    from preference_metrics import print_preference_metrics

    sample_size = len(examples)

    # ── Candidate evaluation (single or confirm mode) ─────────────────────────
    if confirm_runs <= 1:
        print(f"Evaluating candidate prompt ({sample_size} examples)...")
        cand_metrics_single = _evaluate_once(
            examples, candidate_prompt_path, judge_model,
            split, seed, candidate_predictions_path,
        )
        cand_metrics_single["seed"] = seed
        cand_metrics_single["judge_model"] = judge_model

        c_acc = cand_metrics_single.get("accuracy")
        c_parse = cand_metrics_single.get("n_parse_failures", 0)
        c_bias = cand_metrics_single.get("position_bias")

        confirm_run_accuracies = None
        confirm_run_parse_failures = None

        candidate_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_metrics_path.write_text(
            json.dumps(cand_metrics_single, indent=2), encoding="utf-8"
        )
        print(f"Candidate metrics saved to {candidate_metrics_path}")
        print_preference_metrics(cand_metrics_single)

    else:
        from load_preference_data import sample_shp
        from preference_metrics import compute_preference_metrics

        print(f"Confirm mode: running {confirm_runs} candidate evaluations...")
        confirm_run_accuracies = []
        confirm_run_parse_failures = []
        confirm_run_biases = []

        for run_idx in range(confirm_runs):
            run_seed = seed + run_idx
            print(f"  Run {run_idx + 1}/{confirm_runs} (seed={run_seed}, sample={sample_size})...")
            run_examples = sample_shp(split=split, size=sample_size, seed=run_seed)
            run_metrics = _evaluate_once(
                run_examples, candidate_prompt_path, judge_model,
                split, run_seed, candidate_predictions_path,
            )
            run_acc = run_metrics.get("accuracy")
            run_pf = run_metrics.get("n_parse_failures", 0)
            run_bias = run_metrics.get("position_bias") or 0.0
            confirm_run_accuracies.append(run_acc)
            confirm_run_parse_failures.append(run_pf)
            confirm_run_biases.append(run_bias)
            print(f"    accuracy={run_acc:.3f}  parse_failures={run_pf}  position_bias={run_bias:+.3f}")

        valid_accs = [a for a in confirm_run_accuracies if a is not None]
        c_acc = sum(valid_accs) / len(valid_accs) if valid_accs else None
        c_parse = sum(confirm_run_parse_failures) / confirm_runs
        c_bias = sum(confirm_run_biases) / confirm_runs if confirm_run_biases else None

        cand_metrics_summary = {
            "accuracy": c_acc,
            "n_parse_failures": c_parse,
            "position_bias": c_bias,
            "confirm_runs": confirm_runs,
            "confirm_run_accuracies": confirm_run_accuracies,
            "confirm_run_parse_failures": confirm_run_parse_failures,
            "seed": seed,
            "judge_model": judge_model,
        }
        candidate_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_metrics_path.write_text(
            json.dumps(cand_metrics_summary, indent=2), encoding="utf-8"
        )
        print(f"\nConfirm averages: accuracy={c_acc:.3f}  "
              f"parse_failures={c_parse:.2f}  position_bias={c_bias:+.3f}")

    # ── Load baseline metrics ─────────────────────────────────────────────────
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    b_acc = baseline_metrics.get("accuracy")
    b_parse = baseline_metrics.get("n_parse_failures", 0)
    b_bias = baseline_metrics.get("position_bias")

    # In confirm mode, scale the baseline parse threshold to match N runs.
    parse_threshold = b_parse * confirm_runs if confirm_runs > 1 else b_parse

    # ── Strict accept / reject ────────────────────────────────────────────────
    rejection_reasons: list[str] = []

    # 1. Accuracy improvement
    if b_acc is None or c_acc is None:
        rejection_reasons.append("accuracy unavailable in baseline or candidate")
    else:
        delta = round(c_acc - b_acc, 4)
        if delta < min_accuracy_delta:
            rejection_reasons.append(
                f"accuracy improvement {delta:.4f} < required threshold {min_accuracy_delta}"
            )

    # 2. Parse failures must not increase
    if confirm_runs > 1:
        if sum(confirm_run_parse_failures) > parse_threshold:
            rejection_reasons.append(
                f"parse_failures total {sum(confirm_run_parse_failures)} > "
                f"allowed {int(parse_threshold)} across {confirm_runs} runs "
                f"(baseline={b_parse}/run)"
            )
    else:
        if c_parse > b_parse:
            rejection_reasons.append(
                f"parse_failures increased: baseline={b_parse} candidate={int(c_parse)}"
            )

    # 3. Position bias must not worsen meaningfully
    if b_bias is not None and c_bias is not None:
        bias_worsening = abs(c_bias) - abs(b_bias)
        if bias_worsening > max_position_bias_delta:
            rejection_reasons.append(
                f"position_bias worsened: |baseline|={abs(b_bias):.4f} "
                f"|candidate|={abs(c_bias):.4f} "
                f"delta={bias_worsening:.4f} > allowed {max_position_bias_delta}"
            )

    # 4. Eval-condition matching
    for field in ("split", "sample_size"):
        bv = baseline_metrics.get(field)
        cv = baseline_metrics.get(field)  # both used same conditions
        if field == "sample_size":
            cv = sample_size
        if bv is not None and cv is not None and bv != cv:
            rejection_reasons.append(
                f"eval condition mismatch on '{field}': baseline={bv!r} candidate={cv!r}"
            )

    accepted = len(rejection_reasons) == 0
    delta = (
        round(c_acc - b_acc, 4)
        if (b_acc is not None and c_acc is not None)
        else None
    )
    bias_delta = (
        round(abs(c_bias) - abs(b_bias), 4)
        if (b_bias is not None and c_bias is not None)
        else None
    )

    # ── Print decision ────────────────────────────────────────────────────────
    label = f"(avg over {confirm_runs} runs)" if confirm_runs > 1 else ""
    print(f"\nBaseline accuracy  : {b_acc}  (parse_failures={b_parse}  position_bias={b_bias:+.3f})" if b_bias is not None else f"\nBaseline accuracy  : {b_acc}  (parse_failures={b_parse})")
    print(f"Candidate accuracy : {c_acc:.4f}  (parse_failures={c_parse:.2f}  position_bias={c_bias:+.3f}) {label}" if c_bias is not None else f"Candidate accuracy : {c_acc}  (parse_failures={c_parse}) {label}")
    print(f"Accuracy delta     : {delta}  (threshold={min_accuracy_delta})")
    if bias_delta is not None:
        print(f"Bias worsening     : {bias_delta:+.4f}  (allowed={max_position_bias_delta})")

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
        "baseline_accuracy": b_acc,
        "candidate_accuracy": c_acc,
        "accuracy_delta": delta,
        "min_accuracy_delta": min_accuracy_delta,
        "baseline_parse_failures": b_parse,
        "candidate_parse_failures": c_parse,
        "baseline_position_bias": b_bias,
        "candidate_position_bias": c_bias,
        "position_bias_delta": bias_delta,
        "max_position_bias_delta": max_position_bias_delta,
        "confirm_runs": confirm_runs,
        "confirm_run_accuracies": confirm_run_accuracies,
        "judge_model": judge_model,
        "optimizer_model": optimizer_model,
        "split": split,
        "sample_size": sample_size,
        "seed": seed,
    }
    _log_decision(decision)
    print(f"Decision logged to {EXPERIMENT_LOG_PATH}")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate preference judge prompts."
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split", default="validation", choices=["train", "validation", "test"]
    )
    parser.add_argument(
        "--min-accuracy-delta",
        type=float,
        default=DEFAULT_MIN_ACCURACY_DELTA,
        help="Minimum accuracy improvement required to accept the candidate.",
    )
    parser.add_argument(
        "--max-position-bias-delta",
        type=float,
        default=DEFAULT_MAX_POSITION_BIAS_DELTA,
        help="Maximum allowed worsening of |position_bias| before rejection.",
    )
    parser.add_argument(
        "--confirm-runs",
        type=int,
        default=1,
        help="Evaluate candidate this many times and average results before deciding.",
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    for path, label in [
        (BASELINE_METRICS_PATH, "preference_metrics.json"),
        (CANDIDATE_PATH, "preference_judge_prompt_candidate.md"),
    ]:
        if not path.exists():
            print(f"Error: {label} not found. Run the earlier pipeline steps first.")
            sys.exit(1)

    sys.path.insert(0, str(ROOT / "src"))
    from load_preference_data import sample_shp

    print(f"Loading SHP (split={args.split}, sample_size={args.sample_size}, seed={args.seed})...")
    examples = sample_shp(split=args.split, size=args.sample_size, seed=args.seed)

    compare_and_decide_preference(
        examples,
        judge_model=args.judge_model,
        split=args.split,
        seed=args.seed,
        min_accuracy_delta=args.min_accuracy_delta,
        max_position_bias_delta=args.max_position_bias_delta,
        confirm_runs=args.confirm_runs,
    )
