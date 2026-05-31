"""
Print a short summary of the optimization experiment.

Reads:
  results/experiment_log.jsonl      — one record per accept/reject decision
  results/baseline_metrics.json     — metrics for the current judge prompt
  results/final_test_metrics.json   — held-out test results (optional)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

EXPERIMENT_LOG_PATH = ROOT / "results" / "experiment_log.jsonl"
BASELINE_METRICS_PATH = ROOT / "results" / "baseline_metrics.json"
FINAL_TEST_METRICS_PATH = ROOT / "results" / "final_test_metrics.json"

DIMENSIONS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def fmt(val) -> str:
    return str(val) if val is not None else "n/a"


if __name__ == "__main__":
    print("=" * 55)
    print("EXPERIMENT SUMMARY")
    print("=" * 55)

    if not EXPERIMENT_LOG_PATH.exists():
        print(f"\nNo experiment log found at {EXPERIMENT_LOG_PATH}.")
        print("Run the optimization loop first:")
        print("  python src/optimize_loop.py --iterations 5 --sample-size 50")
        sys.exit(0)

    log = load_jsonl(EXPERIMENT_LOG_PATH)
    n_accepted = sum(1 for e in log if e.get("accepted"))
    n_rejected = sum(1 for e in log if not e.get("accepted"))

    print(f"\nIterations logged : {len(log)}")
    print(f"  Accepted        : {n_accepted}")
    print(f"  Rejected        : {n_rejected}")

    # Best accepted candidate on dev.
    accepted = [e for e in log if e.get("accepted") and e.get("candidate_avg_mae") is not None]
    if accepted:
        best = min(accepted, key=lambda e: e["candidate_avg_mae"])
        initial_mae = log[0].get("baseline_avg_mae")
        best_mae = best["candidate_avg_mae"]
        if initial_mae is not None and best_mae is not None:
            abs_imp = round(initial_mae - best_mae, 4)
            rel_imp = round(abs_imp / initial_mae * 100, 1)
            print(f"\nInitial dev avg_mae       : {initial_mae}")
            print(f"Best accepted dev avg_mae : {best_mae}")
            print(f"Absolute improvement      : {abs_imp}")
            print(f"Relative improvement      : {rel_imp}%")
        else:
            print(f"\nBest accepted dev avg_mae : {best_mae}")
        print(f"  Accepted at: {best.get('timestamp', 'unknown')[:19]}")
    else:
        print("\nNo candidates accepted yet.")

    # Rejection reason summary.
    all_reasons: list[str] = []
    for e in log:
        all_reasons.extend(e.get("rejection_reasons") or [])
    if all_reasons:
        from collections import Counter
        counts = Counter(r.split(":")[0].strip() for r in all_reasons)
        print("\nTop rejection reasons:")
        for reason, count in counts.most_common(5):
            print(f"  {count:>3}x  {reason}")

    # Parse failure trend.
    parse_rows = [
        (e.get("baseline_parse_failures"), e.get("candidate_parse_failures"))
        for e in log
        if e.get("baseline_parse_failures") is not None
    ]
    if parse_rows:
        print("\nParse failure trend  (baseline → candidate):")
        for i, (b, c) in enumerate(parse_rows, 1):
            arrow = "✓" if (c is not None and c <= (b or 0)) else "↑"
            print(f"  iter {i:2d}: {fmt(b):>3} → {fmt(c):>3}  {arrow}")

    # Current dev metrics.
    print()
    if BASELINE_METRICS_PATH.exists():
        m = json.loads(BASELINE_METRICS_PATH.read_text())
        meta_parts = []
        for k in ("split", "sample_size", "seed", "judge_model"):
            if k in m:
                meta_parts.append(f"{k}={m[k]!r}")
        print(f"Current dev baseline  avg_mae={m.get('avg_mae')}  "
              f"n={m.get('n_examples')}  parse_failures={m.get('n_parse_failures')}")
        if meta_parts:
            print(f"  Eval conditions: {', '.join(meta_parts)}")
        print(f"  Per-dimension MAE:")
        for dim in DIMENSIONS:
            d = m.get("dimensions", {}).get(dim, {})
            print(f"    {dim:<14} mae={fmt(d.get('mae'))}  bias={fmt(d.get('bias'))}")
    else:
        print("No baseline_metrics.json found. Run:")
        print("  python src/run_judge.py && python src/metrics.py")

    # Held-out test results.
    print()
    if FINAL_TEST_METRICS_PATH.exists():
        m = json.loads(FINAL_TEST_METRICS_PATH.read_text())
        print(f"Held-out test result  avg_mae={m.get('avg_mae')}  "
              f"n={m.get('n_examples')}  parse_failures={m.get('n_parse_failures')}")
        print(f"  Per-dimension MAE:")
        for dim in DIMENSIONS:
            d = m.get("dimensions", {}).get(dim, {})
            print(f"    {dim:<14} mae={fmt(d.get('mae'))}  bias={fmt(d.get('bias'))}")
    else:
        print("No held-out test results yet. Run:")
        print("  python src/final_eval.py --sample-size 100 --seed 42")

    print()
