"""
Compute preference accuracy metrics from preference_predictions.jsonl.

Reads:  results/preference_predictions.jsonl
Writes: results/preference_metrics.json
"""
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
PREDICTIONS_PATH = ROOT / "results" / "preference_predictions.jsonl"
METRICS_PATH = ROOT / "results" / "preference_metrics.json"


def compute_preference_metrics(records: list[dict]) -> dict:
    n_total = len(records)
    n_parse_failures = sum(1 for r in records if r.get("parse_failure", False))
    parsed = [r for r in records if not r.get("parse_failure", False)]
    n_parsed = len(parsed)

    accuracy = a_pick_rate = b_pick_rate = human_a_rate = human_b_rate = position_bias = None

    if n_parsed > 0:
        correct = sum(1 for r in parsed if r["judge_winner"] == r["human_winner"])
        accuracy = correct / n_parsed

        judge_counts = Counter(r["judge_winner"] for r in parsed)
        human_counts = Counter(r["human_winner"] for r in parsed)

        a_pick_rate = judge_counts.get("A", 0) / n_parsed
        b_pick_rate = judge_counts.get("B", 0) / n_parsed
        human_a_rate = human_counts.get("A", 0) / n_parsed
        human_b_rate = human_counts.get("B", 0) / n_parsed
        # positive = judge over-picks A relative to humans; negative = over-picks B
        position_bias = a_pick_rate - human_a_rate

    split = records[0].get("split", "unknown") if records else "unknown"

    return {
        "n_examples": n_total,
        "n_parse_failures": n_parse_failures,
        "n_evaluated": n_parsed,
        "accuracy": accuracy,
        "a_pick_rate": a_pick_rate,
        "b_pick_rate": b_pick_rate,
        "human_a_rate": human_a_rate,
        "human_b_rate": human_b_rate,
        "position_bias": position_bias,
        "split": split,
        "sample_size": n_total,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def print_preference_metrics(m: dict) -> None:
    sep = "─" * 50
    print(f"\n{sep}")
    print("  SHP Pairwise Preference Metrics")
    print(sep)
    print(f"  Split          : {m['split']}")
    print(f"  Examples       : {m['n_examples']}")
    print(f"  Parse failures : {m['n_parse_failures']}")
    print(f"  Evaluated      : {m['n_evaluated']}")
    if m["accuracy"] is not None:
        print(f"  Accuracy       : {m['accuracy']:.3f}")
        print(f"  Judge  A rate  : {m['a_pick_rate']:.3f}   B rate : {m['b_pick_rate']:.3f}")
        print(f"  Human  A rate  : {m['human_a_rate']:.3f}   B rate : {m['human_b_rate']:.3f}")
        bias = m["position_bias"]
        direction = "over-picks A" if bias > 0.02 else ("over-picks B" if bias < -0.02 else "balanced")
        print(f"  Position bias  : {bias:+.3f}  ({direction})")
    print(sep + "\n")


if __name__ == "__main__":
    if not PREDICTIONS_PATH.exists():
        print(f"Error: {PREDICTIONS_PATH} not found.")
        print("Run: python src/run_preference_judge.py --split validation --sample-size 50 --seed 42")
        sys.exit(1)

    records = []
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found in predictions file.")
        sys.exit(1)

    metrics = compute_preference_metrics(records)
    print_preference_metrics(metrics)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")
