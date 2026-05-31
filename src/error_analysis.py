"""
Identify the worst failure cases and save them to data/failed_cases.jsonl.

A "failure case" is any example where the judge's total absolute error
(sum across all five dimensions) is high. The top-N worst cases are saved.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = ROOT / "results" / "predictions.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "failed_cases.jsonl"

DIMENSIONS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
DEFAULT_TOP_N = 20


def score_record(record: dict) -> float | None:
    """Total absolute error for a single prediction record."""
    if record.get("parse_failed") or not record.get("judge_scores"):
        return None
    return sum(
        abs(record["judge_scores"][d] - record["human_scores"][d]) for d in DIMENSIONS
    )


def load_predictions(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def identify_failures(records: list[dict], top_n: int = DEFAULT_TOP_N) -> list[dict]:
    scored = []
    for r in records:
        total_err = score_record(r)
        if total_err is not None:
            scored.append((total_err, r))

    scored.sort(key=lambda t: t[0], reverse=True)

    failures = []
    for total_err, r in scored[:top_n]:
        dim_errors = {d: abs(r["judge_scores"][d] - r["human_scores"][d]) for d in DIMENSIONS}
        failures.append({
            "index": r["index"],
            "prompt": r["prompt"],
            "response": r["response"],
            "human_scores": r["human_scores"],
            "judge_scores": r["judge_scores"],
            "dim_errors": dim_errors,
            "total_error": total_err,
        })
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save worst failure cases.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="Number of worst cases to save.")
    args = parser.parse_args()

    records = load_predictions(Path(args.input))
    failures = identify_failures(records, top_n=args.top_n)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for case in failures:
            f.write(json.dumps(case) + "\n")

    print(f"Saved {len(failures)} failure cases to {out}")
    if failures:
        worst = failures[0]
        print(f"\nWorst case (index {worst['index']}, total_error={worst['total_error']}):")
        print(f"  human_scores: {worst['human_scores']}")
        print(f"  judge_scores: {worst['judge_scores']}")
        print(f"  dim_errors:   {worst['dim_errors']}")
