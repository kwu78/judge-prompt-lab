"""
Compute metrics comparing judge predictions to human labels.

Reads:  results/predictions.jsonl
Writes: results/baseline_metrics.json  (or a custom path via --output)

Metrics per dimension:
  - mae             mean absolute error
  - within_1        fraction of examples where |judge - human| <= 1
  - pearson_r       Pearson correlation coefficient
  - spearman_r      Spearman rank correlation coefficient
  - bias            mean signed error (judge - human); positive = judge over-scores

Summary:
  - avg_mae         average MAE across all five dimensions
  - n_examples      number of examples with valid judge scores
  - n_parse_failures
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = ROOT / "results" / "predictions.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "baseline_metrics.json"

DIMENSIONS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


def load_predictions(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None

    def ranks(vals):
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_vals[j + 1][1] == sorted_vals[j][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-based average rank for ties
            for k in range(i, j + 1):
                r[sorted_vals[k][0]] = avg_rank
            i = j + 1
        return r

    return pearson(ranks(xs), ranks(ys))


def compute_metrics(records: list[dict], metadata: dict | None = None) -> dict:
    valid = [r for r in records if not r.get("parse_failed") and r.get("judge_scores")]
    n_parse_failures = len(records) - len(valid)

    per_dim: dict[str, dict] = {}
    for dim in DIMENSIONS:
        human_vals = [r["human_scores"][dim] for r in valid]
        judge_vals = [r["judge_scores"][dim] for r in valid]
        errors = [abs(j - h) for j, h in zip(judge_vals, human_vals)]
        signed = [j - h for j, h in zip(judge_vals, human_vals)]

        mae = sum(errors) / len(errors) if errors else None
        within_1 = sum(1 for e in errors if e <= 1) / len(errors) if errors else None
        bias = sum(signed) / len(signed) if signed else None

        per_dim[dim] = {
            "mae": round(mae, 4) if mae is not None else None,
            "within_1": round(within_1, 4) if within_1 is not None else None,
            "bias": round(bias, 4) if bias is not None else None,
            "pearson_r": round(pearson(human_vals, judge_vals), 4)
                if human_vals and pearson(human_vals, judge_vals) is not None else None,
            "spearman_r": round(spearman(human_vals, judge_vals), 4)
                if human_vals and spearman(human_vals, judge_vals) is not None else None,
        }

    mae_values = [per_dim[d]["mae"] for d in DIMENSIONS if per_dim[d]["mae"] is not None]
    avg_mae = round(sum(mae_values) / len(mae_values), 4) if mae_values else None

    # Metadata (split, seed, judge_model, etc.) goes first for readability.
    result: dict = {}
    if metadata:
        result.update(metadata)
    result.update({
        "n_examples": len(valid),
        "n_parse_failures": n_parse_failures,
        "avg_mae": avg_mae,
        "dimensions": per_dim,
    })
    return result


def print_metrics(m: dict) -> None:
    print(f"\nn_examples:       {m['n_examples']}")
    print(f"n_parse_failures: {m['n_parse_failures']}")
    print(f"avg_mae:          {m['avg_mae']}\n")
    print(f"{'Dimension':<14} {'MAE':>6} {'within_1':>9} {'bias':>7} {'pearson':>8} {'spearman':>9}")
    print("-" * 58)
    for dim in DIMENSIONS:
        d = m["dimensions"][dim]
        print(
            f"{dim:<14} "
            f"{str(d['mae']):>6} "
            f"{str(d['within_1']):>9} "
            f"{str(d['bias']):>7} "
            f"{str(d['pearson_r']):>8} "
            f"{str(d['spearman_r']):>9}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute judge vs. human metrics.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    records = load_predictions(Path(args.input))
    m = compute_metrics(records)
    print_metrics(m)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to {out}")
