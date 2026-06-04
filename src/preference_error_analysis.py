"""
Read-only error analysis for the SHP preference pipeline.

Reads:  results/preference_predictions.jsonl
Writes: results/preference_failed_cases.jsonl
        results/preference_error_summary.json  (optional)

Enriches all parsed predictions with domain, score_A, score_B, score_ratio,
and the raw labels field by reloading the SHP dataset and matching on history
text. If the dataset is unavailable, those fields are None and the analysis
proceeds with whatever is available.

Note: SHP reflects community preference signals (Reddit upvotes), not
necessarily objective answer quality. Accuracy here measures alignment with
that community signal, not factual correctness.

Usage:
    python src/preference_error_analysis.py
    python src/preference_error_analysis.py --top-domains 10
"""
import argparse
import datetime
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
PREDICTIONS_PATH = ROOT / "results" / "preference_predictions.jsonl"
FAILED_CASES_PATH = ROOT / "results" / "preference_failed_cases.jsonl"
SUMMARY_PATH = ROOT / "results" / "preference_error_summary.json"

RATIO_BUCKETS = [
    (1.00, 1.25, "1.00–1.25"),
    (1.25, 1.50, "1.25–1.50"),
    (1.50, 2.00, "1.50–2.00"),
    (2.00, float("inf"), "2.00+"),
]


# ---------------------------------------------------------------------------
# SHP enrichment lookup
# ---------------------------------------------------------------------------

def _load_shp_lookup(split: str) -> dict[str, dict]:
    """
    Build {history_text → {domain, score_A, score_B, score_ratio, label}} from
    the raw SHP dataset. Matches on exact history string.

    Returns an empty dict if the dataset cannot be loaded so the rest of the
    analysis proceeds without enrichment.
    """
    try:
        from datasets import load_dataset
        hf_token = os.getenv("HF_TOKEN")
        print(f"Loading stanfordnlp/SHP ({split}) for enrichment...")
        ds = load_dataset("stanfordnlp/SHP", split=split, token=hf_token)
        lookup: dict[str, dict] = {}
        for item in ds:
            history = item.get("history", "")
            if history:
                lookup[history] = {
                    "domain": item.get("domain"),
                    "score_A": item.get("score_A"),
                    "score_B": item.get("score_B"),
                    "score_ratio": item.get("score_ratio"),
                    "label": item.get("labels"),  # 1=A preferred, 0=B preferred
                }
        print(f"Enrichment lookup built: {len(lookup)} rows.\n")
        return lookup
    except Exception as exc:
        print(f"[warn] Could not load SHP dataset for enrichment: {exc}")
        print("[warn] Proceeding without domain / score fields.\n")
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _preference_ratio(extra: dict) -> float | None:
    """
    Return a preference-strength ratio >= 1.0.

    Uses score_ratio from SHP if available; falls back to max/min of the raw
    scores. Returns None when scores are missing or both are zero.
    """
    raw = _safe_float(extra.get("score_ratio"))
    if raw is not None:
        return raw if raw >= 1.0 else (1.0 / raw if raw > 0 else None)
    sa = _safe_float(extra.get("score_A"))
    sb = _safe_float(extra.get("score_B"))
    if sa is None or sb is None:
        return None
    lo = min(sa, sb)
    hi = max(sa, sb)
    return hi / lo if lo > 0 else None


def _bucket(ratio: float | None) -> str | None:
    if ratio is None:
        return None
    for lo, hi, label in RATIO_BUCKETS:
        if lo <= ratio < hi:
            return label
    return RATIO_BUCKETS[-1][2]  # catch anything >= 2.0


def _acc_str(n_correct: int, n_total: int) -> str:
    if n_total == 0:
        return "n/a"
    return f"{n_correct / n_total:.3f}  (n={n_total})"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_analysis(
    predictions_path: Path,
    failed_path: Path,
    summary_path: Path,
    top_domains: int,
) -> None:
    if not predictions_path.exists():
        print(f"Error: {predictions_path} not found.")
        print("Run: python src/run_preference_judge.py first.")
        sys.exit(1)

    records = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found.")
        sys.exit(1)

    # ── Basic partition ───────────────────────────────────────────────────────
    n_total = len(records)
    n_parse_failures = sum(1 for r in records if r.get("parse_failure", False))
    parsed = [r for r in records if not r.get("parse_failure", False)]
    n_evaluated = len(parsed)
    correct = [r for r in parsed if r["judge_winner"] == r["human_winner"]]
    incorrect = [r for r in parsed if r["judge_winner"] != r["human_winner"]]
    n_correct = len(correct)
    n_incorrect = len(incorrect)
    accuracy = n_correct / n_evaluated if n_evaluated > 0 else None

    split = records[0].get("split", "validation")

    # ── Enrich ALL parsed records ─────────────────────────────────────────────
    lookup = _load_shp_lookup(split) if parsed else {}

    def enrich(r: dict) -> dict:
        return lookup.get(r["history"], {})

    parsed_enriched = [(r, enrich(r)) for r in parsed]

    # ── Confusion matrix ──────────────────────────────────────────────────────
    human_a_judge_b = sum(
        1 for r in incorrect if r["human_winner"] == "A" and r["judge_winner"] == "B"
    )
    human_b_judge_a = sum(
        1 for r in incorrect if r["human_winner"] == "B" and r["judge_winner"] == "A"
    )

    # ── Accuracy by domain ────────────────────────────────────────────────────
    # Bucket all parsed records by domain, sort by sample count desc.
    domain_stats: dict[str, dict[str, int]] = {}
    for r, extra in parsed_enriched:
        d = extra.get("domain") or "unknown"
        if d not in domain_stats:
            domain_stats[d] = {"n": 0, "n_correct": 0}
        domain_stats[d]["n"] += 1
        if r["judge_winner"] == r["human_winner"]:
            domain_stats[d]["n_correct"] += 1

    domain_rows = sorted(domain_stats.items(), key=lambda kv: -kv[1]["n"])

    # ── Score-ratio bucket analysis ───────────────────────────────────────────
    bucket_stats: dict[str, dict[str, int]] = {label: {"n": 0, "n_correct": 0} for _, _, label in RATIO_BUCKETS}
    n_no_ratio = 0
    for r, extra in parsed_enriched:
        ratio = _preference_ratio(extra)
        b = _bucket(ratio)
        if b is None:
            n_no_ratio += 1
            continue
        bucket_stats[b]["n"] += 1
        if r["judge_winner"] == r["human_winner"]:
            bucket_stats[b]["n_correct"] += 1

    # ── Label / score conflict analysis ──────────────────────────────────────
    # Conflict = SHP labels field disagrees with which score is higher.
    # labels=1 → A preferred; labels=0 → B preferred.
    # score-based: A preferred when score_A > score_B.
    n_conflict = n_agree = 0
    correct_agree = correct_conflict = 0

    for r, extra in parsed_enriched:
        label = extra.get("label")
        sa = _safe_float(extra.get("score_A"))
        sb = _safe_float(extra.get("score_B"))
        if label is None or sa is None or sb is None or sa == sb:
            continue  # can't determine agreement
        label_says_a = label == 1
        score_says_a = sa > sb
        is_correct = r["judge_winner"] == r["human_winner"]
        if label_says_a == score_says_a:
            n_agree += 1
            correct_agree += int(is_correct)
        else:
            n_conflict += 1
            correct_conflict += int(is_correct)

    acc_agree = correct_agree / n_agree if n_agree > 0 else None
    acc_conflict = correct_conflict / n_conflict if n_conflict > 0 else None

    # ── Avg score_ratio: correct vs incorrect ─────────────────────────────────
    correct_set = {id(r) for r in correct}
    correct_ratios = [
        _preference_ratio(extra)
        for r, extra in parsed_enriched
        if id(r) in correct_set and _preference_ratio(extra) is not None
    ]
    incorrect_set = {id(r) for r in incorrect}
    incorrect_ratios = [
        _preference_ratio(extra)
        for r, extra in parsed_enriched
        if id(r) in incorrect_set and _preference_ratio(extra) is not None
    ]

    # ── Write failed cases (incorrect predictions only, enriched) ─────────────
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(failed_path, "w", encoding="utf-8") as f:
        for r, extra in parsed_enriched:
            if r["judge_winner"] == r["human_winner"]:
                continue
            entry = {
                "index": r["index"],
                "split": r.get("split", split),
                "domain": extra.get("domain"),
                "human_winner": r["human_winner"],
                "judge_winner": r["judge_winner"],
                "score_A": extra.get("score_A"),
                "score_B": extra.get("score_B"),
                "score_ratio": extra.get("score_ratio"),
                "preference_ratio": _preference_ratio(extra),
                "label": extra.get("label"),
                "history": r["history"],
                "response_a": r["response_a"],
                "response_b": r["response_b"],
                "analysis_timestamp": ts,
            }
            if r.get("raw_output") is not None:
                entry["raw_output"] = r["raw_output"]
            f.write(json.dumps(entry) + "\n")

    # ── Build summary dict (also written to JSON) ─────────────────────────────
    summary = {
        "n_examples": n_total,
        "n_parse_failures": n_parse_failures,
        "n_evaluated": n_evaluated,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "accuracy": accuracy,
        "confusion": {
            "human_A_judge_B": human_a_judge_b,
            "human_B_judge_A": human_b_judge_a,
        },
        "avg_score_ratio_correct": (
            sum(correct_ratios) / len(correct_ratios) if correct_ratios else None
        ),
        "avg_score_ratio_incorrect": (
            sum(incorrect_ratios) / len(incorrect_ratios) if incorrect_ratios else None
        ),
        "score_ratio_buckets": {
            label: {
                "n": st["n"],
                "n_correct": st["n_correct"],
                "error_rate": round(1 - st["n_correct"] / st["n"], 4) if st["n"] > 0 else None,
            }
            for _, _, label in RATIO_BUCKETS
            for st in [bucket_stats[label]]
        },
        "label_score_conflicts": {
            "n_conflicts": n_conflict,
            "n_agrees": n_agree,
            "accuracy_agrees": acc_agree,
            "accuracy_conflicts": acc_conflict,
        },
        "domain_accuracy": [
            {
                "domain": d,
                "n": st["n"],
                "n_correct": st["n_correct"],
                "accuracy": round(st["n_correct"] / st["n"], 4) if st["n"] > 0 else None,
            }
            for d, st in domain_rows[:top_domains]
        ],
        "split": split,
        "timestamp": ts,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Print report ──────────────────────────────────────────────────────────
    sep = "─" * 54

    print(f"\n{sep}")
    print("  SHP Preference Error Analysis")
    print(sep)
    print(f"  Total examples    : {n_total}")
    print(f"  Parse failures    : {n_parse_failures}")
    print(f"  Evaluated         : {n_evaluated}")
    print(f"  Correct           : {n_correct}")
    print(f"  Incorrect         : {n_incorrect}")
    if accuracy is not None:
        print(f"  Accuracy          : {accuracy:.3f}")

    print(f"\n  Confusion (incorrect only)")
    print(f"    human=A / judge=B : {human_a_judge_b}")
    print(f"    human=B / judge=A : {human_b_judge_a}")

    # Domain accuracy
    has_domain = any(d != "unknown" for d, _ in domain_rows[:top_domains])
    if has_domain:
        print(f"\n  Accuracy by domain  (top {top_domains} by sample count)")
        print(f"    {'domain':<35} {'n':>4}  {'acc':>6}")
        print(f"    {'─'*35}  {'─'*4}  {'─'*6}")
        for d, st in domain_rows[:top_domains]:
            acc_d = st["n_correct"] / st["n"] if st["n"] > 0 else 0.0
            print(f"    {d:<35} {st['n']:>4}  {acc_d:>6.3f}")

    # Score_ratio bucket error rates
    bucket_has_data = any(bucket_stats[label]["n"] > 0 for _, _, label in RATIO_BUCKETS)
    if bucket_has_data:
        print(f"\n  Error rate by score_ratio bucket")
        print(f"    {'bucket':<12} {'n':>4}  {'correct':>7}  {'error_rate':>10}")
        print(f"    {'─'*12}  {'─'*4}  {'─'*7}  {'─'*10}")
        for _, _, label in RATIO_BUCKETS:
            st = bucket_stats[label]
            if st["n"] == 0:
                print(f"    {label:<12} {'':>4}  {'n/a':>7}  {'n/a':>10}")
            else:
                err = 1 - st["n_correct"] / st["n"]
                print(f"    {label:<12} {st['n']:>4}  {st['n_correct']:>7}  {err:>10.3f}")
        if n_no_ratio > 0:
            print(f"    ({n_no_ratio} records had no score data and were excluded)")

    # Label / score conflict
    if n_agree + n_conflict > 0:
        print(f"\n  Label vs score agreement")
        print(f"    Agree    : n={n_agree:<4}  acc={acc_agree:.3f}" if acc_agree is not None else f"    Agree    : n={n_agree}  acc=n/a")
        print(f"    Conflict : n={n_conflict:<4}  acc={acc_conflict:.3f}" if acc_conflict is not None else f"    Conflict : n={n_conflict}  acc=n/a")
        if n_conflict > 0:
            pct = n_conflict / (n_agree + n_conflict) * 100
            print(f"    ({n_conflict} rows where labels ≠ score_A>score_B  [{pct:.1f}% of scoreable rows])")
    else:
        print(f"\n  Label vs score agreement : n/a (enrichment unavailable)")

    # Score_ratio: correct vs incorrect
    if correct_ratios or incorrect_ratios:
        print(f"\n  Avg score_ratio  (higher = clearer human preference signal)")
        if correct_ratios:
            print(f"    Correct   : {sum(correct_ratios)/len(correct_ratios):.3f}  (n={len(correct_ratios)})")
        else:
            print(f"    Correct   : n/a")
        if incorrect_ratios:
            print(f"    Incorrect : {sum(incorrect_ratios)/len(incorrect_ratios):.3f}  (n={len(incorrect_ratios)})")
        else:
            print(f"    Incorrect : n/a")
    else:
        print(f"\n  Avg score_ratio : n/a (enrichment unavailable)")

    print(f"\n  Note: SHP reflects Reddit community preference signals, not")
    print(f"  necessarily objective answer quality. Accuracy measures alignment")
    print(f"  with that community signal, not factual correctness.")

    print(f"\n  Failed cases  → {failed_path}")
    print(f"  Summary JSON  → {summary_path}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read-only error analysis for SHP preference predictions."
    )
    parser.add_argument(
        "--predictions",
        default=str(PREDICTIONS_PATH),
        help="Path to preference_predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(FAILED_CASES_PATH),
        help="Where to write failed cases JSONL.",
    )
    parser.add_argument(
        "--summary",
        default=str(SUMMARY_PATH),
        help="Where to write the summary JSON.",
    )
    parser.add_argument(
        "--top-domains",
        type=int,
        default=10,
        help="How many domains to show (default: 10).",
    )
    args = parser.parse_args()

    run_analysis(
        predictions_path=Path(args.predictions),
        failed_path=Path(args.output),
        summary_path=Path(args.summary),
        top_domains=args.top_domains,
    )
