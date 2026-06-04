"""
Print a concise summary of the SHP pairwise preference experiment.

Reads:
  results/preference_metrics.json           — basic accuracy / pick-rate metrics
  results/preference_error_summary.json     — domain / score_ratio breakdown (optional)
  results/preference_experiment_log.jsonl   — optimization loop history (optional)

Usage:
    python src/summarize_preference_experiment.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
METRICS_PATH = ROOT / "results" / "preference_metrics.json"
ERROR_SUMMARY_PATH = ROOT / "results" / "preference_error_summary.json"
EXPERIMENT_LOG_PATH = ROOT / "results" / "preference_experiment_log.jsonl"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, fmt=".3f") -> str:
    if v is None:
        return "n/a"
    return format(v, fmt)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize() -> None:
    metrics = _load_json(METRICS_PATH)
    if metrics is None:
        print(f"Error: {METRICS_PATH} not found.")
        print("Run: python src/run_preference_judge.py && python src/preference_metrics.py")
        sys.exit(1)

    error_summary = _load_json(ERROR_SUMMARY_PATH)   # optional
    log = _load_jsonl(EXPERIMENT_LOG_PATH)            # optional

    sep = "─" * 56

    # ── Overview ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SHP Pairwise Preference — Experiment Summary")
    print(sep)
    print(f"  Split            : {metrics.get('split', 'n/a')}")
    print(f"  Sample size      : {metrics.get('n_examples', 'n/a')}")
    print(f"  Evaluated        : {metrics.get('n_evaluated', 'n/a')}")
    print(f"  Parse failures   : {metrics.get('n_parse_failures', 'n/a')}")
    print(f"  Accuracy         : {_fmt(metrics.get('accuracy'))}")

    # ── Pick rates ────────────────────────────────────────────────────────────
    print(f"\n  Pick rates")
    print(f"    Judge  A : {_fmt(metrics.get('a_pick_rate'))}   B : {_fmt(metrics.get('b_pick_rate'))}")
    print(f"    Human  A : {_fmt(metrics.get('human_a_rate'))}   B : {_fmt(metrics.get('human_b_rate'))}")
    bias = metrics.get("position_bias")
    if bias is not None:
        direction = "over-picks A" if bias > 0.02 else ("over-picks B" if bias < -0.02 else "balanced")
        print(f"    Position bias : {bias:+.3f}  ({direction})")
    else:
        print(f"    Position bias : n/a")

    # ── Score_ratio: correct vs incorrect ─────────────────────────────────────
    if error_summary:
        rc = error_summary.get("avg_score_ratio_correct")
        ri = error_summary.get("avg_score_ratio_incorrect")
        if rc is not None or ri is not None:
            print(f"\n  Avg score_ratio  (higher = clearer human preference signal)")
            print(f"    Correct   : {_fmt(rc)}")
            print(f"    Incorrect : {_fmt(ri)}")

    # ── Score_ratio bucket error rates ────────────────────────────────────────
    if error_summary:
        buckets = error_summary.get("score_ratio_buckets", {})
        has_data = any(v.get("n", 0) > 0 for v in buckets.values())
        if has_data:
            print(f"\n  Error rate by score_ratio bucket")
            print(f"    {'bucket':<12} {'n':>4}  {'error_rate':>10}")
            print(f"    {'─'*12}  {'─'*4}  {'─'*10}")
            for label, st in buckets.items():
                n = st.get("n", 0)
                er = st.get("error_rate")
                if n == 0:
                    print(f"    {label:<12} {0:>4}  {'n/a':>10}")
                else:
                    print(f"    {label:<12} {n:>4}  {_fmt(er):>10}")

    # ── Domain accuracy ───────────────────────────────────────────────────────
    if error_summary:
        domain_rows = error_summary.get("domain_accuracy", [])
        has_domain = any(r.get("domain") not in (None, "unknown") for r in domain_rows)
        if has_domain:
            print(f"\n  Top domains by sample count")
            print(f"    {'domain':<35} {'n':>4}  {'acc':>6}")
            print(f"    {'─'*35}  {'─'*4}  {'─'*6}")
            for row in domain_rows:
                d = row.get("domain") or "unknown"
                n = row.get("n", 0)
                acc = row.get("accuracy")
                print(f"    {d:<35} {n:>4}  {_fmt(acc):>6}")

    # ── Label / score conflict note ───────────────────────────────────────────
    if error_summary:
        lsc = error_summary.get("label_score_conflicts", {})
        n_conf = lsc.get("n_conflicts", 0)
        n_agree = lsc.get("n_agrees", 0)
        if n_conf + n_agree > 0:
            pct = n_conf / (n_conf + n_agree) * 100
            print(f"\n  Label/score agreement")
            print(f"    Agree    : n={n_agree}   acc={_fmt(lsc.get('accuracy_agrees'))}")
            print(f"    Conflict : n={n_conf}   acc={_fmt(lsc.get('accuracy_conflicts'))}  "
                  f"({pct:.1f}% of scoreable rows)")

    # ── Optimization loop history ─────────────────────────────────────────────
    if log:
        n_accepted = sum(1 for e in log if e.get("accepted"))
        n_rejected = len(log) - n_accepted

        accepted_entries = [e for e in log if e.get("accepted")]
        best_acc = max((e.get("candidate_accuracy", 0) for e in accepted_entries), default=None)

        initial_acc = log[0].get("baseline_accuracy")
        current_acc = metrics.get("accuracy")

        parse_trend = [e.get("candidate_parse_failures", "?") for e in log]

        # Categorise all rejection reasons across every logged iteration.
        reason_categories = Counter()
        for entry in log:
            for reason in entry.get("rejection_reasons") or []:
                r = reason.lower()
                if "parse_failure" in r or "parse failure" in r:
                    reason_categories["parse_failures"] += 1
                elif "position_bias" in r:
                    reason_categories["position_bias"] += 1
                elif "accuracy" in r:
                    reason_categories["accuracy"] += 1
                else:
                    reason_categories["other"] += 1

        # Position bias trend across logged iterations.
        bias_trend = [
            f"{e.get('candidate_position_bias'):+.3f}"
            if e.get("candidate_position_bias") is not None else "n/a"
            for e in log
        ]

        last = log[-1]
        last_reasons = last.get("rejection_reasons") or []

        print(f"\n  Optimization loop  ({len(log)} iterations logged)")
        print(f"    Accepted         : {n_accepted}")
        print(f"    Rejected         : {n_rejected}")
        if initial_acc is not None:
            print(f"    Initial accuracy : {_fmt(initial_acc)}")
        if best_acc is not None:
            print(f"    Best accepted    : {_fmt(best_acc)}")
        if current_acc is not None:
            print(f"    Current baseline : {_fmt(current_acc)}")
        if initial_acc is not None and current_acc is not None and n_accepted > 0:
            abs_imp = current_acc - initial_acc
            rel_imp = abs_imp / initial_acc * 100 if initial_acc > 0 else 0
            print(f"    Improvement      : {abs_imp:+.3f}  ({rel_imp:+.1f}%)")

        # Parse failures and position bias trends.
        print(f"    Candidate parse_failures trend   : {parse_trend}")
        if any(b != "n/a" for b in bias_trend):
            print(f"    Candidate position_bias trend    : {bias_trend}")

        # Rejection reason breakdown by category.
        if reason_categories:
            print(f"    Rejection reasons by category:")
            for cat, count in reason_categories.most_common():
                print(f"      {cat:<20}: {count}")

        if last_reasons:
            print(f"    Latest rejection reasons:")
            for r in last_reasons:
                print(f"      - {r}")

    # ── Interpretation note ───────────────────────────────────────────────────
    print(f"\n  Note: SHP reflects Reddit community preference signals (upvotes),")
    print(f"  not objective answer correctness. Accuracy measures alignment with")
    print(f"  that community signal. Near-random baselines score around 0.50.")

    print(sep + "\n")


if __name__ == "__main__":
    summarize()
