"""
Inspect the Stanford Human Preferences (SHP) dataset.

Prints N examples from the specified split, showing all fields and
an inferred human winner based on score_A vs score_B (when available).
Use this script to confirm the label mapping before relying on
load_preference_data.py.

Usage:
    python src/inspect_shp.py
    python src/inspect_shp.py --split validation --sample-size 5 --seed 42
"""
import argparse
import os
import random

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SHP dataset examples.")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    print(f"Loading stanfordnlp/SHP (split={args.split})...")
    ds = load_dataset("stanfordnlp/SHP", split=args.split, token=hf_token)
    print(f"Loaded {len(ds)} examples.\n")
    print(f"Column names: {ds.column_names}\n")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), min(args.sample_size, len(ds)))

    for rank, idx in enumerate(indices):
        row = ds[idx]
        print("=" * 70)
        print(f"Example {rank + 1}/{args.sample_size}  (dataset index={idx}, split={args.split})")
        print("=" * 70)

        # History / prompt — try likely field names
        history = row.get("history") or row.get("context") or row.get("prompt") or "[FIELD NOT FOUND]"
        print(f"HISTORY (first 400 chars):\n{str(history)[:400]}\n")

        # Response A
        resp_a = row.get("human_ref_A") or row.get("response_A") or row.get("responseA") or "[FIELD NOT FOUND]"
        print(f"RESPONSE A (first 200 chars):\n{str(resp_a)[:200]}\n")

        # Response B
        resp_b = row.get("human_ref_B") or row.get("response_B") or row.get("responseB") or "[FIELD NOT FOUND]"
        print(f"RESPONSE B (first 200 chars):\n{str(resp_b)[:200]}\n")

        # Scores
        score_a = row.get("score_A")
        score_b = row.get("score_B")
        print(f"score_A : {score_a}")
        print(f"score_B : {score_b}")

        # Raw label field
        label = row.get("labels") if "labels" in row else row.get("label")
        print(f"labels  : {label}")

        # Other metadata fields
        for field in ("domain", "score_ratio", "upvote_ratio", "seconds_difference"):
            if field in row:
                print(f"{field} : {row[field]}")

        # Infer winner from scores
        if score_a is not None and score_b is not None:
            try:
                inferred = "A" if float(score_a) > float(score_b) else "B"
                print(f"\nInferred winner (score_A > score_B) : {inferred}")
            except (TypeError, ValueError):
                print("\nCould not infer winner — non-numeric scores.")
        else:
            print("\nCould not infer winner — score fields missing.")

        if label is not None:
            print(
                f"Cross-check: labels={label}  "
                f"— verify whether 1 means A or B using the inferred winner above."
            )

        print()


if __name__ == "__main__":
    main()
