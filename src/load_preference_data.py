"""
Load and normalize Stanford Human Preferences (SHP) examples.

SHP provides pairwise comparisons between two Reddit responses to a post.
Human winner is inferred from score_A vs score_B (Reddit upvote counts).

Label mapping (from SHP paper, verifiable via inspect_shp.py):
    labels=1  →  response A was preferred  (score_A > score_B)
    labels=0  →  response B was preferred  (score_B > score_A)

We use score_A vs score_B as the primary source for human_winner because it
is directly interpretable. The raw labels field is stored for reference.
On exact ties (rare), labels is used as a tiebreak.
"""
import os
import random

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

SHP_AVAILABLE_SPLITS = ("train", "validation", "test")


def load_shp_split(split: str = "validation", seed: int = 42) -> list[dict]:
    """
    Load all rows from one SHP split and return a shuffled, normalized list.

    Each element has:
        index        — position after shuffling
        history      — Reddit post / conversation context (human_ref field)
        response_a   — first candidate response (human_ref_A)
        response_b   — second candidate response (human_ref_B)
        score_a      — Reddit upvotes for A
        score_b      — Reddit upvotes for B
        label        — raw SHP label (1=A preferred, 0=B preferred)
        human_winner — "A" or "B" derived from score_a vs score_b
    """
    if split not in SHP_AVAILABLE_SPLITS:
        raise ValueError(f"split must be one of {SHP_AVAILABLE_SPLITS}, got {split!r}")

    hf_token = os.getenv("HF_TOKEN")
    ds = load_dataset("stanfordnlp/SHP", split=split, token=hf_token)

    rows = []
    for item in ds:
        score_a = item.get("score_A") or 0
        score_b = item.get("score_B") or 0
        raw_label = item.get("labels")  # 1=A preferred, 0=B preferred per SHP paper

        if score_a > score_b:
            human_winner = "A"
        elif score_b > score_a:
            human_winner = "B"
        else:
            # Exact score tie — fall back to the labels field.
            # TODO: re-verify this mapping with inspect_shp.py if SHP schema changes.
            human_winner = "A" if raw_label == 1 else "B"

        rows.append({
            "history": item.get("history", ""),
            "response_a": item.get("human_ref_A", ""),
            "response_b": item.get("human_ref_B", ""),
            "score_a": score_a,
            "score_b": score_b,
            "label": raw_label,
            "human_winner": human_winner,
        })

    rng = random.Random(seed)
    rng.shuffle(rows)

    for i, row in enumerate(rows):
        row["index"] = i

    return rows


def sample_shp(split: str, size: int, seed: int = 42) -> list[dict]:
    """Return up to `size` examples from the specified SHP split."""
    rows = load_shp_split(split=split, seed=seed)
    rng = random.Random(seed)
    if size >= len(rows):
        return rows
    return rng.sample(rows, size)


if __name__ == "__main__":
    import json

    rows = load_shp_split("validation")
    print(f"Loaded {len(rows)} validation examples.")
    winner_counts = {"A": 0, "B": 0}
    for r in rows:
        winner_counts[r["human_winner"]] += 1
    print(f"Human winner distribution: {winner_counts}")
    print("\nFirst example (truncated):")
    ex = rows[0]
    preview = {
        k: (str(v)[:120] + "...") if isinstance(v, str) and len(v) > 120 else v
        for k, v in ex.items()
    }
    print(json.dumps(preview, indent=2))
