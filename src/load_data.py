"""
Load and split the NVIDIA HelpSteer2 dataset.

HelpSteer2 only ships a "train" split; we carve out dev and test ourselves
using a fixed seed so every script sees the same partition.
"""
import argparse
import os
import random

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

DIMENSIONS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

# Fraction of total data reserved for test (never touched by the optimizer).
TEST_FRACTION = 0.15
# Fraction of remaining data used as dev (what the optimizer sees).
DEV_FRACTION = 0.15


def load_helpsteer2(seed: int = 42) -> dict[str, list[dict]]:
    """
    Return {"train": [...], "dev": [...], "test": [...]} where each element is a
    plain dict with keys: prompt, response, helpfulness, correctness, coherence,
    complexity, verbosity.
    """
    hf_token = os.getenv("HF_TOKEN")
    ds = load_dataset("nvidia/HelpSteer2", split="train", token=hf_token)

    rows = []
    for item in ds:
        rows.append({
            "prompt": item["prompt"],
            "response": item["response"],
            "helpfulness": item["helpfulness"],
            "correctness": item["correctness"],
            "coherence": item["coherence"],
            "complexity": item["complexity"],
            "verbosity": item["verbosity"],
        })

    rng = random.Random(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_test = max(1, int(n * TEST_FRACTION))
    n_dev = max(1, int(n * DEV_FRACTION))

    test = rows[:n_test]
    dev = rows[n_test: n_test + n_dev]
    train = rows[n_test + n_dev:]

    return {"train": train, "dev": dev, "test": test}


def sample_split(split: list[dict], size: int, seed: int = 42) -> list[dict]:
    """Return up to `size` examples from `split`, sampled with the given seed."""
    rng = random.Random(seed)
    if size >= len(split):
        return split
    return rng.sample(split, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show HelpSteer2 split sizes.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = load_helpsteer2(seed=args.seed)
    for name, data in splits.items():
        print(f"{name}: {len(data)} examples")
    print("\nSample row (dev[0]):")
    import json
    print(json.dumps(splits["dev"][0], indent=2))
