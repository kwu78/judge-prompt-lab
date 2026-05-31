"""
MVP 3: Ask an optimizer LLM to propose a revised judge prompt.

Reads:
  prompts/judge_prompt.md
  results/baseline_metrics.json
  data/failed_cases.jsonl
  prompts/optimizer_prompt.md

Writes:
  prompts/judge_prompt_candidate.md
"""
import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
JUDGE_PROMPT_PATH = ROOT / "prompts" / "judge_prompt.md"
OPTIMIZER_PROMPT_PATH = ROOT / "prompts" / "optimizer_prompt.md"
METRICS_PATH = ROOT / "results" / "baseline_metrics.json"
FAILED_CASES_PATH = ROOT / "data" / "failed_cases.jsonl"
CANDIDATE_PATH = ROOT / "prompts" / "judge_prompt_candidate.md"

DEFAULT_OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", "claude-opus-4-8")
# Cap how many failed cases we send to the optimizer to keep context manageable.
MAX_FAILED_CASES = 10


def load_failed_cases(path: Path, limit: int = MAX_FAILED_CASES) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    # Already sorted by total_error descending from error_analysis.py
    return cases[:limit]


def build_optimizer_message(
    optimizer_prompt: str,
    judge_prompt: str,
    metrics: dict,
    failed_cases: list[dict],
) -> str:
    """Assemble the full message sent to the optimizer model."""
    cases_text = json.dumps(failed_cases, indent=2)
    metrics_text = json.dumps(metrics, indent=2)

    return (
        f"{optimizer_prompt}\n\n"
        "---\n\n"
        "## Current judge prompt\n\n"
        f"{judge_prompt}\n\n"
        "---\n\n"
        "## Current metrics\n\n"
        f"```json\n{metrics_text}\n```\n\n"
        "---\n\n"
        "## Worst failure cases (sorted by total absolute error, descending)\n\n"
        f"```json\n{cases_text}\n```\n\n"
        "---\n\n"
        "Now return the full revised judge prompt."
    )


def propose_revised_prompt(
    model: str = DEFAULT_OPTIMIZER_MODEL,
    judge_prompt_path: Path = JUDGE_PROMPT_PATH,
    optimizer_prompt_path: Path = OPTIMIZER_PROMPT_PATH,
    metrics_path: Path = METRICS_PATH,
    failed_cases_path: Path = FAILED_CASES_PATH,
    output_path: Path = CANDIDATE_PATH,
) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    judge_prompt = judge_prompt_path.read_text(encoding="utf-8")
    optimizer_prompt = optimizer_prompt_path.read_text(encoding="utf-8")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    failed_cases = load_failed_cases(failed_cases_path)

    message_text = build_optimizer_message(optimizer_prompt, judge_prompt, metrics, failed_cases)

    print(f"Calling optimizer model ({model})...")
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=1,  # some creativity for prompt revision
        messages=[{"role": "user", "content": message_text}],
    )
    revised_prompt = msg.content[0].text.strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(revised_prompt, encoding="utf-8")
    print(f"Candidate prompt saved to {output_path}")
    return revised_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a revised judge prompt candidate.")
    parser.add_argument("--optimizer-model", type=str, default=DEFAULT_OPTIMIZER_MODEL)
    parser.add_argument("--judge-prompt", type=str, default=str(JUDGE_PROMPT_PATH))
    parser.add_argument("--metrics", type=str, default=str(METRICS_PATH))
    parser.add_argument("--failed-cases", type=str, default=str(FAILED_CASES_PATH))
    parser.add_argument("--output", type=str, default=str(CANDIDATE_PATH))
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    for path, label in [
        (args.metrics, "--metrics"),
        (args.failed_cases, "--failed-cases"),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} file not found: {path}")
            print("Run run_judge.py, metrics.py, and error_analysis.py first.")
            sys.exit(1)

    propose_revised_prompt(
        model=args.optimizer_model,
        judge_prompt_path=Path(args.judge_prompt),
        metrics_path=Path(args.metrics),
        failed_cases_path=Path(args.failed_cases),
        output_path=Path(args.output),
    )
