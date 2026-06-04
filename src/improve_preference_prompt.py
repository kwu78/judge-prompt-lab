"""
Ask an optimizer LLM to propose a revised preference judge prompt.

Reads:
  prompts/preference_judge_prompt.md
  prompts/preference_optimizer_prompt.md
  results/preference_metrics.json
  results/preference_error_summary.json  (optional)
  results/preference_failed_cases.jsonl

Writes:
  prompts/preference_judge_prompt_candidate.md
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
JUDGE_PROMPT_PATH = ROOT / "prompts" / "preference_judge_prompt.md"
OPTIMIZER_PROMPT_PATH = ROOT / "prompts" / "preference_optimizer_prompt.md"
METRICS_PATH = ROOT / "results" / "preference_metrics.json"
ERROR_SUMMARY_PATH = ROOT / "results" / "preference_error_summary.json"
FAILED_CASES_PATH = ROOT / "results" / "preference_failed_cases.jsonl"
CANDIDATE_PATH = ROOT / "prompts" / "preference_judge_prompt_candidate.md"

DEFAULT_OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", "claude-opus-4-8")
MAX_FAILED_CASES = 10


def _load_jsonl(path: Path, limit: int) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases[:limit]


def _load_json_optional(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_optimizer_message(
    optimizer_prompt: str,
    judge_prompt: str,
    metrics: dict,
    error_summary: dict | None,
    failed_cases: list[dict],
) -> str:
    metrics_text = json.dumps(metrics, indent=2)
    cases_text = json.dumps(failed_cases, indent=2)

    parts = [
        optimizer_prompt,
        "---",
        "## Current preference judge prompt",
        "",
        judge_prompt,
        "",
        "---",
        "## Current metrics",
        "",
        f"```json\n{metrics_text}\n```",
        "",
    ]

    if error_summary is not None:
        summary_text = json.dumps(error_summary, indent=2)
        parts += [
            "---",
            "## Error analysis summary",
            "",
            f"```json\n{summary_text}\n```",
            "",
        ]

    parts += [
        "---",
        "## Failed cases (examples where judge disagreed with human preference)",
        "",
        f"```json\n{cases_text}\n```",
        "",
        "---",
        "Now return the full revised preference judge prompt.",
    ]

    return "\n".join(parts)


def propose_revised_preference_prompt(
    model: str = DEFAULT_OPTIMIZER_MODEL,
    judge_prompt_path: Path = JUDGE_PROMPT_PATH,
    optimizer_prompt_path: Path = OPTIMIZER_PROMPT_PATH,
    metrics_path: Path = METRICS_PATH,
    error_summary_path: Path = ERROR_SUMMARY_PATH,
    failed_cases_path: Path = FAILED_CASES_PATH,
    output_path: Path = CANDIDATE_PATH,
) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    judge_prompt = judge_prompt_path.read_text(encoding="utf-8")
    optimizer_prompt = optimizer_prompt_path.read_text(encoding="utf-8")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    error_summary = _load_json_optional(error_summary_path)
    failed_cases = _load_jsonl(failed_cases_path, limit=MAX_FAILED_CASES)

    message_text = build_optimizer_message(
        optimizer_prompt, judge_prompt, metrics, error_summary, failed_cases
    )

    print(f"Calling optimizer model ({model})...")
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=1,
        messages=[{"role": "user", "content": message_text}],
    )
    revised_prompt = msg.content[0].text.strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(revised_prompt, encoding="utf-8")
    print(f"Candidate prompt saved to {output_path}")
    return revised_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a revised preference judge prompt candidate."
    )
    parser.add_argument("--optimizer-model", default=DEFAULT_OPTIMIZER_MODEL)
    parser.add_argument("--judge-prompt", default=str(JUDGE_PROMPT_PATH))
    parser.add_argument("--metrics", default=str(METRICS_PATH))
    parser.add_argument("--failed-cases", default=str(FAILED_CASES_PATH))
    parser.add_argument("--output", default=str(CANDIDATE_PATH))
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
            print(
                "Run: run_preference_judge.py → preference_metrics.py → "
                "preference_error_analysis.py first."
            )
            sys.exit(1)

    propose_revised_preference_prompt(
        model=args.optimizer_model,
        judge_prompt_path=Path(args.judge_prompt),
        metrics_path=Path(args.metrics),
        failed_cases_path=Path(args.failed_cases),
        output_path=Path(args.output),
    )
