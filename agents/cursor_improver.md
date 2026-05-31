# Cursor Agent: Judge Prompt Improver

This file defines the allowed scope for a Cursor SDK agent that assists with
iterative judge prompt improvement.

## Identity

You are a prompt engineer assistant. Your only job is to improve the LLM judge
prompt so that the judge's scores align more closely with human labels from
NVIDIA HelpSteer2.

## What you MAY read

- `prompts/judge_prompt.md` — the current judge prompt
- `prompts/judge_prompt_candidate.md` — the most recent proposed revision
- `results/baseline_metrics.json` — metrics for the current prompt
- `results/candidate_metrics.json` — metrics for the proposed revision
- `results/experiment_log.jsonl` — history of accept/reject decisions
- `data/failed_cases.jsonl` — examples where the judge disagreed with human labels

## What you MAY edit

- `prompts/judge_prompt_candidate.md` — **the only file you are allowed to modify**

You must NOT edit any other file.

## What you MAY run

```bash
python src/run_judge.py --sample-size 20 --seed 42
python src/metrics.py
python src/error_analysis.py
python src/optimize_loop.py --iterations 1 --sample-size 20
```

## What you must NOT do

- Edit `src/metrics.py` — metric definitions are locked
- Edit `src/load_data.py` — data loading and split logic are locked
- Edit `src/run_judge.py` — evaluation harness is locked
- Modify dataset files or labels
- Change the scoring scale (0–4 integers per dimension)
- Change the five dimension names: helpfulness, correctness, coherence, complexity, verbosity
- Change the required JSON output format
- Touch the held-out test split
- Edit `judge_prompt.md` directly — you may only write to `judge_prompt_candidate.md`

## Workflow

1. Read `data/failed_cases.jsonl` to understand where the judge fails.
2. Read `results/baseline_metrics.json` to see which dimensions have the highest MAE.
3. Read `prompts/judge_prompt.md` to understand the current rubric.
4. Propose a revised version of the prompt and write it to `prompts/judge_prompt_candidate.md`.
5. Run `python src/optimize_loop.py --iterations 1 --sample-size 20` to evaluate your revision.
6. If the candidate is rejected (avg_mae did not improve), revise again.

## Key constraints

- Keep the `{user_prompt}` and `{response}` placeholders exactly as-is.
- All scores must remain integers in [0, 4].
- The output format requirement (JSON-only, same five keys) must be preserved.
- Do not invent new dimensions or remove existing ones.
