# judge-prompt-lab

Automated calibration loop for LLM-as-judge prompts using
[NVIDIA HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) as ground truth.

## What it does

HelpSteer2 contains (prompt, response, human\_scores) triples scored on five dimensions:
**helpfulness, correctness, coherence, complexity, verbosity** — all on a 0–4 integer scale.

This project uses those human labels to calibrate an LLM judge prompt:

1. **Baseline evaluation** — run the current judge prompt over a dev sample; compute MAE,
   correlation, and within-1 accuracy vs. human labels.
2. **Failure analysis** — save the worst-disagreement examples to `data/failed_cases.jsonl`.
3. **Prompt optimization** — ask an optimizer LLM to revise the judge prompt using the failed
   cases as evidence.
4. **Candidate evaluation** — run the revised prompt on the same dev sample and compare metrics.
5. **Accept / reject** — keep the revision only if average MAE improves; log the decision.
6. **Loop** — repeat for N iterations.

## Current results

| Metric | Value |
|---|---|
| Initial dev avg\_mae | 0.828 |
| Current dev avg\_mae | 0.640 |
| Held-out test avg\_mae | 0.664 |
| Parse failures | 0 |
| Candidates accepted | 1 |
| Candidates rejected | 8 |

The final prompt generalizes well: dev and held-out test MAE are within 0.024 of each other,
indicating the rubric improvements did not overfit to the dev failure cases.

## How to reproduce

```bash
# View experiment summary
python src/summarize_experiment.py

# Run held-out test evaluation (use the same seed as optimization)
python src/final_eval.py --sample-size 100 --seed 42

# Run or continue the optimization loop
python src/optimize_loop.py --iterations 3 --sample-size 50 --min-mae-delta 0.02 --seed 42
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### Run a quick baseline (20 examples)
```bash
python src/run_judge.py --sample-size 20 --seed 42
python src/metrics.py
```

### Run failure analysis
```bash
python src/error_analysis.py
```

### Generate a revised prompt
```bash
python src/improve_prompt.py
```

### Compare baseline vs. candidate prompt
```bash
python src/compare_prompts.py
```

### Full optimization loop
```bash
python src/optimize_loop.py \
    --iterations 5 \
    --sample-size 50 \
    --judge-model claude-sonnet-4-6 \
    --optimizer-model claude-opus-4-8 \
    --seed 42
```

## Project structure

```
judge-prompt-lab/
  prompts/
    judge_prompt.md           # current judge prompt (edited by optimizer loop)
    judge_prompt_candidate.md # proposed revision (overwritten each iteration)
    optimizer_prompt.md       # meta-prompt fed to the optimizer LLM
  data/
    failed_cases.jsonl        # worst disagreements from dev set
  results/
    baseline_metrics.json     # metrics for current judge_prompt.md
    candidate_metrics.json    # metrics for judge_prompt_candidate.md
    final_test_metrics.json   # held-out test evaluation results
    experiment_log.jsonl      # one record per accept/reject decision
  src/
    load_data.py              # load + split HelpSteer2
    run_judge.py              # call judge LLM, parse JSON scores
    metrics.py                # MAE, correlation, within-1 accuracy
    error_analysis.py         # identify + save failure cases
    improve_prompt.py         # call optimizer LLM to revise judge prompt
    compare_prompts.py        # evaluate candidate, accept or reject
    optimize_loop.py          # end-to-end loop
    final_eval.py             # held-out test evaluation (read-only)
    summarize_experiment.py   # print experiment summary
  agents/
    cursor_improver.md        # instructions for a Cursor SDK agent
```

## Guardrails

- The optimizer only sees `data/failed_cases.jsonl` from the **dev split** — never the test set.
- The optimizer may only rewrite `prompts/judge_prompt_candidate.md`.
- `src/metrics.py`, `src/load_data.py`, dataset labels, and split logic are never modified by the loop.
- Evaluation is deterministic: fixed random seed, temperature 0 for the judge.

## Next phase: Cursor SDK orchestration

The Python eval harness is the source of truth and the referee for all accept/reject decisions.
Cursor SDK can be used as an optional orchestration layer to inspect results, propose prompt
revisions, and trigger evaluation runs.

Cursor may read:
- `prompts/judge_prompt.md`, `prompts/judge_prompt_candidate.md`
- `results/*.json`, `data/failed_cases.jsonl`

Cursor may edit:
- `prompts/judge_prompt_candidate.md` only

Cursor may run:
- `python src/optimize_loop.py`, `python src/final_eval.py`

Cursor must not edit:
- `src/metrics.py`, `src/load_data.py`, `src/run_judge.py`, `src/compare_prompts.py`
- Dataset split logic, label mapping, or accept/reject rules

See `agents/cursor_improver.md` for the full agent scope contract.

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `JUDGE_MODEL` | Model used as judge (default: `claude-sonnet-4-6`) |
| `OPTIMIZER_MODEL` | Model used to revise the prompt (default: `claude-opus-4-8`) |
| `HF_TOKEN` | Optional — needed only if HelpSteer2 requires auth |
