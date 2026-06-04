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

## SHP pairwise preference mode

HelpSteer2 mode assigns absolute 0–4 scores across five quality dimensions and
calibrates the judge against human ratings via MAE. **SHP mode is different**: it
asks the judge to choose between two responses and compares that choice to the
human-preferred response (determined by Reddit upvote counts in the SHP dataset).

| | HelpSteer2 | SHP |
|---|---|---|
| Task | Assign scores 0–4 per dimension | Pick the better of two responses |
| Ground truth | Human ratings per dimension | Human-preferred response (A or B) |
| Primary metric | Mean Absolute Error (MAE) | Accuracy |
| Secondary metric | Bias per dimension | Position bias (A pick rate vs human A rate) |

### Current SHP baseline

Evaluated on 300 validation examples with `claude-sonnet-4-6` and the default
`prompts/preference_judge_prompt.md`:

| Metric | Value |
|---|---|
| Sample size (validation) | 300 |
| Parse failures | 1 |
| Accuracy | 0.625 |
| Position bias | +0.027 |

This is a pairwise preference task — the judge picks A or B, not a 0–4 score.
Accuracy measures alignment with the Reddit community preference signal (upvotes),
not factual correctness. A random baseline scores ≈ 0.50.

### Quickstart

```bash
# 1. Inspect the dataset (confirm field names and label mapping before running the judge)
python src/inspect_shp.py --split validation --sample-size 5

# 2. Run the preference judge over a validation sample
python src/run_preference_judge.py --split validation --sample-size 300 --seed 42

# 3. Compute and print metrics
python src/preference_metrics.py

# 4. Run error analysis (domain breakdown, score_ratio buckets, label/score conflicts)
python src/preference_error_analysis.py

# 5. Print experiment summary
python src/summarize_preference_experiment.py
```

**Position bias** is `judge_A_pick_rate − human_A_rate`. A value near zero means the
judge is not systematically biased toward whichever response appears first (A).

> Note: SHP mode is a standalone pipeline. It does not share code with the
> HelpSteer2 optimization loop.

### SHP preference optimization loop

Iteratively improves `prompts/preference_judge_prompt.md` using the same
guarded accept/reject logic as the HelpSteer2 loop, but measured by pairwise
accuracy instead of MAE:

- A candidate is **accepted** only if accuracy improves by ≥ `min_accuracy_delta` (default 0.02), parse failures do not increase, and split/sample_size match.
- All decisions are logged to `results/preference_experiment_log.jsonl`.
- The optimizer never sees the test split.

```bash
# Run 2 optimization iterations
python src/optimize_preference_loop.py \
    --iterations 2 \
    --split validation \
    --sample-size 300 \
    --seed 42 \
    --min-accuracy-delta 0.02

# Print full experiment summary (includes loop history if log exists)
python src/summarize_preference_experiment.py
```

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

See `agents/cursor_orchestrator.md` for the full agent scope contract.

To run the orchestration demo (one guarded optimization iteration):

```bash
npm run orchestrate:demo
```

This can take **5–20 minutes** depending on sample size and API latency — the
orchestrator runs a full optimize\_loop iteration (judge evaluation + optimizer call +
candidate evaluation) before summarizing the outcome. If the default timeout is too
short for your environment, set `ORCHESTRATE_TIMEOUT_MS` to a larger value (in ms).

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `JUDGE_MODEL` | Model used as judge (default: `claude-sonnet-4-6`) |
| `OPTIMIZER_MODEL` | Model used to revise the prompt (default: `claude-opus-4-8`) |
| `HF_TOKEN` | Optional — needed only if HelpSteer2 requires auth |
| `CURSOR_MODEL` | Model used by the Cursor orchestrator (default: `claude-opus-4-8`) |
| `ORCHESTRATE_TIMEOUT_MS` | Shell command timeout for the orchestrator in ms (default: `1200000` = 20 min) |
