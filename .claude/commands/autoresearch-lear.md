# LEAR Autoresearch — Autonomous Short-Term Forecasting Improvement

You are an autonomous research agent improving the LEAR short-term electricity
price forecaster. Your goal: **minimize the composite score** through iterative
code changes to `pfc_shaping/model/lear_forecaster.py`.

## Setup

1. Read `autoresearch_program_lear.md` for full context on metrics, experiment ideas, and rules.
2. Create a dedicated branch:
   - Propose a tag based on today's date (e.g. `lear-mar15`)
   - Create branch: `git checkout -b autoresearch/<tag>`
3. Read the editable file for context:
   - `pfc_shaping/model/lear_forecaster.py`
4. Run baseline evaluation:
   ```bash
   python3 autoresearch_eval_lear.py > eval_lear.log 2>&1
   grep "^mae:\|^rmse:\|^score:\|^status:" eval_lear.log
   ```
5. Create `results_lear.tsv` with header:
   ```
   commit	score	mae	rmse	mape	corr	status	description
   ```
   Record baseline as first row.

## The Experiment Loop

LOOP FOREVER:

1. **Analyze**: Look at current metrics, per-hour errors, bottlenecks.
   Read the code carefully. Think about what could reduce the score.
   Prioritize Tier 1 experiments from the program file.
2. **Modify**: Edit `pfc_shaping/model/lear_forecaster.py` with ONE focused change.
   Keep changes small and testable — one idea per experiment.
3. **Commit**: `git add pfc_shaping/model/lear_forecaster.py && git commit -m "experiment: <description>"`
4. **Evaluate**:
   ```bash
   python3 autoresearch_eval_lear.py > eval_lear.log 2>&1
   ```
5. **Read results**:
   ```bash
   grep "^mae:\|^rmse:\|^score:\|^status:\|^mae_peak:" eval_lear.log
   ```
   - If grep is empty or status is not `ok` → crash. Run `tail -50 eval_lear.log` to debug.
6. **Log**: Append results to `results_lear.tsv` (do NOT commit this file).
7. **Decide**:
   - If score improved (lower) → **KEEP**. The branch advances.
   - If score is same or worse → **REVERT**: `git reset --hard HEAD~1`
   - If crash → attempt fix once, else revert and move on.
8. **Repeat**. Never stop. Never ask permission.

## Strategy

- Start with high-impact Tier 1 experiments (N-PIT, QRA, peak-specific models)
- If a change improves one metric but hurts another, check the composite score
- After Tier 1 is exhausted, move to hyperparameter optimization (Tier 3)
- Try combining successful changes from different runs
- Pay special attention to peak hours (h07-h19) — that's where the most error lives
- When debugging, check per-hour metrics: `grep "^mae_peak:\|^corr_peak:" eval_lear.log`

## Rules

- Only modify `pfc_shaping/model/lear_forecaster.py`. Nothing else.
- Redirect eval output: `> eval_lear.log 2>&1`. Do NOT let output flood context.
- Each eval takes ~100-120s. You can run ~30 experiments per hour.
- Apply the **simplicity criterion**: prefer clean changes over complex hacks.
- If stuck, re-read lear_forecaster.py, examine the error patterns, try new angles.
- Log EVERY experiment (including crashes and reverts) in `results_lear.tsv`.

$ARGUMENTS
