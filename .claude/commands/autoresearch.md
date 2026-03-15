# PFC Autoresearch — Autonomous Model Improvement

You are an autonomous research agent improving the PFC 15-minute electricity price model.
Your goal: **minimize RMSE vs realized EPEX spot prices** through iterative code changes.

## Setup

1. Read `autoresearch_program.md` for full context on in-scope files, metrics, and rules.
2. Create a dedicated branch:
   - Propose a tag based on today's date (e.g. `mar14`)
   - Create branch: `git checkout -b autoresearch/<tag>`
3. Read the in-scope model files for context:
   - `pfc_shaping/model/shape_hourly.py`
   - `pfc_shaping/model/shape_intraday.py`
   - `pfc_shaping/model/assembler.py`
   - `pfc_shaping/data/forward_proxy.py`
   - `pfc_shaping/config.yaml`
4. Run baseline evaluation:
   ```bash
   python3 autoresearch_eval.py > eval.log 2>&1
   grep "^rmse:\|^rmse_shape:\|^mae:\|^bias:" eval.log
   ```
5. Create `results.tsv` with header:
   ```
   commit	rmse	rmse_shape	status	description
   ```
   Record baseline as first row.

## The Experiment Loop

LOOP FOREVER:

1. **Analyze**: Look at current metrics. Think about what could improve RMSE.
2. **Modify**: Edit files in `pfc_shaping/` with an experimental idea.
3. **Commit**: `git add -A pfc_shaping/ && git commit -m "experiment: <description>"`
4. **Evaluate**: `python3 autoresearch_eval.py > eval.log 2>&1`
5. **Read results**: `grep "^rmse:\|^rmse_shape:\|^status:" eval.log`
   - If grep is empty → crash. Run `tail -50 eval.log` to debug.
6. **Log**: Append results to `results.tsv` (do NOT commit this file).
7. **Decide**:
   - If rmse improved (lower) → **KEEP**. The branch advances.
   - If rmse is same or worse → **REVERT**: `git reset --hard HEAD~1`
   - If crash → attempt fix once, else revert and move on.
8. **Repeat**. Never stop. Never ask permission.

## Rules

- Only modify files under `pfc_shaping/`. Never modify `autoresearch_eval.py`.
- Redirect eval output: `> eval.log 2>&1`. Do NOT let output flood context.
- Each eval takes ~30-60 seconds. You can run ~60-120 experiments per hour.
- Apply the **simplicity criterion**: prefer clean, simple changes over complex hacks.
- If stuck, re-read model files for new angles. Try combining near-misses.
- Log EVERY experiment (including crashes and reverts) in `results.tsv`.

$ARGUMENTS
