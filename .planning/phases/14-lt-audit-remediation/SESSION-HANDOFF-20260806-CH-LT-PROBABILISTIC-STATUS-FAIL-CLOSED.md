# Session handoff - CH LT probabilistic status fail-closed

Date: 2026-08-06  
Decision: `D-20260806-242`  
Status: `PASS_FAIL_CLOSED_PROBABILISTIC_CLAIM_MODEL_NO_GO`

## Outcome

D242 closes a latent false-authority path in LT probabilistic output
governance. Before this change, any caller could attach
`probabilistic_status=CALIBRATED_ROLLING_ORIGIN` to a pandas frame and make
finite, non-crossing P10/P90 appear governed. The attribute is mutable caller
metadata and is not a calibration receipt.

`governed_intervals_available` now remains false until an independent,
content-bound probabilistic admission verifier exists. It still rejects
duplicate labels, missing or empty intervals, booleans, non-numeric and
non-finite values, and crossing quantiles, but passing those structural checks
does not create evidence authority. Deterministic LT exports are unchanged.

## Why this matters for the first ambitious PFC

The local PFC can already produce a structurally clean 15-minute curve and
preserve EEX monthly/product constraints, but its current P10/P90 are empty and
its three structural paths are not calibrated probabilities. D242 prevents a
future caller from hiding that gap with a status string. A credible first
probabilistic candidate must instead bind:

1. governed PIT CH EEX and ENTSO-E inputs;
2. direct CH rolling-origin truth and a new independently frozen holdout;
3. frozen quantile/scenario methods and transparent baselines;
4. coverage, sharpness, pinball, WIS/CRPS and trajectory-score evidence;
5. the exact candidate artifacts and calibration receipt.

## Changed files

- `pfc_shaping/pipeline/probabilistic_output_governance.py`
- `tests/test_probabilistic_output_governance.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY, OMPEX, T057, Databricks connector, `H:` or heavy
desk-data file was opened or changed by D242. Existing unrelated worktree
changes were preserved.

## Verification

Every shell action verified cwd and Git top-level as the canonical
`C:\Users\jbattaglia\PFC_LT`. Mutable test state remained below `build/`.

- focused workspace-supervised matrix: `45 passed in 0.27s`;
- pytest tests/failures/errors: `45/0/0`;
- target status: `TARGET_EXIT_ZERO_NOT_AUTHORITY`;
- implementation SHA-256:
  `735ac0cee3b36c582eeedf7170d007ba7018a67e57592775dd3fd6f31e9565ac`;
- tests SHA-256:
  `6bd2fb319bcfe411ef9493a3b4b920e876783ae9e945d9a4b26fd15495d9a602`;
- execution-receipt SHA-256:
  `6779d2b8a0b18906e9f8acd1b5a9fb15efac1b2f687413372ebc3f3aafe34cb8`;
- `git diff --check`: passed;
- Ruff: not qualified; the repo-local scientific runtime does not include the
  `ruff` module. No lint pass is claimed.

No Databricks connection or statement, Warehouse start, network call, `H:`
access or remote write occurred.

## Current position and next permitted step

The D241 Unity Catalog control-plane inventory confirms the real `dev.gold`
tables and exposes their schema gaps without starting a Warehouse. D242
ensures probabilistic intervals cannot self-admit. Neither supplies real
empirical evidence.

The shortest honest route to the first ambitious candidate is:

1. freeze the physical-to-normalized mapping for the observed real schema,
   including cadence, sign and PIT timestamp semantics;
2. implement the bounded streaming verifier for real ENTSO-E Parquet plus the
   independent receipt/signature/time binding;
3. admit one governed real capture and run D239 quality/coverage diagnostics;
4. accumulate trustworthy vintages and direct CH future origins, preserving
   the new holdout;
5. execute the frozen deterministic and probabilistic challengers only after
   those gates open;
6. freeze the candidate, then compare OMPEX read-only from `H:` as an external
   benchmark.

Until then, a technically clean deterministic research curve is possible, but
no claim of calibrated uncertainty, empirical superiority or production
readiness is justified.

Immediate predecessor:
`.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260806-ENTSOE-DEV-CONTROL-PLANE-INVENTORY.md`.
