# Session Handoff - 2026-07-14 - Tier 2 BASE Signed Output Replay

## Status

D137 is GO only for the narrow statement that one chosen SELECTION fold/config
BASE computation exactly replays a pre-existing signed byte commitment inside
the captured trusted Python process.

Campaign, HOLDOUT, hourly PFC and production remain `NO_GO`.

## Changed Files

- `pfc_shaping/calibration/tier2_monthly_eex_base_output_evidence.py`
- `pfc_shaping/calibration/tier2_monthly_eex_base_replay.py`
- `pfc_shaping/calibration/tier2_monthly_eex_evaluation.py`
- `pfc_shaping/calibration/tier2_monthly_eex_fold_evidence.py`
- `pfc_shaping/calibration/tier2_monthly_eex_selection_base_inputs.py`
- `tests/test_tier2_monthly_eex_evaluation.py`
- `tests/test_tier2_monthly_eex_fold_evidence.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI or heavy desk-data file was changed by this slice. No generated
evidence artifact was retained in the repo; all signed workbooks/packages used
by tests were temporary pytest fixtures.

## Implemented Contract

New path-only API:

`verify_signed_selection_fold_base_model_output(...)`

Exact output package inventory:

- `base_output_manifest.v1.json`
- `expected_base_output.v1.json`

The incoming manifest is signed by a distinct model-execution authority and a
Tier 2 trusted-time authority but must declare all model/campaign/promotion
claims false. Independent replay alone creates the ephemeral narrow result.

Environment policy includes:

- `PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH`
- `PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH`
- `PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH`
- `PFC_TIER2_EXECUTION_TRUSTED_PUBLIC_KEY_PATH`
- `PFC_TIER2_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH`
- `PFC_TIER2_BASE_MODEL_EXECUTION_TRUSTED_PUBLIC_KEY_PATH`
- source and Tier 2 timestamp journal IDs

Each PEM is read atomically once and the same bytes determine snapshot hash,
public key and key ID. Individual authority roles, journal IDs, exact receipts,
outer inputs, catalog internals, source files, loaded Python bindings and runtime
distribution files are bound before replay and recaptured after.

The loaded-state closure rejects substitutions of:

- the top-level BASE replay callable;
- the internal monthly solver alias;
- imported classes spoofing provider metadata;
- governed regex constants;
- imported modules such as `math`;
- authority roles preserving the same unordered key set;
- alternate signed receipt bytes restored before final snapshot.

## Explicit Non-Claims

The result fixes these values:

- `configuration_selection_verified=false`
- `process_isolation_verified=false`
- `metrics_verified=false`
- `campaign_eligible=false`
- `production_approved=false`
- `peak_offpeak_verified=false`

The selected config is a signed-grid member, but its choice is not independent
of the masked target. No rolling-origin metric may consume this evidence until
all preregistered configs are replayed for every fold and selection occurs only
at campaign level. PEAK/OFFPEAK still requires a joint product solver.

## Verification

```powershell
python -m pytest tests/test_tier2_monthly_eex_base_replay.py tests/test_tier2_monthly_eex_evaluation.py tests/test_tier2_monthly_eex_fold_evidence.py -q
```

Result after the D138 private input prerequisite: `102 passed, 1 skipped in
139.47s`.

Additional checks:

- targeted `py_compile`: PASS
- targeted Ruff: PASS
- targeted `git diff --check`: PASS
- loaded binding digest in two fresh Python processes: exact SHA match

Permanent Quant, Data and IT agents each returned final GO with no P0/P1 for
the narrow D137 boundary. All three returned NO_GO for campaign/production.

## Next Methodical Slice

The D138 design is fixed in D-20260714-138 and its private config-neutral input
derivation is closed in D-20260714-139. Do not aggregate D137 or call the
existing target-dependent `fold_result` path.

1. Wrap the private derivation in a path-only signed `selection_base_inputs`
   package for every preregistered BASE SELECTION fold. The wrapper must verify
   plan/grid/catalog/trust itself and reject caller-supplied derived objects.
2. Implement the complete signed output inventory derived as
   `F x C` candidate cells plus one canonical baseline per fold. Verify plan,
   grid, catalog and runtime once; derive one immutable context per fold.
3. Reject the whole D138 boundary if any SELECTION fold is PEAK/OFFPEAK, any
   cell fails, inventory differs, paths collide or governed resource caps are
   exceeded.
4. Only after complete inventory closure, implement a separately signed target
   reveal and metric recomputation at campaign level. Its trusted execution must
   be strictly later than the matrix receipt.
5. Freeze selected config under independent governance, then construct disjoint
   HOLDOUT evidence.
6. Before campaign acceptance, move execution into a fresh immutable
   subprocess/container and bind image, OS/native runtime, BLAS/threading,
   locale and floating-point policy.
7. Keep full hourly/PIT, probabilistic, PEAK/OFFPEAK and production promotion
   gates NO_GO until their separate evidence chains exist.

Do not generate individual month patches. Do not use OMPEX in model fitting or
selection. Do not touch CT or Power BI during this LT campaign slice.
