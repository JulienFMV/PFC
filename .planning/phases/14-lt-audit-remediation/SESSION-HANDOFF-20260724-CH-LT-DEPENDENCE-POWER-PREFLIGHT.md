# Session handoff - CH LT dependence/power preflight

Date: 2026-07-24

Branch: `fix/lt-audit-remediation`

HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`

Workspace: `C:\Users\jbattaglia\PFC_LT`

Production status: strict `NO_GO`

Scientific CH status: `NO_GO`

## Outcome

The next scientific blocker was quantified without consuming T057 or any
future holdout. The only available retrospective panel has 16 non-overlapping
14-day DE-LU intrahour origins. It is useful for diagnosing instability but is
not direct CH truth and its target conditions on the realized parent-hour
mean. It therefore cannot determine the sample size for a full CH LT PFC
evaluation.

The current v2 artifact is explicitly a post-selection plug-in sensitivity,
not an acquisition plan. The earlier v1 artifact is immutable but superseded,
and a path-and-hash verifier now rejects its selection.

## Exact files changed in this continuation

Created:

- `pfc_shaping/validation/dependence_power_preflight.py`
- `scripts/audit_ch_lt_dependence_power_preflight.py`
- `tests/test_dependence_power_preflight.py`
- `pfc_shaping/validation/dependence_power_supersession.py`
- `scripts/verify_ch_lt_dependence_power_preflight_supersession.py`
- `tests/test_dependence_power_supersession.py`
- `.planning/phases/14-lt-audit-remediation/CH-LT-DEPENDENCE-POWER-PREFLIGHT-SUPERSESSION-20260724.json`
- this handoff

Updated:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` with
  D-20260724-160
- `.planning/HANDOFF.md` to point here

Local ignored artifacts created append-only:

- superseded v1:
  `output/phase14/ch_lt_dependence_power_preflight/intraday_shape_rolling_origin_20260724_v3.json`
- current v2:
  `output/phase14/ch_lt_dependence_power_preflight/intraday_shape_rolling_origin_20260724_v3_v2.json`

No unrelated dirty file was reset, cleaned, restored, staged or rewritten.

## Quantitative result

Exact source evidence:

- fold CSV SHA-256:
  `910dd2dbe0a2c87f345f05431be14769d45e4113e1f54301176ba25fc7ecfe45`;
- source summary SHA-256:
  `523169da91a3ec3ab46f341608274a449f035488bf671d2af82481112a3d8f09`;
- source status: `LOCAL_DIAGNOSTIC_CANDIDATE_REJECTED_NOT_PRODUCTION`;
- source authority: `COMPLETE_LOCAL_MIXED_AUTHORITY_NO_GO`;
- source target: `quarter_hour_price_given_realized_parent_hour_mean`;
- 16 non-overlapping origins, each 14 days / 1,344 quarter-hours;
- coverage touches winter, spring and summer, but not autumn.

Observed candidate comparison:

- benchmark equal-origin MAE: `5.783128666386267 EUR/MWh`;
- candidate equal-origin MAE: `5.681786842737326 EUR/MWh`;
- observed improvement: `0.10134182364894076 EUR/MWh`;
- relative improvement: `1.7523702046952163%`, below the draft `2%` target;
- fold outcomes: six wins, six losses, four ties;
- observed plug-in sample variance: `0.49395752904413964`;
- maximum observed Bartlett-HAC LRV on lags 0..4:
  `0.46308518347888084`;
- observed plug-in effective origin count: `16.0`; this is not a confidence
  bound.

Conditional normal-known-LRV plug-in sensitivities are 272, 387 and 443
origins for hypothesis-family sizes 1, 4 and 8. Delete-one sensitivity reaches
500. These numbers are not CH acquisition requirements. Five hundred
sequential 14-day windows would represent `19.165349048919552` years; the v2
therefore sets:

- `ch_acquisition_requirement_supported=false`;
- `ch_acquisition_requirement_origins=null`;
- `pilot_reuse_for_confirmatory_test_forbidden=true`.

The 16 origins and their summary are denylisted by exact hash for future
confirmatory use.

## V1 supersession and current identity

The first output overstated `500` as an acquisition floor. Its bytes were not
deleted or overwritten.

- superseded v1 SHA-256:
  `614a7a79cd2d22c1e7abe8303900f861a7c660ca199ce1e656268a291157cadd`;
- current v2 SHA-256:
  `34fc3621ea9082ac6c4c0306c4f7f77e333a33fef83a7f673e62c462ec6d5907`;
- supersession registry SHA-256:
  `9fd9cf706c768716a42967962527a49a9f92c7a16d1e60de0b3d910565312b72`.

Never select these files by glob or modification time. The supported exact
verifier is:

```powershell
python -m scripts.verify_ch_lt_dependence_power_preflight_supersession `
  --registry .planning\phases\14-lt-audit-remediation\CH-LT-DEPENDENCE-POWER-PREFLIGHT-SUPERSESSION-20260724.json `
  --expected-registry-sha256 9fd9cf706c768716a42967962527a49a9f92c7a16d1e60de0b3d910565312b72 `
  --selected-artifact output\phase14\ch_lt_dependence_power_preflight\intraday_shape_rolling_origin_20260724_v3_v2.json `
  --expected-selected-artifact-sha256 34fc3621ea9082ac6c4c0306c4f7f77e333a33fef83a7f673e62c462ec6d5907
```

Current v2 result: exit `0`,
`CURRENT_LOCAL_DIAGNOSTIC_SELECTION_VERIFIED_NO_GO`. Scientific admission,
execution, production and promotion remain false.

Selecting the exact v1 path/hash with the same verifier returns exit `3`,
`SUPERSEDED_DEPENDENCE_POWER_PREFLIGHT_REJECTED`.

The direct-file form `python scripts\verify_...py` does not add the repository
root to `sys.path` in this environment. Use the documented `python -m` form.

## Audit CLI semantics

The safe default is scientific-admission mode and returns exit `3` for the
valid v2 diagnostic:

```powershell
python -m scripts.audit_ch_lt_dependence_power_preflight `
  --fold-csv output\phase14\intraday_shape_rolling_origin_20260724_v3\fold_metrics.csv `
  --expected-fold-csv-sha256 910dd2dbe0a2c87f345f05431be14769d45e4113e1f54301176ba25fc7ecfe45 `
  --source-summary output\phase14\intraday_shape_rolling_origin_20260724_v3\summary.json `
  --expected-source-summary-sha256 523169da91a3ec3ab46f341608274a449f035488bf671d2af82481112a3d8f09 `
  --candidate-model candidate_sparse_price_space_24_dense_envelope `
  --benchmark-model incumbent_ratio
```

Only the explicit `--mode validate-diagnostic` returns exit `0`; it still emits
all scientific/execution/production/promotion authorities as false.

## Hardening completed

- exact single-capture reads for fold CSV and source summary;
- strict JSON with duplicate/non-finite rejection and strict CSV headers/rows;
- bound source schema, target, fold duration, UTC 15-minute alignment and exact
  observation count;
- finite bounded losses and normalized numeric-overflow failures;
- paired model rows, equal windows/counts and non-overlapping origins;
- degenerate paired variance rejected instead of returning required `n=2`;
- post-selection, target mismatch, missing season and confirmatory-reuse
  blockers explicit;
- no CH acquisition claim and no confidence-bound claim;
- append-only local output and default admission exit `3`;
- exact supersession registry, denylist and path/hash verifier;
- same-byte shadow paths and stale v1 selection rejected.

## Tests and checks

Focused final suite:

```powershell
python -m pytest tests\test_dependence_power_supersession.py tests\test_dependence_power_preflight.py -q -p no:cacheprovider --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-dependence-power-20260724-v8
```

Result: `28 passed in 1.84s`.

Final integrated matrix:

```powershell
python -m pytest tests\test_dependence_power_supersession.py tests\test_dependence_power_preflight.py tests\test_ch_lt_pit_preregistration.py tests\test_governed_lt_acquisition.py tests\test_lt_input_sources.py tests\test_governed_lt_input_snapshot_v2.py tests\test_monthly_forward_curve_constraints.py tests\test_monthly_forward_curve_solver.py tests\test_monthly_forward_curve_priors.py tests\test_monthly_forward_curve_integration.py tests\test_monthly_curve_sensitivity.py tests\test_cascading.py tests\test_arbitrage_free.py tests\test_probabilistic_output_governance.py tests\test_assembler_profile_type.py tests\test_intraday_amplitude.py -q -p no:cacheprovider --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-ch-power-matrix-20260724-v3
```

Result: `385 passed, 3 skipped in 114.53s`.

Earlier PowerShell-wrapped pytest attempts hit only the known Windows sandbox
ACL failure on temporary directories. They are non-conclusive and are not
counted.

Ruff, `py_compile`, exact hash checks and `git diff --check` pass on the final
documented state.

## Independent review

- Quant/Data verified the conditional HAC/ESS/required-n algebra, then found
  and required correction of the target mismatch, confidence-bound/acquisition
  overclaims, post-selection reuse and stale-v1 selection. Final review reports
  no residual P0/P1 after schema v2, denylist and opposable supersession.
- Security reproduced the short-window/count and numeric-overflow weaknesses
  in v1. The v2 rejects those cases cleanly and its final review reports no
  residual P0/P1 in the preflight slice.
- IT/Operations reports no residual P0/P1. It verified Windows module-form
  exits `3/0/2` for default diagnostic admission/explicit validation/invalid
  input, and `0/3/2` for current-v2/stale-v1/invalid supersession selection.
  Local GO is limited to the v2 diagnostic; scientific and production remain
  `NO_GO`.
- A separate final governance review verified all five canonical identities,
  mono-link/no-reparse observations, exact v1 rejection, v2 local selection
  and false authorities, with no residual P0/P1. The remaining P2 is that the
  registry hash must stay externally pinned; it is now pinned in D160, this
  handoff and `.planning/HANDOFF.md`.

## Protected-state audit

- branch and HEAD remained exact;
- `data/eex_forwards_history.parquet` was not touched, restored or staged by
  this continuation and retained SHA-256
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- no staged file, commit or production promotion was created;
- `pfc_shaping/ct/*` and Power BI were not touched.

## Next scientific threshold

Do not collect “500 more DE folds” and do not reuse these 16 origins. The next
valid design step is:

1. define the exact delivered CH LT horizon/target tensor, truth and masks;
2. obtain a direct governed CH 15-minute pilot independent of candidate
   selection, with all seasons and preregistered regimes;
3. elicit an economic MDE from exact FMV Fil, Acc, Bloc 13 and dispatch profile
   value errors;
4. specify plausible dependence/structural-break scenarios before looking at
   confirmatory losses;
5. run simulation-based power and Monte-Carlo-error analysis across those
   scenarios, then freeze the test family in a receipt-free plan core;
6. issue a separate external signed admission envelope and one-shot ledger;
7. obtain new Quant/Data, Security and IT/Operations reviews before execution.

The current draft preregistration blocker
`DEPENDENCE_POWER_MDE_AND_EFFECTIVE_SAMPLE_SIZE_STUDY_NOT_FROZEN` remains open.
T057, the prospective CH evaluation and production all remain strict `NO_GO`.
