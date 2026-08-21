# Session handoff - CH LT PIT probabilistic preregistration draft

Date: 2026-07-24

Branch: `fix/lt-audit-remediation`

HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`

Workspace: `C:\Users\jbattaglia\PFC_LT`

Production status: strict `NO_GO`

Scientific evaluation status: `DRAFT_BLOCKED_NOT_EXECUTABLE`

## Outcome

A fail-closed structural preregistration now exists for the next prospective
Swiss LT deterministic and probabilistic evaluation. Schema v1 cannot
authorize execution, production or promotion. It converts the missing
scientific, data-authority and operational evidence into fourteen explicit
blockers instead of allowing a self-attested frozen plan to run.

This work does not execute T057, consume a future holdout, create a new CH
candidate or promote any artifact.

## Exact files changed in this slice

Created:

- `.planning/phases/14-lt-audit-remediation/CH-LT-PIT-PROBABILISTIC-PREREGISTRATION-DRAFT-20260724.json`
- `.planning/phases/14-lt-audit-remediation/CH-LT-PIT-PROBABILISTIC-PREREGISTRATION-DRAFT-20260724.md`
- `pfc_shaping/validation/ch_lt_pit_preregistration.py`
- `scripts/audit_ch_lt_pit_preregistration.py`
- `tests/test_ch_lt_pit_preregistration.py`
- this handoff

Updated:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` with
  D-20260724-159
- `.planning/HANDOFF.md` to point to this handoff

The worktree was already intentionally very dirty. No unrelated change was
restored, reset, cleaned, staged or rewritten.

## Canonical draft identity

- schema: `ch_lt_pit_probabilistic_preregistration.v1`
- protocol: `ch_lt_pit_outer24_future4_20260724_v1`
- lifecycle: `DRAFT_NOT_FROZEN`
- file SHA-256:
  `aba798530084b7031a0ac38b1c48b20cff575d6082edbcf37c9a04528900ba61`
- semantic `plan_id`:
  `ae5557fd7e58a6ee4164e7f8a949cb379fc2d8ac23766e17a1873c4de420c5f6`
- audit schema: `ch_lt_pit_preregistration_audit.v1`
- blockers: `14`
- `execution_authorized=false`
- `production_authorization=false`
- `promotion_gate=false`

The values `n=24`, moving-block length `3`, HAC lag `2`, four seasonal 14-day
episodes, nine marginal quantiles and 1,000 scenarios are hypotheses in this
draft. They are not accepted executable design constants.

## Fail-closed boundary

The following eight blockers are unconditional in schema v1, even if a caller
supplies apparently complete hashes and receipts:

1. receipt-free immutable plan core and separate external admission envelope
   not implemented;
2. dependence, power, MDE and effective-sample-size study not frozen;
3. exact LT horizons, targets, truth, masks and inner-fold inventories absent;
4. direct CH 15-minute truth and post-episode outcome receipts absent;
5. complete statistical/probabilistic hypothesis family and Monte-Carlo error
   policy not frozen;
6. exact FMV profiles, formulas and per-profile economic non-regression absent;
7. durable one-shot attempt seal and ledger not implemented;
8. CPU/GPU reproducibility contract not bound.

The draft additionally reports six current-document blockers: independent
freeze/trusted time, governed PIT manifests, exact origin/regime inventory,
exact future-episode inventory, FMV Risk materiality approval and independent
review receipts.

## Security and correctness hardening

Initial read-only roasts found exploitable or misleading paths. The final code:

- forces `execution_authorized=False` for every schema-v1 document;
- names the only alternate lifecycle
  `FROZEN_STRUCTURE_UNVERIFIED_EXTERNAL_ADMISSION_REQUIRED`;
- rejects mapping/document-byte rebinding;
- performs recursive type-strict comparison so `0 != False` and `1 != True`;
- preserves the lexical input path until the secure single-capture reader;
- requires distinct admitted-manifest hashes across roles;
- requires current unsigned/local evidence flags to be exactly false;
- rejects simultaneous `HYDRO_LOW` and `HYDRO_HIGH` on one origin;
- requires each future episode to stay inside one season and align to
  15-minute boundaries;
- rejects duplicate JSON keys and non-finite canonical JSON.

Marginal quantiles are required to be non-crossing, but they are not each
repriced to one monthly mean because that would collapse the distribution.
Complete scenario paths must preserve monthly-solver means pathwise.

## Exact commands and results

Every PowerShell command was preceded by the exact canonical cwd/workspace
guard required by the user.

Focused adversarial suite:

```powershell
python -m pytest tests\test_ch_lt_pit_preregistration.py -q -p no:cacheprovider --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-ch-pit-focused-20260724-v4
```

Result: `17 passed in 0.19s`.

Authoritative integrated matrix:

```powershell
python -m pytest tests\test_ch_lt_pit_preregistration.py tests\test_governed_lt_acquisition.py tests\test_lt_input_sources.py tests\test_governed_lt_input_snapshot_v2.py tests\test_monthly_forward_curve_constraints.py tests\test_monthly_forward_curve_solver.py tests\test_monthly_forward_curve_priors.py tests\test_monthly_forward_curve_integration.py tests\test_monthly_curve_sensitivity.py tests\test_cascading.py tests\test_arbitrage_free.py tests\test_probabilistic_output_governance.py tests\test_assembler_profile_type.py tests\test_intraday_amplitude.py -q -p no:cacheprovider --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-ch-pit-matrix-20260724-v5
```

Result: `357 passed, 3 skipped in 98.52s`.

The preceding wrapper run in
`build/pytest-ch-pit-matrix-20260724-v4` encountered the known Windows pytest
temporary-directory ACL failure during setup/cleanup. It is non-conclusive and
is not counted as model or contract evidence.

Canonical audit invocation:

```powershell
python -m scripts.audit_ch_lt_pit_preregistration --plan .planning\phases\14-lt-audit-remediation\CH-LT-PIT-PROBABILISTIC-PREREGISTRATION-DRAFT-20260724.json --expected-plan-sha256 aba798530084b7031a0ac38b1c48b20cff575d6082edbcf37c9a04528900ba61 --mode validate-draft
```

Result: exit `0`, status `DRAFT_BLOCKED_NOT_EXECUTABLE`, fourteen blockers.

Changing only the mode to `admit-execution` returns exit `3` with
`execution_authorized=false`.

The direct file form `python scripts\audit_ch_lt_pit_preregistration.py ...`
does not put the repository root on `sys.path` in this environment and raised
`ModuleNotFoundError`. It is not the supported invocation and is not counted.
Use the exact module invocation above.

Quality checks:

```powershell
python -m ruff format pfc_shaping\validation\ch_lt_pit_preregistration.py scripts\audit_ch_lt_pit_preregistration.py tests\test_ch_lt_pit_preregistration.py
python -m ruff check pfc_shaping\validation\ch_lt_pit_preregistration.py scripts\audit_ch_lt_pit_preregistration.py tests\test_ch_lt_pit_preregistration.py
python -m py_compile pfc_shaping\validation\ch_lt_pit_preregistration.py scripts\audit_ch_lt_pit_preregistration.py tests\test_ch_lt_pit_preregistration.py
```

Result: three files already formatted; Ruff and byte-compilation pass.

## Independent final re-roasts

Security:

- no residual P0/P1 in the structure-only v1 scope;
- fake receipts, same-authority hashes, byte rebinding, bool/int confusion,
  shared manifests and season-crossing episodes were replayed and rejected;
- local PASS as a non-executable structural validator;
- evaluation and production `NO_GO`.

IT/Operations:

- no residual P0/P1 in the structure-only v1 scope;
- local GO to retain and audit the draft;
- freeze/evaluation and production `NO_GO`.

Quant/Data:

- no residual P0/P1 in the claims made by the corrected draft;
- confirms that the initial `n=24`, block `3` and HAC `2` values are not an
  executable statistical design;
- scientific draft and production `NO_GO`.

Residual P2 items are deliberately non-authorizing: optional audit persistence
requires a pre-created canonical namespace; collision/concurrency/link tests
can be expanded; descriptive `current_evidence` paths are not independently
read-verified; library callers may omit expected bytes/hash but still cannot
obtain authorization; audit outputs do not yet bind operation ID/runtime/mode;
and the future external envelope does not yet exist.

## Protected-state audit

- branch and HEAD remained exact;
- `data/eex_forwards_history.parquet` was not touched, restored or staged by
  this slice;
- its observed SHA-256 remained
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- no staged file, commit or production promotion was created;
- `pfc_shaping/ct/*` and Power BI were not touched.

## Next best work

Do not run a new candidate or consume a future holdout yet. Build a v2
scientific admission design in this order:

1. define exact LT delivery horizons, target tensors, truth/eligibility masks,
   nested inner folds and regime inventory;
2. acquire independently governed, point-in-time direct CH 15-minute truth and
   post-episode outcome receipts;
3. run a dependence-aware pilot and freeze power/MDE/effective `n`, bootstrap
   block selection, multiplicity and Monte-Carlo error budgets;
4. bind exact FMV Fil, Acc, Bloc 13 and dispatch profiles and materiality rules;
5. specify a receipt-free plan core and an independent signed admission
   envelope with one-shot ledger;
6. bind CPU and optional GPU environments, deterministic fallbacks and
   cross-runtime reproducibility tolerances;
7. obtain fresh Security, IT/Operations and Quant/Data roasts before any
   non-production execution.

The monthly solver remains authority of level, OMPEX remains benchmark-only,
and no production promotion is allowed before the full evidence chain passes.
