# Session handoff - LSEG / ENTSO-E spot reconciliation

Date: 2026-09-01

Branch: `fix/lt-audit-remediation`

## Outcome

The next offline increment is complete while Jérôme's ENTSO-E PRD rebuild
receipt remains pending.

The source design is now executable and fail-closed:

- ENTSO-E Silver vintages remain the homogeneous candidate panel for CH, AT,
  DE-LU, FR and IT-North;
- LSEG EPEX actuals independently cross-check CH, AT, DE-LU and FR;
- IT-North is explicitly ENTSO-E-only because the LSEG configuration contains
  no matching active EPEX actual-price curve;
- EEX-constrained monthly solving remains the sole monthly-level authority;
- no source is silently substituted after a mismatch.

The LSEG extraction uses exact price curves:

| Zone | CurveID | Native cadence |
|---|---:|---:|
| CH | `115688058` | 1 hour |
| AT | `165444048` | 15 minutes |
| DE-LU | `165349556` | 15 minutes |
| FR | `165442712` | 15 minutes |

Volume curves `115689883`, `165444047`, `165442711` and HPFC/continuous-
forward curve `110181967` are excluded by immutable SQL and local validation.

## Implementation

Created:

- `docs/data/sql/databricks_prd_lseg_epex_actuals_pit_extract.sql`
  - exact four-curve binding;
  - PRD Silver vintage source;
  - `value_date` partition pruning;
  - exact EPEX actual-price semantics;
  - `pipeline_first_seen_at_utc <= as_of_utc`;
  - deterministic latest-known revision;
  - 20,001-row rejection sentinel;
  - SHA-256
    `a4cc587d9f8116dd3f20c295ce8edbf291bfcf06a3f70f859d6fc9816c336bed`.
- `pfc_shaping/validation/spot_source_reconciliation.py`
  - immutable SQL verification and typed one-month parameters;
  - exact schema, curve, cadence, interval, finite-value, uniqueness and PIT
    validation;
  - defensive revalidation of both source artifacts before comparison;
  - complete UTC-hour aggregation with duration weights;
  - explicit detection of gaps, overlap, off-grid and cross-hour intervals;
  - explicit content-hashed threshold policy with no defaults;
  - aggregate-only report, IT-North visibility and authority-negative result;
  - SHA-256
    `a630b449cc20da98c6fa4d0a9997d139f94064bd6707760b9a712e223143dda2`.
- `tests/test_spot_source_reconciliation.py`
  - 25 tests covering SQL/cost fences, source binding, PIT leakage, duplicates,
    non-finite and negative prices, defensive copies, weighted cadence
    alignment, missing/overlapping intervals, thresholds, audit tampering,
    IT-North and DST;
  - SHA-256
    `ce5308f573b7ab9afcdd7eff42c851fef8a11bfce6020bd34b915c8be894a382`.

Modified:

- `pfc_shaping/validation/entsoe_day_ahead_prd.py` now declares the LSEG EPEX
  realized-price cross-check role in its layer policy; SHA-256
  `4339e753c1996dacf3db967016ccd16cfb44eef26e6a8ddf4b8403de04b3564d`.
- `tests/test_entsoe_day_ahead_prd.py` asserts that role; SHA-256
  `bc84bf5e607ae012f856a693b7e577b533a44e19a08f40de4d14fd91f5706053`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` records durable
  decision D-20260901-261.
- `.planning/HANDOFF.md` points to this handoff.

## Verification

Final commands were executed from the guarded canonical root through
`scripts.run_workspace_local`:

```text
python -B -m scripts.run_workspace_local --run-id spottest4a --wall-timeout-seconds 600 -- python -B -m pytest tests\test_spot_source_reconciliation.py -q -p no:cacheprovider
25 passed, 1 warning

python -B -m scripts.run_workspace_local --run-id spotmatrix2a --wall-timeout-seconds 1800 -- python -B -m pytest tests\test_spot_source_reconciliation.py tests\test_entsoe_day_ahead_prd.py tests\test_databricks_pfc_layer_acceptance.py tests\test_databricks_pfc_source_cost_preflight.py tests\test_databricks_lt_materialization.py tests\test_entsoe_resolution_regimes.py tests\test_entsoe_bounded_execution_receipt.py -q -p no:cacheprovider -m "not slow"
185 passed, 1 warning

python -B -m scripts.run_workspace_local --run-id spotbound2a --wall-timeout-seconds 600 -- python -B -m pytest tests\test_lt_ct_imports.py tests\test_lt_package_contract.py -q -p no:cacheprovider -m "not slow"
43 passed, 1 skipped, 1 warning

python -B -m scripts.run_workspace_local --run-id spotruff6a -- python -B -m ruff check pfc_shaping\validation\spot_source_reconciliation.py tests\test_spot_source_reconciliation.py
All checks passed

python -B -m scripts.run_workspace_local --run-id spotfmt3a -- python -B -m ruff format --check pfc_shaping\validation\spot_source_reconciliation.py pfc_shaping\validation\entsoe_day_ahead_prd.py tests\test_spot_source_reconciliation.py tests\test_entsoe_day_ahead_prd.py
4 files already formatted
```

`git diff --check` passed. The recurring pytest warning is the existing
unknown `cache_dir` configuration warning.

One initial dedicated test run (`spottest1a`) failed because the synthetic
fixture passed the bare alias `h` to `pd.Timedelta`; the fixture was corrected
to an explicit one-hour duration and all final matrices pass. The first Ruff
supervisor receipt attempt (`spotruff1`) encountered a repo-local Windows
atomic-replace access denial; a fresh governed run ID passed immediately.

Databricks SQL statements / Warehouse starts / business rows / writes:
`0/0/0/0`.

## Residual gates and next bounded action

No real LSEG or ENTSO-E values were opened in this increment. Synthetic test
rows prove code behavior only.

When Jérôme replies:

1. validate his independently supplied full PRD rebuild receipt with the
   existing ENTSO-E gate and confirm zero old/new `SeriesKey` coexistence;
2. run the already cost-fenced value-blind PRD profile to bind the five exact
   production `SeriesKey` values;
3. only after those passes, execute same-window, same-`as_of_utc`, one-month
   bounded ENTSO-E and LSEG extracts;
4. freeze an owner-approved reconciliation policy on a calibration period
   before evaluating any future holdout; never tune thresholds on the holdout;
5. preserve the aggregate reconciliation receipt and source extract manifests
   independently.

The current state remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`. The real cadence proof, new
independently frozen future holdout and promotion evidence are still absent;
T057 remains sealed.

Durable decision: D-20260901-261.
