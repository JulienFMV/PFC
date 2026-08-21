# Session handoff - EEX short-tenor quarantine hardening

Date: 2026-08-05

## Outcome

D216 supersedes the D215 short-tenor audit identity after a roast found that a
matching quarantine row was not independently constrained to CH POWER and a
known reason enum. The audit now fails closed on untrusted reconciliation
metadata. All economic, coverage, nesting, OFFPEAK and gap outputs remain
byte-identical to D215.

All work was performed from the frozen local D212 normalization. No Databricks
request, SQL Warehouse start, Databricks write or network call occurred.

## Changed files

- `pfc_shaping/validation/databricks_eex_short_tenor_audit.py`
  - validates only relevant DAY/WEEK/WEEKEND quarantine rows;
  - requires CH, POWER, BASE/PEAK and a known quarantine reason;
  - validates quotation/load timestamps, finite settlement price, delivery
    bounds and canonical product/type consistency;
  - allows inconsistent delivery bounds only for the explicitly reason-coded
    `DELIVERY_BOUNDARY_MISMATCH` case.
- `tests/test_databricks_eex_short_tenor_audit.py`
  - expands focused coverage from 11 to 18 tests;
  - rejects cross-country, cross-commodity, unsupported-load, unknown-reason,
    time-inconsistent, non-finite and non-boundary metadata mismatches.
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` (D216)
- `.planning/HANDOFF.md`
- this handoff.

No CT, AFRY numeric, Power BI, protected heavy desk-data or monthly-solver file
was modified.

## Selected evidence

Bundle root:

`build/databricks-eex-daily/2026-08-05/short-tenor-audits/b09eb3250df5a3c0616eb169c512319c514ddf540251b405023d9351bd5d8bde/`

- content ID:
  `b09eb3250df5a3c0616eb169c512319c514ddf540251b405023d9351bd5d8bde`;
- manifest SHA-256:
  `5978fd0383c64138cac0486f891361a07fb4160ee0e6fb22465ffe60ef5bb63d`;
- audit module SHA-256:
  `715717cafee80bf659b6090842c8a646cf0b85b71e50014d222c33c32c259485`;
- audit tests SHA-256:
  `a2258f97271055ca5adb999bd444086bac2408c9ef0ca02d760c1eb613f362f9`;
- materializer SHA-256:
  `a6c1b145fc23fc3106e706b6cf876bd29b087a62e3ea423847f619de75539445`.

The selected source bindings remain D212 normalization
`2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f`,
source snapshot
`593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812`,
all-product history SHA-256
`896f0c9f839b7fc9364398ed0848b0f1886c2ee54d277bf327ae1a414833c06e`
and quarantine SHA-256
`d1328caac9fb860fc2bb0f4bf518ec59e792338f1d154d01c0067b937f2a21e5`.

## Exact analytical continuity

D215 bundle
`f84ac6c9461bf9b8a0c5e36618f74b9b155b9c3050969f42ba780602014f433e`
is superseded. D216 preserves its exact artifact bytes:

- `daily_coverage.parquet`: 5,813 rows, SHA-256
  `68138ae62372678e18b8668917ccab3b9e4d9b7151e59938c46c57c976ca4694`;
- `horizon_profile.parquet`: 86 rows, SHA-256
  `ab029600314be0eb70903866b7f9761c94646a1348329da84477392ba0bf2429`;
- `product_lifecycle.parquet`: 5,054 rows, SHA-256
  `0b99198b4446baed6b63adf1a878b7bb61b595f53fd8a8dd733b994be5942510`;
- `gap_diagnostics.parquet`: 53 rows, SHA-256
  `be5c8008b0a0d641fc2f82eb7e0bb7e4d976ac62077f8d274fd78eddb2ca19d6`;
- `nesting_diagnostics.parquet`: 4,900 rows, SHA-256
  `8b845ce14f353531eb99c3e7892686a65e19ec3c9342516317231a94eebc0b91`;
- `offpeak_diagnostics.parquet`: 12,170 rows, SHA-256
  `c8a06d7584eba16e11c8cef1c4c14f61c524e0e44128af5933f040e69b3b2dc7`;
- `summary.json`: SHA-256
  `0364358c0f0de00541643c88b06e3e307eedd3db47c7de1b4a2e14291449544a`.

The verdict remains
`PASS_LOCAL_INTEGRITY_WITH_LOCALIZED_TEMPORAL_GAPS`: one diagnostic is
explained by the valid D212 boundary quarantine and 52 candidates remain
unexplained. No missing quote was filled.

## Commands and verification

Every command first verified exact cwd and Git top-level
`C:\Users\jbattaglia\PFC_LT`; `TEMP`, `TMP` and pytest basetemp stayed below
`build/`.

- Focused:
  `build\pytest-runtime-v2-final\python.exe -m pytest -q tests/test_databricks_eex_short_tenor_audit.py`
  -> `18 passed in 0.74s`.
- Adjacent EEX/LT matrix covering D212-D216, governed history, cascading,
  monthly audit/constraints/integration/priors/solver and LT/CT boundaries:
  -> `226 passed, 4 skipped in 25.31s`.
- The exact materializer with explicit normalization/output/hash arguments ran
  twice and returned the same D216 content ID, byte-checking existing outputs.
- One initial materializer invocation omitted its mandatory CLI arguments and
  failed closed at argument parsing. It produced no selected artifact and is
  not a product failure.
- `git diff --check` must be green at final handoff.

## Authority and next safe step

The monthly solver remains sole monthly-level authority. Short-tenor quotes may
only become solver-neutral within-month shape diagnostics under a separately
governed feature contract. Local quote chronology is not signed PIT evidence;
rolling-origin, model selection, candidate assembly, promotion and production
remain false. Governed ENTSO-E, signed EEX vintages and a new independent
future holdout remain required. AFRY and OMPEX stay benchmark-only and T057 is
sealed.

Next offline batch: define a dormant, zero-monthly-mean short-tenor feature
contract and prove algebraically that it cannot alter solver monthly levels.
