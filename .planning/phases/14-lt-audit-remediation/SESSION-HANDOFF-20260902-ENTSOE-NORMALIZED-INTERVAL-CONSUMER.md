# Session handoff - ENTSO-E normalized-interval consumer

Date: 2026-09-02

## Outcome

The LT day-ahead consumer now accepts the real producer-normalized Silver
contract without requiring, inferring or emitting `curve_type`. Positive,
native-grid-aligned blocks whose duration is an exact integer multiple of the
declared resolution are expanded deterministically in UTC into half-open
native-cadence intervals. Source bounds, provenance, availability, finality,
quality and effective-dated selection evidence remain preserved.

The bounded PIT validator and the synthetic ENTSO-E/LSEG reconciliation are
versioned v2 for the new normalized-block semantics. The two SQL templates and
their hashes are unchanged. Historical profile and temporal-diagnostic
captures replay exactly under their original v1 semantics and are not
relabelled as v2 admission evidence.

This correction removes only the obsolete downstream `curve_type` blocker.
It does not authorize real PRD model use or change the global status
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.

Durable decision: D-20260902-267.

## Critical review of the initiating prompt

The prompt correctly protected hash-bound evidence and separated the temporal
normalization issue from availability and auction-selection blockers. Four
prescriptions required narrowing before implementation:

- `IntervalStartUtcUtc` was a typo; the actual invariant is
  `IntervalEndUtc > IntervalStartUtc`.
- Jerome's answer alone did not establish every alignment and half-open
  implementation detail, so those were checked against deployed producer code
  and the existing LT contract.
- gap rejection was scoped to the explicitly requested field/window in the
  consumer; the bounded PIT validator still makes no cadence-completeness
  claim.
- the 21,586-line decision log was inspected structurally and at the relevant
  durable decisions rather than copied wholesale into active context.

## Independent producer verification

Read-only GitHub inspection used repository `FMVSA/opendata-lakehouse`:

- deployed SHA:
  `a7e920d95b94b2db59180412f31213f917e8d8a3`;
- parser correction:
  `8549319ff944bfcf2e8123b05907ac025a8f23b8`;
- compare result: deployed SHA is 25 commits ahead, zero behind, with the
  parser correction as merge base;
- `parse_entsoe_points` reads TimeSeries `curveType`, gives A01 one resolution,
  gives A03 the next point start or final Period end, and writes that end as
  the FMV right-edge timestamp;
- the common Silver stage assigns `Date_Time_UTC` from the normalized value
  timestamp and `IntervalEndUtc` from the normalized interval end;
- both Silver vintages and latest select the normalized bounds and resolution
  but omit `curve_type`.

No GitHub mutation was made. The installed connector could not see the private
repository, so the already-authenticated read-only `gh api` path was used.

## Files changed

- `pfc_shaping/validation/entsoe_day_ahead_consumption.py`
  - removed `curve_type` from exact input and output schemas;
  - changed audit schema to `fmv_entsoe_day_ahead_consumption.v2`;
  - validates positive exact cadence multiples and native UTC alignment;
  - expands without any A01/A03 branch and preserves original source bounds;
  - SHA-256
    `866af2d50b37a56907a92cad4b8ee2994551198d81fc35c88744d532cb2412a2`.
- `tests/test_entsoe_day_ahead_consumption.py`
  - realistic Silver fixtures without `curve_type`, multi-cadence expansion,
    missing/inverted/nonmultiple/off-grid bounds, overlap/gap, exact coverage,
    DST, final/causal availability, auction rules, source bounds, zero-mean
    shape and LT/CT boundary;
  - SHA-256
    `ceb58dda819e211dc4765d565879276c8edca094d223f9d6fc761d8890960a3d`.
- `pfc_shaping/validation/entsoe_day_ahead_prd.py`
  - changed PIT audit schema to
    `fmv_entsoe_day_ahead_prd_pit_extract.v2`;
  - accepts normalized multi-cadence blocks and rejects unsupported,
    nonpositive, nonmultiple, off-grid, out-of-window, duplicate and
    overlapping rows;
  - retains `cadence_completeness_authorized=false`;
  - SHA-256
    `dfa1d5fd1ce8d56a6d05bbb1fedab1763a60319cc01fa109838617d2b39b6fbf`.
- `tests/test_entsoe_day_ahead_prd.py`
  - covers PIT v2 multiple acceptance and all new fail-closed predicates;
  - SHA-256
    `8b86f5c15817c17941981ea7028efb7d7290207a0888d4ac76fbde55981e45ce`.
- `pfc_shaping/validation/spot_source_reconciliation.py`
  - changed synthetic report schema to
    `fmv_lseg_entsoe_spot_reconciliation.v2`;
  - expands normalized ENTSO-E blocks before complete UTC-hour comparison;
  - SHA-256
    `7c7d87daccaa68f380dccf5163c776140fd5d3ddbb617231dc3b54d56f34b45e`.
- `tests/test_spot_source_reconciliation.py`
  - proves a normalized two-hour ENTSO-E block reconciles after deterministic
    native-cadence expansion;
  - SHA-256
    `c4ea6e237abdabe33d0ab45372dafae214a6fda9c806cdee494778e9d93574b5`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - appended D-20260902-267; D-20260901-266 was not rewritten.
- `.planning/HANDOFF.md`
  - updated the current consumer contract and residual blockers.
- this handoff.

No CT, Power BI, heavy data, Parquet, DuckDB or monthly-solver file changed.

## Immutable SQL and historical evidence

Unchanged SQL hashes:

- PIT extract:
  `7b444430db6b62a7bcd9f0bc6e5f05858a56b9756be68f70016c107a4b271ebf`;
- profile:
  `e48bc8b09d6f3676616ed42966d50a3f44d9eaf3649f9ce9c9543a0bc024259e`.

Real offline replay results:

```text
python -B -m scripts.capture_entsoe_day_ahead_prd_profile
  --verify-capture build/entsoe-day-ahead-prd-profile/20260901-july-profile-api/capture.json
PASS_CAPTURE_REPLAY
file SHA-256 661f0efaaaab19a4654a526d7f7c71c8782a3c7ebf20c7af5d740fdf2e5383d0
content ID abef7f875e43358cdd663f1f17fd45a862907a79ddedb6aef9dd42da121bad2c
Databricks statements 0

python -B -m scripts.capture_entsoe_day_ahead_temporal_diagnostic
  --verify-capture build/entsoe-day-ahead-prd-profile/20260901-july-temporal-diagnostic/diagnostic.json
PASS_DIAGNOSTIC_CAPTURE_REPLAY
file SHA-256 9f6b798ec89e507ebd26524dbbc439abf347b9e36a9762855c00521f0841919d
content ID a5f5f3b948212d65112cd2243633744cb1ef1872d827f7ff28312b2dfa80e3d4
duration mismatch count 404
Databricks statements 0
```

The profile v1 remains blocked under its historical exact-duration predicate.
That result is preserved as evidence of what was executed, not treated as a v2
failure and not used as v2 admission. No real PIT v1 receipt exists because
the profile never authorized that extraction.

## Test record

All pytest/Ruff commands ran from the canonical root with mutable state below
repo-local `build/` via `scripts.run_workspace_local`, except the two
read-only replay module calls, which are not allowlisted by that harness.

```text
entnorm0902a
initial focused matrix
87 passed, 3 failed
failures: diagnostic order plus pandas integer timestamp unit assumption

entnorm0902b
focused matrix after correction
90 passed

entnorm0902cap
profile/temporal capture tests
10 passed

entnorm0902c
final focused consumer/PIT/reconciliation matrix
95 passed

entnmat1
all non-slow tests whose path contains entsoe plus spot reconciliation
840 passed

entnlt1
test_arbitrage_free.py, test_cascading.py, test_water_value.py,
test_lt_ct_imports.py
58 passed, 1 skipped

entnpkg1
test_lt_package_contract.py
26 passed

entnorm0902ruff1
targeted Ruff check
pass

entnruff2 / entnfmt5
final targeted Ruff check / format check
pass / 6 files already formatted

entnfinal1
post-format focused consumer/PIT/reconciliation matrix
95 passed
```

The common pytest warning is the existing unknown `cache_dir` configuration.
The single LT skip is the existing optional TensorFlow import boundary.

## Databricks and external cost

- SQL statements: 0.
- Warehouse starts/resizes/creates: 0/0/0.
- business rows opened: 0.
- writes: 0.
- incremental DBU/Azure exposure: zero.

Read-only GitHub API calls inspected code and ancestry only.

## Residual blockers and invariants

1. `publication_timestamp_utc` remains returned-document `createdDateTime`, not
   proof of original historical day-ahead publication.
2. `realized_final` and `causal_asof` remain economically and temporally
   distinct; unknown backfills remain unusable for causal history.
3. AT and DE-LU sequences 1 and 2 remain distinct. Effective-dated selection
   and independent LSEG evidence must be admitted explicitly.
4. The governed real export/snapshot chain and a new independently frozen
   future holdout are still missing.
5. Visible source data currently end on 2026-08-31 due to the reported upstream
   outage.
6. LSEG remains an independent control for CH, AT, DE-LU and FR; IT-North is
   explicitly ENTSO-E-only.
7. The monthly BASE solver remains sole level authority; spot may supply only
   realized truth, control or duration-weighted zero-mean shape.
8. T057 remains sealed; LT imports no `pfc_shaping.ct.*`.

No further request to Jerome about `curve_type` or interval normalization is
needed. A future producer-wide formal promotion may still require the missing
platform-signed top-level/post-backfill receipt, independently of this bounded
consumer correction.
