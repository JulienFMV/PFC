# Session handoff - LT evaluation protocol v1 - 2026-09-02

## Outcome

Point 2 is locally complete. A compact immutable LT contract now freezes the
next CH hourly-shape comparison without loading truth, fitting a model or
granting operational authority.

The contract reuses the existing estimand, origin-registry v2 and
dependence/power design. It does not create another registry, metric authority
or monthly-level path.

## Changed files

- `pfc_shaping/lt/evaluation_protocol.py`
  - typed candidate, origin, holdout, binding and authority definitions;
  - canonical five-family comparison;
  - canonical 12-origin prospective cohort;
  - fail-closed semantic-hash verification.
- `tests/test_lt_evaluation_protocol.py`
  - ten synthetic inventory, authority, binding, schedule and adversarial
    tests.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`
  - operator-readable statistical and governance boundary.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - added D-20260902-272.
- `.planning/HANDOFF.md`
  - updated current state and reading order.
- this handoff.

No CT, heavy desk data, T057, model weights, production flags, solver logic or
data-source implementation was opened or changed.

## Frozen comparison

Candidate inventory:

1. incumbent `current_unweighted_mlp`;
2. challenger `recency_weighted_mlp` with true observation-level 180-day
   decay weighting;
3. challenger `ridge`;
4. challenger `spline_ridge_gam`;
5. challenger `lightgbm`, deterministic CPU only.

The incumbent source/config identities use UTF-8/LF-normalized SHA-256 to
avoid Windows/Linux line-ending drift:

- `shape_hourly_mlp.py`:
  `8f199d8075d1cd0e1d0231c3999a819d2663c4ca0a6af8ebb1178b1e6bab4aef`;
- `pfc_shaping/config.yaml`:
  `f06bb9d101289e2750f72eae10cd8726aed525645455bf23b4ae524b1f1e972d`.

Challenger implementations remain specification-only and have no
implementation hashes. Tuning is nested-origin-only; holdout tuning is
forbidden.

## Metrics and cohort

- primary metric: `MONTHLY_LEVEL_NEUTRALIZED_MAE_EUR_MWH`;
- secondary metrics: exact seven-metric inventory from the existing estimand;
- horizon buckets: `M01_M06`, `M07_M12`, `M13_M24`, `M25_M36`;
- origin cadence: monthly;
- scheduled cohort: 12 slots, October 2026 through September 2027;
- delivery support per slot: lead months 1 through 36;
- missed slot: not shifted, backfilled or reweighted.

The scheduled count is 12 but the countable origin count is zero. External
registration, trusted time, admitted inputs, FMV-risk MDE and power calibration
remain missing. Twelve slots do not constitute a power claim.

Canonical protocol semantic SHA-256:
`c2705a8d175bfe7421e2722d316bee4a5eb5631506f284dde03363ab561cb26b`.

## Verification

Focused protocol suite:

```text
10 passed in 0.14s
```

Expanded contract matrix covering protocol, estimand, dependence/power,
origin registry, curve products, outage-aware acquisition and LT/CT imports:

```text
96 passed, 1 skipped in 6.97s
```

The skip is the pre-existing optional TensorFlow boundary. Targeted Ruff check
and format check passed. The initial protocol-hash test intentionally exposed
the hash before it was frozen. A later cross-platform audit found mixed line
endings in the historical MLP/config files; raw-byte bindings were replaced by
explicit UTF-8/LF-normalized hashes before final qualification.

## Authority and execution

- data acquisition: `0`;
- model training/retraining: `0`;
- truth rows opened: `0`;
- Databricks statements/Warehouse starts: `0/0`;
- CT changes: `0`;
- CH monthly solver-authority changes: `0`.

Training, truth opening, selection, scientific claims, monthly-level changes,
publication and production all remain false. T057 remains sealed and absent
from the canonical manifest.

## Next action

1. Externally register the schedule and exact origin information-set contract;
   missed October or later slots must remain missed.
2. Complete the governed EEX and ENTSO-E source admission needed at each
   origin.
3. Bind the unchanged incumbent replay artifact and implement each challenger
   behind a new version with exact code/runtime hashes.
4. Run development-origin nested selection only after those gates pass.
5. Keep all future cohort truth closed until its separately registered
   maturity event.

Durable decision: D-20260902-272.
