# Phase 14 LT Audit Remediation Summary

## Baseline Fixture Refresh

`tests/fixtures/baseline_pfc_seed42_phase05.parquet` was regenerated after the
P0 water-value sign and block-neutral delta fix. This is an intentional math
change under the Phase-5 fixture policy: the canonical baseline must match the
current default negative-ready LT math rather than the pre-remediation water
value semantics.

Generation command:

```powershell
python tests/fixtures/_generate_phase05_fixture.py --generate-baseline
```

Measured against the previous committed baseline:

| metric | old | new |
|---|---:|---:|
| rows | 2976 | 2976 |
| columns | 14 | 14 |
| `price_shape` min | 6.521363855860 | 5.299584714849 |
| `price_shape` max | 25.380877405702 | 25.380877405702 |
| `price_shape` mean | 20.076203687143 | 20.007939085413 |
| `price_shape` max abs diff |  | 1.339021444862 |
| `price_shape` mean abs diff |  | 0.072378675599 |
| changed `price_shape` rows |  | 1935 / 2976 |

The largest changed point is `2027-07-03 08:30:00+00:00`:
old `7.209918237992`, new `5.870896793130`.

The generator also rewrote `forwards_phase05_seed42.parquet` byte-for-byte, but
the DataFrame was identical; that file was restored to avoid unrelated fixture
churn.
