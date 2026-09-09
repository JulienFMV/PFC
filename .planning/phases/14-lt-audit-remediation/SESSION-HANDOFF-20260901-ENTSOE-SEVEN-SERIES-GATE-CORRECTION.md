# Session handoff - ENTSO-E seven-series gate correction

Date: 2026-09-01

Branch: `fix/lt-audit-remediation`

## Outcome

Point 1 is complete and audited offline. No Databricks SQL, Warehouse start or
write was issued.

The day-ahead contract now distinguishes:

- source inventory: seven exact PRD SeriesKeys;
- PIT selection: five explicit SeriesKeys, one per market field;
- model authority: still false.

Exact inventory:

| Slot | SeriesKey |
|---|---|
| CH | `day_ahead_prices||ch_price` |
| AT sequence 1 | `day_ahead_prices||at_price||1` |
| AT sequence 2 | `day_ahead_prices||at_price||2` |
| DE-LU sequence 1 | `day_ahead_prices||de_lu_price||1` |
| DE-LU sequence 2 | `day_ahead_prices||de_lu_price||2` |
| FR | `day_ahead_prices||fr_price` |
| IT-North | `day_ahead_prices||it_nord_price` |

Sequences 1 and 2 are legitimate A44 multi-auction identities. They are not
technical vintages, duplicates or old/new key coexistence.

## Changed files

- `pfc_shaping/validation/entsoe_day_ahead_prd.py`
  - seven-slot exact inventory contract;
  - exact field/classification identity checks;
  - explicit five-field PIT selection with no AT/DE-LU default;
  - accepts only sequence 1 or 2 for AT and DE-LU;
  - emits value-blind per-series resolutions, row counts and temporal bounds;
  - keeps cadence/model/selection/production authorities false;
  - SHA-256
    `50033f4340276f196c94ea355a7a1ea193c72b5413c319954b2a516361af97a5`.
- `pfc_shaping/validation/spot_source_reconciliation.py`
  - revalidates the renamed `series_selection` PIT audit binding;
  - SHA-256
    `3cb404a93c2d67be5b53dcd659d1f30b4e69c0e1f8f26d0d434140707ba4a04e`.
- `tests/test_entsoe_day_ahead_prd.py`
  - exact seven-key evidence;
  - both multi-auction sequences required in inventory;
  - missing sequence detected even when all five fields exist;
  - unexpected sequence 3 rejected;
  - sequence 1/2 selection accepted only when explicit;
  - base AT/DE-LU selection rejected;
  - effective-dated multi-resolution profiles accepted without false cadence
    authority;
  - SHA-256
    `02e7d4834cbc1c0c0b7bef2d89c7e15b562b3be150c63f677c30927219546246`.
- `tests/test_spot_source_reconciliation.py`
  - uses an explicit sequence-1 synthetic selection; no production preference
    is inferred;
  - SHA-256
    `645ffd11666f010ba6bb5bdf737ba6f7e6c333cc07a25296e52d5ac45409afa6`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - durable decision D-20260901-262.
- `.planning/HANDOFF.md`
  - current handoff pointer and state.

The hash-bound SQL templates did not require modification:

- profile SQL SHA-256:
  `d89bc5f42b1ec6cefcfb1cfb5ef044f22a647941c598f9f4530c86c5d1295603`;
- selected five-series PIT SQL SHA-256:
  `7b444430db6b62a7bcd9f0bc6e5f05858a56b9756be68f70016c107a4b271ebf`.

## Audit evidence

All commands ran from the guarded canonical root through
`scripts.run_workspace_local`.

```text
dapseventest3
pytest tests/test_entsoe_day_ahead_prd.py tests/test_spot_source_reconciliation.py
59 passed

dapsevenmatrix1
pytest reconciliation + ENTSO-E gate + Databricks layer/cost/materialization + resolution + receipt
191 passed

dapsevenentsoe1
pytest all tests/test_*entsoe*.py -m "not slow"
769 passed

dapsevenbound1
pytest tests/test_lt_ct_imports.py tests/test_lt_package_contract.py
43 passed, 1 skipped

dapsevenruff4
ruff check corrected modules and tests
All checks passed

dapsevenfmt3
ruff format --check corrected modules and tests
4 files already formatted

git diff --check
pass
```

The recurring pytest warning is the pre-existing unknown `cache_dir`
configuration warning.

## Audit conclusion

No remaining code defect was found in point 1. The gate no longer confuses
multi-auction identities with source revisions and cannot silently select or
average AT/DE-LU sequences.

The profile intentionally does not prove cadence completeness. It reports the
evidence needed to derive that rule. The next external action, only after this
handoff is accepted, is the low-cost value-blind profile on the already-running
or separately authorized PBI SQL Warehouse. Price extraction remains later.

Databricks SQL statements / Warehouse starts / writes for this correction:
`0/0/0`.

Model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`; T057 remains sealed.

Durable decision: D-20260901-262.
