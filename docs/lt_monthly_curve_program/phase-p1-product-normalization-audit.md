# Phase P1 - CH Product Normalization Audit

This phase adds a read-only gate for the delivered hourly CH LT curve. It does
not regenerate curves, patch months, promote production flags, or infer missing
evidence from broad data/output scans.

## Scope

The audit compares an explicitly supplied hourly CSV against an explicitly
supplied EEX CH forwards source.

Checked gates:

- UTC-aware hourly timestamp integrity: monotone, duplicate-free, no gaps.
- Exact local Europe/Zurich product windows, including DST and leap years.
- EEX CH peak calendar: 08:00-20:00 local, Monday-Friday, excluding CH holidays.
- Direct BASE and PEAK quote means for the active Month > Quarter > Calendar
  quote set.
- Implied OFFPEAK where same-product BASE and PEAK quotes exist.
- Quote-aware residual buckets with Month > Quarter > Calendar priority.
- Dropped parent quotes when a complete finer child set exists, including
  `dropped_reason`, child products and parent/child residual. Conflicting
  dropped parents remain blocking source evidence.
- Optional explicit CSV-window scoping for quoted products with no overlap with
  the delivered CSV. Scoped-out quotes are `INFO`; partial product windows
  remain `UNSUPPORTED`.
- Conflicting duplicate quotes and missing/partial evidence.

## CLI

Example diagnostic run:

```powershell
python scripts/audit_ch_product_normalization.py `
  --csv output/local_test_ch_pfc_hourly_20260613_20301231.csv `
  --forwards data/eex_forwards_history.parquet `
  --manifest output/local_test_ch_pfc_hourly_20260613_20301231.monthly_curve_manifest.json `
  --tolerance-eur-mwh 0.000001 `
  --report .planning/phases/14-lt-audit-remediation/product-normalization-audit.md `
  --gates-output .planning/phases/14-lt-audit-remediation/product-normalization-gates.csv `
  --json-output .planning/phases/14-lt-audit-remediation/product-normalization-audit.json
```

By default the CLI returns non-zero if any `CRITICAL` or `UNSUPPORTED` gate is
present. `--allow-failed-gates` is diagnostic only: it changes the process exit
code, not the gate statuses.

Use `--scope-forwards-to-csv-window` only when the selected audit population is
the delivered CSV window. It excludes quoted products entirely outside that
window and records `INFO out_of_scope_quote` rows. It does not exclude partially
covered products.

## Evidence Rules

The audit records hashes for the CSV, forwards source, and selected manifest.
If the manifest is omitted or the explicit path is missing, the result is
`CRITICAL`. Synthetic fixtures are only used by tests.
