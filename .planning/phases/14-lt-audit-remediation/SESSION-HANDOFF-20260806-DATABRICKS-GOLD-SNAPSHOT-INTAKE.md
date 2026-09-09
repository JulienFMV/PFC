# Session handoff — Databricks Gold snapshot intake

Date: 2026-08-06  
Decision: D-20260806-292  
Status: local integrity gate ready; real Gold export pending; modelling paused

## Outcome

The future D291 Gold delivery now has a fail-closed local intake boundary. One
snapshot must be delivered below
`build/databricks-exports/<snapshot_id>/` with an exact manifest, cost receipt,
source semantics and ten Parquet files representing the complete Spot,
weather, Swissgrid and ENTSO-E Gold surface.

The validator streams every artifact through SHA-256 and inspects only its
Parquet footer. It verifies file identity, single-link status, size, row count,
column count, row-group count and a canonical flat-schema fingerprint. It does
not decode business values and grants no semantic, PIT, model or production
authority.

## Exact Gold inventory

- `prd.gold.dimspotproduct`;
- `prd.gold.factspotpriceinterval`;
- `prd.gold.factweather`;
- `prd.gold.factweatherforecasthistms`;
- `prd.gold.factweatherforecasthistom`;
- `prd.gold.factswissgridbalancingquarterhourly`;
- `prd.gold.factswissgridtenderofferresult`;
- `prd.gold.dimentsoeseries`;
- `prd.gold.factentsoetimeserieslatest`;
- `prd.gold.factentsoetimeseriesvintages`.

Any `dev` or Silver substitution fails before artifact admission.

## Files and hashes

- contract:
  `.planning/phases/14-lt-audit-remediation/DATABRICKS-GOLD-SNAPSHOT-INTAKE-CONTRACT-V1.json`;
  raw/content SHA-256
  `5153a3a7398352b0d76042990a7d02d731f9999ac8eef7a716ed6ea5fbd95f91` /
  `1235c2cb9cc15750d2bcbe693e414928b9fa9d9b99dc1d51b8c8b0bd88db8522`;
- validator:
  `pfc_shaping/validation/databricks_gold_snapshot_intake.py`;
  SHA-256
  `af247f3c6239198f2fbeba8dcdf20fa813c7818deb19a998ebc67e18d1e02f5e`;
- adversarial tests: `tests/test_databricks_gold_snapshot_intake.py`;
  SHA-256
  `e6ff28a9ddee4d6ae445fda690f97ef8365751f3cd6a07ede3f3092459bfe150`;
- operational documentation:
  `docs/data/DATABRICKS-GOLD-SNAPSHOT-INTAKE.md`;
  SHA-256
  `2f65297db304a9a910d5c853002a1616a354dd8d8430c0460b593d31e82e14a6`.

## Verification

- focused D292 suite: `29 passed in 2.46s`;
- D289-D292 Databricks matrix: `76 passed in 2.47s`;
- LT/CT import boundary: `17 passed, 1 skipped in 6.31s`;
- Ruff format/check: pass;
- Python compilation: pass;
- final `.d292-test-*` residue count: `0`.

The first Windows fixture run left Parquet handles open, causing directory
rename failures. Handles were closed explicitly, fixture generation was moved
to in-memory Parquet bytes, and 29 exact synthetic residue directories were
removed from `build/databricks-exports`. They contained only reproducible test
fixtures and no real or user data.

D292 made zero Databricks connections/statements, business-row reads,
Warehouse starts, writes, network calls or `H:` accesses. CT, Power BI, AFRY,
OMPEX, T057 and heavy desk data were untouched.

## Next action

Wait for either the D291 Phase 0 receipt or the governed Gold export. If a real
export arrives, run D292 first. Only after integrity succeeds should a new
batch decode values and test schemas, keys, units, signs, timestamps, vintages,
duplicates, temporal coverage, revisions and PIT leakage. Modelling must remain
paused until those real-data gates are complete.
