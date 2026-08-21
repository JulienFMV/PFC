# Session handoff - Databricks local LT materializer

Date: 2026-08-21

Starting commit:
`97eefc96df`

Branch:
`fix/lt-audit-remediation`

## Outcome

An offline deterministic building block now converts already-exported Gold
spot and Gold/Silver ENTSO-E frames into the canonical raw and derived frames
used by the LT replay layer. No Databricks or GitHub connection was made.

## Implemented boundary

- Gold spot: exact market/product selection, observation/delivery cutoff,
  EUR/MWh validation, atomic 15/30/60-minute expansion and DST-safe UTC grid.
- Gold ENTSO-E: current-serving materialization from dimension/latest with
  known availability.
- Silver ENTSO-E: origin-safe vintage selection with availability basis,
  publication/first/last-seen ordering, DQ exclusion, revision selection and
  ambiguous-tie rejection.
- Mapping: exact `fmv_entsoe_feature_mapping.v1`, bound to the Gold dimension
  semantic SHA-256 and explicit weighted SeriesKeys.
- Features: actual load, solar B16, wind B18/B19 and optional signed physical
  flows, then the existing causal replay transforms.
- Production climatology: no neutral fill; incomplete Swiss-local slot
  coverage now fails closed and fall-back DST remains distinct in UTC.

Every audit returned by the materializer keeps layer, model-input,
calibration and production authorities false.

## Changed files

- `pfc_shaping/data/databricks_lt_materialization.py`
- `pfc_shaping/pipeline/production_phases.py`
- `tests/test_databricks_lt_materialization.py`
- `tests/test_quality_gate.py`
- `docs/data/DATABRICKS-LT-MATERIALIZATION.md`
- `docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md`
- `docs/data/README.md`
- `.github/workflows/lt-model.yml`
- `README.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

## Verification

All mutable paths were below the canonical workspace `build/` boundary.

- initial materializer tests: `7 passed`;
- materializer plus climatology quality: `38 passed`;
- expanded materializer/layer/replay/input/quality/assembler matrix:
  `177 passed, 2 skipped in 157.19s`;
- final signature-focused matrix: `53 passed`;
- exact bounded CI data/PIT matrix: `135 passed, 2 skipped in 36.67s`;
- targeted Ruff: pass;
- `git diff --check`: pass before the final decision/handoff update;
- Databricks connections/statements/Warehouse starts: `0/0/0`;
- remote writes: `0`.

## Residual work

1. Export and admit the real Gold dimension, then author the exact SeriesKey
   mapping contract. No production IDs are guessed in the repository.
2. Integrate the pure materializer into a governed Databricks-export publisher
   with raw Parquet, query, watermark, code/config and source-manifest bindings.
3. Extend the v3 provider-raw contract for Databricks exports rather than
   impersonating Energy-Charts, ENTSO-E XML or SFOE responses.
4. Convert the existing EEX Gold normalization into the signed vintage catalog
   required by the monthly solver and resolve the separate workbook evidence.
5. Keep hydro separate until an exact Gold source is approved. Weather,
   Swissgrid and LSEG remain separately admitted candidates/benchmarks.
6. Run real-data layer acceptance and mapping/materialization only after the
   bounded local export exists. That future extraction requires explicit user
   authorization and a cost fence.

Durable decision: D-20260821-252.
