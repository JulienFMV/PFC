# Session handoff — local export ownership and provisional bootstrap

Date: 2026-08-06  
Decision: D-20260806-293  
Status: ownership corrected; local engineering can advance; modelling remains paused

## Outcome

The data engineer owns the governed Databricks Gold publication, backfill,
vintages and source semantics. The PFC team owns the later selective read-only
export, immutable Parquet snapshot, manifest and hashes below
`build/databricks-exports/<snapshot_id>`. The data engineer must not write to
the workstation `C:` drive.

The interim strategy is not to rebuild the whole data universe. Reuse existing
local captures first, mark incomplete/legacy inputs as
`PROVISIONAL_ENGINEERING_ONLY`, and acquire an official external source only
for a demonstrated gap or reconciliation. Provisional inputs may support
mapping, adapters, time/calendar logic, quality checks and exploratory feature
code; they cannot support final calibration, model selection, OMPEX comparison
or promotion.

## Local inventory observed without decoding heavy values

- Databricks EEX capture already exists under
  `build/databricks-eex-daily/2026-08-05/`, including
  `eex_ch_power.parquet` (542,030 bytes),
  `eex_ch_cal_q_m_history.parquet` (148,621 bytes) and
  `eex_ch_cal_q_m_latest.parquet` (11,654 bytes).
- Existing provisional local inputs include
  `pfc_shaping/data/entso_15min.parquet` (5,430,291 bytes),
  `epex_15min.parquet` (1,284,397 bytes),
  `epex_de_15min.parquet` (1,307,770 bytes) and
  `hydro_reservoir.parquet` (74,477 bytes).
- The staged Swissgrid month-ahead source remains
  `build/external-inputs/swissgrid-ntc/NTC-202609.pdf` (86,483 bytes).
- This inventory read filenames, sizes and timestamps only. It did not decode
  the heavy Parquet contents.

## D293 contract hardening

The former D292 ten-Parquet package was insufficient for ENTSO-E acceptance.
The current intake requires sixteen Parquet artifacts and four companions. It
adds resolution history, zone/EIC history, per-series quality, gap report,
source reconciliation and exclusions, plus ENTSO-E quality and family JSON.
Gap and exclusion Parquets may contain zero rows with a valid schema; core data
and reconciliation artifacts must be non-empty.

The two data-engineer documents now state explicitly that the engineer
publishes `prd.gold`; our team creates the local snapshot. The detailed
ENTSO-E request also names the required Gold resolution and zone-history
surfaces and leaves quality-report generation to our local exporter.

## Changed files and final hashes

- Gold/source request:
  `1ab1f396fb37455d6b91f1bd2c374870dc3910214d894294e4e8c81397056b6b`;
- ENTSO-E Gold-publication request:
  `65116313a73fca108d1e8e1a406119513a689f5f14c65772764ab58371d89828`;
- Gold intake contract raw/content:
  `663b9cbf5dec1de0ff5071ae7876cf6f5849677c7eef8f95ff639466f6298ba9` /
  `1ed6b11e19fee7a31be976699230906eaac73d9f344e4cac17191b5623282869`;
- cost-preflight contract raw/content, rebound to the corrected request:
  `0beac8ef83e31f039b7e862987785694889ec5f946b082342931cfa4bf1a768c` /
  `4d8857e09eb4fd2acadcd51faa9110a3e406cd1a04afea439d7f10c6e9da8771`;
- Gold intake validator:
  `3baacd2953e5827aedbb16e95cc7bb10b4765a760d994271e798fd8d5394936a`;
- Gold intake tests:
  `963d7ef324d9983d13d845ecb9bf401e732575f6ab6ac75b9ea4277fcdfb895f`;
- request tests:
  `128670808ed75449493983f80667a914df271b9b20c05031fab325724109cd96`.

The decision log contains D293. Its hash will change again when future
decisions are appended and is therefore not used as a stable input binding.

## Verification

- final request, SQL-profile, cost-preflight and Gold-intake matrix:
  `83 passed in 4.14s`;
- one transient Windows fixture-directory lock occurred in the first run; the
  isolated replay passed, and the complete rerun passed;
- Ruff format/check: pass;
- 29 exact synthetic test directories left by transient Windows fixture locks
  were removed; no real or user data was affected;
- final synthetic export fixture residue below `build/databricks-exports`: 0;
- Databricks connections/statements/business rows/Warehouse starts/writes,
  network calls and `H:` accesses in D293: all zero.

## Next small batch

Create a compact local source registry for the existing EEX, ENTSO-E, EPEX,
hydro and Swissgrid inputs using manifests and Parquet footers only. Classify
each source as governed, provisional or missing. Then profile the already-local
EEX capture first and use the local ENTSO-E only to validate mappings and
quality code. Do not download a full duplicate dataset and do not resume model
calibration until the final Gold export passes semantic, temporal, coverage,
revision and PIT gates.
