# D302 — structural shaping first executable lot

Date: 7 September 2026. Status: implemented, executed and independently checked.
User instruction: "alors cmpact et continue". No tool can trigger `/compact`;
the user was told this and authorized local work continued in the same root.
No renewed calibration/benchmark permission is needed for later local work.

## Outcome and exact scope

Implemented a read-only audit of five pinned retained public-source scenario
inventories, plus a numerical signed hourly shape function. This completes the
first 2030+ input/target milestone. It does not implement dispatch, fit a model,
produce a forecast, integrate signed shapes into assembly, or admit a scenario.

Canonical artifacts: `build/lt-structural-shaping-20260907/audit-v2/`.
Open `RAPPORT-SHAPING-2030.md`, `source-summary.csv`, `annual-coverage.csv`,
`field-matrix.csv`, `mechanism-coverage.csv`, `signed-history-months.csv`,
`signed-historical-targets.parquet`, `audit.json`, `manifest.json`.
The task root also contains `verify.py`, `verification.json` and test XMLs.

135 tests passed, four optional CT dependency skips. Arithmetic checked on
91 complete Swiss months, February 2019 through August 2026: 66,455 native
CH hours, including 1,032 negative original-price hours. All four transport
quarter-hours agree exactly before being represented as one observation.
January 2019 and September 2026 are incomplete and excluded. No negative-day
filter, positive ratio or clipping. Signed mean residual maximum
5.558023177501228e-14 EUR/MWh; independent scalar-sum recalculation maximum
1.1368683772161603e-13 EUR/MWh. These are arithmetic checks, not forecast errors.

## Source findings and remaining structural evidence

Audit origin `2026-09-07T12:00:00Z`; countries CH/DE/FR/IT/AT; requested exact
annual years 2030–2035. Scenario labels stay per source. A cell means one
source/scenario/country/year, not an independently validated observation.

| Retained source under data/ | Rows read | Exact/requested cells |
| --- | ---: | ---: |
|electrification_scenarios_tyndp2024_supply.parquet|30|15/90|
|electrification_scenarios_tyndp2024_demand.parquet|16|0/60|
|electrification_scenarios_ep2050.parquet|6|4/60|
|electrification_scenarios_composed_p0_public_sources_2030.parquet|15|15/90|
|electrification_scenarios_prod_candidate_neutralized_2030.parquet|15|15/90|

- TYNDP supply has 2030/2040 rows with locally mapped slow/central/fast labels.
  CH 2030 has battery energy but missing power, demand and several other fields.
- TYNDP demand has 2040/2050 rows for neighbours under different source labels;
  it supplies no exact 2030–2035 cell. Never silently merge scenario names.
- EP2050 has CH WWB/ZERO_Basis rows in 2025/2030/2035; only 2030/2035 enter
  this audit. It lacks ingestion timestamps and battery/flexibility parameters.
- Composed public 2030 rows contain partial/proxy components. The neutralized
  derivative fills missing DSM/managed charging/coal with explicit zeros;
  this cannot establish that the future system has no demand flexibility.
- Battery operation fields (efficiencies and initial/terminal state shares)
  are absent even in the most filled composed table. Complete weather profiles,
  annual/hourly reconciliations, hydro inflows/carry-over and bidding behaviour
  are not provided by annual non-null coverage.

All these files were read in memory after exact hash checks. No protected desk
data was overwritten, no table fetched, no interpolation/clamping/neutralizing
performed, no AFRY numerical source loaded. Hashes are in `audit.json` and
the fixed allowlist in the script. Partial source metadata is retained, not
upgraded to signed availability or scenario admission.

## Files changed in this lot

- `pfc_shaping/lt/structural_readiness.py`: exact annual field diagnostics;
  complete-Swiss-month signed shape centering using existing UTC validation.
- `scripts/audit_lt_structural_readiness.py`: fixed five-source read-only audit,
  D300 native-hour arithmetic verification, source/code/output hashes, report.
- `tests/test_lt_structural_readiness.py`: 35 tests including invalid inputs,
  timing, duplicates, missing/zero distinctions, DST/leap, monthly invariance,
  hash tampering, transport identity and months not closed at audit origin.
- `docs/model/LT-STRUCTURAL-SHAPING-CONTRACT.md`: input/numerical contract and
  required chronological physical mechanisms.
- `.planning/HANDOFF.md`, this handoff and Phase 14 `DECISION-LOG.md` D302.
- Task-local `build/lt-structural-shaping-20260907/verify.py` and generated
  evidence/runtime directories. Prior uncommitted user work is preserved.

## Commands, failures and verification

Every shell command used the canonical cwd plus `git rev-parse --show-toplevel`
guard. No H: command, elevation, exception or external writable path.
Interpreter `build/conda-runtime-v41-model-source/python.exe -B`; cache/TEMP/TMP/
APPDATA/LOCALAPPDATA/MPLCONFIGDIR/XDG_CACHE_HOME/NUMBA_CACHE_DIR/JOBLIB_TEMP_FOLDER/
PYTHONUSERBASE/PIP_CACHE_DIR below task-root `runtime/`; BLAS threads=4,
`CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`. No dependency installation.

Commands after those guards/environment settings:

1. `-m pytest tests/test_lt_structural_readiness.py -q -p no:cacheprovider
   --basetemp build/lt-structural-shaping-20260907/pytest-unit
   --junitxml build/lt-structural-shaping-20260907/tests-unit-v1.xml`:
   31 passed/one failed. Mixed timestamp strings without seconds were wrongly
   coerced to NaT by pandas inferred-format parsing, hiding a duplicate release.
   Fixed both timestamp parsers with `format="mixed"`. Retained failing XML.
   That first run also warned about cache_dir with cacheprovider disabled.
2. Matrix: new test file plus `test_arbitrage_free.py`, `test_cascading.py`,
   `test_water_value.py`, `test_lt_ct_imports.py`, `test_electrification_shape.py`,
   `test_electrification_scenarios_data.py`, `test_lt_local_benchmark.py`.
   `-q -o cache_dir=build/lt-structural-shaping-20260907/pytest-cache
   --basetemp build/lt-structural-shaping-20260907/pytest-matrix-v2
   --junitxml build/lt-structural-shaping-20260907/tests-matrix-v2.xml`:
   final 135 passed/4 skipped, no warning. Earlier matrix-v1 was 132/4 before
   adding runner cutoff/hash/native-cadence regression tests.
3. `-m scripts.audit_lt_structural_readiness --as-of 2026-09-07T12:00:00Z
   --output build/lt-structural-shaping-20260907/audit-v2`: exit 0. Earlier
   audit-v1 retained; v2 adds the explicit complete-month-before-origin guard.
4. `build/lt-structural-shaping-20260907/verify.py`: exit 0. Independent saved
   shape recalculation, exact annual-cell counts, all artifact/source hashes,
   XML results, D300 exported curve/prepared manifest and D301 plan/scores.
5. `git diff --check` and trailing whitespace checks on new files pass.

| Evidence | SHA-256 |
| --- | --- |
|audit-v2/manifest.json|edee3eea45294fc4151d46f8952bea3d6ef11b9ef9754bdc63f271977fadf03b|
|audit-v2/audit.json|c5d94bbb6526dcdeeae310cfd0280bfa4166ccdc3aa02ac72d273e59889e5e38|
|audit-v2/signed-historical-targets.parquet|7f071d36973bceed035d11fbab678683d2a7569d65b76138fb898ebe07551d21|
|verification.json|23a229a1fcadb442d17781d3dbb40cf4d0fed1c93109dc9dbf65e1a452ba5529|
|tests-matrix-v2.xml|74b4268d8505e23bed6cf2464c4ddb3c914c6655c2fc77ca65a7b3b50f8f4bb9|

## Next authorized act and practical limit

No physical 2030 forecast can be claimed from these inputs alone. This is a
data/formulation gap for that forecast, not a reason to halt useful code work.
The next smallest code lot is a reduced chronological storage/flexible-demand
kernel with explicit power/energy bounds, efficiencies, energy balance and
initial/terminal conditions, tested on labelled mathematical unit fixtures.
Those fixtures must never substitute for real EEX/ENTSO-E or count as forecast
validation. The actual scenario inventory and weather/flow/hydro trajectories
must be reconciled before running that kernel as an FMV future prediction.

Continue to reuse existing components. Signed EUR/MWh shape is not native
multiplicative f_H; final integration requires its own versioned numerical
tests at the existing assembly boundary, preserving monthly levels and quoted
BASE/PEAK. No new adapter or per-month post-solver patches.

D300/D301, frozen scientific v6, CT, AFRY model/calendar gates and T057 remain
unchanged. All production/scientific/promotion/trading authorities false.
No active process remains. This lot makes no new predictive superiority claim.
