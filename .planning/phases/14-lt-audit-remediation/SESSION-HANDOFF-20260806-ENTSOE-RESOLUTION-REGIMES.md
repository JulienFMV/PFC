# Session handoff - ENTSO-E effective-dated resolution regimes

Date: 2026-08-06
Decision: D-20260806-260
Status: `PASS_SYNTHETIC_ALGORITHM_REAL_SIDECAR_REQUIRED`

## Outcome

D260 closes a cadence-model defect in the future normalized ENTSO-E package.
The original dimension had one `native_resolution` per series. That is safe
only for a series whose cadence never changes. A physical price, capacity or
fundamental series may instead cross an hourly/half-hourly/quarter-hourly
market transition.

The new sidecar uses grain `series_id, valid_from_utc` and left-closed,
right-open intervals. It requires `valid_to_utc`, resolution, source document,
HTTPS locator and owner confirmation time. The dimension resolution is treated
only as the current or single-regime value and must match the regime active at
the last observed target.

Expected targets are reconstructed piecewise. Earlier hourly truth does not
become 15-minute because a later regime changes, and no quarter hours are
upsampled. Missing slots are reported without fill. Different products and
borders may retain different schedules at the same time.

## Source interpretation

EPEX SPOT treats Swiss 15-minute introduction separately from the SDAC rollout
and documents cross-product handling across 15/30/60-minute products. Swissgrid
describes distinct planned rollouts by Swiss auction and border. Neither source
is used as proof that a particular physical series changed on a chosen date.
Only the future owner/source-backed sidecar may establish that fact.

Sources:

- `https://www.epexspot.com/en/new-15-minute-products-market-coupling`
- `https://www.swissgrid.ch/content/dam/swissgrid/about-us/newsroom/publications/balancing-roadmap-fr.pdf`

## Exact changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-RESOLUTION-REGIME-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_resolution_regimes.py`
- `tests/test_entsoe_resolution_regimes.py`
- `build/databricks-eex-daily/materialize_entsoe_resolution_regime_proof.py`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-REAL-MAPPING-DATA-ENGINEER-REQUEST-20260806.md`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-GAPS-20260805.md`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff.

Concurrent D252-D254 files and their latest temporal-zone handoff were
preserved. No CT, AFRY numeric, Power BI or heavy desk data file was modified.

## Canonical evidence

- contract raw SHA-256:
  `561679a368f054e2117a9912540eb0fb5c049e8e4fd90006a011fd9b2775d112`;
- contract canonical content ID:
  `3d412d38dc76b2d7eb6e2441b64bfc1b7323f7258af20a0752d7bfd0feea4185`;
- validator SHA-256:
  `487af0c3ac9769c3649470a6b4486c026714270b5dc9d83ebcd7c9a00f6339b9`;
- tests SHA-256:
  `c884060734e886a9c3f463b05ded11c7f49412307dec7c74172362f0fde72011`;
- materializer SHA-256:
  `b670e44f8c4fa65d2980eaf65aeb0cfb46d31a83c8f7eb02e78102d4ba0b3cd7`;
- deterministic proof/content ID:
  `740de518f8f4afc2b135f7cb3217f301c883897d39b78fa8a83d80b61fbf2739`;
- proof manifest SHA-256:
  `f767373192ce83d85620fe7bfec0680714086ef4fbac9c30b51e36dae42ee881`;
- proof path:
  `build/databricks-eex-daily/2026-08-06/entsoe-resolution-regime-proofs/740de518f8f4afc2b135f7cb3217f301c883897d39b78fa8a83d80b61fbf2739/`.

The selected proof contains two synthetic series, three regimes and 168 target
timestamps, but zero value columns, zero market/vendor values and no clear
series identifiers. Two final materializations returned identical bytes.
Earlier D260 proof IDs are superseded and not selected.

## Roast and commands

Every command verified exact cwd and Git root
`C:\Users\jbattaglia\PFC_LT`. Mutable test paths and `TEMP/TMP` were below
`build/`.

- focused regime roast: `18 passed in 0.19s`;
- adjacent D239/D245/D250/inventory/directional-mapping/D260 matrix:
  final exact-hash rerun `92 passed in 6.91s`;
- isolated replay of a concurrency-sensitive D245 staging-cleanup test:
  `1 passed in 0.47s`;
- Ruff 0.15.12: all validator, tests and materializer checks pass;
- two final materializer runs: identical proof ID.

An initial adjacent run inherited workstation `TEMP/TMP` under `AppData`, so
ten D245 tests failed their intended workspace guard while 82 other tests
passed. The conforming rerun used repo-local `TEMP/TMP`. A following run saw a
temporary staging directory from a concurrent process between its before/after
snapshots; the directory disappeared and the isolated test passed. Neither is
reported as a code regression or hidden as a green authoritative receipt.

## Cost and authority

D260 made zero Databricks connections or statements, zero control-plane GETs,
zero Warehouse starts, zero Databricks writes, zero network calls, zero `H:`
accesses and opened zero real rows. The D247 2026-08-06 reservation remains
consumed and no same-day retry is allowed.

Real source evidence, owner identity, PIT, value quality, model input,
selection, candidate, promotion and production authority are all false.

## Next safe actions

1. Ask the data engineer for the effective-dated resolution sidecar together
   with the D250 right-edge mapping and D254 zone-history sidecar.
2. Extend the future immutable normalized export package from the D239/D245
   single-regime fixture to bind and stream this sidecar; do not reinterpret
   the current synthetic proof as real admission.
3. On a later reserved day, run the bounded D243 dimension inventory only if
   the Warehouse is already running; never start it.
4. Once the governed local package passes integrity, mapping, cadence, zone,
   quality and PIT gates, freeze a new independent holdout and resume empirical
   rolling-origin selection.

Predecessor:
`SESSION-HANDOFF-20260806-ENTSOE-TEMPORAL-ZONE-CONFIGURATION.md`.
