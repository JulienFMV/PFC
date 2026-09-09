# Session handoff — EEX short-tenor contrast construction (D220)

Date: 2026-08-05  
Scope: LT only, offline local EEX Parquet, no Databricks/network action

## Outcome

D220 admits a dormant deterministic constructor for separate CH EEX
DAY/WEEK/WEEKEND additive contrast observations. It does not fit, blend,
select, assemble or activate a model feature. The selected local panel bundle
is:

`build/databricks-eex-daily/2026-08-05/short-tenor-contrast-panels/7a4c2a601e169d8c471e0939919f12cf56b03d132d8691a82d6bfef5b392bd03/`

Status:
`PASS_LOCAL_DESCRIPTIVE_CONSTRUCTION_ONLY_NO_MODEL_AUTHORITY`.

The constructor:

- requires exact normalized D212 schema and one source snapshot hash;
- requires CH, EEX, EUR/MWh, Databricks PRD Gold source metadata and all
  authority flags false;
- rejects non-live, duplicated, malformed or mixed-snapshot inputs;
- constructs DAY-within-WEEK, DAY-within-WEEKEND,
  WEEKEND-versus-implied-WEEKDAY BASE and BASE/PEAK implied-OFFPEAK families;
- keeps every component separate and hour-weighted zero-parent-mean;
- emits no component for incomplete or conflicting same-vintage strips;
- retains negative implied OFFPEAK in additive price space;
- materializes exactly one component on complete Europe/Zurich months at 1h or
  15min, with exact support-hour reconciliation and additive zero outside
  support;
- remains absent from `production_phases`.

## Selected inputs and outputs

Inputs:

- D212 normalization content ID:
  `2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f`;
- all-product history SHA-256:
  `896f0c9f839b7fc9364398ed0848b0f1886c2ee54d277bf327ae1a414833c06e`;
- D216 audit content ID:
  `b09eb3250df5a3c0616eb169c512319c514ddf540251b405023d9351bd5d8bde`;
- source snapshot SHA-256:
  `593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812`.

Selected bundle:

- content ID:
  `7a4c2a601e169d8c471e0939919f12cf56b03d132d8691a82d6bfef5b392bd03`;
- manifest SHA-256:
  `a62e5656febf38b088e18b9e463c2eb9851cb0d0c90ac7584e7d527dfa4e89f1`;
- `contrasts.parquet`: 95,783 rows, SHA-256
  `99933b4e4fce7e208f3f57017f6ba1ad6b1342e97804958b64465cdc4e550dcc`;
- `diagnostics.parquet`: 48,324 rows, SHA-256
  `f06f024de5e8f8d1e758195c7a260f64ff49bd9a488f10092c87294ee199e049`;
- `summary.json`: SHA-256
  `663bbc40827efe51e3a0f8e7b32b60b1f1679ea097a8d1e3a8816257bff42694`.

Two exact materializations returned the same content ID.

## Data-quality findings

From 38,070 short-tenor quote rows across 1,938 quotation dates:

- 17,635 accepted separate components;
- 12,170 BASE/PEAK implied-OFFPEAK components accepted of 25,900
  diagnostics (46.99%);
- 1,123 DAY-within-WEEK accepted of 10,884 (10.32%);
- 3,777 DAY-within-WEEKEND accepted of 3,821 (98.85%);
- 565 WEEKEND-versus-WEEKDAY BASE accepted of 7,719 (7.32%);
- zero parent/child quote-conflict component;
- maximum accepted parent/child residual:
  `0.0043712574850331976 EUR/MWh`;
- maximum zero-parent-mean residual:
  `8.526512829121202e-14 EUR/MWh`;
- two negative implied OFFPEAK observations retained.

Rejected diagnostics remain missing rather than filled: 9,805 incomplete DAY
strips, 13,730 missing PEAK pairs and 7,154 missing WEEKEND parents. The
accepted diagnostic rate is about 4.7% in 2019-2022, 33.38% in 2023, 65.13%
in 2024 and 72.5% in 2025-2026. This matches the documented coverage-regime
change but is not signed point-in-time availability.

## Changed files and hashes

- `pfc_shaping/lt/model/short_tenor_contrasts.py`
  SHA-256 `95ca6087f9ea213111abed2c06ba87978d21cb4d4c4221bb40e953abba626b0b`;
- `tests/test_short_tenor_contrasts.py`
  SHA-256 `4f2986eba1e91b1b68f057b68a36567570c119b2173cbf600bed55cd80ae9b6a`;
- `.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-CONTRAST-CONSTRUCTION-V1.json`
  SHA-256 `8d18107356ae60e1379baaee5392ff7f2fc2c79895140c6e4423105ad970a802`;
- `build/databricks-eex-daily/materialize_short_tenor_contrasts.py`
  SHA-256 `481314d1aa07375bcf02c290306723384f35620542098b21200d0ed858958435`;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`: D220 added;
- `docs/research/forwards_sources.md`: D220 construction section added and
  D219 proof label corrected;
- `.planning/HANDOFF.md`: latest handoff pointer/status updated;
- this handoff added.

## Verification

Focused tests:

```text
python -m pytest tests/test_short_tenor_contrasts.py -q
19 passed in 1.47s
```

Adjacent EEX/solver/LT matrix:

```text
200 passed, 4 skipped in 16.65s
```

The matrix covered constructor, D219 projector, EEX short-tenor audit,
normalization, horizon/surface audits, monthly solver constraints/integration,
LT/CT import boundary and LT package contract.

Independent deterministic property roast:

- 12 ISO weeks across 2026-2029;
- spring/fall DST and cross-month weeks;
- 1h and 15min cadence;
- 336 individual component projections through D219;
- maximum active-constraint residual:
  `2.6645352591003757e-14 EUR/MWh`;
- maximum monthly residual:
  `5.4737133025664514e-15 EUR/MWh`.

Python byte-compilation passed. Ruff was unavailable in the repo-local test
runtime (`No module named ruff`); this is a tooling absence, not a test pass.

## Failures encountered and disposition

- Initial test collection failed because the pre-existing parametrization used
  names instead of string parameter names. Corrected to
  `("mutator", "message")`.
- The initial autumn-DST assertion treated every Week diagnostic as 169 hours,
  incorrectly including the 60-hour PEAK strip. Corrected to assert 169 hours
  for BASE/BASE_PEAK and 60 hours for PEAK; Weekend similarly distinguishes 49
  total hours and 24 PEAK hours.
- Initial content-addressed bundles
  `816b993244fa427d17e33a52c4c073bce4ddd23e3a3ddd6f887a91f9c9d7bc6c`
  `ae7b962ca5f85ff75223ad45fd578c29b49a1d66dfbdd418b941c337dda38147`
  and `309b9f07236d1cfb32b3d92c1ed5413bde6966fa383629882a539ccbd60e9cb5`
  were superseded after the contract gained the single-component materializer
  rules and final test bytes were bound. Select only `7a4c...bd03`.

## Authority and risks

- Databricks request count: 0.
- SQL Warehouse start count: 0.
- Network call count: 0.
- Remote write count: 0.
- Point-in-time availability proven: false.
- Model input / assembly / selection / promotion / production: false.
- No component may be combined or weighted until a future preregistered
  amplitude/clipping policy is admitted without OMPEX/AFRY/T057.
- Governed ENTSO-E, signed EEX vintages and a new independent future holdout
  remain mandatory. Current empirical gate remains
  `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.
- Monthly solver remains sole level authority; no individual month patch.
- LT/CT separation unchanged.

## Next safe batch

Do not activate the panel. A safe offline continuation is limited to consumer
schema/manifest hardening or preregistration scaffolding that does not choose
weights from local outcomes. Empirical amplitude selection and candidate
assembly must wait for signed EEX PIT vintages, governed ENTSO-E and the new
independent holdout.

## Post-handoff byte audit

An ambient-runtime Ruff pass subsequently reordered imports only in
`tests/test_short_tenor_contrasts.py`. The current test SHA-256 is
`505d30aa8608af24619b5fcccf68d3e20f3b9520dc31ee73a97eb74c1d2ba858`;
the constructor module, contract and canonical materializer hashes remain
unchanged. The current focused test still reports `19 passed`, and the broader
ambient LT matrix reports `254 passed, 1 skipped`.

Two ambient-runtime replays produced byte-identical bundle ID
`293e7b126164cce4e705589a5fdb9a3c1e5b456a32c8a0f5f580f2a392256927`
with manifest SHA-256
`447a23ee0900455f8987a9de69f3cbd0fe1205c7e22affe5a1d735f2292e1b47`.
Their Parquet bytes differ from the governed-runtime D220 bundle, so the
ambient replay is explicitly non-selected. The D220 selection remains
`ae7b962ca5f85ff75223ad45fd578c29b49a1d66dfbdd418b941c337dda38147`,
which retains the governed-runtime materialization and the 336-projection
roast. This note does not change formulas, results, authority or activation
state.
