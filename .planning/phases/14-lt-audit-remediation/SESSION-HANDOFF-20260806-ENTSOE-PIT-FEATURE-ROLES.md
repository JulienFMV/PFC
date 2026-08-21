# Session handoff - ENTSO-E PIT-safe LT feature roles

Date: 2026-08-06  
Decision: D-20260806-273  
Scope: LT feature timing and missingness only; zero-query and value-free

## Outcome

D273 defines seven explicit information roles and rejects ENTSO-E feature rows
that were not knowable at the model origin. It separates realized actuals,
operational forecasts, forecast errors, lagged actuals, known calendar,
origin-frozen climatology and governed scenario shape.

The roast found and closed three generic temporal bypasses:

1. future target rows relabelled as training covariates;
2. future target rows relabelled as lagged covariates;
3. contemporaneous prediction rows whose target had already started before the
   forecast origin.

Temporal leakage is reported as `CRITICAL`; other support/missingness/policy
rejections are `HIGH`; clean structural rows are `INFO`. Passing remains
strictly non-authoritative.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-LT-FEATURE-AVAILABILITY-CONTRACT-V1.json`
  - exact seven-role contract and added training/lagged/contemporaneous target
    timing invariants.
- `pfc_shaping/validation/entsoe_lt_feature_availability.py`
  - exact hash-bound contract validation;
  - value-free role/timing/dependency/missingness assessment;
  - critical/high/info severity;
  - zero-authority and zero-execution receipt.
- `tests/test_entsoe_lt_feature_availability.py`
  - 33 focused cases including actual leakage, operational horizon, forecast
    error dependencies, backfills, climatology cutoff, scenario timing,
    missingness, zero-mean monthly effects, canonical time and the three roast
    bypasses.
- `.planning/phases/14-lt-audit-remediation/ENTSOE-REAL-MAPPING-DATA-ENGINEER-REQUEST-20260806.md`
  - exact origin, target, availability, forecast-horizon and dependency-time
    fields requested from the future governed export.
- `docs/research/forwards_sources.md`
  - plain-language distinction between historical actuals, operational
    forecasts and N+3-eligible information.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - finalized D-20260806-273.
- `.planning/HANDOFF.md`
  - latest pointer and current D273/D272 status.

## Verification

- focused:
  `python -B -m scripts.run_workspace_local --run-id entpit273e python -B -m pytest tests/test_entsoe_lt_feature_availability.py -q`
  - `33 passed in 0.14s`.
- complete ENTSO-E:
  `python -B -m scripts.run_workspace_local --run-id entall273b python -B -m pytest <all tests/test_*entsoe*.py> -q`
  - `588 passed in 21.03s`;
  - receipt `TARGET_EXIT_ZERO_NOT_AUTHORITY`;
  - target/runner exit `0`, output complete, 588 tests, zero failures/errors.
- Ruff:
  `python -B -m scripts.run_workspace_local --run-id entprf273d python -B -m ruff check pfc_shaping/validation/entsoe_lt_feature_availability.py tests/test_entsoe_lt_feature_availability.py`
  - `All checks passed!`.

Earlier `entall273a` was invalidated because the D273 source changed during the
run; its failures were not treated as evidence. The stable `entall273b` run
supersedes it.

## Hashes and identities

- contract file SHA-256:
  `3cb013cf11ef787538473fdcceef6287259106a4eb7cf97bfba27ef26693dfb6`
- contract canonical content ID:
  `c7826b4ad2fa5cdb6baff5d077f00ee5fd8d98108cebef9920c147d787df2ab0`
- validator SHA-256:
  `89a9d90195db78f5f82ffd715c7e94a41e5eef1ecd177345b601b7eff6dc75b5`
- tests SHA-256:
  `9b7f217ee66121a517f92359e168665a39994e393929b07e8fb3e444ffadccd2`

## Cost and authority

- Databricks connections/statements/Warehouse starts/writes: `0`;
- network calls: `0`;
- `H:` access: `0`;
- real ENTSO-E value rows opened: `0`;
- real availability, PIT, predictive, model, candidate and production
  authorities: all false.

## Next safe batch

D280 is reserved to compose D272 series semantics, D254/D255 zones,
D260/D261/D270 cadence, D245 quality/package evidence and D273 feature timing
without collapsing their independent authorities. It must remain synthetic and
zero-query until a governed local ENTSO-E package exists.
