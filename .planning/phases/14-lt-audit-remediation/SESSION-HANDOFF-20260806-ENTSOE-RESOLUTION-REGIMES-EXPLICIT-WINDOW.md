# Session handoff - ENTSO-E cadence explicit assessment window

Date: 2026-08-06  
Decision: D-20260806-261  
Status: `PASS_SYNTHETIC_EXPLICIT_WINDOW_ALGORITHM_REAL_SIDECAR_REQUIRED`

## Outcome

D261 closes the remaining completeness loophole in D260. Cadence is assessed
over an explicit immutable UTC interval rather than the first and last
observed rows. Missing leading and trailing slots, and a series with no rows,
therefore remain visible.

`metadata_as_of_utc` is a provenance cut-off, not a delivery timestamp. It
must be at or after the assessment end but may be off the delivery grid. The
dimension's current resolution is checked against the effective regime at
that instant. Owner confirmation must not be later than the same cut-off and
source locators must be clean HTTPS URLs.

## Exact changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-RESOLUTION-REGIME-CONTRACT-V2.json`
- `pfc_shaping/validation/entsoe_resolution_regimes.py`
- `tests/test_entsoe_resolution_regimes.py`
- `build/databricks-eex-daily/materialize_entsoe_resolution_regime_proof.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- this handoff.

The D260 v1 contract and its prior proof remain historical evidence. D261 v2
supersedes observed-bound completeness. Concurrent D255/D271 work was
preserved.

## Canonical evidence

- contract SHA-256 / content ID:
  `a661cd047ab498e2286ce4f082bca345b75318a2e94aebf8f54d325279fd0e7d` /
  `450d3628bf7dc83e16d630fe001709b58561f74bdad66d73b155b199f5d96821`;
- validator / tests / materializer SHA-256:
  `e687f895829ebdb0a148105a1281c023a2e06e4e8c422064e1b961d71b2078bc` /
  `0e4d2556aa38e9ef38512f39868ad68898eb168a34381f3305258402abc3121d` /
  `e5c8ccae583fa65e8326f09ea277e059ac2fa928f634310eb633bd0c8e89378a`;
- deterministic proof ID:
  `681089851d35af5fc67a86c77319daa3508355260ff9498aa86b1ee18bd94f05`;
- proof manifest / assessment SHA-256:
  `2da79c8b4363bc8843584a86c3f660297c8b5f7e5489a79b730626f4690dfeae` /
  `3bd29d0c6c86ef319f7b486396333da9fada392c33c7381bcbd50f0051c65d54`;
- proof path:
  `build/databricks-eex-daily/2026-08-06/entsoe-resolution-regime-proofs/681089851d35af5fc67a86c77319daa3508355260ff9498aa86b1ee18bd94f05/`.

## Verification

- focused D261 roast: `30 passed`;
- adjacent D244/D245/D261/D270 matrix later in the same session: `70 passed`;
- Ruff passes on validator, tests and materializer;
- two v2 materializations returned the same proof ID.

An earlier D270 adjacent launch exposed only a Pytest cross-module fixture
registration defect (`61 passed, 9 setup errors`). The D270 fixture was made
local and the exact matrix then passed; this was not a D261 regression.

## Cost and authority

Zero Databricks connections/statements/writes, zero Warehouse starts, zero
network calls, zero `H:` accesses and zero opened real rows. No real-source,
owner, PIT, value, model, candidate, promotion or production authority is
granted.

## Next safe action

Use D270 to bind this exact v2 cadence contract and a content-addressed regime
sidecar to the immutable D244/D245 Parquet package. Real admission still
requires the governed owner/source-backed sidecar and all independent family,
zone, sign, lineage, quality, revision and PIT gates.

