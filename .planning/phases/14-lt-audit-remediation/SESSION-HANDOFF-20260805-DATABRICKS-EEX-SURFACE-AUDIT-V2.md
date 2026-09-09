# Session handoff — Databricks EEX offline surface audit v2

Date: 2026-08-05  
Decision: D-20260805-211  
Scope: CH LT EEX live quote-surface quality, offline only

## Outcome

D211 retains D210's live all-product normalization and quarantine but corrects
the nesting estimand. The selected live CAL/Q/M audit bundle is:

`build/databricks-eex-daily/2026-08-05/surface-audits-live-cqm/c3891655e5cf043c904dba9d5584ce0adf845807e019ce4195697866d9d7e8d7/`

Status: `PASS_LOCAL_INTEGRITY_NO_MARKET_QUOTE_CONFLICTS`.

The D209 audit spliced one or two direct months with the overlapping
full-quarter quote as if that quarter represented only the uncovered month.
This was not a valid non-overlapping child partition and created 437 false
positive CAL conflicts. D210's live filter remains semantically correct but
was not the cause of their disappearance: the corrected live and unfiltered
histories produce the same nesting diagnostic bytes.

## Inputs and immutable bindings

- Raw snapshot SHA-256:
  `593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812`
- D210 live all-product normalization content ID:
  `bb258371a96f19a8e08b54f9126635c3a478314e3284b7bf1249dba3feaaaeb5`
- Live CAL/Q/M history SHA-256:
  `fc5de85d1870937955dcc93cbf1cea0e1d2f85d892b286f490a6de11650d7a25`
- V2 live audit content ID:
  `c3891655e5cf043c904dba9d5584ce0adf845807e019ce4195697866d9d7e8d7`
- V2 manifest SHA-256:
  `15deddb8a31c56a9bff2c1c6cf1dc5cf99bd2e09e53479f9b7ef14f410ba5766`
- Auditor source SHA-256:
  `2c7a52149d41f09f6fd611c6352ea9ea77d402ebd3438786e3818f8ee65ac94b`
- Audit-test source SHA-256:
  `4f61b23cc634409063e74e05a9ebb2ffa16c0286399f2a48803803325b86d6bd`

Superseded audit IDs:

- `080b3f58...acbcdd`: false-positive overlapping v1 audit on unfiltered CQM.
- `8d301f96...cbab3f`: correct result bytes, but its v1 manifest did not bind
  the corrected non-overlapping estimand or separate it from live filtering.

## Corrected data-quality results

- 34,105 live normalized CAL/Q/M rows over 1,939 quotation dates.
- PEAK first appears on 2023-06-26; direct earlier PEAK calibration or
  validation is unsupported.
- 3,255 complete non-overlapping parent/child comparisons.
- 1,615 CAL comparisons: zero conflict above 0.01 EUR/MWh; maximum absolute
  residual 0.005370 EUR/MWh.
- 1,640 quarter/month comparisons: zero conflict; maximum absolute residual
  0.004433 EUR/MWh.
- Latest 2026-08-04 surface: 38 CAL/Q/M rows, 19 BASE/PEAK pairs, four
  comparable parents, zero conflict and maximum absolute nesting residual
  0.002337 EUR/MWh.
- 10,766 live BASE/PEAK pairs; zero negative implied OFFPEAK in the observed
  sample and zero recomposition failure.
- Strict latest BASE solve: 77 delivery months, 18 active constraints, two
  `redundant_consistent` parents and maximum hard residual
  `8.526512829121202e-14` EUR/MWh. No conflict-policy bypass survived the roast.

The live and unfiltered corrected nesting Parquets both have SHA-256
`dc3876c4c8d6c6a1ff26596381f65e2a7eecd6c8bcac03cb33bdd200e18a9652`.
Therefore live eligibility and nesting correctness are distinct contracts.

## Changed files

- `pfc_shaping/validation/databricks_eex_surface_audit.py` — schema v2 and
  complete non-overlapping child-partition logic.
- `tests/test_databricks_eex_surface_audit.py` — six tests, including the
  partial-month/whole-quarter anti-overlap regression.
- `docs/research/forwards_sources.md` — corrected causality and v2 audit ID.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` — D211 and partial
  D210 supersession.
- `.planning/HANDOFF.md` — D211 pointer and current status.
- This handoff replaces the D209 surface-audit handoff.

Existing unrelated worktree changes, including monthly solver objective work,
were preserved. The experimental solver conflict-policy path built during the
roast was removed after the corrected real data showed it was unnecessary.

## Verification

- Focused surface-audit tests: `6 passed`.
- Focused audit plus monthly-constraint matrix: `42 passed` after removal of
  the experimental conflict path.
- Independent fresh-process replay: exact summary and three-Parquet equality,
  all source/artifact hashes valid, comparison policy bound, and every PIT,
  rolling-origin, selection, promotion and production authority flag false.
- Adjacent LT/product-normalization matrix:
  `193 passed, 4 skipped in 23.19s`.
- `git diff --check`: pass on final batch files.
- Ruff was not installed in the repo-local runtime and was not fetched.

Every audit and test used the repo-local normalized Parquet. No Databricks
connector, SQL statement, Warehouse start or remote read occurred.

## Invariants and next small batch

- Do not query Databricks for checks reproducible from the selected local
  bundles.
- Never compare overlapping child contracts as though they partitioned their
  parent delivery period.
- Preserve D210 live-delivery eligibility and the reason-coded quarantine.
- Keep strict solver failure on a genuine conflicting redundant quote.
- Do not backfill missing PEAK history or substitute legacy/synthetic EEX or
  ENTSO-E evidence.
- Do not use this audit for PIT, rolling-origin, model selection, promotion or
  production claims.
- The monthly CH solver remains sole level authority; hourly shaping may not
  rewrite monthly means.
- AFRY and OMPEX remain benchmark-only; restricted AFRY values stay outside
  Git/docs/prompts; T057 remains sealed; LT stays independent from CT.
- Governed ENTSO-E, signed EEX vintages and a new independently frozen future
  holdout remain prerequisites for empirical candidate selection.

Recommended next offline batch: quantify live quote availability and staleness
by delivery horizon using quotation dates only, without claiming historical
point-in-time availability. This can inform robust horizon-dependent input
selection while preserving all current authority gates.
