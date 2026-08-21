# Session handoff — Databricks EEX short-tenor audit — 2026-08-05

## Outcome

The selected D212 offline normalization was audited at DAY/WEEK/WEEKEND grain
without any Databricks request or write. The selected result is:

`PASS_LOCAL_INTEGRITY_WITH_LOCALIZED_TEMPORAL_GAPS`

Selected content ID:

`f84ac6c9461bf9b8a0c5e36618f74b9b155b9c3050969f42ba780602014f433e`

The short products are internally coherent and useful as future local
day/week/weekend shape diagnostics. They remain short-horizon settlements, not
hourly or 15-minute truth, monthly-level authority, PIT evidence, model-
selection evidence or production evidence.

## Changed files

- `pfc_shaping/validation/databricks_eex_short_tenor_audit.py`
  - SHA-256
    `604a867bf502fb0a41e226bf1198edf74a6f1904f51cc35b207dbb7356dc68e3`;
  - exact normalized 16-column schema and authority validation;
  - canonical DAY/WEEK/WEEKEND parsing and Europe/Zurich DST hours;
  - quote-horizon, daily coverage and product-lifecycle diagnostics;
  - same-vintage WEEK/WEEKEND versus complete DAY-strip comparisons;
  - BASE/PEAK implied-OFFPEAK energy recomposition;
  - official EEX holiday-calendar adjustment;
  - reason-coded normalization-quarantine reconciliation;
  - no connector, network, fill-forward, fitting or model-selection path.
- `tests/test_databricks_eex_short_tenor_audit.py`
  - SHA-256
    `21b5cd45c4dab2c4cb93062c3d6bee888c550ca6255f2398503169491d6310b5`;
  - 11 tests for complete/incomplete strips, conflicts, DST, BASE-only
    history, authority overclaim, delivery mapping, product-ID reuse, EEX
    holidays, candidate gaps and quarantine reconciliation.
- `docs/research/forwards_sources.md`
  - records the selected audit, horizons, coverage regimes, conflicts, gap
    evidence and data-engineer asks without price values;
  - SHA-256 before this handoff
    `863236a696307aa94919c41dec9ef278a43661f7dfaebbd7a21e96a708988db3`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D-20260805-215;
  - SHA-256 before this handoff
    `4c439f34ca87ffa805b8ca4e8ad0d60151bdbffdf0326c2f895c0bb7282d411d`.
- build-only utilities and evidence:
  - `build/databricks-eex-daily/profile_short_tenor_live.py`;
  - `build/databricks-eex-daily/materialize_short_tenor_audit.py`, SHA-256
    `a6c1b145fc23fc3106e706b6cf876bd29b087a62e3ea423847f619de75539445`;
  - `build/eex-public-docs/eex-holiday-calendar-20250619.pdf`, SHA-256
    `17f170941ad6765b2e53d06733a49afcc40895acaf28510b39c528677f9fed8d`.
- this handoff.

No CT file, protected heavy desk data, Power BI file, AFRY numeric artifact,
monthly solver production flag or source snapshot was modified.

## Selected source bindings

Normalization:

`build/databricks-eex-daily/2026-08-05/normalizations-all-products/2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f/`

- D212 normalization content ID
  `2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f`;
- normalization manifest SHA-256
  `f91805f3004e746ac588e19aa6745ae0dc0f490a9ef5c49aae0918e7ec3f8f53`;
- frozen source snapshot SHA-256
  `593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812`;
- all-product live history SHA-256
  `896f0c9f839b7fc9364398ed0848b0f1886c2ee54d277bf327ae1a414833c06e`;
- quarantine SHA-256
  `d1328caac9fb860fc2bb0f4bf518ec59e792338f1d154d01c0067b937f2a21e5`.

Public documents:

- EEX Contract Details 2026-07-22 SHA-256
  `e03e51125b5e0b76668bc66bd736351799323abcb57b85e940ea574cd9ff1232`;
- EEX Holiday Calendar 2025-06-19 SHA-256
  `17f170941ad6765b2e53d06733a49afcc40895acaf28510b39c528677f9fed8d`;
- holiday-calendar URL
  `https://www.eex.com/fileadmin/EEX/Downloads/Trading/Calendar/Holiday_Calendar/EEX_Trading_Calendar_Emissions_Spot_Derivatives.pdf`;
- official wording states that exchange days are Monday-Friday excluding the
  listed closures and that the calendar applies continuously until further
  notice.

## Selected audit artifacts

Path:

`build/databricks-eex-daily/2026-08-05/short-tenor-audits/f84ac6c9461bf9b8a0c5e36618f74b9b155b9c3050969f42ba780602014f433e/`

- manifest SHA-256
  `5a3d2c6af24a7f6ecc6c577f1eee5a3a80ba4aea2ed8c882424c716af78da703`;
- summary SHA-256
  `0364358c0f0de00541643c88b06e3e307eedd3db47c7de1b4a2e14291449544a`;
- `daily_coverage.parquet`: 5,813 rows, SHA-256
  `68138ae62372678e18b8668917ccab3b9e4d9b7151e59938c46c57c976ca4694`;
- `horizon_profile.parquet`: 86 rows, SHA-256
  `ab029600314be0eb70903866b7f9761c94646a1348329da84477392ba0bf2429`;
- `product_lifecycle.parquet`: 5,054 rows, SHA-256
  `0b99198b4446baed6b63adf1a878b7bb61b595f53fd8a8dd733b994be5942510`;
- `gap_diagnostics.parquet`: 53 rows, SHA-256
  `be5c8008b0a0d641fc2f82eb7e0bb7e4d976ac62077f8d274fd78eddb2ca19d6`;
- `nesting_diagnostics.parquet`: 4,900 rows, SHA-256
  `8b845ce14f353531eb99c3e7892686a65e19ec3c9342516317231a94eebc0b91`;
- `offpeak_diagnostics.parquet`: 12,170 rows, SHA-256
  `c8a06d7584eba16e11c8cef1c4c14f61c524e0e44128af5933f040e69b3b2dc7`.

All Parquets were exact-replayed in memory before publication. Two independent
materializations from D212 returned the same content ID.

The prior short-audit attempts `bbb99589...140e6ed` and
`b2e8c6b1...888da821` remain under `build/` as superseded forensic evidence.
The first bound the superseded D210 normalization; the second preceded explicit
quarantine reconciliation. Neither is selected or deleted.

## Data-quality findings

Dataset and grain:

- 38,070 live settlement observations over 1,938 quotation dates;
- 5,054 product/load lifecycles;
- DAY BASE 15,679 rows, DAY PEAK 7,686;
- WEEK BASE 7,719 rows, WEEK PEAK 3,165;
- WEEKEND BASE 2,502 rows, WEEKEND PEAK 1,319;
- zero quote-identity duplicate and zero non-positive live horizon.

Observable quotation horizons:

- DAY: J-1 through J-13, median J-5;
- WEEK: J-3 through J-28, median J-14;
- WEEKEND: J-1 through J-12, median J-4 BASE and J-5 PEAK.

Coverage regime:

- PEAK first appears on 2023-06-26;
- 12,170 complete BASE/PEAK pairs and zero PEAK-only observation;
- aggregate pair coverage 46.9884%, depressed by the pre-PEAK era;
- DAY and WEEK pair coverage is 100% in 2024;
- 2025-2026 coverage is approximately 99-100% by family except documented
  WEEKEND exceptions.

Economic consistency:

- 4,900 complete, non-overlapping parent-versus-DAY-strip comparisons;
- zero conflict above 0.01 EUR/MWh;
- maximum absolute residual 0.0043712574850331976 EUR/MWh;
- 12,170 implied-OFFPEAK identities and zero recomposition failure;
- two negative implied OFFPEAK observations, both for the same Sunday DAY
  delivery at PEAK launch; preserved as economically possible.

Temporal diagnostics:

- 53 initial holiday-adjusted lifecycle diagnostics over seven dates;
- one 2025-09-03 WEEK PEAK diagnostic is exactly explained by the D212
  reason-coded boundary quarantine;
- 52 unexplained candidates remain over six dates;
- 2026-07-08: 30 affected product/load identities;
- 2026-07-20: 18 affected identities and the entire short-tenor date absent;
- 2020-05-20, 2025-09-26, 2025-09-29 and 2025-09-30: one affected identity
  each;
- one `ProductID` is reused across two valid WEEKEND BASE delivery identities;
  normalized product and delivery-period identities remain unambiguous.

These are localized upstream questions, not permission to fill or discard
observations.

## Commands and results

Every shell action first verified exact cwd and Git top-level
`C:\Users\jbattaglia\PFC_LT`. All mutable temporary and test paths remained
below `build/`.

1. Initial focused test roast exposed a raw-versus-normalized schema constant
   mismatch: `8 failed`. The module and tests were corrected before any real
   artifact was selected.

2. Final focused short-tenor audit tests:

   Result: `11 passed in 0.61s`.

3. Combined normalization, CAL/Q/M audit and short-tenor audit tests before the
   final quarantine-reconciliation addition:

   Result: `23 passed in 1.01s`.

4. Final adjacent matrix covered normalization, both EEX audits, governed
   forward history, LT/CT import boundaries, LT package contract, cascading,
   monthly audit, constraints, integration, priors and solver:

   Result: `209 passed, 4 skipped in 12.42s`.

5. The public EEX holiday PDF first failed to download under Windows Schannel
   with `CRYPT_E_NO_REVOCATION_CHECK`. It was then fetched from the exact
   official HTTPS URL with curl's public-download `--ssl-no-revoke` option,
   inspected independently through the web reader and bound by the exact hash
   above. No credential or private endpoint was involved.

6. D212 materialization was run twice after final quarantine reconciliation.
   Both runs returned `f84ac6c9...14f433e` and byte-checked all existing
   artifacts.

## Authority and invariants

- The CH monthly solver remains sole monthly-level authority.
- DAY/WEEK/WEEKEND may support a future solver-neutral, zero-monthly-mean shape
  feature contract; they cannot rewrite monthly means or bypass the solver.
- The local history does not prove signed provider-time availability. No
  rolling-origin, model selection, candidate, promotion or production claim is
  authorized.
- Missing quotations remain missing. Standard EEX closures and normalization
  quarantine are reconciled before identifying candidate gaps.
- Governed ENTSO-E Databricks evidence and a new independently frozen future
  holdout remain mandatory. Local ENTSO-E remains schema/tooling only.
- AFRY and OMPEX remain benchmark-only; T057 remains sealed; production is
  strict `NO_GO`; LT/CT separation is unchanged.

## Next safe batch

Stay offline. Define and test a solver-neutral short-tenor feature contract
that can express DAY-versus-WEEK and WEEKEND-versus-WEEK shape residuals with
hard zero monthly mean and no price fitting or selection. Keep it dormant until
signed PIT vintages and governed ENTSO-E evidence permit honest rolling-origin
evaluation. Separately send the six unexplained gap dates to the data engineer
for lineage confirmation.
