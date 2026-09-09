# Session Handoff - T057 Quality and Reproducible Local CH PFC

> **SUPERSEDED / REVOKED ON 2026-07-24.** The historical `12/12` claim below
> counted twelve identical copies of one cutoff. Do not use this handoff or
> any generated `t057_local_quality_report_20260724*.json` or
> `t057_local_quality_report_20260724*.md` as accepted quality evidence. All
> six report paths (JSON and Markdown v1/v2/v3) are explicitly superseded in
> the registry below.
> The authoritative correction is
> `SESSION-HANDOFF-20260724-T057-FOLD-INTEGRITY-RECLOSURE.md` and the append-only
> hash-bound sidecar is v2 at
> `output/phase14/t057_fold_integrity_correction_20260724_run2/t057_fold_integrity_correction_v2.json`
> (SHA-256 `2cc0b67a509fe79baf4136da65e5eec3cc424a8f1d8739c300357a26b282e1c6`).
> The v1 sidecar is itself superseded. The discoverable fail-closed authority is
> `.planning/phases/14-lt-audit-remediation/T057-EVIDENCE-SUPERSESSION-REGISTRY.json`.

Date: 2026-07-24  
Repository: `C:\Users\jbattaglia\PFC_LT`  
Branch: `fix/lt-audit-remediation`  
Starting HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Production status: strict `NO_GO`  
Commit/promotion performed: none

## Scope and invariants

Every command used the exact canonical-root guard. The old `H:` repository was
never used. The dirty worktree was preserved; no reset, clean or restore was
performed. `data/eex_forwards_history.parquet` was read but not written or
staged. No `pfc_shaping/ct/*` or Power BI file was touched. The monthly solver
remains sole level authority and OMPEX was not used as a model input.

The immutable T057 directory
`output/phase14/t057_locked_t056_future_holdout/energy_charts_locked_runner_20260724`
was not rerun, deleted, overwritten or extended.

## Locked T057 quality report

The reporter now:

- checks lexical absolute paths before resolution and rejects links/reparse
  points;
- uses descriptor-stable mono-link reads;
- requires caller-held `--expected-run-summary-sha256`;
- recomputes positive improvement arithmetic with absolute tolerance `1e-12`;
- requires all rolling folds and all registered bucket folds to be positive and
  complete before stating the positive result.

Caller-held run-summary SHA-256:
`7b1f8613ffdf8c5771c6d493299fbeec5ac8fc15d136d3ed428282e4e081ffc7`.

Command shape:

```powershell
python -m scripts.build_local_pfc_quality_report `
  --run-summary C:\Users\jbattaglia\PFC_LT\output\phase14\t057_locked_t056_future_holdout\energy_charts_locked_runner_20260724\energy_charts_locked_holdout_run_summary.json `
  --expected-run-summary-sha256 7b1f8613ffdf8c5771c6d493299fbeec5ac8fc15d136d3ed428282e4e081ffc7 `
  --repo-root C:\Users\jbattaglia\PFC_LT `
  --output-json C:\Users\jbattaglia\PFC_LT\output\phase14\t057_local_quality_report_20260724_v3.json `
  --output-markdown C:\Users\jbattaglia\PFC_LT\output\phase14\t057_local_quality_report_20260724_v3.md
```

Artifacts:

- JSON SHA-256
  `d16787c8d28d4c0ddd0cfd346cda786b09385684c66a5663454af584fcfcd389`;
- Markdown SHA-256
  `7fd80eedb98a206faf555ab465a956105b2824e9f37fac3aabfd67ccfe3917a7`;
- UTF-8 reads pass with no replacement character.

Superseded scientific result: the future holdout remains one positive 336-hour
episode, MAE `18.2796823169` to `18.0108897194`, uplift
`0.2687925975 EUR/MWh` or `1.470445%`. The historical file contains twelve
identical rows for cutoff `2026-06-24T00:00:00Z`; its effective historical
sample size is one. The former `12/12` claim is withdrawn. Historical
robustness and economic materiality are not established.

Reporter checks: Ruff and `py_compile` pass; reporter tests `7 passed`.

## Local CH PFC runner hardening

Changes:

- the monthly solver is always enabled and a missing solver authority is a hard
  runtime error;
- every output path must remain inside the canonical repository, all output
  paths must be distinct, and any pre-existing target rejects the run before
  work;
- Parquet artifacts use flushed same-directory temporary bytes plus exclusive
  hardlink publication; JSON/Markdown evidence uses the shared durable exact
  writer;
- a completion manifest is written last and binds stable pre/post hashes for
  inventory, governance manifest, hourly and intraday EPEX, forward history and
  relevant code, plus every generated artifact and all CLI arguments;
- the summary exposes `monthly_level_authority=solver`,
  `forward_source_kind=TEST_FIXTURE`, `hard_quote_eligible=false`,
  `promotion_eligible=false`, internal null forward hash and the actual consumed
  forward file SHA-256
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- the direct legacy `build_ep2050_multi_scenario_pfc.py` CLI is disabled;
- maturity labels are `M+1..M+6`, `M+7..M+12`, `Y+2/Y+3`, then `Y+4+` beyond
  36 months;
- `f_Q` horizon flattening uses the floored hour so each four-quarter hourly
  mean remains one at numerical precision.

A two-day run without an annual solver override failed as expected on the
partial Cal-2030 delivery grid. It left intermediate diagnostic artifacts but
no completion manifest. A second two-day run with the complete 2030 solver
window passed; its completion manifest SHA-256 is
`c84ae2c56b572aadc83574b56b8247f11d530fe0e5b9f72ba972ab03fef977e8`.
Attempted reuse of that output set was rejected before overwrite.

## Full local runs and reproducibility

Both commands used:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python -m scripts.build_local_test_ch_pfc `
  --inventory data\electrification_scenarios_prod_candidate_neutralized_2030.parquet `
  --manifest .planning\phases\13-lt-electrification-scenario-shape\SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml `
  --vintage 2026-06-12 --market CH `
  --start-date '2026-07-24 22:00:00' --horizon-days 1621 `
  --epex-hourly data\epex_hourly.parquet `
  --intraday-epex pfc_shaping\data\epex_de_15min.parquet `
  --intraday-market DE --intraday-cutoff 2025-10-01T00:00:00Z `
  --require-nonflat-intraday-shape `
  --forwards data\eex_forwards_history.parquet `
  --enable-monthly-forward-curve-solver `
  --disable-cascade-trend-for-annual-only `
  --governance-report <run>\LOCAL-TEST-GOVERNANCE-GATE.md `
  --expanded-output <run>\scenario_expanded.parquet `
  --features-output <run>\hpfc_scenario_features.parquet `
  --output-dir <run> `
  --output-prefix local_pfc_smoke_20260724_intraday `
  --fan-chart-output <run>\structural_fan_chart.parquet `
  --summary <run>\LOCAL-PFC-SMOKE-SUMMARY.md `
  --completion-manifest <run>\completion_manifest.json
```

Run roots:

- `output/phase14/local_pfc_smoke_20260724_intraday_run4`, 46.09 s,
  completion SHA-256
  `99e908ea856c22b9c3901fbc577efab4c7c8fd7191bfdb37ca779fe89212e509`;
- `output/phase14/local_pfc_smoke_20260724_intraday_run5`, 59.26 s,
  completion SHA-256
  `1079cace8feb468e04956ff239a60bfab96b0a16a91b36f4d49603a06dc26817`.

Each scenario contains 155,616 ordered, unique 15-minute rows from
`2026-07-24T22:00Z` through `2030-12-31T21:45Z`. Eight model/governance
artifacts are byte-identical between runs:

- expanded scenarios
  `45b16dfac5de1d1d2cee1738044875692c88f9cf6404b3b22ee161ebc50764eb`;
- scenario features
  `86a9289d31b4ca966ab427e32259cb72e6fe1ae3233f48b5c8dd39b4b5a4824e`;
- governance report
  `9f18bcd2f01592f3bd7b4180813fd3e19beaff18d7030b724f71d2265848b719`;
- monthly manifest
  `1e5642341daf05baf4d9abe9d602b2f505672c8ff72cfa1e9060fff8fd59c861`;
- slow PFC
  `4b8b0a2c04396affef2cd22f26da2dd3a64eb18799b581f96c7f891399863ed1`;
- central PFC
  `5c97931c0c0bdc7da42085642c954071507b09440f72193fa7c37988edccb5a5`;
- fast PFC
  `e4a0744c45e0dce7b8e827b8fd676bf597a942e4eb1e985f5941e504dbae72a6`;
- structural fan
  `ef95fec1e0d3117ea7fb64c8fbc1d24e3041f8e2a0556a6c68fe0449506c9688`.

Quality checks:

- 52 complete local months, maximum solver monthly-mean error
  `5.491074261954054e-11`;
- solver max constraint residual `2.842170943040401e-14`, stationarity residual
  `1.5719360463372172e-13`, no ridge/lstsq fallback;
- maximum hourly `f_Q` mean error `2.220446049250313e-16`;
- numeric fields finite, fan ordered, no duplicate timestamps;
- profile labels include 49,728 `Y+4+` quarters;
- zero negative-price intervals and all 155,616 p10/p90 rows null;
- central mean `85.4836`, min `12.4673`, max `237.3975 EUR/MWh`; this is a
  diagnostic description, not a goodness claim.

Intraday evidence remains limited: 96/480 cells are direct fits and 384/480
are fallback cells. Direct coverage is Hiver 72, Automne 24, Printemps 0,
Ete 0. The DE-LU proxy has not been validated against CH. The strict all-season
gate correctly fails.

## Test and packaging evidence

- targeted local-runner/model tests: `14 passed`;
- LT/model/solver/reporter matrix: `183 passed, 1 skipped in 117.64s`;
- packaging/runtime/acquisition matrix:
  `158 passed, 14 skipped in 104.94s`;
- publication/external-CAS matrix:
  `165 passed, 13 skipped in 115.46s`;
- explicit captured-root-only `sys.path`: `3 passed in 0.30s`;
- `git diff --check`: pass, only expected LF/CRLF notices.

Fresh wheels:

- A:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\local-pfc-hardening-wheel-20260724-103005-a\fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- B:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\local-pfc-hardening-wheel-20260724-103005-b\fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- both: 79 members, 426,438 bytes, contract PASS, byte-identical SHA-256
  `efe917d2d1cc2a3df3aeeeeab69a7560e05cf480d7b5c84c4c2e4133e0e26d76`,
  source revision
  `60ac2c8f10e3cba6eddecb0129155e1f89a0843911e4f054c4e018cd29bdee62`,
  `promotion_eligible=false`.

The retained optimized provider-raw zipapp test already passed on the exact
runtime paths recorded in the 2026-07-23 handoff. Publisher runtime code was
not changed in this slice. Docker remains unavailable; no image claim is made.

## Read-only roasts and corrections

Security found reporter lexical-link erasure, weak stat/read/stat reads, no
caller-held anchor and unchecked positive wording. All four are corrected and
covered by negative tests. IT/Operations found sequential mutable local output,
optional solver authority, hidden fixture provenance, a non-governed legacy
entrypoint, absent post-fix replay and the inaccurate far-horizon label. These
are corrected and the replay is recorded above. Quant/Data found the 480-cell
claim misleading: the runner now reports 96 direct and 384 fallback cells and
can require named direct-fit seasons.

No roast reported a demonstrated P0. Remaining release/science findings are
real blockers, not waived findings.

## Exact changed files in this slice

- `scripts/build_local_pfc_quality_report.py`
- `tests/test_build_local_pfc_quality_report_script.py`
- `scripts/build_local_test_ch_pfc.py`
- `tests/test_build_local_test_ch_pfc_script.py`
- `scripts/build_ep2050_multi_scenario_pfc.py`
- `tests/test_build_ep2050_multi_scenario_pfc_script.py`
- `pfc_shaping/lt/model/shape_intraday.py`
- `tests/test_intraday_amplitude.py`
- `pfc_shaping/lt/model/assembler.py`
- `tests/test_assembler_profile_type.py`
- `tests/test_candidate_evidence_assembler.py` (timestamp fixture correction)
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- this handoff.

## Remaining blockers and next work

Production remains strict `NO_GO` because:

1. the forward input is still a local `TEST_FIXTURE`, not a fresh signed PIT
   production snapshot with exact EEX history/catalog authority;
2. the DE-LU intraday proxy lacks CH validation and complete seasonal direct
   coverage;
3. probabilistic p10/p90/scenario calibration and proper-score/coverage tests
   are absent;
4. one 336-hour T057 future holdout is too short for materiality and regime
   robustness;
5. independent external CAS, service ACL/freeze, HSM/KMS, Docker/CI image,
   monitoring, rollback and DR drills remain unproved.

Next scientific slice: acquire a fresh prospective point-in-time source set
through the governed external handoff, construct a new auditable CH candidate,
extend future holdout evidence, validate CH-vs-DE intraday transfer and fit
calibrated probabilistic/scenario outputs. Do not promote before independent
Security, IT/Operations and Quant/Data evidence is complete.
