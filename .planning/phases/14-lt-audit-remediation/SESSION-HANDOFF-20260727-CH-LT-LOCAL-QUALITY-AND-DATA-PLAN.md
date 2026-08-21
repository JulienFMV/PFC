# Session handoff - CH LT local quality and prospective data plan - 2026-07-27

## Scope and outcome

This session resumed after packaging/runtime closure and addressed the next
scientific boundary without promoting anything:

1. formally revoked the obsolete pathwise-fixed monthly scenario
   preregistration;
2. independently recomputed what the current local run15 can and cannot prove;
3. defined a fail-closed prospective CH acquisition plan that distinguishes
   current hourly day-ahead truth, continuous intraday 15-minute data and the
   planned explicit 15-minute auction;
4. reran the integrated acquisition, compute, dependence and T057 controls.

Final status remains `NO_GO`. No holdout or T057 outcome was newly consumed,
and no snapshot, candidate or production object was promoted.

## Swiss market boundary frozen in this session

- Current CH day-ahead truth is native hourly (`3600s`).
- EPEX CH Continuous 15-minute data is a separate intraday product. It may
  become a labelled auxiliary diagnostic after licensed, point-in-time and
  external admission, but never day-ahead absolute-level truth.
- The Swiss explicit day-ahead/intraday 15-minute auction transition is only
  recorded as planned for Q3 2026. Canonical status is
  `PLANNED_NOT_CONFIRMED_UNSUPPORTED_15MIN` until exact official go-live bytes,
  first delivery date, product/session/settlement identity, licensed schema and
  sufficient post-go-live history are captured and admitted.
- Hourly duplication is never native quarter-hour truth. Neighbor data is
  never direct CH truth. The monthly solver remains level authority and OMPEX
  remains benchmark-only.

Primary official references consulted for planning:

- Energy Charts API: `https://api.energy-charts.info/`
- EPEX Spot market results:
  `https://www.epexspot.com/en/market-results?market_area=CH&modality=Continuous&product=15&data_mode=table`
- Swissgrid Balancing Roadmap 2026-2030:
  `https://www.swissgrid.ch/content/dam/swissgrid/about-us/newsroom/publications/balancing-roadmap-en.pdf`

Their exact web bytes were not captured, signed or sealed. They are planning
references, not admission evidence.

## Exact files added

- `.planning/phases/14-lt-audit-remediation/CH-LT-PIT-PREREGISTRATION-SUPERSESSION-20260727.json`
- `.planning/phases/14-lt-audit-remediation/CH-LT-PROSPECTIVE-DATA-ACQUISITION-PLAN-20260727.json`
- `pfc_shaping/validation/ch_lt_preregistration_supersession.py`
- `pfc_shaping/validation/ch_lt_local_candidate_quality.py`
- `pfc_shaping/validation/ch_lt_prospective_acquisition_plan.py`
- `scripts/verify_ch_lt_preregistration_supersession.py`
- `scripts/build_ch_lt_local_candidate_quality_report.py`
- `scripts/audit_ch_lt_prospective_acquisition_plan.py`
- `tests/test_ch_lt_preregistration_supersession.py`
- `tests/test_ch_lt_local_candidate_quality.py`
- `tests/test_ch_lt_prospective_acquisition_plan.py`
- `output/phase14/ch_lt_local_candidate_quality_20260727_v3/quality.json`
- `output/phase14/ch_lt_local_candidate_quality_20260727_v3/QUALITY.md`

The earlier non-versioned report and the directory-v2 report that still
declared schema v1 are superseded by directory v3. V3 declares
`ch_lt_local_candidate_quality.v2`. All remain local untracked diagnostics and
none is authority.

## Exact files updated

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/HANDOFF.md`

No `pfc_shaping/ct/*`, Power BI or protected data file was touched by this
slice.

## Canonical identities and hashes

- preregistration supersession registry:
  `6bfa49831c91693bd355beace39a6fdd2a74cfa1d25517292634a71dbcfe2282`
- acquisition plan ID:
  `a0faf97bc10b4d51e23b21c200a73cdd95f92ffa9f8ea7976bc4a10859a8688f`
- acquisition plan document SHA-256:
  `99361b488e5a031f66c64866c7d69f21a7d8d031e30cf37b097a452bb5627c5a`
- local quality JSON:
  `bbba2480cdc8ae7d08c6722f7d4a37680da40055f08aaacf2a83a48b6dab9d55`
- local quality Markdown:
  `9a4d75a6a312b0668f1eaec784d013b6da7f2fb2a55259eb0e02ee8121b3d3c8`
- bound run15 completion manifest:
  `7ff70c65f49d53f0f1ceba29dacf71826df45547bf3c334fd327b3d03573540d`

Implementation SHA-256 values:

- `ch_lt_preregistration_supersession.py`:
  `4eefe89cac578cef615e104c06ce4d1742b3d75854eb914e0df4c549faf6db6e`
- `ch_lt_local_candidate_quality.py`:
  `39367ab9281fbb335042705cebc427d2f05ba31433eac77ff8c6d2b926595773`
- `ch_lt_prospective_acquisition_plan.py`:
  `ab4acd5a95b25f83214c944698de54e37239d12d63eecf8f823a7f22a7e5e4b5`

## Local quality result

Status: `LOCAL_STRUCTURAL_DIAGNOSTIC_ONLY_NO_GO`.

- 155,616 rows per scenario, exact 15-minute UTC grid from
  `2026-07-24T22:00:00Z` to `2030-12-31T21:45:00Z`.
- Maximum scenario monthly-mean residual versus solver:
  `5.5010218602546956e-11 EUR/MWh`.
- Maximum solver constraint residual: `2.842170943040401e-14`.
- Central curve mean/min/max: `85.48362297717911`,
  `4.468397139799421`, `237.39736787425164 EUR/MWh`.
- DE-LU pilot candidate MAE/RMSE: `5.681786842737326` /
  `11.318385916301677`; incumbent: `5.783128666386267` /
  `11.533733187339516`.
- Relative MAE gain is 1.752%, below the draft 2% target; outcomes are 6 wins,
  6 losses and 4 ties; latest-fold noninferiority is false.
- P10/P90 contain zero observations. Slow/central/fast are structural paths,
  not calibrated probabilities. Their displayed weights are explicitly named
  `structural_display_weights_not_probabilities`.
- Corrected T057 supplies one historical fold and one 336-hour future episode;
  robustness and materiality are not established.
- The run used `TEST_FIXTURE` forwards and a mixed-authority DE-LU proxy with
  61.46% fallback cells. Direct governed CH 15-minute validation is absent.
- Exact run code bytes were not captured. The observed runner SHA-256
  `8e4f91824d73b89d58faa44757718c8a3effc4b03a811dd0b8c5d1562c3dde98`
  differs from current checkout SHA-256
  `08f270bc3587d14c2ff7fdd347f342fa9edc3ef4af1cf6158e3ce2a7f6dbf316`.
- Rolling MAE/RMSE are recomputed from the hash-bound fold rows and checked
  against the bound summary. Intraday fallback-cell counts remain copied from
  the bound run15 quality summary and are explicitly labelled as not
  independently recomputed.
- `t057_consumed=false` means no outcome was opened or reused by this slice;
  the report separately states `t057_supersession_metadata_read=true` and
  `t057_reused_for_selection=false`.

Therefore run15 is usable only for local visual inspection, structural monthly
level diagnostics and smoke tests. It is not usable for model selection,
scientific CH 15-minute claims, probabilistic valuation or promotion.

## Commands and results

Every command was preceded by the exact canonical-workspace guard required by
the user and ran from `C:\Users\jbattaglia\PFC_LT`.

Focused implementation tests during construction:

```powershell
python -m pytest tests\test_ch_lt_preregistration_supersession.py tests\test_ch_lt_pit_preregistration.py tests\test_ch_lt_compute_runtime.py -q -p no:cacheprovider
```

Result: `36 passed in 0.33s`.

Local quality tests:

```powershell
python -m pytest tests\test_ch_lt_local_candidate_quality.py -q -p no:cacheprovider
```

Result: `4 passed in 63.19s`.

Exact v3 local-report generation:

```powershell
New-Item -ItemType Directory -Path output\phase14\ch_lt_local_candidate_quality_20260727_v3 -ErrorAction Stop
python scripts\build_ch_lt_local_candidate_quality_report.py --completion-manifest output\phase14\local_pfc_prospective_intraday_20260724_run15\completion_manifest.json --expected-completion-sha256 7ff70c65f49d53f0f1ceba29dacf71826df45547bf3c334fd327b3d03573540d --output-json output\phase14\ch_lt_local_candidate_quality_20260727_v3\quality.json --output-markdown output\phase14\ch_lt_local_candidate_quality_20260727_v3\QUALITY.md
```

Result: exit `0`, status `LOCAL_STRUCTURAL_DIAGNOSTIC_ONLY_NO_GO`.
This builder is deliberately checkout-only because it imports the local T057
supersession script; it is not a governed-wheel service entrypoint.

Corrected prospective-plan and supersession tests:

```powershell
python -m pytest tests\test_ch_lt_prospective_acquisition_plan.py tests\test_ch_lt_preregistration_supersession.py -q -p no:cacheprovider
```

Result: `15 passed in 0.19s`.

Direct fail-closed audits:

```powershell
python scripts\verify_ch_lt_preregistration_supersession.py --registry .planning\phases\14-lt-audit-remediation\CH-LT-PIT-PREREGISTRATION-SUPERSESSION-20260727.json --expected-registry-sha256 6bfa49831c91693bd355beace39a6fdd2a74cfa1d25517292634a71dbcfe2282
python scripts\audit_ch_lt_prospective_acquisition_plan.py --plan .planning\phases\14-lt-audit-remediation\CH-LT-PROSPECTIVE-DATA-ACQUISITION-PLAN-20260727.json --expected-plan-sha256 99361b488e5a031f66c64866c7d69f21a7d8d031e30cf37b097a452bb5627c5a
```

Results: supersession
`ACTIVE_FAIL_CLOSED_V1_SUPERSEDED_NO_EXECUTABLE_SUCCESSOR`; acquisition plan
`STRUCTURE_VALID_BLOCKED_NOT_EXECUTABLE`; all authority flags false.

Final integrated matrix:

```powershell
python -m pytest tests\test_ch_lt_preregistration_supersession.py tests\test_ch_lt_local_candidate_quality.py tests\test_ch_lt_prospective_acquisition_plan.py tests\test_ch_lt_pit_preregistration.py tests\test_ch_lt_compute_runtime.py tests\test_governed_lt_acquisition.py tests\test_capture_public_energy_charts_lt_script.py tests\test_dependence_power_supersession.py tests\test_verify_t057_evidence_supersession_script.py -q -p no:cacheprovider --basetemp build\pytest-local-quality-data-plan-final-20260727-a
```

Initial pre-roast result: `130 passed, 1 skipped, 1 warning in 13.29s`. The warning is the known
timezone-to-Period conversion warning in the explicitly non-admitted legacy
Energy Charts loader.

Final post-roast matrix with the identity, semantic, budget and schema fixes:
`136 passed, 1 skipped, 1 warning in 47.89s`. The warning is unchanged.

Targeted Ruff over all new modules, scripts and tests: `All checks passed!`.

## Independent read-only roasts and fixes

Security initially demonstrated two P1 findings:

1. the supersession verifier accepted a caller-selected registry hash;
2. the acquisition validator accepted a caller-rehashed shadow plan whose
   nested allowed/forbidden uses or T057/pilot action could be changed.

Both are corrected. The supersession registry now requires canonical path and
the frozen `REGISTRY_SHA256`. The prospective plan now requires canonical path,
frozen physical `DOCUMENT_SHA256`, captured bytes matching the parsed mapping
and frozen `PLAN_ID`; reserialized, rehashed, unknown-key, changed-use,
changed-action and changed-observation countertests fail closed.

Post-fix focused result: `20 passed in 0.17s`; combined local quality plus both
identity suites: `24 passed in 34.98s`.

IT/Operations found no demonstrated P0/P1. Its P2 observations are retained:
the JSON/Markdown pair is not transactionally published as one directory, the
output namespace is local rather than governed, and the future acquisition
runbook still lacks frozen external values. The cumulative bound-output budget
is now 128 MiB, the exact generation command and checkout-only boundary are
documented, and CI/operators must interpret exit `0` as structural validation
only by checking `status` and every authority flag.

Quant/Data confirmed the canonical hourly/continuous/future-explicit product
separation but reproduced the same shadow-plan P1, now fixed. Its P2 semantic
findings were addressed by renaming display weights, recomputing fold metrics
and separating T057 metadata read from T057 selection reuse. The prospective
sample-size/power/multiplicity design remains intentionally blocked and must be
frozen before admission.

A final IT/Operations re-roast then demonstrated one schema-identity P1: the
directory-v2 report had changed shape while still declaring schema v1. The
contract is now `ch_lt_local_candidate_quality.v2`, the schema identity and
cumulative-budget rejection have dedicated tests, and directory v3 was
generated without overwriting either earlier local artifact. Focused result:
`5 passed in 52.87s`.

Final independent verdicts: Security `no demonstrated P0/P1`, IT/Operations
`no demonstrated P0/P1`, Quant/Data `no demonstrated P0/P1` in this explicitly
local, non-authoritative slice.

## Decisions recorded

- `D-20260727-164`: revoke pathwise-fixed monthly scenarios and preserve solver
  authority at ensemble-expectation level.
- `D-20260727-165`: classify run15 as structural local diagnostic only.
- `D-20260727-166`: current CH day-ahead is hourly; continuous intraday 15 min
  is separate; future explicit 15 min stays unsupported until proven live.

## Remaining blockers and next work

1. Capture fresh CH hourly day-ahead source bytes point-in-time through the
   existing Energy Charts route, then obtain independent trusted-time,
   signature, immutable retention and external-CAS receipts.
2. Obtain fresh licensed CH EEX forward bytes with frozen product and quote
   convention for monthly solver repricing.
3. Monitor and archive exact Swissgrid/EPEX official 15-minute go-live notice
   bytes. Do not infer go-live from the roadmap date alone.
4. After confirmed go-live, obtain the exact licensed explicit auction feed and
   accumulate sufficient post-go-live history. Continuous 15-minute data may be
   acquired separately but remains product-labelled diagnostic evidence.
5. Build a new hash-bound candidate from exact code/input/runtime bytes. Do not
   reuse the DE-LU pilot or T057 for selection.
6. Preregister unique rolling origins, a future holdout ledger, calibrated
   probabilistic scenarios and FMV economic materiality before opening truth.
7. Repeat independent Security, IT/Operations and Quant/Data roasts on the new
   candidate and external admission chain.

Production remains strict `NO_GO`. Do not commit before a final scope audit.
