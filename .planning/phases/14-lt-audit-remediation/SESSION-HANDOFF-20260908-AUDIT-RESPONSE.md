# D317 — Response to Claude audit, 2026-09-08

Status: local implementation and verification complete; publication/remote CI
recorded separately in the local closure receipts. Canonical workspace only.

Audit `719a18975229d5d13477c78a45c1aff6e3eebcfe` was fetched and its two
document files cherry-picked as `c2680d7f7a`. The audited code remains
`dee652bc919d06345f71304d1f1eaacf0edf7bc7`; predecessor documentation is
`e8313f771c03641fc554a0ef8128a5c67e93591e`. Do not edit Claude's report.

Objective: fix demonstrated collection/causality defects before pilot day 2,
preserve D304, review all 33 findings and make the next model experiment
conditional on explicit evidence. No model fit, SQL, Warehouse, GPU, AFRY or
T057 use. All six authorities remain false. Pilot stays 1/20 observations.

Local evidence root: `build/lt-audit-response-20260908/`.
`before.json` binds 1,568 existing workspace files before this work. The
D316 closure hash is
`91f0e3f7ad2df1b8930d3c3d261d84808abaaf86d339028cff53ba947a657f7e`.
Six richer business/governance files already differed from Git; never stage
them wholesale. Stage only the new public governance text over public HEAD.

Implemented: EEX accepted-history protection, exact merge schemas and day-2
history binding; quotation-date metadata for new v2 daily records with v1
read compatibility; actual stop-attempt and early/secondary-failure receipts;
full solver quote diagnostics with unsigned tolerance/source explanation;
strict timezone/epoch/grid validation; rejection of unproven document-created
PIT; causal tie ordering; downstream PIT mode and spot cadence checks;
publisher test dependencies and optional-local-fixture test portability.

Tests: initial new regressions 22 failed/68 deselected, followed by 95 passed,
177 passed, 49 passed. First broad matrix: 715 passed/1 failed/5 skipped.
Failure was the new Databricks provenance being sent to the Energy Charts
quality validator. Shared Databricks validation now covers replay and quality;
focused integration run: 70 passed. Final run: 716 passed, 0 failed, 5 skipped,
1 existing warning, 117.42 seconds. It is recorded separately as
`tests-matrix-v2*`; do not overwrite the failed-run evidence.

Independent verification `independent-v1/verification.json` executes archived
real functions versus current functions on five adversarial probes. All five
old behaviours reproduced and current versions reject them. Clean PIT values
are unchanged. Frozen real day-1 solver replay reproduces all 75 monthly
levels exactly (max error 0.0); all D304 recipe bindings are unchanged.
Important horizon correction: that extended export covers October 2026 to
December 2032, not only the core 2026–2029 comparison window.

Open: signed EEX conflict policy; real PRD atomic/block PIT qualification;
physical-series resolution weighting; F-01/F-05/F-06/F-10 safeguards before
any new model benchmark; future holdout; remote CI result and existing lint
debt. Local runtime lacks both ruff and pip; attempted module calls failed
without installing anything. Do not claim the repo-wide lint debt is fixed.

## Exact changed public files

- `.gitattributes` (follow-up from actual Windows CI: preserve operations JSON LF bytes)
- `.github/workflows/publisher-runtime-v6.yml`
- `README.md`
- `pfc_shaping/data/databricks_lt_materialization.py`
- `pfc_shaping/data/databricks_lt_snapshot.py`
- `pfc_shaping/data/lt_input_replay.py`
- `pfc_shaping/data/lt_input_sources.py`
- `pfc_shaping/validation/lt_benchmark_snapshots.py`
- `scripts/collect_lt_benchmark_day.py`
- `scripts/run_lt_benchmark_day.py`
- `tests/test_collect_lt_benchmark_day.py`
- `tests/test_databricks_lt_materialization.py`
- `tests/test_databricks_lt_snapshot.py`
- `tests/test_lt_benchmark_day.py`
- `tests/test_lt_benchmark_snapshots.py`
- `tests/test_lt_source_acquisition_outage_plan.py`
- `docs/data/DATABRICKS-LT-MATERIALIZATION.md`
- `docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md`
- `docs/data/CH-STRUCTURAL-EVENT-REGISTRY-PROPOSAL.md` (horizon erratum/link only)
- `docs/data/CH-STRUCTURAL-EVENT-WATCHLIST-20260908-V2.json` (new)
- `docs/model/PFC-CH-AUDIT-RESPONSE-20260908.md` (new)
- this handoff (new)
- `.planning/HANDOFF.md` and Phase 14 `DECISION-LOG.md` (public D317 text only)

The two Claude-authored documents were imported in the preceding separate
commit; their bytes are unchanged. Four private business notes are unchanged;
the two shared governance notes receive the same public D317 addition while
keeping their richer local prior content.

## Commands and configuration

Before each shell invocation, verify both cwd and Git top-level equal
`C:\Users\jbattaglia\PFC_LT`. Interpreter:
`build/conda-runtime-v41-model-source/python.exe -B`.
Set TEMP, TMP, APPDATA, LOCALAPPDATA, MPLCONFIGDIR, XDG_CACHE_HOME,
NUMBA_CACHE_DIR, JOBLIB_TEMP_FOLDER, PYTHONUSERBASE and PIP_CACHE_DIR to
separate subdirectories of `build/lt-audit-response-20260908/runtime/`.
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONIOENCODING=utf-8`,
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4`,
`CUDA_VISIBLE_DEVICES=-1`; Git terminal/credential interaction disabled.

Executed verification modules under that interpreter:

- `build/lt-audit-response-20260908/independent_replay.py`
- `build/lt-audit-response-20260908/run_checks.py` (first integration failure)
- `build/lt-audit-response-20260908/run_checks_v2.py` (final matrix pass)
- `build/lt-audit-response-20260908/build_watchlist.py`
- `build/lt-audit-response-20260908/prepare_governance.py`
- `build/lt-audit-response-20260908/review_and_stage.py` (preservation/public review)

The final exact pytest argv and its 38 paths are stored in
`tests-matrix-v2-command.json`; basetemp is `pytest-matrix-v2`, cache is
`pytest-cache`, both under this task root. JUnit receipts retain earlier
regression runs too. No source values were copied into the public report.
Ruff and pip module attempts failed because neither exists in this runtime;
no global installation or workstation executable was launched. A PowerShell
brace-list parse error and an initial console encoding error did not mutate
project inputs; corrected commands completed successfully.

Eight documentary events now exist in V2. Its previous-source SHA-256 is
`8de0e436a2177de4ecc9fe1202ccff4d311271bbf1bcc78e9123bedc4ace10de`;
the original V1 is unchanged. Official OFEN/JAO/ENSI pages were consulted,
not archived as publisher byte evidence. Effect dates, price coefficients,
commercial MW and probabilities remain unadmitted/unknown as appropriate.

Publication sequence: explicit public paths, public-stage variants of the
two shared notes, `git diff --cached --check`, normal commit, normal push to
`origin fix/lt-audit-remediation`, then verify the remote head. Full hashes,
staged file manifest, exact Git outputs and CI observation are stored in
`public-review.json`, `publication.json` and `closure.json` below the task root.
Remote Actions is an independent check, not implied by local success.
No blanket merge, model promotion, scheduler or automatic day-2 capture.

## Remote CI follow-up

Implementation commit `c5b80a80f38905788fa93c2103b58b7eeb8874cd` pushed and
remote head verified. `lt-model` run34236508592 passed. Publisher run34236508625
passed dependency installation and static checks, then failed2/29tests because
Git's Windows CRLF conversion changed the hash-bound operations JSON. This
was hidden behind the earlier pandas import failure. It is a checkout issue,
not a reason to relax the byte guard.

Reproduced using `check_checkout_bytes.py before` with real
`git -c core.autocrlf=true checkout-index` into a fresh local build directory.
The checkout SHA was
`a60e335701bfe92f1e9db087f260086d988c962595ddc4c41521f154c86ff883`,
while the required Git blob SHA is
`8d1af62295835df056cedb7faab8040ddbf0fdcfdeaba99f5288f066d33bf433`.
Added only `deploy/publisher/operations-contract.json text eol=lf` to
`.gitattributes`; `check_checkout_bytes.py after` verifies exact checkout bytes
and acceptance by the existing strict contract. The contract JSON, its hash
constant and publisher implementation are unchanged. See `checkout-before/`
and `checkout-after/` receipts plus the follow-up publication and CI receipts.

Additional preservation verification: `verify_prior_artifacts.py` confirms
1,063 bound build files across D304, D305, D306, D307 and D312 closures, and
the original one-record pilot registry still validates as v1. No new day.
