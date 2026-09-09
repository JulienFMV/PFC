# D310 hourly maturity experiment — completed, no adoption

## Outcome and invariants

D310 completed: bounded origin/delivery residual maturity experiment rejected.
D304 retained; no model/month or model/horizon switching, no further tuning.
Calendar-only residual shape MAE/RMSE21.182405/29.983503 versus D304
20.345234/30.180010; maturity21.453508/30.241637. Maturity MAE worsens5.447341%
versus D304 and1.279849% versus calendar-only; RMSE worsens0.860923% versus
calendar-only. Two origin wins each;23/25 regime regressions and5 per-origin
shape/ramp regressions each. Both seam screens pass, both combined screens fail.
Monthly level errors identical47.260956/55.960385. All six authorities false.

User explicitly authorizes GPU if useful. This permission persists; CPU four
threads chosen for54/108-dimensional ridge,14 fits and14 new assemblies.
No GPU compute, Warehouse, AFRY, T057, protected desk data or CT work. Existing
assembler and EEX projection unchanged; monthly solver remains level authority.
No production promotion, registered origin or independent holdout claim.

## Frozen design and data

Protocol: docs/model/LT-MATURITY-LOCAL-EXPERIMENT.md.
Plan frozen 2026-09-08T08:51:01.122254+00:00, SHA256 15a03996c361602220a1c7714afad1e018d0a38432a2ffc810711cc65c8c2e6e.
Seven retained outer origins2021–2026/current; tuning2021/2022, exposed
assessment2023–2026, descriptive current. Latest-observed revised local data,
not historical point-in-time. CH observed hourly repeated four times in QH.
Current valuation September7,2026 08:00UTC, not a new September8 snapshot.

Quarterly inner origins twelve hours before Swiss quarter start;12 complete
months minimum at inner origin. Pair delivery labels must be closed before
outer origin. Fixed leads1–36, inverse pair count gives total weight1 per unique
delivery hour. Fixed harmonic1/2/3 sine/cosine by4FMV seasons and5day types,
monthly-centered54 columns; maturity adds columns multiplied by min(lead,36)/36.
Weighted mean squared residual +0.1 squared coefficient norm, no intercept,
standardization, hyperparameter search or monthly choice. D304 at each inner
origin reconstructed using strictly earlier complete-month signed targets.

Pairs by outer origin:10995/56286/136617/240993/346401/451521/530421.
Distinct hours:5857/14617/23377/32137/40921/49681/56256; inner origins3/7/11/15/19/23/26.
Maximum learned lead8/20/32/36/36/36/36 months.2021/2023 have2929 delivery hours
outside training lead support; current full horizon has28513 beyond36 months.
Fixed capped formula emits descriptive curves there, not qualified maturity.

## Tests and independent checks

10 new tests pass;48 focused pass. Full matrix282pass/5skip/2 known Phase5
failures. Exact two failure messages match D309 tests-matrix-v1.xml;
no baseline fixtures rewritten. No product source edited.
Independent numerical replay reconstructs calendar means, residuals, weighted
normal matrices with independent feature implementation and eigen solve;
checks21 curves, 3288 metrics and 22170 seam events.
Maximum solver monthly residual 2.39879227593e-11.
Current product gates all80PASS/9QUOTE_CONFLICT/0CRITICAL. Diagnostics separate
shape/level/full error before/after projection; years/seasons/horizons, negative,
near-zero, raw and signed tails, ramps, seams, revision decomposition retained.
Scales exactly match frozen D309 pre-origin scales. No new post-result gate.

First independent review failed on empty-overlap Parquet index.freq metadata:
None versus Hour. Frozen source remains unchanged. review_v2.py wraps only
pd.testing.assert_frame_equal(check_freq=False); numeric values, actual timestamp
index and timezone checks remain. review-v1/failure.json preserves failure.
Second review passes. additional_review.py verifies all expected quarterly pairs,
label coverage/cutoffs/weights,6CSV values/timestamps and733D301/746D308/294D309
frozen files; shared current handoff/decision log excluded from old closures.

## Exact changed files and artifacts

New source/test/protocol:
- scripts/lt_maturity_experiment.py
- scripts/run_lt_maturity.py
- scripts/verify_lt_maturity.py
- tests/test_lt_maturity.py
- docs/model/LT-MATURITY-LOCAL-EXPERIMENT.md
New handoff: .planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260908-HOURLY-MATURITY.md
Updated only shared governance: .planning/HANDOFF.md and Phase14/DECISION-LOG.md.
Prior proposal, D304–D309 source snapshots/outputs and other dirty user work preserved.

Task root build/lt-maturity-20260908:
- run-v1/: plan/source-snapshot,7pairs/supports,21curves,14fits/receipts,
  diagnostic-metrics.csv,events.parquet,seam-metrics.csv,revisions.csv,
  delivery-support.csv,complete.json,manifest.json.
- review-v1/failure.json; review-v2/: comparison.csv,regime-gates.csv,
  origin-gates.csv,seam-gates.csv,decision.json,verification.json,manifest.json.
- additional-review-v1/: pair-coverage.csv,exports.json,verification.json,manifest.json.
- revision-regimes.csv,external-benchmark-follow-up.json,RAPPORT-MATURITE.md,
  report-evidence.json,tests-initial.xml,tests-focused.xml,tests-matrix.xml.
- additional_review.py,review_v2.py,finish_report.py,close_session.py,
  validate_closure.py,closure.json,closure-validation.json; repo-local runtime/pytest dirs.

Current candidate exports under run-v1/current/residual-calendar/ and
residual-maturity/: pfc-fmv-ch-1h.csv(28513rows) and pfc-fmv-ch-15min.csv(114052rows).
Unchanged D304 control under signed-equal/. All six exports verified; CSV period
Oct2026–Dec2029. Full curve.parquet219268QH rows throughDec2032. Research candidates
only, no trading/adoption authority. Hashes in additional-review-v1/exports.json.

Run manifest SHA256 451b3a8fdb07744852a3b97d96351ed5e4a5ced897e54845ecc6e947cec07bd9.
Review manifest SHA256 809dd5659aafd136f8c44097f853bb74c5e1600ddccdd19330a0b54e34e3ea92.
Report SHA256 d93f7d3ca12c7967a0bec1f2964c999f7f7adf6b413902c6a45516328b05f44b.
Source/input pins:106/1796.
Final closure binds all final task evidence plus source/handoff/decision files.

## Commands and execution

Every shell call verified cwd and Git top-level exactly C:\Users\jbattaglia\PFC_LT;
workdir always canonical. TEMP/TMP/APPDATA/LOCALAPPDATA/MPLCONFIGDIR/XDG_CACHE_HOME/
NUMBA_CACHE_DIR/JOBLIB_TEMP_FOLDER/PYTHONUSERBASE/PIP_CACHE_DIR were created under
build/lt-maturity-20260908/runtime/<name>. OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
MKL_NUM_THREADS=4; PYTHONDONTWRITEBYTECODE=1; CUDA_VISIBLE_DEVICES=-1 for this run.
No detached/external process or active process remains after closure.

All Python commands use & build/conda-runtime-v41-model-source/python.exe -B:
1. -m pytest tests/test_lt_maturity.py -q -o cache_dir=build/lt-maturity-20260908/pytest-cache --basetemp=build/lt-maturity-20260908/pytest-initial --junitxml=build/lt-maturity-20260908/tests-initial.xml :10pass.
2. -m pytest tests/test_lt_maturity.py tests/test_hourly_stability.py tests/test_hourly_revisions.py -q -o cache_dir=build/lt-maturity-20260908/pytest-cache --basetemp=build/lt-maturity-20260908/pytest-focused --junitxml=build/lt-maturity-20260908/tests-focused.xml :48pass.
3. -m scripts.run_lt_maturity --output build/lt-maturity-20260908/run-v1 :COMPLETE.
4. -m pytest tests/test_lt_maturity.py tests/test_hourly_revisions.py tests/test_hourly_stability.py tests/test_hourly_recency.py tests/test_price_conditioned_intraday.py tests/test_signed_hourly_assembly.py tests/test_signed_composition.py tests/test_signed_benchmark.py tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py tests/test_lt_evaluation_curve_assembly.py tests/test_lt_local_benchmark.py tests/test_assembler_profile_type.py tests/test_phase05_negative_prices.py tests/test_lt_structural_readiness.py tests/test_intraday_persistence.py -q -o cache_dir=build/lt-maturity-20260908/pytest-cache --basetemp=build/lt-maturity-20260908/pytest-matrix --junitxml=build/lt-maturity-20260908/tests-matrix.xml :282pass/5skip/2known.
5. -m scripts.verify_lt_maturity --run build/lt-maturity-20260908/run-v1 --output build/lt-maturity-20260908/review-v1 :metadata failure above.
6. build/lt-maturity-20260908/additional_review.py :VERIFIED.
7. build/lt-maturity-20260908/review_v2.py :VERIFIED.
8. build/lt-maturity-20260908/finish_report.py :report and revision/source follow-up written.
9. build/lt-maturity-20260908/close_session.py :D310/governance/closure written.
10. build/lt-maturity-20260908/validate_closure.py :final validation; git diff --check.
Read-only exploration: relevant prior handoffs/scripts, source target start/end,
aggregate saved metrics, rg on compare_hfc.py. No external source queried.

## User-provided next benchmark sources and missing contracts

User reports a long-term price curve in Databricks under lseg; exact catalog/
schema/table, country, valuation/availability timestamps and horizon not yet
identified. User reports daily OMPEX PFC files at10:18 in
H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC\_OMPEX\_15min.
Availability time zone is unverified. These hints are preserved in
external-benchmark-follow-up.json; no source access or governed qualification
has occurred here, no Warehouse started. Do not work from the legacy H: checkout.

Next expert lot: qualify these external curves and matched actual forecast
vintages, calendar/DST, currency, country, delivery horizon and publication
cutoffs. Pin read-only source evidence into canonical build before use. Compare
monthly levels separately from zero-mean hourly form. Existing compare_hfc.py
uses latest mtime and averages duplicate timestamps; do not use it unchanged as
proof of vintage or DST correctness. External benchmark never overrides solver
levels. Retain D304; no more tuning of exposed origins. Complete D309 future
holdout custody/truth-finalization requirements when independently available;
same-value rerun is not a new vintage. AFRY/T057 stay excluded.
