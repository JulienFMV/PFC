# D311 — external PFC descriptive benchmark completed

Read AGENTS.md and .planning/HANDOFF.md, then this handoff. D310 remains closed.
Report: build/lt-external-benchmark-20260908/RAPPORT-BENCHMARK-EXTERNE.md.

## Outcome / decision / invariants

OMPEX Sep8 captured and LSEG PRD Sep8 queried; D304 Sep7 reference unchanged.
This is a different-vintage descriptive comparison, not forecast accuracy,
scientific admission, trading, training, tuning or model selection. All6false.
OMPEX39complete months28513hours; LSEG27months19753hours, no2029. Monthly
levels separated from within-month shape; unchanged solver/assembler/EEX.
Full-window level/shape mean absolute differences:OMPEX3.982965/10.632228,
LSEG18.093715/20.547824EUR/MWh. Same27month comparison in common-window-comparison.csv.
No model fitted, no source-code change, no scenario/AFRY/T057/CT/protected data work.

## User steering and access

User corrected/provided exact OMPEX file path:
H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260908_101700.xlsx.
Initial directory with ER -HFC\_OMPEX\_15min gavePathNotFound; no retries there.
Exact supplied file accessed read-only and hash-captured in capture-v1.
SHA2568d5f6ce6778f814cf15f964f6ff864bf940bb8a0525e9ca413ff717ed16ce512,894212bytes,
52584hourly rows. Hour-ending interpretation structural only; timezone/filename
availability not vendor-authenticated. No H: write or metadata mutation.

User explicitly authorized Warehouse start if needed, superseding no-Warehouse
restriction. GPU permission also persists, but no GPU compute used. Started
configured Warehouse6da42442d4303908 (Classic2X-Small) once fromSTOPPED.
SixSELECTs, dimension1/profil1/exports2209+8760+8784+0rows; no table/config writes.
Attempted stop returnedHTTP403. auto-stop45min configured, no further SQL after
capture. Do not claimSTOPPED; permission blocker recorded and user informed.
No elevation or ACL/security exception requested. No local process active.

PRD Gold dimension/latest available, vintage endpoint404; no broad alternative
history scan. Curve110181967/CHE/continuous_forward,scenario0,EUR/MWh,1h,CETvendor
metadata; UTC interval end-start1h independently checked. OneissueSep8T00:00Z,
pipeline firstseen/pull07:06:32.475Z. Latest has no Sep7 asof authority.
Old August metadata and sample retained only as qualification evidence.

## Artifacts and hashes

Taskroot build/lt-external-benchmark-20260908/:
- prepare.py; qualification-v1/request.json,source-pins.json,retained-gold-schema.json,
  RAPPORT-QUALIFICATION.md,manifest.json. PREPARED_NOT_EXECUTED is superseded;
  preserved original evidence, not final status.
- capture_sources.py; capture-v1/plan.json,exactXLSX,source-receipt.json,
  ompex-hourly.parquet,ompex-structure.json,warehouse/dimension/latest metadata,
  vintage404error,manifest.json.
- lseg_live.py; lseg-live-v1/plan.json,warehouse-before.json,6SQLtexts/responses/
  Parquet outputs,complete.json,warehouse-stop-error.json,manifest.json.
- compare.py; comparison-v1/plan.json,3hourlyParquetexports,2pairedParquets,
  monthly-levels.csv,curve-distances.csv,forecast-extremes.csv,2seamCSVs,
  coverage.json,lseg-vintage-status.json,complete.json,manifest.json.
- verify_comparison.py; review-v1/verification.json,manifest.json.
- report_close.py,validate_closure.py,common-window-comparison.csv,
  RAPPORT-BENCHMARK-EXTERNE.md,tests.xml,tests-lt.xml,closure.json,closure-validation.json.
- runtime/* and pytest caches/basetemps task-local only.
Capture manifest SHA256 9c80fe31987dc41749f2be583340d4a44297d619b59b896f5ce37b19a1268c23.
LSEG manifest SHA256 70ea891c0ef87d0997ba976256cdee5ab2288255c690577c0edfe2c768587b3f.
Comparison manifest SHA256 13560c3518fec22dffcd0918b5d1a65a60da31dff89c9d71ec3554ff86abc5d0.
Review manifest SHA256 8cdd1be380a6aecad76b6ddd7b33b5706eb476efc83451ba4cffd36fb5e26708.
Report SHA256 80185e01e538d91d75c79f65f46839a8769672d14e51f7aa9498dd8d1ff26826.
Candidate SHA256fa5b9d22cb76e6a1eb9a7355db8f1318c37d3d678c4ef77c8247b8e957471369.

New governance:this file; updated .planning/HANDOFF.md and Phase14/DECISION-LOG.md.
No other tracked source changed; preserve all preexisting dirty work.

## Commands / checks / failures

Every shell action verified cwd and Git top-level exactly C:\Users\jbattaglia\PFC_LT.
PowerShell cwd canonical. Runtime TEMP,TMP,APPDATA,LOCALAPPDATA,MPLCONFIGDIR,
XDG_CACHE_HOME,NUMBA_CACHE_DIR,JOBLIB_TEMP_FOLDER,PYTHONUSERBASE,PIP_CACHE_DIR
under task/runtime/<name>; OMP/OPENBLAS/MKL threads4; PYTHONDONTWRITEBYTECODE=1;
CUDA_VISIBLE_DEVICES=-1. Credentials loaded in memory via existing helper;
no credential values emitted or placed in artifacts.

Read-only commands: Get-Content AGENTS/relevant handoffs/contracts/scripts;
rg files/metadata; Get-ChildItem initial external folder (PathNotFound),
Get-Item exact corrected file (success). Early rg shell wildcard arguments
were invalid Windows paths; corrected to --glob searches. No source mutation.
All Python commands below use & build/conda-runtime-v41-model-source/python.exe -B:
- -m pytest tests/test_ompex_external_benchmark_access.py tests/test_ompex_archive_inventory.py tests/test_ompex_benchmark.py tests/test_databricks_gold_snapshot_intake.py -q --tb=short -o cache_dir=build/lt-external-benchmark-20260908/pytest-cache --basetemp=build/lt-external-benchmark-20260908/pytest-v1 --junitxml=build/lt-external-benchmark-20260908/tests.xml :49pass.
- build/lt-external-benchmark-20260908/prepare.py :prepared, oldsample/hashverified.
- build/lt-external-benchmark-20260908/capture_sources.py :OMPEXcaptured;4metadataGETs,1HTTP404preserved.
- build/lt-external-benchmark-20260908/lseg_live.py :6queriescaptured,1start,1stopHTTP403.
- additional read-only warehouse GET during start reportedSTARTING/HEALTHY;
  no metadata request changes SQL idle accounting or config.
- -m pytest tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py tests/test_compare_hpfc_ompex_benchmark_script.py -q --tb=short -o cache_dir=build/lt-external-benchmark-20260908/pytest-cache --basetemp=build/lt-external-benchmark-20260908/pytest-lt --junitxml=build/lt-external-benchmark-20260908/tests-lt.xml :62pass/4skip.
- build/lt-external-benchmark-20260908/compare.py :COMPLETE_DESCRIPTIVE_NO_ACCURACY_CLAIM.
- build/lt-external-benchmark-20260908/verify_comparison.py :VERIFIED_DESCRIPTIVE.
- build/lt-external-benchmark-20260908/report_close.py :report/governance/closure.
- build/lt-external-benchmark-20260908/validate_closure.py :final closure; git diff --check.
One read-only inline Python produced provisional OMPEX aggregate before final
report; thresholds and alignment had already been frozen in capture-v1/plan.json.
Independent numerical replay and746D308/294D309/357D310 frozen files pass.
Shared current handoff/decisionlog excluded from priorclosures by design.

## Next work / remaining boundaries

Do not train or select using these external benchmarks. Obtain matched
authenticated vintages and finalize Swiss hour-ending and availability semantics.
Capture genuinely new D304 valuations with EEX/solver provenance before external
outcomes, then independently collect post-delivery truth. Current future curve
distances are not forecast errors. LSEG2029 unsupported; no synthetic extension.
Warehouse shutdown403 remains operational follow-up for an entitled user or
auto-stop; do not keep retrying or requesting higher rights. No monetary/DBU
cost claim: only actual statement IDs and limits preserved.
