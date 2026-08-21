# Session Handoff - T057 Fold-Integrity Reclosure

Date: 2026-07-24  
Repository: `C:\Users\jbattaglia\PFC_LT`  
Branch: `fix/lt-audit-remediation`  
HEAD throughout: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Production: strict `NO_GO`  
Commit, staging or promotion: none

## Outcome

The former T057 `12/12` rolling-origin claim remains revoked. The canonical
CSV contains twelve identical rows for one cutoff, so effective historical
sample size is one. The immutable canonical one-shot directory was not rerun,
modified, deleted or repaired.

Three further P1 findings from independent Security and IT/Operations roasts
are now closed in local source and tests:

1. fold and bucket metrics can no longer be forged through a mutually
   rehashed CSV/JSON chain; every replayable fold column and every bucket row is
   reconstructed independently from captured spot and candidate bytes;
2. the direct locked runner and backtest refuse pre-existing evidence before
   any overwrite, with exclusive-create CSV publication and durable summary
   publication;
3. revocation is machine-discoverable through a canonical workspace registry
   and verifier; the new registry remains untracked until an audited commit,
   and correction v2 admits only the exact canonical T057 sources/configuration
   and recomputes its metrics from CSV evidence.

The retained T057 result is only one partial prospective episode: 336 unique
hours, baseline MAE `18.27968231685799`, adjusted MAE
`18.010889719387755`, uplift `0.26879259747023454 EUR/MWh` or
`1.470445%`. It proves neither historical/regime robustness nor economic
materiality and grants no production authority.

## Invariants preserved

- Every shell command was guarded for exact cwd and Git root
  `C:\Users\jbattaglia\PFC_LT`; the old `H:` repo was never used.
- The intentionally dirty worktree was preserved: no reset, clean or restore.
- `data/eex_forwards_history.parquet` was not edited or staged by this work.
- `pfc_shaping/ct/*` and Power BI were untouched.
- Monthly solver remains the level authority; hourly shaping may not rewrite
  solver monthly means.
- OMPEX remains benchmark-only and never enters model, selection or holdout.
- No production publication, promotion, commit or staging occurred.

## Defect evidence and immutable sources

- canonical wrapper:
  `output/phase14/t057_locked_t056_future_holdout/energy_charts_locked_runner_20260724/energy_charts_locked_holdout_run_summary.json`,
  SHA-256
  `7b1f8613ffdf8c5771c6d493299fbeec5ac8fc15d136d3ed428282e4e081ffc7`;
- legacy spot summary SHA-256
  `b0c253993cb785dcfc7e338f15e5763e2100277ab774789f840906283f039a91`;
- defective fold CSV SHA-256
  `9fc671dcb5be86cc1402ac7930647b6839008e936c74b48ad8f2310c24c4d607`:
  12 rows, one unique cutoff `2026-06-24T00:00:00Z`, 11 duplicates,
  overlapping evaluations;
- superseded quality report v1 SHA-256
  `d16787c8d28d4c0ddd0cfd346cda786b09385684c66a5663454af584fcfcd389`;
- non-promotable forensic summary SHA-256
  `0102105bc2320e0df99882e4104616c281c5cdd9b6e10ce24a9f9a49ac102624`;
- corrected one-fold CSV SHA-256
  `34d160327bfa44d78b31e8c6b5bc443f8d40ca6754d70c4719d5cd79bb9f117b`.

Correction sidecar v1, SHA-256
`58bb50ea0efa1528164b9681b11e808d797b21d5427a42e67070a83e167967bf`,
is preserved but superseded because it accepted arbitrary caller-selected
sources with a conforming 12-to-1 shape.

## Authoritative correction and discoverability

Authoritative append-only correction:

`output/phase14/t057_fold_integrity_correction_20260724_run2/t057_fold_integrity_correction_v2.json`

- schema `t057_fold_integrity_correction.v2`;
- SHA-256
  `2cc0b67a509fe79baf4136da65e5eec3cc424a8f1d8739c300357a26b282e1c6`;
- size `4418` bytes;
- canonical wrapper/quality/forensic/plan paths and hashes required;
- baseline, adjusted candidate and spot bytes cross-bound to the plan and both
  replay summaries;
- valuation, lookback, embargo, evaluation length and exact holdout window
  cross-bound;
- rolling metrics recomputed from the corrected fold CSV;
- future metrics recomputed from the exact 336-row post-valuation CSV;
- production and promotion authority both false.

Canonical workspace registry (intended to become tracked only in an audited
commit):

`.planning/phases/14-lt-audit-remediation/T057-EVIDENCE-SUPERSESSION-REGISTRY.json`

- registry SHA-256
  `0efec38b768b5e14add6cbc35c9b0cf9f10eb23f2cbf040813e68fe734ca4cf6`;
- ten superseded paths bound by path and SHA-256, including all six generated
  T057 quality JSON/Markdown v1/v2/v3;
- effective claims fixed to historical `n=1`, one 336-hour prospective episode,
  no materiality and production `NO_GO`;
- `scripts/verify_t057_evidence_supersession.py` reports
  `T057_SUPERSESSION_VERIFIED`;
- `scripts/build_local_pfc_quality_report.py` rejects a registered superseded
  run hash before stale schema/PASS interpretation.

## Code and test changes in this reclosure

Source:

- `scripts/audit_epex_lab_locked_holdout.py`
- `scripts/backtest_epex_shape_lab_against_spot.py`
- `scripts/run_epex_lab_locked_holdout.py`
- `scripts/build_t057_fold_integrity_correction.py`
- `scripts/verify_t057_evidence_supersession.py`
- `scripts/build_local_pfc_quality_report.py`

Tests:

- `tests/test_audit_epex_lab_locked_holdout_script.py`
- `tests/test_backtest_epex_shape_lab_against_spot_script.py`
- `tests/test_run_epex_lab_locked_holdout_script.py`
- `tests/test_build_t057_fold_integrity_correction_script.py`
- `tests/test_verify_t057_evidence_supersession_script.py`
- `tests/test_build_local_pfc_quality_report_script.py`

Governance/docs:

- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260724-T057-LOCAL-PFC-QUALITY.md`
- `.planning/phases/14-lt-audit-remediation/T057-EVIDENCE-SUPERSESSION-REGISTRY.json`
- this handoff.

## Exact verification commands and results

Every direct pytest command below was preceded by the exact workspace guard.

```powershell
python -m pytest tests\test_audit_epex_lab_locked_holdout_script.py -q -p no:cacheprovider
```

Result: `17 passed in 79.47s`.

```powershell
python -m pytest tests\test_backtest_epex_shape_lab_against_spot_script.py tests\test_run_epex_lab_locked_holdout_script.py -q -p no:cacheprovider
```

Result: `18 passed in 9.41s`.

```powershell
python -m pytest tests\test_build_t057_fold_integrity_correction_script.py tests\test_verify_t057_evidence_supersession_script.py tests\test_build_local_pfc_quality_report_script.py -q -p no:cacheprovider
```

Result after correcting one misplaced test assertion: `14 passed in 17.52s`.
The prior run was `13 passed, 1 failed` solely from that test-code `NameError`
and is not counted as product evidence.

```powershell
python -m scripts.verify_t057_evidence_supersession
```

Result before the exhaustive quality-report inventory amendment:
`status=T057_SUPERSESSION_VERIFIED`, five paths, authority SHA-256
`2cc0b67a...`, production/promotion false.

Final result after inventory hardening: `status=T057_SUPERSESSION_VERIFIED`,
ten exact paths, registry SHA-256
`0efec38b768b5e14add6cbc35c9b0cf9f10eb23f2cbf040813e68fe734ca4cf6`,
authority SHA-256 `2cc0b67a...`, production/promotion false. The ten-kind set
is exact and omission or addition fails verification. The registry is a new
untracked file in the intentionally dirty worktree until an audited commit;
clean-clone portability and external-CAS durability are not claimed.

```powershell
python -m pytest tests\test_backtest_epex_shape_lab_against_spot_script.py tests\test_audit_epex_lab_locked_holdout_script.py tests\test_run_epex_lab_locked_holdout_script.py tests\test_run_energy_charts_epex_locked_holdout_script.py tests\test_fetch_energy_charts_epex_spot_hourly_script.py tests\test_epex_lab_locked_holdout_policy.py tests\test_build_local_pfc_quality_report_script.py tests\test_build_t057_fold_integrity_correction_script.py tests\test_verify_t057_evidence_supersession_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_build_epex_lab_adjusted_production_manifest_script.py tests\test_build_epex_lab_adjusted_production_chain_script.py -q -p no:cacheprovider
```

Initial result before the final single-capture/integer/inventory delta:
`145 passed in 142.42s`. The same full matrix was then rerun after that delta:
`146 passed in 148.74s`. After the final immutable kind-to-canonical-path
mapping and shadow-path regression, the full matrix was rerun once more:
`146 passed in 130.90s`; this last run is authoritative.

```powershell
python -m pytest tests\test_snapshot_publisher_runtime_closure.py tests\test_snapshot_publisher_artifact.py tests\test_snapshot_publisher_container_contract.py tests\test_lt_package_contract.py -q -p no:cacheprovider
```

Result: `92 passed, 13 skipped in 66.38s`. Skips are contract-declared optional
platform/container cases, not failures.

```powershell
python -m pytest tests\test_snapshot_publication_external_contract.py tests\test_atomic_promotion.py -q -p no:cacheprovider
```

Result: `134 passed, 2 skipped in 130.34s`.

```powershell
python -m pytest tests\test_candidate_bundle.py tests\test_candidate_evidence.py tests\test_candidate_evidence_assembler.py -q -p no:cacheprovider
```

Result: `65 passed in 102.54s`.

```powershell
python -m pytest tests\test_check_monthly_curve_promotion_from_manifests.py tests\test_run_governed_lt_release_script.py tests\test_governed_release.py tests\test_quality_gate.py -q -p no:cacheprovider
```

Result: `267 passed in 185.65s`.

Final targeted Ruff and `py_compile` both pass. `git diff --check` exits zero;
the displayed LF-to-CRLF notices are warnings for the dirty Windows worktree,
not whitespace errors. Final perimeter audit confirms:

- cwd/root `C:\Users\jbattaglia\PFC_LT`;
- branch `fix/lt-audit-remediation`, HEAD
  `2f68125bff869ccb21c1e20df0201ad024ed27d3`;
- staged file count `0`;
- protected parquet SHA-256
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- CT/Power BI tracked or untracked delta count `0`;
- legacy wrapper, quality report, forensic replay and sidecar v1 hashes remain
  respectively `7b1f8613...`, `d16787c8...`, `0102105b...` and `58bb50ea...`;
- sidecar v2 and registry remain `2cc0b67a...` and `0efec38b...`.

Earlier non-conclusive executions remain excluded: one monolithic publication
matrix timed out after 304 seconds, and earlier pytest basetemp attempts hit
Windows ACL setup errors. The successful isolated matrices above are the
authoritative results.

## Independent review

The first read-only re-roasts found no P0 but demonstrated:

- Security P1: partial fold determinant comparison and producer-declared bucket
  aggregation allowed coherent forgery;
- IT/Operations P1: direct runner overwrite window, non-canonical sidecar v1
  sources, and documentary-only supersession.

Final read-only re-roasts on the exact closing source:

- Security: no residual P0/P1; local PASS for this delta. It verified single
  registry-byte capture/hash, exact finite integer comparison, the adversarial
  `1.9` rejection, exact ten-item inventory and dynamic verifier PASS.
- IT/Operations: no residual P0/P1; local GO for this delta. It verified
  pre-write runner admission, exclusive backtest output, durable summary,
  canonical sidecar v2 bindings, exact ten-path registry mapping and rejection
  of a same-byte shadow path.
- Quant/Data: no residual P0/P1 in revocation/supersession; local GO for this
  slice. It confirmed all six stale quality JSON/Markdown reports are named by
  canonical path/hash and the old `12/12` claim is unambiguous everywhere.

These are slice-local verdicts only. T057 remains scientific `NO_GO` because
historical effective `n=1`; production remains strict `NO_GO` because global
scientific, CAS, identity, deployment and operations evidence is incomplete.

## Residual risks and next protocol

Production remains strict `NO_GO`. Local green tests do not prove external CAS
service identity, cross-principal ACL/freeze, independent clean build/signing,
SBOM/provenance, registry deployment, alert delivery, rollback, backup/restore,
power-loss durability or disaster recovery.

T057 cannot be repaired into a robust multi-origin proof. The next plan must
use a new non-colliding ID and preregister, before observation:

1. an exact ordered list and hash of scientifically sufficient PIT origins,
   with no generic `n=1` diagnostic accepted as robustness;
2. full acquisition, solver, feature fit, shaping fit and selection replay at
   each origin using only then-available vintages;
3. multiple seasons and price/renewables/hydro regimes plus genuinely future
   episodes;
4. deterministic gates for monthly solver exactness, cascade invariance,
   quote-to-curve sensitivity, complexity and capture-factor materiality;
5. coherent non-crossing probabilistic/scenario outputs preserving monthly
   means, evaluated with pinball, WIS/CRPS, coverage, sharpness and PIT or
   reliability diagnostics;
6. dependence-aware uncertainty for metrics using block bootstrap or HAC and
   explicit minimum effect/materiality thresholds.

Before that plan is frozen or any candidate is evaluated, its quantitative
appendix must additionally fix: exact `n` and per-season/regime minima, origin
as the inference unit, an untouched nested-selection final holdout, alpha and
multiplicity policy, block/HAC lengths and seeds, tie/non-regression rules,
quantile grid and scenario count, energy and variogram scores, pathwise monthly
coherence tolerance, and an FMV capture-factor/economic materiality threshold.
Those values are not yet frozen; no new holdout observation is authorized
until the appendix is independently Quant/Data-roasted and hash-registered.

Fresh point-in-time prospective data and a new auditable CH candidate may be
built only under that new plan. OMPEX remains benchmark-only. Nothing in this
handoff authorizes production promotion.
