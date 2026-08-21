# Session handoff - 2026-07-31 - CH LT final repricing and runtime reclosure

## Outcome

The local standard-user CH LT path now runs reproducibly through the canonical
repo-local Python runtime without administrator rights, Defender exceptions,
project executables, Playwright, or mutable inputs outside
`C:\Users\jbattaglia\PFC_LT\build`. The obsolete runtime-v9 command using an
`AppData` publisher wheelhouse is forbidden and was not resumed.

This slice closes the demonstrated local P1 defects in the delivery window and
final product replay. It does not authorize production. The resulting evidence
uses `TEST_FIXTURE` forward levels, reports source-quote conflicts, has no fresh
governed CH EEX authority, no direct-CH prospective shaping validation, no
calibrated probabilities, and no future holdout evidence. Production remains
strictly `NO_GO`; no publication or promotion occurred.

## Canonical workspace and protected state

- workspace/cwd: `C:\Users\jbattaglia\PFC_LT`;
- branch: `fix/lt-audit-remediation`;
- HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`;
- worktree intentionally dirty; no reset, clean, restore, stage, commit, or
  protected-data edit was performed;
- `data/eex_forwards_history.parquet` SHA-256 remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- no `pfc_shaping/ct/*` or Power BI file was touched by this slice;
- monthly solver remains the monthly level authority; OMPEX remains
  benchmark-only; T057 outcome was not read.

Every shell command started by comparing both `Get-Location` and
`git rev-parse --show-toplevel` to the exact canonical root.

## Implemented closure

### Exact delivery window and Cal2030

The local build accepts an exact local end-exclusive timestamp. The frozen
window is:

- start: `2026-08-01T00:00:00 Europe/Zurich`;
- end-exclusive: `2031-01-01T00:00:00 Europe/Zurich`;
- total: 154,948 quarter-hours;
- Cal2030: exactly 35,040 quarter-hours / 8,760 local hours;
- last delivered local interval: `2030-12-31T23:45:00 Europe/Zurich`.

This replaces the prior horizon-day truncation that omitted the final local
hour of Cal2030.

### Final quote-aware product projection and replay

Eligible PEAK quotes are now passed into the canonical assembler projection.
The monthly BASE solver remains the level authority. The final joint
BASE/PEAK/implied-OFFPEAK projection emits an explicit decomposition via
`price_pre_final_projection` and `delta_final_product_projection`.

The immutable completion role `final_product_replay` replays the captured
forward bytes against the final hourly delivered prices and fails closed on an
unsupported/partial/critical product. Current local results across slow,
central and fast are:

- source quote counts: BASE 20, PEAK 20, explicit OFFPEAK 0;
- Cal2030 replay hours: 8,760;
- maximum supported absolute residual: `6.465938895416912e-12 EUR/MWh`
  against tolerance `1e-9`;
- maximum final non-quarter factor parent-hour range:
  `3.694822225952521e-13 EUR/MWh`;
- maximum final counterfactual hourly-mean residual:
  `2.842170943040401e-14 EUR/MWh`;
- maximum final price decomposition residual: `0.0`;
- 18 source-quote conflict gate rows are retained explicitly;
- replay status: `PASS_LOCAL_FINAL_PRODUCT_REPLAY_PRODUCTION_NO_GO`.

The conflicts are not silently reconciled into production evidence. Approved
source-hierarchy/conflict policy remains a production blocker.

### Baseline and supervisor hardening

The parent-hour fixture generator refuses existing outputs, validates exact
expected SHA-256 and size before writing, and uses exclusive create plus
`fsync`. Tests pin both committed fixture identities.

The current-selection auditor now binds the qualification audit supervisor's
single terminal write, deadline and zero overshoot, one-shot consumed/absent
capability, worker admission, kill-on-close process tree, zero active
processes, execution receipt, and all false authority flags.

## Current evidence

### Reproducible model pair and qualification V5

Run A:

- model/supervisor ID `mdl31n1c`;
- completion SHA-256
  `fbc8278e9ae51be2d2bb54387b939540c7da97577025026440d4eebaecf0202c`;
- execution receipt SHA-256
  `409132c223419423d6940bbde8b474a73e36be1e324b40827a4595542ca11a80`;
- supervisor receipt SHA-256
  `1d37bd1191e5fcbfb435b663c1690e98e2ac84f407d268d74dc16aa8a31941c8`.

Run B:

- model/supervisor ID `mdl31n2`;
- completion SHA-256
  `2404fc3ad59442b414ab90f951425e9d70d9b09cdb95e4603f620f2620674c04`;
- execution receipt SHA-256
  `ed68eaba3bf1b7607e298f653cf8a7d71dfdf4628fd5408786989c44e2ce0b84`;
- supervisor receipt SHA-256
  `b89f9cc3209667bc7341e1f1c57e5f7321d9507e5f7b8000308c213b7ebcee1b`.

Nine material roles are byte-identical. Both runs bind source tree
`5ee25178ffa27028f0b532c7c75f12df64bda6001a528b908ba4f24ceaa5db01`,
capture import closure `BOUND_REPO_LOCAL_PTH`, contain the canonical workspace
root exactly once in `sys.path`, and leave zero active processes.

Qualification V5:

- directory: `build/local-model-qualification/ch-lt-laptop-model-pair-v5`;
- qualification ID
  `c90dba9deaa3de1938353843aa5dc713a5aebcef40a02e61d5157fed341a5a36`;
- `quality.json` SHA-256
  `ce2f16c37e2bbd465e8b4000c77231ea0040643b4c00b520068d61d247b0bbe6`;
- `QUALITY.md` SHA-256
  `e73c402312d9c9fefa881197da56e10109f6eb49c875efed585e8b80af05d390`;
- qualification receipt SHA-256
  `cb55186ce5dad68ab30bb644f936384b1f72876c7884cf00c1b326ac732a5fdf`;
- supervised audit `qa31pairv5`: execution/supervisor SHA-256
  `e6da9c167b3c490168a895414e1a883b4bed8c22a9b0ae48484e36bb485ac7db` /
  `efb8e17cc6b10e7e1f67ac9d8587be27bd65fbf94023d1f8bdad28f9cc933ebd`;
- status `PASS_LOCAL_STRUCTURAL_REPRODUCIBILITY_ONLY_NO_GO`.

### Current selection V5

- path:
  `.planning/phases/14-lt-audit-remediation/CH-LT-LOCAL-QUALITY-CURRENT-SELECTION-V5-20260731.json`;
- SHA-256
  `f61509834d692d64ee17dad6030f3672969788aac39d3cb3d56b7d5db9b7207b`;
- selection ID
  `aa4233b4b42159bc4d0b6868fe3c3f0ec3e2449db593dbb14cd363c5e27336c6`;
- direct and supervised audit status
  `CURRENT_LOCAL_ENGINEERING_QUALITY_SELECTION_VALID_NO_GO`;
- supervised audit `qsel31v5`: execution/supervisor SHA-256
  `326dc97bc8e265fda529ef97818b405a8315c5681e29a0116e348e88dc1e5f2a` /
  `4123bea7b83b9f731fa25b594127371649985652a357d5f8fcb572cee18f4937`.

### Non-countable structural commitment V3

- directory:
  `build/ch-lt-structural-prediction-commitments/structural-dry-run-20260730T120000Z-v3`;
- commitment ID
  `954b95559f71aa8aa3454482ee985d697588392bc07934a58806256a6e969d86`;
- `commitment.json` SHA-256
  `9808bccdf11598eb9be311250688f82f5ed34d25f285d86d94c36262eb9082e3`;
- `predictions.json` SHA-256
  `b9291f6dc7eae45f5ad41c8aa3c9b462df7fc22b0ba4a0db2251a29634dcb9c5`;
- 36 targets / 108 structural scenario predictions;
- status `SEALED_LOCAL_STRUCTURAL_DRY_RUN_NONCOUNTABLE_NO_GO`;
- supervised run `pcmt31v3`: execution/supervisor SHA-256
  `c548f93d748113e717cc45de9e3ef6a6d37486edae02db16654aeadd8f792f9c` /
  `7a764041ee5714848d0218efc964f11d0d6e8983c4261ff7b9bc3e920218a415`.

## Commands and results

All commands used `build\pytest-runtime-v1\python.exe`; every mutable root was
redirected below `build/` by `scripts.run_workspace_local`.

Targeted regression:

```text
pytest tests/test_audit_ch_lt_local_quality_current_selection_script.py
5 passed

pytest tests/test_build_ch_lt_structural_prediction_commitment_script.py \
       tests/test_run_workspace_local_script.py
142 passed

targeted local/model/monthly/shape selection suite
185 passed

local/multi/runner suite excluding Windows E2E
158 passed, 4 deselected
```

Final supervised model/runtime matrix `lmqmat31v5a`:

```text
691 passed, 16 skipped, 2 deselected, 3 warnings in 111.72s
execution SHA 318315743374548a91047485e427a1807678ee6334d09ff91d7c1b8de0b66ade
supervisor SHA e45ca00db0bacd79b024f12947f193666b9387fb2169ed67544c13b06406a4b6
source tree 374c1c7cfb77ff0ccca2f6df271e6b8ab788533dd6d99c98c38b80e2caf54510
```

Final supervised packaging/publication matrix `pkgpub31v5a`:

```text
489 passed, 18 skipped, 1 deselected in 239.87s
execution SHA 552f8d1101d1b9c58758bec07d442b359243d88722af3bbd62cc1ea9dfdfe1e9
supervisor SHA 6656be18b12da13da1c4b94af4abc36e52f92f1ad4ab3b24165d69e75826498d
source tree 374c1c7cfb77ff0ccca2f6df271e6b8ab788533dd6d99c98c38b80e2caf54510
```

For both matrices: target exit zero, status
`WORKER_EXIT_ZERO_NOT_AUTHORITY`, import closure `BOUND_REPO_LOCAL_PTH`,
canonical root count one, deadline not exceeded, one terminal receipt write,
and zero active processes.

Other checks:

```text
ruff check --no-cache <changed Python scope>
All checks passed!

git diff --check
PASS (line-ending warnings only)
```

Negative evidence retained:

- `mdl31n1` was rejected at precheck because PowerShell split the unquoted
  comma-separated weights; no model target or output ran, and the ID was not
  reused;
- `mdl31n1b` used the corrected quoting but the 300-second supervisor budget
  expired before target admission. Status is
  `SUPERVISOR_WALL_TIMEOUT_TREE_TERMINATED_NO_AUTHORITY`, active processes are
  zero, no completion manifest exists, and execution/supervisor receipt
  SHA-256 values are
  `ea29c9cbc7dbb473afe7ef8ba8bf847cab0b10e1513cccb756f83cd39df1d271` /
  `67daeaf69c72b314e2de01b6ee54c51f65f3925a3c19f27af8c9ca42cc82ddc1`;
- the accepted fresh run rotated to `mdl31n1c` with a 900-second governed
  budget. No timeout evidence was reclassified as a passing run.

The matrix skips are retained evidence, not silently converted to passes. They
cover unavailable wheelhouse/symlink attack capabilities on this managed
workstation and optional CT dependencies; they block full installed-wheel and
production packaging qualification.

## Read-only independent re-roasts

Security/Governance V5: P0/P1/P2 = `0/0/2`. The two P2 limits are checkout
qualification rather than immutable installed wheel, and unavailable
wheelhouse/symlink attack capabilities. The prior stale-matrix, overwriteable
fixture, incomplete supervisor-binding, Cal2030 and final-replay findings are
closed.

IT/Operations V5: P0/P1/P2 = `0/0/4`. Accepted operational P2 are missing
installed-wheel qualification, environment-denylist rather than filesystem
sandbox plus unavailable symlink drills, rotate-to-fresh-ID rather than model
checkpoint recovery/RPO, and missing formal capacity/SLO/alert thresholds.
The staged source identities were independently rechecked: A/B share
`5ee25178...`; their successor qualification binds their exact receipts; final
selection, commitment and both matrices share `374c1c7c...`. No binding break
exists and no artifact claims one source hash for every generation.

Quant/Data V5: P0/P1/P2 = `0/0/0` for the corrected local structural claims.
The stale docstring and JSON residual-rounding findings are closed. The roast
independently confirmed Cal2030 8,760 hours, BASE/PEAK/OFFPEAK residuals below
`1e-9`, monthly solver authority, parent-hour neutrality and byte-identical A/B
replay. Production-readiness claims remain fail-closed, not counted as local
defects. No roast agent edited files or opened T057.

## Exact source files changed by this closure

- `pfc_shaping/lt/model/assembler.py`;
- `scripts/build_first_ep2050_pfc.py`;
- `scripts/build_ep2050_multi_scenario_pfc.py`;
- `scripts/build_local_test_ch_pfc.py`;
- `scripts/audit_ch_lt_laptop_model_pair.py`;
- `scripts/audit_ch_lt_local_quality_current_selection.py`;
- `scripts/build_ch_lt_structural_prediction_commitment.py`;
- `scripts/run_workspace_local.py`;
- `tests/fixtures/_generate_parent_hour_baselines.py`;
- `tests/fixtures/baseline_pfc_seed42_parent_hour_v1.parquet`;
- `tests/fixtures/baseline_pfc_seed42_bowl_parent_hour_v1.parquet`;
- `tests/test_shape_hourly_bowl.py`;
- corresponding local-build, qualification, selection, commitment and runner
  tests;
- current selections V4/V5, with V5 current, and this documentation.

## Next safe work

1. Preserve production `NO_GO` and resolve the source-quote conflict policy
   through an independently approved hierarchy; do not hide conflicts.
2. Acquire fresh governed point-in-time CH EEX levels and fresh prospective CH
   truth under external receipts. Swiss truth remains hourly until the native
   15-minute transition is independently verified and admitted.
3. Execute direct-CH rolling-origin shaping validation and dependence-aware
   probabilistic/scenario calibration. Keep the old T057 outcome closed and
   use only a new externally frozen independent holdout.
4. Build and qualify an immutable installed-wheel runtime, external CAS,
   trusted time/signatures, CI/ASR/SBOM, observability and rollback on an IT
   runner with the missing wheelhouse/symlink capabilities.
5. Assemble a new auditable CH candidate only after those gates pass. Do not
   promote production before complete manifest-backed evidence.
