# Session handoff — publisher post-admission and T057 final re-closure

Date: 2026-07-23  
Canonical repo: `C:\Users\jbattaglia\PFC_LT`  
Branch: `fix/lt-audit-remediation`  
HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Production: strict `NO_GO`  
Commit/staging: none

## Outcome

The demonstrated P1 software slice for publisher dependency-import TOCTOU and
the T057 one-shot holdout is closed on current source. Independent read-only
Security, IT/Operations and Quant/Data re-roasts report no remaining P0/P1
software finding in the reviewed scope. Production was not promoted and no
snapshot or T057 data was acquired.

The manifest-locked dependency closure is explicitly part of the local runtime
TCB. Removing dependencies from that TCB is an external authority problem and
requires a crypto broker/HSM, dedicated service identity, signed artifact/SBOM
attestation and real CAS/WORM controls. Those proofs do not exist, so production
remains `NO_GO`.

## Publisher corrections

- The initial worker capability contains only
  `deferred_environment_expected`; it contains no private-key path.
- Request-signing and mTLS paths are delivered in a separate one-shot
  PID/token-bound capability only after dependency verification/import and
  admission-metric sealing.
- Metric and post-admission capability bytes are written to a private temporary
  mono-linked file, flushed and `fsync`ed, then published atomically by hardlink.
  Readers retry only the transient `nlink > 1` window and reject mono-linked
  invalid/divergent JSON immediately.
- A barrier test covers both pre-`fsync` invisibility and the exact interleaving
  after the final hardlink exists with two links.
- The admitted worker still revalidates the exact post-import `sys.path`; only
  the process-private captured dependency root is admitted.

## T057 corrections

- Wrapper, direct runner, audit and policy require the frozen plan SHA and exact
  canonical paths. A T057-named plan in a temporary directory fails before
  backtest/audit.
- The canonical T057 route rejects any caller-supplied `as_of_utc`; maturity is
  based on the system clock.
- An exclusive `xb` + flush + `fsync` attempt seal is created before any
  provider call and remains blocking after crash/failure.
- The attempt seal hash is included in the capture seal and independently
  replayed by runner and audit.
- Relative candidate paths in the frozen plan resolve from repo root.
- The audit redrives holdout residuals from the hash-bound baseline, adjusted
  candidate and semantic raw-to-Parquet replay. A forged residual CSV with a
  recomputed hash fails.
- The audit derives the exact expected rolling cutoffs and replays
  `_evaluate_fold` from candidates plus spot. Order/count, bounds, train/eval
  hours, common profile cells, MAE, no-leak and eligibility must match, and every
  evaluation ends no later than valuation. A one-fold cherry-pick with rewritten
  summary counts and hash fails.

Frozen T057 identity was rechecked at `2026-07-23T15:52:00Z`:

- plan SHA-256:
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`;
- baseline exists, SHA-256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`;
- adjusted exists, SHA-256:
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`;
- canonical output directory does not exist;
- therefore no provider attempt, capture or T057 scoring has occurred.

Only the command and paths in
`T057-LOCKED-ONE-SHOT-EXECUTION-SIDECAR-20260723.md` are authoritative after
`2026-07-24T00:00:00Z`. Old direct plan commands remain obsolete.

## Files changed in this re-closure

- `pfc_shaping/publisher_runtime_admission.py`
- `deploy/publisher/README.md`
- `scripts/epex_lab_locked_holdout_policy.py`
- `scripts/run_energy_charts_epex_locked_holdout.py`
- `scripts/run_epex_lab_locked_holdout.py`
- `scripts/audit_epex_lab_locked_holdout.py`
- `tests/test_snapshot_publisher_artifact.py`
- `tests/test_run_energy_charts_epex_locked_holdout_script.py`
- `tests/test_run_epex_lab_locked_holdout_script.py`
- `tests/test_audit_epex_lab_locked_holdout_script.py`
- `tests/test_epex_lab_locked_holdout_policy.py`
- `tests/test_check_monthly_curve_promotion_from_manifests.py`
- this handoff, RFC, decision log and `.planning/HANDOFF.md`

The monthly-capstone test change accepts either the original supporting-artifact
hash-mismatch classification or the earlier bounded-read size rejection. Both
are fail-closed responses to the same post-candidate Parquet mutation.

## Commands and results

Every command began by asserting exact cwd and Git root
`C:\Users\jbattaglia\PFC_LT`.

### Final targeted matrices

- `python -m pytest tests\test_snapshot_publisher_runtime_closure.py tests\test_snapshot_publisher_artifact.py -q -p no:cacheprovider`
  - final: `48 passed, 12 skipped in 92.07s` (`111.7s` shell wall while run
    beside T057).
- `python -m pytest tests\test_run_energy_charts_epex_locked_holdout_script.py tests\test_run_epex_lab_locked_holdout_script.py tests\test_audit_epex_lab_locked_holdout_script.py tests\test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider`
  - final: `47 passed in 259.74s`.
- Exact `sys.path` subset:
  `test_runtime_admission_appends_only_private_captured_dependency_root`,
  `test_isolated_sys_path_removes_runtime_root_from_import_search`, and
  `test_dependency_import_sys_path_mutation_is_rejected`
  - `3 passed in 1.22s`.

### Integrated matrices

- Publication/CAS matrix over anchor client/reference, bootstrap signer,
  external publication contract, governed snapshot/acquisition and publisher
  artifact:
  - `201 passed, 13 skipped in 197.94s`.
- Packaging/governance matrix over package contract, governed release,
  candidate evidence/bundle, atomic promotion, quality/capstone gates, governed
  forward history and LT sources:
  - first pass: `543 passed, 4 skipped, 1 failed in 438.24s`; the sole failure
    expected the old hash message although bounded-read size admission rejected
    the mutation earlier;
  - isolated corrected case: `1 passed in 3.81s`;
  - final sequential matrix: `544 passed, 4 skipped in 829.06s`.
- One parallel integrated rerun reached its orchestration timeout at `604s` and
  was discarded without a pytest verdict. A process audit confirmed no orphaned
  pytest remained; the sequential matrices above supersede it.

### Real optimized artifact

Environment:

- wheelhouse:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252`;
- dependency root:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages`;
- receipt:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\dependency-closure-receipt.json`.

Command selector:

`python -m pytest -q tests\test_snapshot_publisher_artifact.py -k "optimized_publisher_zipapp_prepares_real_provider_raw_v3_bundle" -p no:cacheprovider`

Result: exit `0`, wall `394.66066s`. `Measure-Command` captured the pytest text,
so exit status plus the selected single test is the durable result.

### Static verification

- targeted Ruff: `All checks passed!`;
- targeted `py_compile`: pass;
- `git diff --check`: pass (only existing LF/CRLF warnings);
- staged file list: empty.

## Independent final roasts

- Security: no P0/P1 software finding remains under the explicit locked-wheel
  TCB boundary. A hostile same-interpreter dependency requires external
  authority separation, not another Python handshake.
- IT/Operations: post-`fsync` atomic publication, transient two-link retry and
  T057 attempt binding are closed; no P0/P1 software finding remains.
- Quant/Data: exact cutoffs and fold replay close cherry-pick/leakage; no P0/P1
  remains in the reviewed T057 scope.

## Residual blockers and next actions

Production remains `NO_GO`. Required external proofs include signed release and
SBOM/wheelhouse attestation, exact service-account ACLs, crypto broker/HSM/KMS,
real external CAS/WORM and trusted timestamp, provider authenticity, monitoring,
multi-host/power-loss, backup/restore and DR.

Scientific P2s remain: T057 is one 14-day July holdout, the frozen pass threshold
is only MAE improvement `>= 0`, and it lacks confidence-interval, tail, negative
hour and probabilistic-calibration gates. Do not change the frozen T057 plan.
After the maturity boundary, perform at most the canonical first provider attempt,
audit without retuning, and only then return to fresh prospective data and a new
auditable CH candidate. No production promotion is authorized.
