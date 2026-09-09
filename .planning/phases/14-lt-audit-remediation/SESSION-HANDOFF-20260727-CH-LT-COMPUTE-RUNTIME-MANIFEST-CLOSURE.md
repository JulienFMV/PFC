# Session Handoff - CH LT compute/runtime-manifest structural closure (2026-07-27)

## Status

- Canonical repo: `C:\Users\jbattaglia\PFC_LT`.
- Branch: `fix/lt-audit-remediation`.
- HEAD observed before the slice: `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree remains intentionally very dirty. Do not reset, clean or restore.
- No commit, staging, data write, snapshot publication, candidate promotion or
  production promotion occurred.
- Security, IT/Operations and Quant/Data final read-only roasts find no residual
  P0/P1 in this explicitly local, structural and non-authoritative slice.
- Scientific evaluation, T057, runtime admission and production remain strict
  `NO_GO`.

## Protected invariants

- Every shell command was preceded by an exact cwd/Git-root guard for
  `C:\Users\jbattaglia\PFC_LT`; the old `H:` repo was never used.
- `data/eex_forwards_history.parquet` remained pre-existing modified, was not
  touched or staged and retains SHA-256
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- No `pfc_shaping/ct/*` or Power BI file was changed by this slice.
- Monthly solver remains the level authority; no post-solve month patch exists.
- OMPEX remains benchmark-only.

## Durable decision

`ch_lt_compute_runtime.v1` is a frozen structural policy, never execution
authority. CPU float64 owns monthly solve, EEX repricing, cascade and
quote-sensitivity, ensemble-monthly consistency, final projection and every
hard gate. GPU v1 is limited to scenario transform/scoring and shaping
inference from frozen weights. GPU fit/model selection, monthly solve,
repricing, cascade calibration, acquisition, publication and promotion abort.

Full-price scenarios preserve monthly solver authority at the energy-weighted
ensemble expectation, not by forcing every path to a deterministic monthly
mean. Scenario-specific monthly risk requires a frozen zero-mean ensemble. The
PIT preregistration v1 remains non-executable and carries
`PREREGISTRATION_V1_PATHWISE_SCENARIO_LEVEL_POLICY_SUPERSEDED`.

The packaged runtime-manifest validator is relational rather than declarative:

- exact top-level and nested inventories;
- exact compute-contract ID/document hash;
- canonical execution-context SHA-256 covering all runtime facts except the
  eight cyclic receipt-file hashes, required in every receipt;
- exact CPU/GPU scope, fallback-before-RNG/output and frozen-weight relations;
- candidate-independent CPU PCG64 shock policy and exact order/count/chunk
  bindings;
- physically distinct mono-linked paths below one absolute evidence root;
- stable bytes and SHA-256 verification for all receipts and every bound
  payload;
- exact pre-freeze scenario/MC design, MC error study and non-cyclic local
  freeze receipt;
- CRN, non-holdout, candidate-count, selected count/chunk/shock, finite positive
  scale/margin, positive-margin and zero-margin half-width, confidence and
  authority checks;
- local receipt explicitly denies signature, trusted time and ledger sequence.

Successful output says only `structure_valid=true` and
`receipt_and_bound_payload_bytes_verified=true`; execution, production and
promotion flags remain false. An independent signed admission envelope is
still mandatory.

## Files added by this slice

- `pfc_shaping/validation/ch_lt_compute_runtime.py`
- `pfc_shaping/validation/ch_lt_compute_runtime_manifest.py`
- `pfc_shaping/cli/audit_ch_lt_compute_runtime.py`
- `pfc_shaping/cli/audit_ch_lt_compute_runtime_manifest.py`
- `scripts/audit_ch_lt_compute_runtime.py`
- `scripts/audit_ch_lt_compute_runtime_manifest.py`
- `tests/test_ch_lt_compute_runtime.py`
- `tests/test_ch_lt_compute_runtime_manifest.py`
- `.planning/phases/14-lt-audit-remediation/CH-LT-COMPUTE-RUNTIME-DRAFT-20260727.json`
- this handoff.

## Existing files changed by this slice

- `pfc_shaping/validation/ch_lt_pit_preregistration.py`
- `pfc_shaping/package_contract.py`
- `pyproject.toml`
- `scripts/check_lt_wheel_contract.py`
- `tests/test_lt_package_contract.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`

## Canonical identities

- compute contract ID:
  `d06710ba8ebee2364b81930fce51d17768f206521edfe62336ff2abdef60930a`;
- compute document SHA-256:
  `b231345e96e7664ae02b7dbf3514af87d47ded7783034eaab1f8d449a28fe96f`;
- preregistration v1 document SHA-256, unchanged:
  `aba798530084b7031a0ac38b1c48b20cff575d6082edbcf37c9a04528900ba61`;
- manifest-validator policy SHA-256:
  `7d86b9a94d9ab176ff30db80ca6943693362717a24ee1e045691344e7c5fc99f`;
- manifest-validator implementation SHA-256:
  `bdbca66585f378a38a16aaa1e4aa809536a76ce1c879bfb5068df6d0e49d7ee2`;
- structural validator implementation SHA-256:
  `0ab96514f5881a2b0ba3c66619fd789c20cc3e163a47f965dc5b8b8bce3a8746`.

## Final governed wheel proof

Build commands, repeated with independent fresh bases `o` and `p`:

```powershell
python setup.py --quiet build `
  --build-base build\compute-manifest-final-build-20260727-o `
  bdist_wheel `
  --bdist-dir build\compute-manifest-final-bdist-20260727-o `
  --dist-dir build\compute-manifest-final-wheel-20260727-o
```

The second command substitutes suffix `p` everywhere.

Results:

- wheel: `fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- size: 456,904 bytes;
- members: 85;
- byte-identical SHA-256:
  `7d985c9fd7b77d253f0924f9b7dda04172930b12560b26e8743c86dfc582a577`;
- embedded source revision:
  `6e52bcfd8700e425dd684807cc42c27805e332516995e75c44dac89c6a927989`;
- both `python -m scripts.check_lt_wheel_contract <wheel>` audits: `PASS`,
  `promotion_eligible=false`.

The direct setuptools backend emits its expected deprecation warning. PEP 517
`pip --target` remains non-conclusive locally because the known sandbox-created
`%TEMP%` ACL denied `pip-target-*`. This is not counted as a PASS. Two clean
independent PEP 517 builders and hash-pinned frontend/backend remain external
IT evidence.

An isolated extraction at
`build/compute-manifest-final-extracted-20260727-q` imported both CLIs, both
validators and build identity only from the extracted root. Checkout was absent
from `sys.path`; no `scripts` or `pfc_shaping.ct` module loaded; both entrypoints
were exact. Result: `ISOLATED_COMPUTE_MANIFEST_FINAL_WHEEL_PASS`.

Wheel audit resource gates are enforced before member reads:

- wheel bytes `<=16 MiB`;
- members `<=128`;
- each uncompressed member `<=4 MiB`;
- total uncompressed `<=32 MiB`;
- per-member compression ratio `<=200`.

## Verification commands and results

Focused final contract/package suite:

```powershell
python -m pytest `
  tests\test_ch_lt_compute_runtime_manifest.py `
  tests\test_ch_lt_compute_runtime.py `
  tests\test_ch_lt_pit_preregistration.py `
  tests\test_lt_package_contract.py `
  -q -p no:cacheprovider `
  --basetemp C:\Users\jbattaglia\PFC_LT\build\p27
```

Result: `70 passed in 11.42s`.

Final runtime/packaging matrix:

```powershell
python -m pytest `
  tests\test_lt_provider_verifier_artifact.py `
  tests\test_snapshot_publisher_artifact.py `
  tests\test_snapshot_publisher_runtime_closure.py `
  tests\test_lt_package_contract.py `
  tests\test_audit_provider_acquisition_quarantine_script.py `
  tests\test_audit_legacy_provider_resolution_script.py `
  tests\test_ch_lt_compute_runtime.py `
  tests\test_ch_lt_compute_runtime_manifest.py `
  tests\test_ch_lt_pit_preregistration.py `
  -q -p no:cacheprovider -m "not slow" `
  --basetemp C:\Users\jbattaglia\PFC_LT\build\p28
```

Result: `148 passed, 12 skipped, 2 deselected in 94.39s`.

The final monolithic publication command exceeded its 15-minute wall timeout
without a test failure. The exact same inventory was then executed in four
direct pytest groups using fresh short basetemps:

- snapshot publication + anchor client/reference/bootstrap signer:
  `77 passed in 9.82s`;
- atomic promotion: `116 passed, 2 skipped in 371.32s`;
- candidate bundle/evidence/assembler: `65 passed in 266.62s`;
- governed release/script/monthly-manifest promotion:
  `241 passed in 256.09s`.

Aggregate over the exact original inventory: `499 passed, 2 skipped`. The
split was necessary because observed cumulative time was ~904 seconds. A
dynamic node-list bisect using `build/p36` is non-conclusive: the known Windows
sandbox ACL denied that basetemp before test execution. It is not counted.

Targeted Ruff passed. Final `git diff --check` returned exit 0 after this
handoff, with only informational LF-to-CRLF warnings.

## Independent roasts and demonstrated fixes

Security demonstrated and then verified closure of:

- receipt reuse after wheel/input/GPU/flag/seed/order/backend mutation;
- receipt hashes without actual payload bytes;
- MC-study/freeze hashes without actual bytes or semantic checks;
- physical path aliases.

IT/Operations demonstrated and then verified closure of:

- unbounded wheel/member/decompression/ratio resource use;
- absent service-host runbook and ambiguous exit-0 semantics.

Quant/Data verified the CPU/GPU authority split, ensemble-level monthly
consistency, CRN/seed policy, deterministic bootstrap, parity, fallback,
anti-leakage, pre-freeze scenario design and all-false authority boundary.

Final verdict from all three: no P0/P1 in the local structural slice; local GO
to retain/audit it only. Evaluation, T057 and production remain `NO_GO`.

## Residual risks and required next actions

1. Do not execute the preregistration v1. Create a v2 plan whose scenario-level
   estimand matches the ensemble-mean solver authority and bind it to an
   independently signed admission envelope.
2. Recompute MC half-widths from bound raw replications or a reproducible
   calculation artifact; the local v1 validator checks the frozen summary and
   relationships but does not independently reproduce the statistic.
3. Obtain external trusted time, monotone attempt ledger, seal-before-truth
   proof, WORM/external CAS, service identities and signatures.
4. Qualify the real GPU in a fresh process on admitted non-holdout data. Do not
   use CUDA for solver/repricing/calibration/hard gates or fitting.
5. Run two independent clean PEP 517 builders; bind build-tool/wheelhouse hashes,
   SBOM, vulnerability/license scans, signature and Windows service launchers.
6. Return to fresh governed prospective CH acquisition. Current direct CH
   native-quarter-hour truth remains unavailable/unsupported; do not substitute
   Swissgrid imbalance, duplicated hourly values or DE proxy as direct truth.
7. Produce a new non-T057 local quality status and then a fresh auditable CH
   candidate only after the data/admission gates are real. T057 effective
   historical `n=1` remains inadmissible.

No production promotion is authorized by this handoff.
