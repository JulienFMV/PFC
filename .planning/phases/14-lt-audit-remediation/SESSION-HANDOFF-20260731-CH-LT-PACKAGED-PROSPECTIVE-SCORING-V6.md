# Session handoff - CH LT packaged prospective scoring V6

Date: 2026-07-31  
Branch: `fix/lt-audit-remediation`  
HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Workspace: `C:\Users\jbattaglia\PFC_LT`  
Production: strict `NO_GO`

## Outcome

Future-origin selection V6 is the current local structural rehearsal for the
existing V7 origin. It supersedes V5 without changing the sealed predictions
or creating another origin. The prospective scorer, native-hourly truth
builder, structural commitment builder and selection auditor now live in the
packaged `pfc_shaping.*` namespace; checkout scripts are compatibility wrappers.

V6 is source-bound and fail-closed, but it is deliberately non-countable and
non-authoritative. A byte-identical installed-wheel-equivalent unpack now
reproduces the full 105-file source inventory with exactly one wheel root and
zero checkout roots. This is local source/import evidence, not an independent
attestation of already-loaded code, a fresh supervised full-runtime receipt,
a supervised service, trusted time, independent registry or external-CAS
proof. No truth outcome, T057, candidate, publication or production promotion
was consumed.

## Selected identities

- V6 path:
  `.planning/phases/14-lt-audit-remediation/CH-LT-LOCAL-FUTURE-ORIGIN-SELECTION-V6-20260731.json`
- V6 SHA-256:
  `11966d1ee85ace46e97006fa74f8aab4789c71f7438de909ed71541aab480df7`
- selection ID:
  `5340bb8c5cd0c2ba65f346c70e75ba7aa6bb17a5553df353f476ef8c50f03984`
- origin timestamp / first delivery:
  `2026-07-31T07:20:27Z` / `2026-07-31T22:00:00Z`
- commitment SHA-256 / ID:
  `67cc6fe722586cd4a02149252d1af7fbbb35e90a89e8bc7965f8d0bf87eda9b4` /
  `799ed71cdeddb9bf81fdc63574f352174bd2ac432c83008a25c63b4cf4094d2a`
- predictions SHA-256 / ID:
  `859c4622eb5b699f22242af0121141729d6c00472b6b9070bdb87c2ba9132302` /
  `fb2f7dc3f5024e5cca92923c4f3a2e47b763a4528ad1df7bb7cc49eaf4f8ec84`
- scoring contract SHA-256 / ID:
  `d84d7ebc2f3c6aa4b820fab26f0a5de18940a6d35af8e796323025eab694ea4d` /
  `ae85de71a3745b4ea8b9704090f9bd742e26a41161c2d0b804f4888fb372653e`
- functional-contract SHA-256:
  `a8cb845385d7da16b0e486b2fa16330d312c2d2f6a8fba2028e2e0d08a11fbbe`

The commitment contains 36 complete delivery months and 108 predictions
(slow/central/fast). The current prediction set contains no negative price:
0/108 rows, minimum approximately `10.9366 EUR/MWh`. This is a model-risk
diagnostic, not evidence that negative prices are impossible.

## Bound package sources

- `pfc_shaping/validation/ch_lt_prospective_hourly_scoring.py`:
  `7faf157e6e83c78c5de3b1419a8418ea1ad7fc8886297f0e7322ee08e37c1cac`
- `pfc_shaping/cli/score_ch_lt_structural_prediction_commitment.py`:
  `481a7c21dfeefbf7a7f998d634a4924632320b12d21b620fe55fb30170992cb0`
- `pfc_shaping/validation/ch_lt_native_hourly_truth_bundle.py`:
  `698a06b698277d1fe542337c93df5f77831e7f3f74f7767ffa58dd2d05caaedf`
- `pfc_shaping/validation/ch_lt_structural_prediction_commitment.py`:
  `9949f8e818aba79d198853a03208e526fe90ef4c6ef937917bfe676aacbf8bef`

Related current delta also includes
`pfc_shaping/validation/ch_lt_local_future_origin_selection.py`, the four
`scripts/*ch_lt*` compatibility wrappers, `scripts/run_workspace_local.py`,
`tests/test_build_ch_lt_native_hourly_truth_bundle_script.py`,
`tests/test_ch_lt_prospective_hourly_scoring.py` and
`tests/test_lt_package_contract.py`.

## Fail-closed lifecycle

Before any outcome-bearing path is opened, the builder verifies V6, the
commitment, sealed predictions, target inventory and scoring contract. Both
the actual local wall clock and the caller-declared read time must be after the
full target-month end, and the declared time cannot be in the future relative
to that wall clock. The ledger/bronze bytes are read only after those checks.

The truth bundle and `truth-publication-receipt.json` are published together
through same-parent staging and rename. The receipt is post-read evidence, not
authorization to read. The scorer revalidates the chain, maturity and exact
fixture/real biconditional before opening truth, and score publication supports
only exact empty-staging resume.

Swiss scientific truth remains native hourly. Quarter-hour values are model
transport/output until an independently admitted Swiss market transition.

## Terminal selected evidence

All commands below ran from the canonical workspace through
`scripts.run_workspace_local`, with repo-local mutable paths and every authority
flag false.

1. V6 audit, run `fvaudit6`: target exit 0 and status
   `VALID_LOCAL_FUTURE_ORIGIN_REHEARSAL_V7_PACKAGED_MODULE_SOURCES_BOUND_NONCOUNTABLE_NO_GO`.
   Execution/supervisor receipt SHA-256:
   `2ebe566e52e25c0cbfcc336400585d7a86f8f4b2cf81b7779b0df1121f33c3a6` /
   `28940f7e3f0ba681accb5dc0adb0c105f5f719751c06f3b88638bd1f3855f5ff`.
   Target command:
   `build/pytest-runtime-v1/python.exe -B -m scripts.audit_ch_lt_local_future_origin_selection --registry <absolute-V6-path> --expected-registry-sha256 11966d...df7`.
2. Truth/scoring tests, run `ftruth36`:
   `17 passed in 12.28s`. Execution/supervisor receipt SHA-256:
   `fdf0d8ab2a88544d0527916a166708bf43e91b42b51fad3a2652c409f0fc8820` /
   `b029fe90c38e469ffeabc6a8aeed0f18dde5326586501c913548deb6c8dfd119`.
   Files:
   `tests/test_build_ch_lt_native_hourly_truth_bundle_script.py` and
   `tests/test_ch_lt_prospective_hourly_scoring.py`.
3. Package contract, run `fpack38`:
   `26 passed in 2.69s`. Execution/supervisor receipt SHA-256:
   `996ac994f66cb430777345867634c855328dfcf2407faea9469e7ac00c46ce71` /
   `e2ade6c2ff54c97cb2290c63131d317b47d5c56e1a3e94d66ad4873e63897517`.

After final wrapper lint, the current bytes were replayed directly with the
physically distinct repo-local `build/pytest-runtime-v2-final/python.exe`:
V6 audit exit zero, truth/scoring `17 passed in 12.06s`, package contract
`26 passed in 2.79s`. Captured `sys.path` contains five repo-local entries;
the canonical checkout root occurs exactly once and every entry is below the
workspace. This direct replay is current local evidence, not a supervised or
installed-wheel authority receipt.

## Historical non-conclusive runs and terminal reruns

- Publication run `fpub38`: `235 passed, 2 skipped, 1 failed`. The sole
  candidate-assembler capstone failure followed a repo-local temporary product-
  replay access failure and produced a fail-closed `CRITICAL` gate. This run is
  not green and does not demonstrate a product regression. Execution/supervisor
  SHA-256:
  `7f62ce7858913c084ea45c8aecfc2f197f11d092e4ddc0b8b7781baf722f9c78` /
  `3fd6b0ff2f7a086bdd3b9c033f9c7f77887a78c53234ca5c363835ce33f2264f`.
- Runtime/publisher run `fpack36`: `69 passed, 12 skipped, 1 deselected,
  6 failed`. The failures are Windows access/cleanup failures below the
  repo-local run temporary tree; one cleanup failure masks the expected
  fail-closed assertion. This run is not green and is not product-failure
  evidence. Execution/supervisor SHA-256:
  `cb2698b93f554db24da475fa199261780d1aa8b4db02dd87557fc75ce5a5a6b8` /
  `b716b691761535bfb5656cf00d9a04e933514a62328e20d6f17b762b87541e3b`.
- Later supervisor attempts `ftruth39`/`fpack39` and
  `ftruth40`/`fpack40` resolved a relative target through the external parent
  interpreter and are non-selectable. Parallel `ftruth41`/`fpack41` then
  overlapped a concurrent source rewrite: tests themselves reached `17/17`
  and `26/26`, but one receipt stayed non-terminal and the other rejected an
  execution-identity change. The final direct replay above occurred only after
  the V6 source hashes were stable again. None of these attempts is authority.

Do not retry these ACL-sensitive broad matrices repeatedly on the managed
laptop. Qualify the skipped/ACL-sensitive branches on the governed standard-
user CI runner with its own checkout and policy.

The demonstrated inherited-ACL scratch defects were then corrected without
changing the monthly capstone formulas, tolerances, products or gates. Final
repo-local split reruns are green:

- prospective scoring, truth and package: `63 passed in 32.12s`;
- runtime, packaging and publisher: `110 passed, 12 skipped, 2 deselected in
  13.90s`;
- publisher ACL targets: `7 passed in 0.44s`;
- candidate, evidence and quality: `92 passed in 101.41s`;
- governed release: `37 passed in 27.79s`;
- atomic promotion: `116 passed, 2 skipped in 135.59s`.

The exact governed truth/scoring runtime was also replayed after the final
roast and reported `17 passed in 12.61s`. A run under the ambient Python
(numpy 2.1.3, pyarrow 20.0.0, tzdata 2025.3) failed on the sealed provider
runtime fingerprint. The selected runtime (Python 3.11.13, numpy 2.0.2,
pandas 2.3.3, pyarrow 21.0.0, tzdata 2026.3) passed. IT/Operations reclassified
the ambient failure as the expected fail-closed behavior, not a product defect.

Two fresh wheels at
`build/wheel-dist-prospective-g/fmv_pfc_lt-0.14.0-py3-none-any.whl` and
`build/wheel-dist-prospective-h/fmv_pfc_lt-0.14.0-py3-none-any.whl` are byte-
identical, 109-member artifacts at SHA-256
`0e7d69b14493a6959222709835a0ca00ea373b99df50a24344cf35bd97e80f37`.
Both wheel audits pass with `promotion_eligible=false` and embedded source
revision
`5f544d90a81077d7aec8d893339c5f2ca0928489e9e950ee02d65c9797b44da7`.
The isolated unpack at
`build/wheel-unpack-prospective-g/fmv_pfc_lt-0.14.0` reports
`SEALED_INSTALLED_FULL_SOURCE_IMPORT_CLOSURE_VERIFIED`, 105 sources, one wheel
root, zero checkout roots, no `scripts.*` or `pfc_shaping.ct.*` module, and the
exact pinned dependency versions. Wheel installation used the standard
`wheel unpack` purelib layout because pip's private temporary directory hit a
managed Windows ACL; this is not a pip-installer receipt.

The selected closure is
`.planning/phases/14-lt-audit-remediation/CH-LT-FUTURE-SCORING-IMPLEMENTATION-CLOSURE-V3-20260731.json`,
closure ID
`cedd9356744eddda6aec87666a64017f3946b5561219800b31c74e4e35d6fa1f`,
SHA-256
`a7d6c37044845fcb5a162d89c4aa6c8fafc4a2b64afb3616538051bc88f9652e`.
It deliberately records that the prior launcherless runtime receipt binds an
older project wheel and is substrate context only. It does not claim a fresh
full-runtime admission receipt for the current wheel.

## Independent roasts and disposition

- Security/Governance final delta: P0/P1/P2 = `0/2/2`. The two P1s are
  attestation boundaries, not demonstrated outcome leakage: the source closure
  does not independently prove the bytes of code already loaded before the
  check, and Closure V3 lacks a fresh supervised execution receipt binding the
  current wheel, command, interpreter, `sys.path`, stdout and exit code. ACL
  scratch hardening closes the observed Windows defect but is not hostile-
  same-identity isolation or a proof against every ABA/reparse race.
- IT/Operations final delta: P0/P1/P2 = `0/2/4`, decomposed as zero product P1,
  one fresh installed-runtime-attestation P1 and one external operational
  blocker treated as P1 for absent scheduler/SLA/owner. Wheels and isolated
  import are local-engineering GO; complete runtime admission and production
  remain NO_GO. Lease/retry/watermark/alerting, crash/rollback drills, CI/ASR,
  SLOs and observability remain open.
- Quant/Data final delta: P0/P1/P2 = `0/0/0`. V5/V6 preserve the same origin,
  36 targets and 108 predictions; solver level authority, hourly/DST contract,
  metric separation and scenario non-pooling remain unchanged. There are still
  zero independent countable origins and therefore zero scientific quality
  proof. Fresh PIT CH/EEX evidence, rolling-origin power/multiplicity, negative-
  price support, probabilistic calibration/scenario coherence and economic
  capture remain open.

## Next actions

1. Admit the current wheel in a fresh immutable runtime through a minimal
   independent bootstrap, verify before project/pandas imports, and emit a
   supervised receipt binding command, interpreter, wheel, dependencies,
   loaded code, `sys.path`, stdout and exit code.
2. Define an owner-approved native-hourly capture schedule, provider-lag SLA,
   watermark, retry/lease rules and alerting; keep truth closed until full-month
   maturity.
3. Register fresh outcome-blind origins through an independent linearizable
   authority with trusted time, signature and builder-inaccessible CAS/WORM.
4. Bind fresh point-in-time CH EEX level inputs and exact origin-available
   product inventories; monthly solver remains the sole level authority.
5. Accumulate the preregistered number of independent rolling origins, then
   score level, shape, tails, calibration, scenario coherence and economic
   capture. Keep T057 sealed until its successor protocol permits one-shot use.
6. Treat absent negative predictions as a testable support/calibration issue;
   do not force negative values merely to match a narrative.

No commit or staging was performed. The protected
`data/eex_forwards_history.parquet` remains untouched at SHA-256
`21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

## Final changed implementation surface

- Packaged prospective modules:
  `pfc_shaping/validation/ch_lt_structural_prediction_commitment.py`,
  `pfc_shaping/validation/ch_lt_local_future_origin_selection.py`,
  `pfc_shaping/validation/ch_lt_native_hourly_truth_bundle.py`,
  `pfc_shaping/validation/ch_lt_prospective_hourly_scoring.py` and
  `pfc_shaping/cli/score_ch_lt_structural_prediction_commitment.py`.
- Compatibility wrappers under `scripts/` for the selection audit, truth
  builder, commitment builder and scorer.
- Package contracts:
  `pfc_shaping/package_contract.py`, `scripts/check_lt_wheel_contract.py` and
  `tests/test_lt_package_contract.py`.
- Standard-user scratch hardening:
  `tests/conftest.py`, `pfc_shaping/publisher_runtime_admission.py`,
  `pfc_shaping/calibration/monthly_curve_capstone.py` and the associated
  publisher artifact tests.
- No file under `pfc_shaping/ct/` or Power BI was changed. No commit or staging
  was performed.
