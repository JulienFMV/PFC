# PFC Operations Runbook

## 1) Preconditions
- IT provisions one absolute, independently admitted CPython interpreter and
  one read-only positive-inventory `fmv-pfc-lt` purelib closure under each
  phase-specific service identity. The interpreter must reach that closure on
  its default isolated `sys.path`; editable installs, user/system site,
  `PYTHONPATH`, `.pth` files and checkout imports are forbidden.
- Generated `pfc-lt*.exe`, `.cmd`, `.bat` and `.ps1` launchers are forbidden.
  On Windows, invoke only the absolute admitted interpreter with `-I -B -m`.
- Governed data, release, workflow, evidence, failure, journal and trust-key
  roots are explicit deployment mounts, not workstation or repository paths.
- The exact EEX workbook and its signed acquisition contract are readable by
  the builder identity.
- Each mutating identity receives a pre-provisioned failure root that is
  disjoint from every release, candidate, data, workflow, evidence, journal and
  key path. It must permit directory creation, file creation, hardlink creation,
  read/list and deletion below the exact `<failure-root>/<run-id>/<phase>` path.
- Builder, finalizer, registrar, auditor, promoter and rollback identities have
  separate least-privilege ACLs and only the key material required by their role.
- Before Builder starts, an IT provisioning identity creates and pins both
  domain markers. Before each run, IT precreates that run's `requests`,
  `audit-results` and `promotion-results` namespaces with separate ACLs. Runtime
  service identities must not have create rights on the common run directory.

### 1.1) Runtime artifact admission

IT admits one immutable wheel hash, not a mutable source checkout. CI must:

1. build the wheel twice from the same reviewed Git commit in two clean,
   independent source trees and require byte-identical SHA-256 values;
2. run `scripts/check_lt_wheel_contract.py` on both wheels and retain both JSON
   reports, the Git commit, embedded source revision and build-tool versions;
3. generate and retain an SPDX or CycloneDX SBOM for the wheel and its complete
   transitive wheelhouse, with every dependency pinned by filename and SHA-256;
4. scan the wheel, SBOM and wheelhouse under the approved FMV vulnerability and
   license policy, then sign the wheel hash and admission manifest with the
   organizational release-signing service;
5. build a content-addressed purelib closure directly from the retained offline
   wheelhouse with hash verification, without invoking pip installation and
   without generating console scripts; reject `.pth`, editable-install residue
   and any project `.exe`, `.cmd`, `.bat` or `.ps1`;
6. attest under every Builder, Finalizer, Registrar, Auditor, Promoter and
   Rollback identity that
   `C:\FMV\PFC-LT\runtime\python.exe -I -B -m pfc_shaping.cli.governed_release --version`
   reports the admitted embedded source revision and that the closure inventory,
   interpreter hash, ABI and distribution hash equal the signed admission.

The wheel auditor's `promotion_eligible=false` is intentional: package
structure alone never authorizes a candidate or production deployment.

### 1.1.1) Corporate-workstation local-quality runtime (not production)

The launcherless runtime under `C:\Users\jbattaglia\PFC_LT\build` is a local
engineering surface for model execution and quality inspection. It requires no
admin rights or Defender exception and must never be mounted as a production
runtime. The current evidence is v22. V14's local byte-provenance closure claim
is invalidated by D170; v15 and v16 are fresh failed admissions retained as
negative evidence. V17 first closed archive payload provenance; v18 hardened
lifecycle scripts, stable reads, physical aliases and exact `sys.path`; v19
also binds the top-level caller-held receipt physically outside the prefix.
V20 is a retained failed prefix audit because `PYTHONPYCACHEPREFIX` displaced
406 Conda-generated files outside the prefix. V21 retained those files inside
a fresh prefix and embedded the durable EEX signing-request workflow, but its
reused lock/spec still named archive payloads under the user `.conda` cache;
it is retained as a local P1 negative and is not selectable. V22 rebuilds the
lock and explicit spec over the retained archive bytes under
`build/conda-pkgs-runtime-v6`, creates a fresh exact prefix from those
repo-local URIs and supersedes v21 for local quality.
V7, v9, v11 and v12 are superseded, while v8, v10 and the
timed-out v13 prefix are incomplete/rejected evidence. Preserve all
superseded/rejected prefixes and receipts until an IT incident/retention
decision; do not repair or relabel them in place.

That revocation is a governance decision, not a technical execution deny. The
preserved v11 prefix and receipt remain executable by the workstation user and
contain the superseded transition check. Never select them operationally. IT
must independently quarantine/deny execution or enforce an external immutable
runtime allowlist before this revocation can be treated as effective. This
user-space project must not change ACLs, delete evidence, request admin rights
or weaken Defender/ASR to simulate that control.

Before every local command, the caller must bind the exact external receipt
bytes through both variables:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH = `
  'C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260729-v22.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256 = `
  '2e45ce409c027395b38096ab5718425d459917c83b66910eb7ddbf13e1d766bf'
```

Run only the absolute Python module route:

```powershell
& 'C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v22-repolocal-archive-base\python.exe' `
  -I -B -m pfc_shaping.cli.governed_acquisition_builder --help
```

The v22 receipt schema is `fmv_lt_launcherless_local_runtime.v4`. It must bind
the caller-held Conda-prefix replay receipt SHA-256
`8155d0878a669437a91072ce71f4083b14bd33be31846e0df0c7c832776db571`,
and report `local_quality_authorization=true` and
`production_authorization=false`. The public promote and rollback APIs are
hard-disabled for this runtime even if all local receipt bytes match. Do not
change the receipt or code to bypass the guard. A future transition runtime
requires a distinct independently signed, IT-admitted attestation.

The installed v22 wheel includes the production-transition guard at the CLI,
high-level workflow and atomic public-API layers. The launcher admission probe
must run before even `--version`; the recorded v22 probe passed with the exact
wheel source revision. The installed EEX explicit-receipt probe crossed
admission, then failed closed on an intentionally absent trusted-time key with
exit 50 and no output. The observed warm local probes took 22.8 and 21.9
seconds respectively. Their import/admission ordering still requires an
explicit operations SLO and an
independently admitted pre-import supervisor or read-only service identity;
they do not authorize skipping or caching admission under the workstation
identity.

Current local-quality successor v40 uses runtime receipt schema
`fmv_lt_launcherless_local_runtime.v5` and Conda prefix receipt schema
`fmv_lt_launcherless_conda_prefix_build_receipt.v3`. The v3 prefix identity
must incorporate the exact SHA-256 of the standard-user confinement receipt
and its external-guard digest. The v5 runtime receipt must expose that same
identity, and the installed launch-time validator must reread the caller-held
confinement receipt, verify its exact bytes/policy/prefix/spec/guard/mutable
paths, and derive the v3 prefix ID again. Path chronology or a matching prefix
string is never sufficient provenance.

Build order is fail-closed:

1. build and independently replay one canonical Conda archive lock over every
   retained `.conda` or `.tar.bz2` archive; parse and validate `info/index.json`,
   `info/paths.json`, `info/link.json` and every payload byte; materialize its
   exact local `@EXPLICIT` spec; reject every pre/post link or unlink lifecycle
   script before any prefix is accepted. Every cache root, archive path and
   materialized file URI must resolve under one governed repo-local `build/`
   archive root; a readable `.conda`, `AppData` or `ProgramData` payload path is
   still forbidden;
2. create one new prefix only through
   `python -B -m scripts.build_repo_local_conda_prefix`; never invoke `conda
   create` directly on this workstation. The recipe redirects `HOME`,
   `USERPROFILE`, `CONDARC`, Conda env/cache paths, `TEMP` and `TMP` to a fresh
   repo `build/` namespace, sets `CONDA_REGISTER_ENVS=false`, and guards the
   real user `.conda/environments.txt` and `.condarc` before and after the
   child. Never solve, use network or target an existing runtime namespace. Do
   not set `PYTHONPYCACHEPREFIX` during prefix creation: Conda-generated
   bytecode belongs inside the prefix inventory;
3. capture `scripts.build_launcherless_python_runtime_manifest` before the
   first execution of the target `python.exe`, then compare installed bytes
   with the independently parsed archive payloads and their explicit Conda
   transformations; segregate generated noarch files as non-runtime and write
    a caller-held prefix-build receipt. Pass the exact standard-user build
    receipt path/SHA to `audit-prefix`; require schema v3 and reject any prefix
    receipt that does not derive its ID from that confinement receipt;
4. build two byte-identical project wheels and audit both;
5. copy retained publisher inputs into a fresh repo-local staging namespace
   such as `build/runtime-inputs-<date>-repolocal-*`; point `TEMP` and `TMP` to
   fresh `build/` children. Do not write to `AppData`, request elevation or use
   an ASR/Defender exception. On this workstation, archive audit and runtime
   assembly require the existing read-only Anaconda interpreter that provides
   `zstandard`; do not install that dependency into another environment;
6. run `scripts.build_launcherless_local_runtime` with the exact prefix,
   project wheel, repo-local publisher closure/receipt, original additional
   wheels, caller-held Python manifest/hash, caller-held Conda prefix-build
    receipt/hash, `uv.lock` and a new runtime-receipt path. Require runtime
    schema v5 to carry the exact standard-user receipt identity through the
    Conda prefix record;
7. validate the runtime receipt independently, require prefix-root count zero
   and governed-closure count one, require the top-level receipt and all nested
   caller-held evidence to be physically outside the prefix, then run the
   installed admission and public promote/rollback rejection probes;
8. retain stdout/stderr, durations, exit codes and every partial/rejected
   namespace.

The closure staging directory is resumable only when every existing byte is
exact; divergence fails closed. Conda-prefix creation is still neither atomic
nor resumable: v13 timed out with a partial 19-record prefix and is retained as
negative evidence, while v14 was created in a new namespace. A parent timeout
is not a terminal failure proof: check for a still-running child and wait for
its terminal state before retrying. Never overlap retries against the same
prefix or receipt path. V15 rejected valid `noarch: generic`; v16 rejected
Conda-patched target dependency strings. Those false rejections were corrected
only in fresh source/wheel/runtime artifacts. V22 assembled from the repo-local
lock/spec and then passed installed launch admission. Its exact
`sys.path` is `Lib`, `DLLs`, then one `governed-site-packages`; no phantom
`python311.zip`, prefix root, checkout or user/system site is allowed.

### Standard-user Conda prefix recipe and incident handling

After the mandatory canonical cwd/Git-root guard, use a fresh namespace:

```powershell
python -B -m scripts.build_repo_local_conda_prefix `
  --conda-python C:\ProgramData\anaconda3\python.exe `
  --explicit-spec C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-explicit-20260729-v22.txt `
  --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-<fresh>-base `
  --work-root C:\Users\jbattaglia\PFC_LT\build\conda-prefix-work-<fresh> `
  --receipt-output C:\Users\jbattaglia\PFC_LT\build\conda-prefix-build-<fresh>.json `
  --timeout-seconds 600
```

The external Anaconda interpreter is read-only. The receipt must report
`PASS_REPO_LOCAL_MUTABLE_PATHS_NOT_PRODUCTION`, target exit zero,
`prefix_ready=true`, `external_guard_unchanged=true`, and every item under
`mutable_paths` below canonical repo `build/`. It never grants runtime,
scientific, publication or production authority. Continue with manifest and
archive-prefix audit only after independently hashing this receipt. Every new
runtime-prefix, work-root and receipt path must pass the no-link/no-reparse
check on all lexical parents before any directory creation or subprocess.

If the process times out, exits nonzero, leaves a partial prefix or reports an
external-guard change, retain the work root, prefix and failure receipt as
quarantined negative evidence. Do not repair or delete the real user registry,
do not retry the same prefix/work/receipt names, and do not assemble a runtime
from those bytes. Record the external path, before/after evidence, command
identity and terminal cause in the session handoff and notify IT/Security when
the real profile changed. No admin, ACL or Defender exception is permitted.

### CH LT origin-registry protocol v2

Protocol v1/runtime v37 are negative, superseded evidence because the v1 wire
contract omitted slot-window cross-field checks and its direct Conda build
touched the real user environment registry. Audit only canonical protocol v2:

```powershell
python -B -m scripts.audit_ch_lt_origin_registry_protocol `
  --protocol C:\Users\jbattaglia\PFC_LT\.planning\phases\14-lt-audit-remediation\CH-LT-ORIGIN-REGISTRY-PROTOCOL-DRAFT-V2-20260730.json `
  --expected-protocol-sha256 6ea896ccdb35414b52237f2bcf1065755c3c10444b308ce905b60f472e68c697
```

The expected local status remains an incomplete outcome-blind `NO_GO` with
zero countable origins and every authority false. Do not create a local JSON,
SQLite or file-lock substitute for the missing external registry. Before a
real origin can count, IT must commission the independent compare-and-append
service, FMV must freeze the exact UTC schedule, and a conformance verifier
must recompute request ID, request-byte hash, schedule entry, trusted origin
time, slot-window inequalities and registry commit deadline. Structural target
universe and ex-ante evaluation-mask rule are origin commitments; a realized
maturity/truth mask is forbidden until a separate externally admitted maturity
event. Missed slots remain missed and nonreplaceable.

The source-only module
`pfc_shaping.data.ch_lt_origin_registry_reference` is the executable protocol
reference-only protocol-semantics drill. It is deliberately excluded from the governed LT wheel
and carries classification `NON_PRODUCTION_TEST_ONLY_NEVER_COUNTABLE`. Its
SQLite state may be created only below repo `build/` by tests or an explicit
local drill. Its registry-domain hash is deliberately derived from
`FMV_CH_LT_CONFIRMATORY_ORIGINS_V2_NON_PRODUCTION_REFERENCE_V1`, never the
future production logical domain. A production verifier that correctly binds
its own domain must reject these bytes independently of the signer. It
exercises four
disjoint Ed25519 roles (request, schedule,
trusted time, registry), exact canonical signed request bytes, request-ID and
request-byte hashes, caller-resolved commitment hashes, the frozen schedule
entry, origin/commit/delivery chronology, linearizable compare-and-append,
global slot/origin/operation/request uniqueness, immutable rejection records,
exact retry/lookup and fresh nonce-bound signed HEAD observations.

Run the conformance and packaging-boundary tests only through the workspace
supervisor, with a fresh run ID:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe -B `
  -m scripts.run_workspace_local --run-id <fresh-origin-reference-id> `
  --wall-timeout-seconds 900 -- `
  C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe -I -B -m pytest `
  tests\test_ch_lt_origin_registry_reference.py `
  tests\test_ch_lt_origin_registry_protocol.py `
  tests\test_lt_package_contract.py `
  -q -p no:cacheprovider
```

Do not replace either interpreter with generic `python`, a user `.conda`
interpreter, or an external path when producing the selected local receipt.
The qualified source-test runtime binds the canonical checkout exactly once
through its repo-local `python311._pth`; production packaging must continue to
exclude this reference module.

The signed reference receipt uses the deliberately incompatible schema
`ch_lt_origin_registration_receipt.non_production_reference.v1`, the separated
reference registry domain, and the signed value
`countable_prospective_origin=false`. The local assessment therefore exposes
both `receipt_claimed_countable_prospective_origin=false` and effective
`countable_prospective_origin=false`, with external/scientific/production/
promotion authorities false. It exercises selected v2 request, schedule and
chronology semantics; it is not an exact production receipt-wire
implementation. Never project a reference receipt into the prospective
ledger, candidate evidence or external registry cache. A real origin still
requires an independently operated remote service, WORM/CAS, trusted time,
approved schedule, active/historical keyrings and fresh HEAD; the reference
database proves local interface semantics only.

For explicitly allowlisted Python build/audit/test modules, use the laptop
standard-user harness to redirect the known mutable destinations of Python,
pip, Conda, pytest and the listed scientific libraries:

For `scripts.build_launcherless_local_runtime`, the harness also rejects every
explicit runtime input or output outside `build/` before target execution
(`uv.lock` is the sole root-level exception). In particular, an `AppData`
wheelhouse or receipt is invalid even when it is readable by the current user.

### CH origin/target/mask installed dry-run (runtime v36)

The selected local-quality runtime is
`build/conda-runtime-v36-origin-mask-schema-v2-base`; its caller-held receipt
is `build/launcherless-runtime-receipt-20260730-v36-origin-mask-schema-v2.json`
with SHA-256
`a23f1af134aa374db038aba5c37935167d4fd2f6b471a64ef72fe6fdf605bbe4`.
Run the installed module only from the canonical repo cwd and only with
`-I -B -m`. Supply the exact caller-held receipt path/hash explicitly to the
child and keep `TEMP`/`TMP` below `build/`:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH = 'C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260730-v36-origin-mask-schema-v2.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256 = 'a23f1af134aa374db038aba5c37935167d4fd2f6b471a64ef72fe6fdf605bbe4'
$env:TEMP = 'C:\Users\jbattaglia\PFC_LT\build\temp-origin-mask-v36'
$env:TMP = $env:TEMP

build\conda-runtime-v36-origin-mask-schema-v2-base\python.exe -I -B -m pfc_shaping.cli.audit_ch_lt_origin_target_mask_inventory --evidence-root C:\Users\jbattaglia\PFC_LT build --origin-as-of-utc 2026-07-30T12:00:00Z --output C:\Users\jbattaglia\PFC_LT\build\ch-lt-origin-target-mask-inventories\structural-dry-run-20260730T120000Z-schema-v2.json

build\conda-runtime-v36-origin-mask-schema-v2-base\python.exe -I -B -m pfc_shaping.cli.audit_ch_lt_origin_target_mask_inventory --evidence-root C:\Users\jbattaglia\PFC_LT audit --inventory C:\Users\jbattaglia\PFC_LT\build\ch-lt-origin-target-mask-inventories\structural-dry-run-20260730T120000Z-schema-v2.json --expected-inventory-sha256 0dcc3411cb962e5ba4df2e36ea7bf67d97e0c56d1dcf3cb32a2684e26016c7d1
```

The workspace runner deliberately scrubs ambient runtime-authority variables;
do not weaken that isolation. Durable installed build-then-audit provenance is
the `otm36p3` pytest receipt, which supplies the exact variables only to its
v36 child. The artifact is a non-countable structural dry-run with every
scientific/execution/publication/production authority false. Never substitute
the obsolete v9/AppData route, request admin/Defender rights, or interpret
successful local generation as promotion evidence.

```powershell
python -B -m scripts.run_workspace_local --run-id <fresh-portable-id> `
  --wall-timeout-seconds <governed-cycle-budget> -- `
  python -B -m pytest <tests> -q -p no:cacheprovider
```

The harness verifies the laptop's literal canonical cwd/Git root. Execution
receipt schema v6 and supervisor receipt schema v1 require a fresh portable
run ID of at most 16 characters. The minimal parent captures the exact runner
source into `build/workspace-local-supervisors/<id>`, writes a one-shot random
capability bound to its PID/start token, direct parent relation, captured
source hash, exact worker argv hash and wall budget, then starts that worker
suspended. The worker publishes an exclusive hash-bound admission sidecar
before consuming and deleting the capability. On Windows the parent assigns
the worker to a
`KILL_ON_JOB_CLOSE` Job Object before resume; on POSIX it creates a new process
group. Direct injection of the internal-worker flag or reserved capability
environment fails closed.

The worker creates the descriptive receipt/cache namespace below
`build/workspace-local-runs/<id>` and a separate short pytest root at
`build/wpt-<id>`. Supervisor bootstrap and worker target `HOME`, `USERPROFILE`,
`APPDATA`, `LOCALAPPDATA`, `PROGRAMDATA`, Conda/pip/uv caches and all temporary
paths remain below their respective repo-local roots. The external
`C:\ProgramData` interpreter may still be addressed explicitly for read-only
capture; its ambient environment variable is never inherited. All mutable
roots are fresh, preflighted and no-reuse; never repair ACLs or retry a failed
ID in place. Retain supervisor, worker, short pytest and terminal-output roots
together until a governed retention operation.

`--wall-timeout-seconds` is a mandatory finite cycle budget with default 1800
seconds and maximum 86400 seconds. It is the complete-cycle admission budget:
supervisor source/capability preflight, worker preflight, target execution and
post-execution byte verification must all finish before it. The worker uses
the same upper bound for its target. If postflight or the first terminal fsync
crosses the deadline, the parent rewrites the terminal decision to exit 124
with no authority. Job termination, evidence fsync and handle cleanup may
overshoot the budget; the receipt records the measured overshoot and explicitly
sets `strict_return_bound=false`. Timeout or interrupt terminates the complete
Job/process group, requires zero active members and writes a non-authoritative
terminal receipt. A zero target exit is recorded as
`TARGET_EXIT_ZERO_NOT_AUTHORITY`, never as scientific or production `PASS`;
nonzero, timeout, interrupt and descendant leakage remain terminal negative
evidence.

Both receipt schemas set production, promotion, scientific, evaluation and
runtime authorities to false. Parent and worker scrub known credential/
authority variables, ambient Python/pytest injection paths and external FMV
data-root overrides, force pip/uv/Conda offline, and persist only a redacted
command plus its hash. This is environment-denylist isolation, not a
filesystem sandbox.

The child stdout and stderr are now drained concurrently into exclusive
repo-local files, capped at 64 MiB per stream, fsynced, replayed to the
terminal and recorded by retained/full-stream hash and byte count. Any
truncation produces `TARGET_OUTPUT_CAPTURE_LIMIT_EXCEEDED` and runner exit 2,
even when the child returned zero. For pytest, the harness forbids a
caller-selected JUnit/native-result path and arbitrary explicit plugins,
injects its own JUnit plus append-only native-result writer, then independently
cross-checks target exit, selected, passed, failed, error, skipped, xfail,
xpass and deselected counts. Missing, malformed, re-bound, inconsistent or
oversized evidence produces `TARGET_EVIDENCE_INVALID` and runner exit 2. The
exact pytest basetemp remains a fresh child of the short preflighted/
identity-bound parent.

Caller-selected pytest config/ini/root overrides are forbidden so `addopts`
cannot inject an extra plugin. This includes separated and compact `-c`, the
pinned `--config-file` alias, `-o`/`--override-ini`, root/confcutdir overrides
and pytest `@argsfile` expansion. Evidence reads compare path `stat` with
handle `fstat` before and after the read. The runner source, selected
interpreter, Git HEAD and branch are hashed/recorded before the child and
revalidated after it.

Execution receipt v6 also records a canonical aggregate over the current dirty execution
tree: every Python file below `pfc_shaping/`, `scripts/` and `tests/`, plus the
root package/test/lock configuration. Each file is bounded, checked against
reparse and path/descriptor rebinding, and contributes path, size, SHA-256 and
physical identity to `execution_identity.source_tree`. The aggregate is
recomputed after the child; editing any covered source or test during a run
makes the evidence fail closed. This same-user aggregate is local byte
identity, not an independent signature.

Output-drain threads have a bounded post-child EOF grace. Any descendant that
keeps inherited streams open, or remains active after the direct child exits,
is terminated through the Job/process group and makes the run fail as
`TARGET_DESCENDANT_LEAK_TERMINATED_NO_AUTHORITY`. Timeout returns 124;
supervised interrupt returns 130; both require `tree_termination_confirmed`.
Windows Job accounting records aggregate user/kernel CPU, peak process/job
memory, page faults and process counts. I/O operations/bytes come from
`JobObjectBasicAndIoAccountingInformation`, not the zero-prone extended-limit
structure. Worker receipts
also record bounded logical regular-file usage before/after without following
links or reparse points.

Retain `target-stdout.log`, `target-stderr.log`, `pytest-results.xml`,
`pytest-native-result.json`, `execution-receipt.json` and
`supervisor-receipt.json` together. The supervisor receipt hashes the terminal
execution receipt and proves the admitted one-shot capability is absent. The
Windows E2E suite exercises zero exit, timeout before admission, timeout after
admission, abrupt supervisor death with append-only reconciliation, and abrupt
Job-owner death with both child and descendant. If an external controller dies
while a v6 receipt is pending, Job-handle closure kills the tree. After exact
runner/target PID+start-token absence is observed,
`--reconcile-stale-run-id <id>` writes a new exclusive
`reconciliation-receipt.json`; it never edits the pending receipt, reuses the
namespace or treats PID absence without reuse protection as sufficient.

These same-user local observations make terminal claims machine-verifiable but
are not independent CI, signing, scientific or release authority. Interpreter/
runner hashes are self-observed local identity evidence, not an independent
attestation. Same-user pre-import replacement or target-code evidence forgery
remains outside this local harness's trust model and is forbidden for
production evidence.

The harness does not turn arbitrary explicit command arguments into a sandbox.
External preinstalled tools and source archives may be read without mutation
only for verified capture/copy below `build/`; all subsequent mutable work must
consume the repo-local bytes. Project executables and Playwright remain
forbidden on this laptop; required E2E/ASR qualification belongs on governed
CI.

Do not use this laptop harness for Conda exact-prefix creation, PEP 517/pip
wheel construction or installed v22 admission. Those flows retain their exact
dedicated recipes above/below: all mutable paths are repo-local, while the
installed admission deliberately consumes its hash-bound runtime receipt and
therefore must not pass through a helper that strips authority variables.
Likewise, governed CI uses its own path-portable runner and checkout; it does
not invoke this laptop-specific literal-root harness.

Current blockers that forbid production qualification:

- target code is imported from the same-user writable closure before the guard
  can verify it;
- the repo-local archives, preinstalled read-only Conda executable, lock,
  prefix receipt and their hashes remain same-user/local inputs without an
  independent signature, durable CAS/WORM or read-only service boundary;
- atomic base-prefix staging, kill recovery, active-runtime CAS and rollback
  are not implemented;
- Windows CI, ASR qualification, SBOM/signatures, structured logs and SLOs are
  absent for the main runtime;
- attempt5 still needs an immutable execution-provenance sidecar and formal
  closure of its first verifier scratch-residue incident.

Until those blockers are closed, a local `PASS` means only that observed bytes
were coherent at admission time. It is not an IT deployment approval.

#### Structural CH LT estimand audit

`pfc_shaping.cli.audit_ch_lt_estimand_contract` is shipped in the governed
wheel. It validates only the
hash-bound structural CH LT estimand/economic design; schema v1 can never admit
an evaluation or authorize production. Always pin the externally recorded
source document SHA-256:

```powershell
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.audit_ch_lt_estimand_contract `
  --contract C:\governance\CH-LT-ESTIMAND-DRAFT.json `
  --expected-contract-sha256 4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931 `
  --mode validate-draft
```

Exit 0 means structural validation only. The default `admit-evaluation` mode
must exit 3 with all authorities false; invalid bytes/hash/schema exit 2. When
retaining an audit, IT pre-provisions an absolute audit root and the
`phase14/ch_lt_estimand_contract_audits` child with least-privilege ACLs, then
passes absolute `--audit-root` and `.json` `--output` paths. The CLI refuses
implicit install-relative output roots, links/reparse points, wrong namespaces,
wrong suffixes and overwrite. Retain source-document, semantic, policy,
operation and installed-runtime source-revision identities from its JSON.

#### Structural CH LT compute-runtime audits

The governed wheel ships two read-only commands:

- `pfc_shaping.cli.audit_ch_lt_compute_runtime` validates the frozen CPU/GPU
  policy bytes;
- `pfc_shaping.cli.audit_ch_lt_compute_runtime_manifest` validates one closed
  local runtime manifest,
  every local receipt and every payload byte named by those receipts.

Neither command admits an evaluation. Exit `0` means only that the claimed
local structure and supplied bytes satisfy the packaged validator. Their JSON
must still report `execution_authorized=false`,
`production_authorization=false` and `promotion_gate=false`. Exit `2`, missing
output, timeout, malformed JSON, a changed blocker inventory or any true
authority flag is an operational alert and a hard `NO_GO`. A scheduler must
never translate exit `0` or `structure_valid=true` into admission, publication
or promotion.

Run only the module form through the absolute independently admitted
interpreter. Before each audit, verify the interpreter and closure hashes,
installed distribution hash, exact `sys.path`, module origins and embedded
source revision against the signed release manifest; use a clean allow-listed
environment, disable network and signing credentials, and record the service
identity, resolved interpreter, closure root, wheel hash, source revision,
command, start/end timestamps, exit code and exact stdout/stderr in an external
append-only operations journal. The generated console launchers and checkout
wrappers under `scripts/` are forbidden on a service host.

Policy validation always pins caller-held bytes:

```powershell
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.audit_ch_lt_compute_runtime `
  --contract C:\governance\CH-LT-COMPUTE-RUNTIME-DRAFT.json `
  --expected-contract-sha256 <externally-recorded-lowercase-sha256>
```

Manifest validation additionally pins the manifest and one absolute evidence
root. Repeat `--evidence ROLE=ABSOLUTE_PATH` exactly once for every required
role. V1 requires the eight receipts
`parity_qualification_report`, `fresh_process_bootstrap`,
`deterministic_operation_inventory`, `cpu_oracle_implementation`,
`cpu_oracle_runtime_fingerprint`, `prediction_seal`, `scenario_seal` and
`truth_open_receipt`; their eight bound payloads `parity_report`,
`bootstrap_report`, `operation_inventory`, `cpu_oracle_source`,
`cpu_oracle_runtime`, `predictions`, `scenarios` and `truth_dataset`; and
the three pre-freeze design artifacts
`probabilistic_scenario_and_mc_design_manifest`,
`monte_carlo_error_study` and `design_freeze_receipt`. Add
`frozen_shaping_weights` only for
`SHAPING_INFERENCE_FROM_FROZEN_WEIGHTS`.

```powershell
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.audit_ch_lt_compute_runtime_manifest `
  --contract C:\governance\CH-LT-COMPUTE-RUNTIME-DRAFT.json `
  --expected-contract-sha256 <contract-sha256> `
  --manifest C:\runtime-evidence\attempt-id\runtime-manifest.json `
  --expected-manifest-sha256 <manifest-sha256> `
  --evidence-root C:\runtime-evidence\attempt-id `
  --evidence parity_qualification_report=C:\runtime-evidence\attempt-id\parity-receipt.json `
  --evidence parity_report=C:\runtime-evidence\attempt-id\parity-report.json `
  <repeat-for-the-complete-exact-role-inventory>
```

All evidence paths must be absolute, physically distinct, mono-linked regular
files below the evidence root, with no symlink, junction or reparse-point
component. The service wrapper must enforce a wall timeout, process/job memory
limit and stdout/stderr size limit. The packaged validator bounds policy and
manifest JSON at 512 KiB, receipts/design at 2 MiB and each supplied bound
payload or frozen-weight object at 64 MiB. Larger scenario data must be reduced
to an independently specified and verified chunk/Merkle evidence protocol;
raising these limits ad hoc is forbidden.

Local receipt status strings do not prove trusted time or causal
seal-before-truth order. Before any future evaluation admission, an independent
identity must verify the payloads again, prove the pre-frozen scenario/Monte
Carlo design, enforce attempt uniqueness in a monotone ledger, attest
seal-before-truth ordering, sign the complete execution-context hash and issue
an external admission envelope. Production remains `NO_GO` without that
envelope and the ordinary SBOM, vulnerability, service-host and rollback gates.

### 1.1a) Local CH hourly curl diagnostic

`scripts/run_ch_lt_hourly_curl_retry_from_spec.py` is a checkout-only local
diagnostic, never a production acquisition entrypoint. For v2 contracts,
PREPARE and FINALIZE require a caller-held contract SHA-256 and exact prepared
metadata keys/false claims. The final output is
`ch_lt_hourly_curl_capture_receipt.v2` plus
`ch_lt_hourly_local_transform_record.v2`; it deliberately does not emit
`capture-spec.json` and therefore is not input to the governed acquisition
builder.

Current v2 contracts allow only `LOCAL_HOURLY_SHAPE_DIAGNOSTIC` and explicitly
forbid `EXTERNAL_ADMISSION_INPUT_CANDIDATE`. Curl execution is not supervised
or hash/version-bound, the CA bytes are not pinned for the full process, and
Windows certificate revocation checking is disabled by `--ssl-no-revoke`.
Every receipt must therefore state `transport_attestation=false`,
`certificate_revocation_check_performed=false`,
`builder_input_authorized=false` and
`external_admission_input_candidate=false`. Signing or copying these bytes
later does not cure those missing transport/product/PIT proofs; an external
admission requires a separately qualified capture chain.

The 15-minute frame exists only in memory as a fourfold forward-filled
transport proxy. It is not persisted bronze truth, must not be counted as
2,880 independent observations, and cannot support quarter-hour volatility,
tail, duration, uncertainty or sample-size claims. Retain all attempts and use
the frozen supersession registry to identify the one selected local diagnostic.

### 1.1b) Standard-user native-hourly CH capture for unsigned replay

`scripts.capture_public_energy_charts_lt` is the only checkout capture route
allowed by the workspace harness to produce a builder-compatible
`lt_provider_capture_spec.v1`. The harness accepts only role `epex_ch`, raw
cadence `60`, one explicit acquisition ID and an output below canonical repo
`build/`. `--ca-bundle` is required and must name an existing absolute file on
the canonical `C:` drive; the harness removes ambient
`PFC_REQUESTS_CA_BUNDLE` before launch. The capture is still a networked local workstation
observation: its clock, transport, product/session identity and namespace are
not external authorities.

Use one new run ID and one new output namespace. The window must be complete,
UTC-aligned and at most 32 days; do not use the T057 namespace or any sealed
future-holdout target:

```powershell
python -B -m scripts.run_workspace_local --run-id <fresh-id> -- `
  python -B -m scripts.capture_public_energy_charts_lt `
  --role epex_ch `
  --start-utc <complete-start-UTC> `
  --end-utc <complete-end-exclusive-UTC> `
  --raw-cadence-minutes 60 `
  --acquisition-id <portable-unique-id> `
  --output-directory C:\Users\jbattaglia\PFC_LT\build\prospective-captures\<id> `
  --ca-bundle C:\certs\git-ca-plus-fmv.crt
```

Success must retain `capture-attempt.json`, exact provider body,
`capture-spec.json`, `capture-summary.json` and the harness receipt. Require
`COMPLETE_LOCAL_UNTRUSTED_CAPTURE`,
`timestamp_authority=LOCAL_WORKSTATION_CLOCK_UNTRUSTED`,
`production_authorization=false` and `promotion_gate=false`. For current CH
day-ahead data, source observation count is hourly; the 15-minute output is
only `UPSAMPLED_STEPWISE_PROXY`, with native-quarter-hour truth and product
identity both unproven.

This network capture is not resumable. If the process fails after creating its
output directory, preserve the entire partial namespace as negative evidence,
quarantine that acquisition/run/output identity and use new names for any
later attempt. Never delete, overwrite or retry in place to manufacture a
complete capture.

The installed networkless Builder may replay the exact capture spec into a
new repo-local quarantine, and the isolated verifier zipapp may audit that
quarantine. Neither exit zero upgrades the capture. A candidate, rolling-origin
confirmation or T057 run must consume only an independently timestamped,
signed, Builder-inaccessible and external-CAS-admitted successor. Never sign or
copy this local namespace and claim the missing capture-time controls were
created retroactively.

### 1.2) Unsigned provider-raw acquisition builder

`pfc_shaping.cli.governed_acquisition_builder` is the only admitted production
module for turning
already captured provider response bodies into an unsigned exact-byte envelope,
replayed bronze Parquet and deterministic build manifest. Run it from the
reviewed wheel, never from `scripts/` or a source checkout:

```powershell
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.governed_acquisition_builder `
  --capture-spec C:\capture\acquisition-uuid\capture-spec.json `
  --output-directory C:\quarantine\provider-raw\acquisition-uuid
```

The builder identity is distinct from network capture, data timestamp,
acquisition-signing, journal, publication-anchor and production identities. It
has read/traverse access to one immutable mono-linked capture specification and
its exact mono-linked body files, and no private keys, network route,
shared-data write right or production mount. The pre-provisioned quarantine
parent ACL must grant list/read-attributes and add-file/add-subdirectory only.
Fresh staging objects created by the service identity must grant that identity
read/write, hardlink creation, delete-self for its temporary files, and DELETE
on its staging directory so the validated staging handle can be renamed. Do not
grant DELETE_CHILD on the quarantine parent or overwrite/delete rights on a
pre-existing output; exact retry reads an existing output and divergent output
requires an independent repair identity. The renamed builder output deliberately
retains its creator ACL and is therefore untrusted mutable quarantine, not a
final authority namespace.

Capture paths and output paths must be lexically absolute and must not contain
`~`; every existing ancestor must be local policy-approved and free of symlinks,
junctions and reparse points. IT must provision the output parent before the
run. On the admitted Windows runtime, the process requires a local filesystem
that returns non-zero `FILE_ID_INFO` (128-bit file ID plus volume serial), pins
the full existing ancestor chain with directory handles that omit
`FILE_SHARE_DELETE`, pins the staging directory, and renames it through that
validated handle. It validates the promoted output identity and exact bytes
before success. A filesystem that cannot supply those identities fails closed.
The service account must be validated against this ACL matrix in the deployment
drill; `create/write/list` alone is insufficient.

Capacity admission is performed before body loading: each provider has a
document-size cap, all documents share a 64 MiB raw cumulative cap, and Parquet
replay is admitted from flat non-repeated metadata under role-specific row,
column, cell and row-group budgets. IT must additionally reserve at least four
times the cumulative body budget plus the expected bronze size on the target
volume and enforce a process/job memory ceiling. An allocation-ceiling kill,
disk-full event or malformed provider payload is an integrity failure, never a
partial success.

Success prints the absolute `acquisition-build-manifest.json` path. The manifest
must state `UNSIGNED_REQUIRES_INDEPENDENT_AUTHORITIES`, `NOT_PUBLISHED`,
`BUILDER_MUTABLE_UNTRUSTED_QUARANTINE`, and
`REQUIRES_BUILDER_INACCESSIBLE_COPY_AND_INDEPENDENT_SIGNATURE`.
Schema `lt_provider_acquisition_build.v2` also binds
`lt_observation_resolution_provenance.v2` for every Energy Charts price role:
the cadence present in the provider bytes, the 15-minute transport cadence,
the resampling method, source/output observation counts and separate native
hourly/native-quarter-hour eligibility. A 15-minute index produced by
forward-filling hourly CH prices is
`NATIVE_HOURLY_PRICE_QH_TRANSPORT_PROXY_ONLY` and must have
`native_quarter_hour_truth_eligible=false`. It may support an hourly layer but
must never satisfy a native-quarter-hour truth, calibration, MDE, holdout or
promotion gate. The independent quarantine audit must reproduce the same
metadata from provider bytes; missing, changed or relabelled metadata is an
integrity failure.
Failure exits `50` and emits one JSON error with `status=INTEGRITY_FAILURE`, an
error type/message, retry disposition and `production_authorization=false`.
Alert on every exit 50, any staging residue, free space below the reserved
budget, unexpected files, concurrent attempts for one acquisition ID or output
hash divergence.

Exact retry is allowed only with the same immutable capture spec, body hashes,
admitted wheel and output path. A complete matching output is idempotent; exact
staging and exact durable-writer hardlink/temporary residue are resumed.
Divergent or ambiguous residue and unparseable output/staging are never
overwritten or deleted by the service identity: revoke its write token, move
the whole acquisition namespace to quarantine under an IT repair identity, and
open an incident. Concurrent writers are forbidden operationally; schedule one
lease per acquisition ID and treat a divergent race as compromise. Rollback is
quarantine/removal of the unsigned namespace before authority handoff—there is
no production pointer to roll back.

After success, stop the Builder job and revoke its namespace token/lease. An
independent handoff identity must copy the exact manifest, envelope, parser,
config and bronze bytes into a pre-provisioned namespace on which the Builder
has no rights, verify every hash and replay after that copy, then emit the signed
acquisition/journal evidence there. It must never sign a path in the mutable
Builder namespace. Retain an ACL/freeze receipt (`icacls` plus service-identity
negative write/delete probe) as deployment evidence. A separate publisher
identity may later consume only that independently signed, builder-inaccessible
bundle; publication rechecks every signed content hash. The builder output alone
cannot enter the shared data root and can never authorize publication or
production.

### 1.2a) Standard-user EEX forward-vintage signing request

`pfc_shaping.cli.eex_forward_vintage_builder` appends one independently
timestamped EEX workbook to a verified cumulative bitemporal history and emits
an immutable local signing-request bundle. It does not sign a catalog, admit a
CAS object, authorize calibration, publish a snapshot or promote production.
All mutable inputs and outputs on the managed workstation must remain below
the canonical repo `build/`; do not use `AppData`, an administrator shell, a
Defender exception, a project `.exe` or Playwright.

Run the sealed repo-local runtime through the standard-user harness. Use a
fresh run ID and a pre-provisioned parent for `--output-directory`:

```powershell
python -B -m scripts.run_workspace_local --run-id eexreq01 -- `
  C:\Users\jbattaglia\PFC_LT\build\conda-runtime-vN\python.exe -I -B -m `
  pfc_shaping.cli.eex_forward_vintage_builder `
  --runtime-receipt C:\Users\jbattaglia\PFC_LT\build\runtime-receipts\runtime-vN.json `
  --expected-runtime-receipt-sha256 <caller-held-runtime-64-hex> `
  --intake-spec C:\Users\jbattaglia\PFC_LT\build\eex-intake\intake-spec.json `
  --expected-spec-sha256 <caller-held-64-hex> `
  --source-document C:\Users\jbattaglia\PFC_LT\build\eex-intake\source.xlsx `
  --trusted-time-receipt C:\Users\jbattaglia\PFC_LT\build\eex-intake\trusted-time-receipt.json `
  --trusted-time-public-key C:\Users\jbattaglia\PFC_LT\build\eex-intake\trusted-time-public.pem `
  --trusted-time-journal-id <governed-journal-id> `
  --output-directory C:\Users\jbattaglia\PFC_LT\build\eex-requests\<intake-id>
```

For a non-genesis append, also supply all three arguments together:

```text
--previous-catalog <build-path-to-signed-catalog.json>
--previous-history <build-path-to-history.parquet>
--previous-catalog-public-key <build-path-to-acquisition-public.pem>
```

Public verification keys are explicit read-only inputs; no private key may be
present. The builder binds their exact bytes and the journal ID into
`catalog-signing-request.json`. Success prints
`eex-vintage-build-manifest.json`, whose status must be
`COMPLETED_UNSIGNED_EXTERNAL_AUTHORITY_HANDOFF`; `signing_status` must remain
`UNSIGNED_REQUIRES_INDEPENDENT_ACQUISITION_AUTHORITY`, `cas_status` must remain
`NOT_ADMITTED_REQUIRES_BUILDER_INACCESSIBLE_EXTERNAL_CAS`, and calibration,
promotion and production flags must all be false.

An exact retry to the same output is idempotent. Exact staging residue can be
resumed after process interruption. Any divergent output, unexpected file,
ambiguous hardlink residue, trust-key change or concurrent loser is an
integrity failure: preserve the entire namespace, stop scheduling that intake
ID and use a separate repair/quarantine identity. Never edit or delete evidence
to make a retry pass.

The local bundle remains writable by the Builder and is therefore not WORM or
calibration evidence. The next authority must copy the exact bundle into a
Builder-inaccessible namespace, revalidate the signing request and every
artifact hash, sign only the proposed catalog payload under an independent
acquisition identity, then perform external CAS/WORM admission with a fresh
expected HEAD and append-only receipt. Until that independently governed
consumer exists and succeeds, snapshot publication, T057 evaluation and
production remain `NO-GO`.

### 1.3) LT input snapshot publication

The snapshot publisher is a separate deterministic `.pyz` and is deliberately
absent from the governed calculation/promotion wheel. Build and audit it twice
from the curated revision using `scripts/build_snapshot_publisher_zipapp.py`;
require byte-identical hashes and retain both audit JSON documents. Its positive
inventory excludes the bootstrap signer and test-only SQLite authority. Apply
the phase contract in `deploy/publisher/environment-contract.json`.

Before either build, use
`scripts/build_snapshot_publisher_runtime_closure.py` against the eleven exact
wheel archives retained in the offline wheelhouse. Do not install them first.
The builder verifies archive hash/size against `uv.lock`, complete wheel
`RECORD`, metadata, tags and paths before extracting a new empty closure and
emitting its canonical receipt. Pass the closure, receipt and wheel directory
to both zipapp builds. Each build independently replays archive and extracted
bytes, then binds the receipt hash, per-wheel hash/`RECORD` evidence, build
interpreter, ABI libraries and complete closure tree into the artifact.

Production must run the three commands under separate publisher phase and
IT-anchor identities:

```powershell
# One-time only, after the offline IT authority signs the exact migrated seed.
C:\runtime\python.exe -I -S -B C:\runtime\fmv-lt-snapshot-publisher.pyz prepare `
  --bootstrap-authorization <signed-bootstrap-authorization.json> `
  --data-root <data-root> --operation-id <authorized-uuid> `
  --operation-created-at-utc <authorized-ISO-8601-UTC>

# Every later prospective publication must name the existing external HEAD.
C:\runtime\python.exe -I -S -B C:\runtime\fmv-lt-snapshot-publisher.pyz prepare `
  --source-bundle <signed-bundle> --data-root <data-root> `
  --operation-id <uuid> --operation-created-at-utc <ISO-8601-UTC>

C:\runtime\python.exe -I -S -B C:\runtime\fmv-lt-snapshot-publisher.pyz cas `
  --intent <intent.json> --receipt-output <receipt.json>

C:\runtime\python.exe -I -S -B C:\runtime\fmv-lt-snapshot-publisher.pyz finalize `
  --data-root <data-root> --intent <intent.json> `
  --anchor-receipt <receipt.json> `
  --observation-directory <operation-root>\observations
```

PREPARE discovers the authenticated external HEAD unless an exact expectation
is supplied. A normal `PUBLISH` can never create genesis. The only sequence-1
transition is `BOOTSTRAP`, authorized by a short-lived, one-shot Ed25519 IT
document that binds the exact legacy pointer, generation, contract, inventory,
operation and empty external HEAD. It must preserve `MIGRATED_UNVERIFIED` and
`calibration_eligible=false`; `ADOPT_LEGACY` remains forbidden. There is no
fallback to genesis after a HEAD failure. The signer command is executed only
on the offline IT host:

```powershell
$env:PFC_DATA_PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_PATH = '<offline-key.pem>'
python -m scripts.sign_snapshot_bootstrap_authorization `
  --input <canonical-unsigned-authorization.json> `
  --output <signed-bootstrap-authorization.json>
```

The signing host must not receive publisher mTLS, request, anchor, acquisition,
journal or promotion credentials. CAS or FINALIZE exit `51` means the external
commit is proven durable but a local receipt archive, observation archive or
pointer projection requires exact repair.

The bootstrap authorization private key and signing tool belong to an offline
IT role. They are forbidden from publisher, model, candidate, audit and
promotion environments. The publisher receives only the signed authorization
and active plus historical public trust. The authorization is archived through
the intent and receipt chain; a candidate pointed directly at a bootstrap
receipt is rejected independently by serialization, assembly and capstone.

Exact retries after CAS recover and reuse the committed receipt. CAS never
creates an observation. FINALIZE performs exact operation lookup and requests a
new nonce-bound observation on every attempt, then archives it under its
content-addressed ID. Expired observations remain immutable history and are
never reused as current evidence. A truncated local receipt is never replaced:
exact authority lookup only stages a content-addressed repair candidate for an
external operator/CAS decision. Exit `40` is a determinate conflict,
`50` an authentication/integrity failure, `51` committed projection repair, and
`52` an indeterminate or retryable authority outcome.

All commands require `PFC_SNAPSHOT_PUBLISHER_DEPENDENCY_ROOT` to point to the
read-only, content-addressed closure bound into the audited zipapp. Invocation
without `-I -S -B`, with a different Python executable/ABI library, or after
adding, removing, linking or changing any closure file exits `50` before model
or publisher business modules load. The environment root is verification input
only: exact bytes are copied to a process-private root, both trees and the
captured distribution inventory are revalidated, and only the capture enters
`sys.path`. Runtime schema v6 starts that admitted worker under a minimal parent
which owns a random supervisor scratch and a consumed one-shot capability bound
to that scratch, the parent PID and artifact path. On Windows the parent starts
the worker suspended, assigns it to a `KILL_ON_JOB_CLOSE` Job Object, then
resumes it. The reserved internal
worker variables in the environment contract must never be injected by the
service definition. The parent removes its scratch after worker exit, once
Windows native extensions are unloaded. Alert on
`RUNTIME_CAPTURE_CLEANUP_FAILED` and free-space pressure. Old
`fmv-pfc-publisher-runtime-*` directories and abnormal-parent
`fmv-pfc-publisher-supervisor-*` residue must be cleaned only during a proven
publisher outage after process inventory; never recursively delete a candidate
computed only from a glob while publisher workers may be live.

The artifact audit reads one stable mono-linked byte image and reports the hash
of those same audited bytes. Zipapp binding also requires the bind-time closure
tree hash and file count to equal the independently replayed receipt. For
`prepare` and `finalize`, `FMV_DATA_ROOT` is mandatory and authoritative; a
supplied `--data-root` must resolve to exactly the same location.

The signed IT release attestation must separately bind the source wheel hashes,
Python image, standard library, SBOM, exact zipapp, source closure, supervisor
scratch parent, captured tree and target ACL for the complete worker lifetime.
It must require a dedicated service identity with no untrusted co-process under
the same Windows token. Application read-only attributes are defense in depth,
not storage enforcement; runtime self-verification is not a substitute for host
trust.

Every exit `51` result uses `lt_snapshot_publication_cli_result.v1` and retains
`commit_status=COMMITTED`, operation/intent/receipt identities, sequence,
receipt path/hash, projection status, retry disposition and structured error.
Do not rerun PREPARE with a new operation ID. Follow `retry_disposition` and
reuse exact operation lookup or FINALIZE with a fresh HEAD as instructed.

The durable writer never replaces any different target, including
noncanonical or truncated local bytes. Exact authoritative bytes are staged
under `.pfc-repair-candidates` using their SHA-256 and returned through
`durable_artifact_repair.v2`. `LOCAL_ARTIFACT_REPAIR_REQUIRED` with
`AUTHORITATIVE_BYTES_DIFFER_FROM_EXISTING_TARGET`,
`PENDING_REPAIR_CANDIDATE`, `UNOWNED_TEMPORARY_RESIDUE` or
`UNOWNED_REPAIR_LOCK` is an operator stop. The software neither deletes the
target nor promotes a repair candidate automatically.

Required publisher configuration includes the HTTPS anchor URL, pinned CA,
mTLS certificate/key, publication domain and the complete public trust registry
from `deploy/publisher/environment-contract.json`, including all five Tier2
authorities. Every active and historical public key is captured once into a
private process directory, registered as immutable in-memory bytes with a
sealed directory inventory before use, and must be globally disjoint. No
downstream verifier reopens captured trust bytes from disk. Active
request/bootstrap keys authorize new admissions; their historical keyrings are
replay-only. The SQLite reference authority is test-only and must never be
configured as the production endpoint.

| Capability | Offline bootstrap | PREPARE | CAS | FINALIZE | IT anchor |
|---|---:|---:|---:|---:|---:|
| bootstrap signing private key | `R` | `-` | `-` | `-` | `-` |
| request signing private key | `-` | `R` | `-` | `-` | `-` |
| anchor signing private key | `-` | `-` | `-` | `-` | `R` |
| anchor mTLS client key | `-` | conditional `R` | `R` | `R` | `-` |
| migrated pointer / source bundle | metadata `R` | `R` | `-` | `R` | `-` |
| immutable intent | `-` | `CW` | `R` | `R` | `R` |
| immutable receipt | `-` | `-` | `CW` | `R` | authoritative `CW` |
| immutable observations | `-` | `-` | `-` | `CW` | authoritative `CW` |
| mutable `current.json` projection | `-` | `-` | `-` | `WX` | `-` |

All public trust stores are read-only and captured at process startup. A key
identity appearing in more than one active or historical role is a startup
failure. `CW` means create/write only in the exact operation namespace; it does
not grant delete or overwrite rights over another operation.

## 2) Governed LT release

The former `python -m pfc_shaping.pipeline.rolling_update` command is disabled
because it read mutable repo caches and published directly. The canonical
command surface is:

```powershell
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.governed_release --help
```

One invocation performs exactly one phase. There is no `all`, `run`,
`release`, `auto-promote` or failed-gate override command.

```powershell
# Builder identity: public trust anchors only. Exit 30 is the expected policy handshake.
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.governed_release build `
  --run-id <run-id> --release-root <release-root> --failure-root <failure-root> `
  --source-revision <source-revision> `
  --reference-timestamp <ISO-8601-UTC> --build-timestamp <ISO-8601-UTC> `
  --config <lt-config.yaml> --config-sha256 <config-sha256> `
  --input-snapshot-contract <lt_input_snapshot.json> `
  --input-snapshot-sha256 <input-snapshot-sha256> `
  --input-pointer-contract <pfc-lt-current.json> `
  --input-pointer-sha256 <input-pointer-sha256> `
  --input-generation-id <generation-id> `
  --publication-head-observation <signed-head-observation.json> `
  --publication-head-observation-sha256 <head-observation-sha256> `
  --publication-head-challenge-nonce <256-bit-caller-nonce> `
  --data-root <fmv-data-root> `
  --historical-thresholds <thresholds.csv> `
  --historical-thresholds-receipt <thresholds-receipt.json> `
  --selected-lambda-decision <selected-lambda.json> `
  --selected-lambda-decision-receipt <selected-lambda-receipt.json> `
  --eex-report-path <Price_Report_EEX.xlsx> `
  --eex-acquisition-contract <signed-acquisition.json> `
  --peak-source-policy same_first `
  --use-seasonal-hourly-shape

# Independent policy/finalizer identity.
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.governed_release finalize `
  --run-id <run-id> --release-root <release-root> --failure-root <failure-root> `
  --source-hierarchy-policy <signed-source-policy.json>

# Registration identity: REGISTER private key only.
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.governed_release register `
  --run-id <run-id> --release-root <release-root> --failure-root <failure-root> `
  --workflow-root <workflow-root> --evidence-root <evidence-root> `
  --expect-no-current

# For a non-empty journal, replace --expect-no-current with:
# --expect-current-event-id <current-event-id>

# Save request_id from registration. Audit and promotion address that immutable request.
# If its signed receipt expires, register a new immutable request with
# --request-nonce <unique-operator-value>, then audit that new request.
# The signed receipt binds the full request ID, operation ID, expected current
# event, request timestamp, release-root hash, workflow-domain hash and exact
# signed REGISTER document hash. None may be rebased at promote.

# Audit identity: receipt private key only; event private key must be absent.
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.governed_release audit `
  --run-id <run-id> --release-root <release-root> --failure-root <failure-root> `
  --workflow-root <workflow-root> --evidence-root <evidence-root> `
  --request-id <request-id> `
  --data-root <fmv-data-root> --signing-private-key <receipt-private.pem>

# Promoter identity: event private key only; receipt private key must be absent.
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.governed_release promote `
  --run-id <run-id> --release-root <release-root> --failure-root <failure-root> `
  --workflow-root <workflow-root> --evidence-root <evidence-root> `
  --request-id <request-id>

# Rollback is never a promote shortcut. It requires a fresh, expiring,
# independently signed authorization bound to the current and target event IDs.
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.governed_release rollback `
  --release-root <release-root> --failure-root <failure-root> `
  --rollback-authorization <signed-rollback-authorization.json>

# Read-only status.
C:\FMV\PFC-LT\runtime\python.exe -I -B -m `
  pfc_shaping.cli.governed_release status `
  --run-id <run-id> --release-root <release-root> `
  --workflow-root <workflow-root> --evidence-root <evidence-root>
```

Exit codes currently guaranteed by this command surface:

| Code | Meaning |
|---:|---|
| `0` | Phase completed successfully; audit approved when phase is `audit`. |
| `2` | CLI usage or argument error; no phase executed. |
| `30` | Expected governance stop: build awaits independent source policy, or audit rejected. |
| `40` | Compare-and-swap conflict. Nothing was committed; register and audit a new request. |
| `41` | Another live transition still holds the lock after the configured wait budget. Retry the same immutable operation. |
| `50` | Integrity or governed-release contract failure. No promotion may be inferred. |
| `51` | Projection reconciliation is required. Read `commit_status`: `COMMITTED` proves the immutable transition exists; `INDETERMINATE` means it may exist and no commit outcome may be inferred before journal reconciliation. |

All successful and expected-governance outputs are versioned JSON on stdout.
Unhandled execution failures remain non-zero and production-blocking; inspect
the preserved workflow failure evidence before retrying.

## 3) Scheduler setup

The former Windows Task Scheduler registration script is disabled. IT must
schedule the governed build, audit and promotion phases under separate service
identities with explicit mounted data and key paths. It must also schedule
finalization and registration as distinct processes. Never expose
`PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH` outside the registrar identity.
Never expose
`PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH` and
`PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH` to the same service identity.
The rollback service must expose the event private key and the rollback
authorization public key, never the rollback authorization private key.
Configure `PFC_PROMOTION_LOCK_WAIT_SECONDS` explicitly for promoter and
rollback services. No existing lock is ever deleted by the application,
including a same-host lock whose PID appears dead. IT must first prove that its
owner is gone and recover it under a documented storage incident procedure;
automatic stale-lock recovery permits a TOCTOU double-writer race.
Configure one canonical `PFC_PROMOTION_RELEASE_DOMAIN_ID` UUID for the logical
release. The exact same value is mandatory through drive-letter, short UNC,
FQDN UNC and container mounts. Signed requests, receipts, events and journal
namespaces bind to its hash rather than to the host-local path spelling; using
different UUIDs intentionally creates different release domains. Generate it
once with PowerShell `New-Guid`, record it in the governed deployment secret,
and never reuse it between dev, test, prod or distinct physical release roots.
IT provisioning pins it before the first governed use in the immutable,
exclusive, fsynced
`<release-root>/release_domain.json`; any later configuration drift, linked
marker, hardlink residue not attributable to an exact writer temporary, or
marker loss after transition activity fails closed.

Configure a second canonical UUID in `PFC_RELEASE_WORKFLOW_DOMAIN_ID` for the
logical REGISTER/AUDIT/PROMOTE workflow namespace. It is independent from the
release-domain UUID and must be identical for those three identities. The
IT provisioning exclusively creates and fsyncs
`<workflow-root>/workflow_domain.json` before REGISTER runs.
Every request contains the marker hash and an Ed25519 REGISTER signature. Audit
and CAS reload the request from its exact canonical path and verify both before
using the signed receipt. A caller-supplied alternate workflow root therefore
cannot manufacture registration authority.

Phase outputs are physically separated below each run: REGISTER owns
`requests/<request-prefix>/release_request.json`, AUDIT owns
`audit-results/<request-prefix>/`, and PROMOTE owns
`promotion-results/<request-prefix>/promotion_result.json`. A failure to write
the PROMOTE projection after the immutable journal commit returns exit `51`
with `commit_status=COMMITTED`; an exact retry repairs the projection without a
second event.

### 3.1) Required identity and ACL matrix

IT owns every root and file. Service identities must never own governed roots,
change ACLs, take ownership or inherit access through a broad operator group.
In the table, `T` is traverse, `L` list, `R` read data/attributes, `C` create a
child, `W` write/append, `X` rename/replace/delete child, and `H` create a
hardlink. `-` means no allow ACE; remove inherited access rather than adding a
broad explicit deny ACE. Private-key files have inheritance disabled and one
read ACE for the named identity only.

| Path or capability | Builder | Finalizer | Registrar | Auditor | Promoter | Rollback | Status |
|---|---:|---:|---:|---:|---:|---:|---:|
| approved wheel and interpreter | `TR` | `TR` | `TR` | `TR` | `TR` | `TR` | `TR` |
| data root and selected immutable snapshot | `TLR` | `-` | `-` | `TLR` | `-` | `-` | `-` |
| governed config, thresholds, lambda and source policy | `TLR` | `TLR` | `-` | `TLR` | `-` | `-` | `-` |
| EEX workbook and acquisition contract | `TLR` | `-` | `-` | `TLR` | `-` | `-` | `-` |
| release root traversal and domain marker | `TLR` | `TLR` | `TLR` | `TLR` | `TLRCWXH` | `TLRCWXH` | `TLR` |
| candidate staging/new run before finalization | `TLRCWXH` | `TLRCWXH` | `-` | `-` | `-` | `-` | `TLR` |
| finalized `release/candidates/<run>` after IT ACL freeze | `TLR` | `TLR` | `TLR` | `TLR` | `TLR` | `TLR` | `TLR` |
| release lock file create/delete at release root | `-` | `CWXH` | `-` | `-` | `CWXH` | `CWXH` | `-` |
| release events, archived receipts and mutable projections | `-` | `-` | `TLR` | `TLR` | `TLRCWXH` | `TLRCWXH` | `TLR` |
| workflow request namespace | `-` | `-` | `TLRCWXH` | `TLR` | `TLR` | `TLR` | `TLR` |
| workflow audit-results namespace | `-` | `-` | `TLR` | `TLRCWXH` | `TLR` | `TLR` | `TLR` |
| workflow promotion-results namespace | `-` | `-` | `TLR` | `TLR` | `TLRCWXH` | `TLR` | `TLR` |
| evidence root and per-run capture | `-` | `TLR` | `TLRCWXH` | `TLR` | `TLR` | `TLR` | `TLR` |
| identity-specific failure root | `TLRCWXH` | `TLRCWXH` | `TLRCWXH` | `TLRCWXH` | `TLRCWXH` | `TLRCWXH` | `-` |
| external promotion journal | `-` | `-` | `-` | `TLR` | `TLRCWXH` | `TLRCWXH` | `TLR` |
| public trust stores required by the phase | `TLR` | `TLR` | `TLR` | `TLR` | `TLR` | `TLR` | `TLR` |
| REGISTER-signing private key | `-` | `-` | `R` | `-` | `-` | `-` | `-` |
| receipt-signing private key | `-` | `-` | `-` | `R` | `-` | `-` | `-` |
| event-signing private key | `-` | `-` | `-` | `-` | `R` | `R` | `-` |
| rollback-authorization private key | `-` | `-` | `-` | `-` | `-` | `-` | `-` |

Each mutating identity receives a different failure root. No service identity
may read another identity's private key or failure root. The rollback
authorization private key belongs to an external authorizer and is absent from
all runtime hosts. Journal history additionally requires storage-enforced
retention/WORM: application `CWXH` rights are necessary for publication but do
not themselves prevent a compromised identity from deleting old bytes.

Provision roots with inheritance removed, an IT-admin full-control ACE and only
the role ACEs above. Example inspection commands, run from an administrative
shell, are:

```powershell
icacls '<root>' /inheritance:r
icacls '<root>' /grant:r 'FMV-PFC-IT-Admins:(OI)(CI)F'
icacls '<root>'
(Get-Acl -LiteralPath '<root>').Access |
  Select-Object IdentityReference,FileSystemRights,AccessControlType,IsInherited
```

Using the admitted wheel and the configured UUIDs, IT pins the empty domain
roots before granting service access:

```powershell
python -c "from pathlib import Path; from pfc_shaping.pipeline.atomic_promotion import _release_root_id; print(_release_root_id(Path(r'<release-root>')))"
python -c "from pathlib import Path; from pfc_shaping.pipeline.release_request_contract import ensure_workflow_domain; print(ensure_workflow_domain(Path(r'<workflow-root>'), create=True)[1])"
New-Item -ItemType Directory -Force `
  '<workflow-root>\<run-id>\requests', `
  '<workflow-root>\<run-id>\audit-results', `
  '<workflow-root>\<run-id>\promotion-results' | Out-Null
```

IT then disables inheritance separately on the three namespaces and grants
`CWXH` only to Registrar, Auditor and Promoter respectively. After FINALIZE,
IT removes Builder and Finalizer write/delete/rename/hardlink rights from the
final `<release-root>/candidates/<run-id>` tree, verifies all service roles are
read-only there, and only then enables REGISTER. Storage-enforced immutability
or WORM is preferred where available. These ACL transitions and their
`icacls` output are retained as release evidence.

Do not test create/delete semantics in a governed release. Under each service
identity, use a fresh dedicated `<acl-probe-root>/<identity>/<probe-id>` on the
same target volume and verify the exact primitives before removal:

```powershell
$probe = '<acl-probe-root>\<identity>\<probe-id>'
New-Item -ItemType Directory -Path $probe -ErrorAction Stop | Out-Null
$source = Join-Path $probe 'source.tmp'
$link = Join-Path $probe 'published.bin'
$moved = Join-Path $probe 'moved.bin'
[IO.File]::WriteAllBytes($source, [byte[]](1,2,3))
New-Item -ItemType HardLink -Path $link -Target $source -ErrorAction Stop | Out-Null
Remove-Item -LiteralPath $source -ErrorAction Stop
Move-Item -LiteralPath $link -Destination $moved -ErrorAction Stop
[IO.File]::ReadAllBytes($moved) | Out-Null
Remove-Item -LiteralPath $moved -ErrorAction Stop
Remove-Item -LiteralPath $probe -ErrorAction Stop
```

Run the D144 storage drill separately on the exact release appliance and aliases.
Archive `icacls`, probe identity, exit code, timestamps and D144 report hash as
deployment evidence. A local administrator run is not evidence for a service
identity.

## 4) Success criteria
- Candidate bundle is finalized and passes byte/hash replay.
- Audit receipt is signed, approved and contains the exact required gate set.
- Promotion event is signed and linked to the approved receipt.
- External immutable journal history is contiguous and signed. Its latest
  immutable head is the commit authority; mutable head and `current.json` are
  repairable projections.
- Delivered BASE, PEAK and implied OFFPEAK products reprice within sanctioned
  tolerances from the promoted bundle bytes.

## 5) If run fails
- Inspect the governed workflow status using explicit release, workflow and
  evidence roots.
- Preserve candidate, evidence, receipt, failure and journal bytes for replay.
- Do not edit a failed receipt, candidate manifest or promoted pointer.
- Fix source acquisition, solver specification, priors, objective or audit
  gates, then generate a new run ID.
- For `POINTER_REPAIR_REQUIRED` with `commit_status=COMMITTED`, reconcile the
  immutable journal and retry the exact request/authorization to repair mutable
  projections. With `commit_status=INDETERMINATE`, stop all transitions and
  reconcile the immutable journal first; do not infer success or failure and do
  not issue a replacement operation until the original outcome is proven.
- If an existing `promotion_result.json` differs from the committed event, stop
  all transitions. Prove the immutable journal and archived receipt first,
  hash and retain the divergent projection in the incident record, then have IT
  move it to a write-protected forensic quarantine. Only after that controlled
  action may the Promoter retry the exact request to recreate the projection.
  The application never overwrites or deletes divergent projection bytes.
- Never hand-edit a journal head, current pointer or failure manifest.
- A CAS conflict requires a new immutable registration and a new signed audit
  receipt. Never mutate or rebase the old request.
- A receipt renewal requires a new request nonce and therefore a new request
  ID. Reusing or replacing bytes below an existing request ID is forbidden.

## 6) Release policy
- Audit and promotion run under separate service identities and private keys.
- Historical public keys remain verification-only files under the configured
  keyring directories. Rotation changes the active private/public pair but
  never removes a public key still referenced by retained signed history. A
  historical key can replay committed history but cannot authorize a new
  promotion or rollback; new transitions must verify against the active primary
  key. A request signed while its REGISTER key was current remains replayable
  for an exact already-audited promotion and exact retry; it cannot be mutated
  or used to register a new operation. REGISTER, event, receipt and
  rollback-authorizer keyrings must remain pairwise disjoint, including retained
  historical keys.
- Retained signed receipts remain replayable after an active policy rotation
  only while their exact policy version and gate inventory remain in the
  reviewed historical policy registry. Removing a retained policy version is a
  breaking migration and must be rejected while any journal, request or receipt
  still references it. Historical policy replay never authorizes a new receipt;
  AUDIT creates new receipts only under the active policy.
- `<release-root>/candidates` must be a real directory on the governed volume,
  never a symlink or Windows junction. A linked candidate root is rejected
  before candidate creation, verification or finalization.
- Governance JSON/YAML is parsed fail-closed: duplicate keys, YAML merge
  collisions, non-finite numbers, cycles, excessive depth/cardinality and
  alias-DAG amplification are rejected before receipt emission.
- Failed gates cannot be bypassed for production promotion.
- Shared data, release roots, trust keys and journal roots are explicit mounts;
  repo-local heavy caches are never production inputs.
- Threshold or trust-policy changes require a reviewed decision-log entry.
- The external journal mount must enforce IT retention/ACL controls preventing
  deletion or rewind of immutable head history. Application signatures detect
  tampering but cannot make an ordinary filesystem WORM by themselves.
- On POSIX, immutable files, replacements and parent directories are fsynced.
  On Windows, file contents are flushed and same-volume `os.replace` is used,
  but directory persistence after power loss is explicitly unproven. Publisher
  results report
  `PROCESS_CRASH_RECOVERABLE_POWER_LOSS_UNPROVEN_ON_WINDOWS`; storage durability
  and write-through guarantees remain an IT mount contract.
- Before production enablement, IT must run kill/restart and concurrent-writer
  drills on the exact Windows/SMB volume and service identities. Local NTFS and
  POSIX tests do not prove the target appliance's durable-link semantics.
- Schema-v1/v2 workflow directories are read-only migration evidence. New
  registrations use signed request-v3 directories keyed by the first 128 bits of the
  full request hash; the complete 256-bit request ID remains hash-bound inside
  every request and receipt and is compared after lookup. Existing request-v2
  documents lacking REGISTER signature, workflow-domain, release-root or operation bindings are pre-D143 evidence
  only and cannot be audited or promoted by this controller.

## 7) Target-volume storage drill (D144)

Run this non-promotional drill before provisioning a production release root.
The selected root must be an existing, dedicated directory on the exact target
volume, outside every governed release and candidate. The command writes only
below `<drill-root>/.pfc-lt-storage-drill/<run-id>` and preserves the full run.
No promotion, rollback, candidate, receipt or private signing key is used.

Clear every `PFC_*PRIVATE_KEY*` variable from the drill identity, then run from
the canonical drive spelling with the actual short and FQDN UNC aliases:

```powershell
python scripts/run_lt_release_storage_drill.py `
  --drill-root 'R:\PFC-LT-storage-qualification' `
  --alias-root '\\fileserver\share\PFC-LT-storage-qualification' `
  --alias-root '\\fileserver.example.ch\share\PFC-LT-storage-qualification' `
  --require-alias-class drive `
  --require-alias-class unc_short `
  --require-alias-class unc_fqdn `
  --timeout-seconds 60 `
  --global-timeout-seconds 600
```

The CLI supervises the complete run. Each probe executes in its own process;
Windows descendants are owned by a `KILL_ON_JOB_CLOSE` Job Object and POSIX
descendants by a process group. The report exercises the production immutable
hardlink writer/recovery, exclusive writer collision, live and abandoned lock,
atomic JSON replacement with concurrent readers, and final directory rename.
It also proves cross-alias `samefile`, visibility and lock exclusion.

Exit codes are `0` for application-level probe PASS, `1` for a recorded probe
FAIL and `2` for unsafe setup, internal failure or supervised timeout. The
versioned report is archived at
`<drill-root>/.pfc-lt-storage-drill/<run-id>/storage_drill_report.v1.json`.
Verify its SHA-256 from stdout and preserve the whole run directory. Never
reuse a run ID and never delete an abandoned `.promotion.lock` automatically.

Even on exit `0`, the report always states:

- `evidence_scope=STORAGE_DRILL_ONLY`;
- `promotion_ready=false` and `production_authorization=false`;
- `production_qualification_status=UNSUPPORTED`.

This is deliberate. A single-host process drill does not attest multi-client
SMB leases, WORM/retention, service ACLs, HSM/KMS custody, appliance write-
through after power loss, backup/restore or disaster recovery. IT must attach
those independent attestations before production enablement.

## 8) Isolated provider-evidence verifier

The provider verifier is a separate read-only zipapp, not a command embedded
in the governed LT wheel. Its authoritative operator procedure is
`deploy/verifier/README.md`. Operators must use `python -I -S -B`, an exact
content-addressed dependency closure, a dedicated scratch root and physically
disjoint new-name mutable-quarantine destinations for the business audit and
runtime receipt. These local files are not immutable evidence. The runbook's
ordered handoff to an independent WORM/external-CAS retention identity is a
separate mandatory gate before any durable-evidence claim.

Current local selection is pinned to
`build/provider-verifier-20260729-v18.pyz`, SHA-256
`7f17be8de8e78ba5a063903c7ea459baed0372b70a128ffab3fb8b17f69b19c5`.
V17 is byte-identical and retained only as a reproducibility witness. Provider
verifier v14, v15 and v16 are non-selectable historical evidence. Audit v2
(the unversioned CH audit) and audit v3 are also non-selectable; audit v4 SHA
`638a6fc8887b867957fb8cb0ba2cafcf07c00c33fd2620c51e3bca366c2bfc02`
plus runtime receipt SHA
`f28bd6d4e93235467ad4a8c4a2102d6aee5ef7231a96ae3f3a512ccbcb12d82e`
are the current local-quarantine pair. Preserve all superseded files without
repair or relabelling. This selection is local-only and never production
authority.

The required commands are:

```powershell
python -m scripts.build_lt_provider_verifier_zipapp `
  '<absolute-new-verifier.pyz>' `
  --dependency-root '<absolute-locked-site-packages>' `
  --dependency-receipt '<absolute-dependency-closure-receipt.json>' `
  --wheel-directory '<absolute-approved-wheelhouse>'

python -I -S -B '<absolute-verifier.pyz>' runtime-check
python -I -S -B '<absolute-verifier.pyz>' audit-acquisition `
  --acquisition-directory '<absolute-v2-acquisition>' `
  --expected-manifest-sha256 '<caller-held-sha256>' `
  --output-json '<absolute-new-audit.json>' `
  --runtime-receipt-json '<absolute-new-runtime-receipt.json>'

python -I -S -B '<absolute-verifier.pyz>' audit-legacy-resolution `
  --acquisition-directory '<absolute-v1-acquisition>' `
  --expected-manifest-sha256 '<caller-held-manifest-sha256>' `
  --prior-audit '<absolute-prior-audit.json>' `
  --expected-prior-audit-sha256 '<caller-held-prior-sha256>' `
  --output-json '<absolute-new-legacy-result.json>' `
  --runtime-receipt-json '<absolute-new-legacy-runtime-receipt.json>'
```

Exit `0` is local verification only. Exit `50` is fail-closed; preserve stderr
and inputs and do not overwrite outputs or patch legacy evidence. After every
run, inventory `vv-*` below `PFC_LT_VERIFIER_SCRATCH_ROOT`. New residue stops
scheduling and starts the IT incident procedure. The fixed worker timeout is
900 seconds. Every output and receipt remains
`production_authorization=false`; the runtime receipt additionally states
`runtime_authority=false`.

## 9) Local FMV EEX workbook capture on a standard-user workstation

This procedure reads exactly one governed FMV workbook and preserves an exact
local handoff. It establishes neither PIT freshness nor scientific/production
authority. The UNC is read-only; declared mutable destinations and outputs are
redirected below repo `build/`. The runner is a harness, not a filesystem
sandbox.

Run the canonical workspace guard as a separate command first. Then retain the
caller-held hash and use fresh IDs:

```powershell
$source='\\fmvfs1\data\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx'
$sha=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()

python -B -m scripts.run_workspace_local --run-id '<fresh-run-id>' -- `
  python -B -m scripts.capture_eex_forward_local `
  --source-document $source `
  --expected-source-sha256 $sha `
  --capture-id '<fresh-capture-id>'
```

The capture performs a bounded stable double read, hash check, XLSX preflight,
CH inventory and before/after source identity check. Both API and runner reject
every source except the exact canonical UNC. The runner also rejects unknown
or duplicate options, non-lowercase SHA and non-portable IDs. It uses an
existing user Conda interpreter read-only and never launches a project `.exe`
or Playwright. Do not use `AppData`, `ProgramData`, `H:`, admin rights or a
Defender/ASR exception.

On failure, preserve the complete runner and capture namespaces. Never delete,
repair or reuse either ID. Inspect the receipt and child terminal state, then
retry only with fresh run and capture IDs. Attempt-only, empty and partial
namespaces remain incident evidence.

Current local selection is resolved only through
`.planning/phases/14-lt-audit-remediation/CH-EEX-CURRENT-LOCAL-CAPTURE-SELECTION-20260730.json`,
SHA-256
`5f0b99aa04fabcb8219cfa34f20ea262a705940cbc3db3ab2e114ba99bb4a778`,
selection ID
`3be51903d7ed2774d464f8bfd49b20fe283fb5d1b2c0bb93f677e99fe4884667`.
It selects only:

- capture `eex-ch-20260730-v1`, manifest SHA-256
  `823eeb1095e48a49db6df28cda4fd6f96e0f054154252016be76cdbeca4e1801`;
- source copy SHA-256
  `fb71338f51334128878526877b802e48819b555639913a786a35d710a6b151e5`,
  62,490 bytes;
- latest CH row `2026-07-29`, 40 quotes;
- runner `eexcap30v1`, receipt SHA-256
  `d191a275fa73828ffa5085e8e92090fa2636e0e2f3fda0a7882100561b9c67ba`;
- non-disclosing quote-identity commitment
  `74ba58f6d00c8734ea668d487c8fb48d6e12e35642045be7a2c834daddbdfc95`
  and quote-value commitment
  `399d524c2c002b55bf844a5410791d0fa33ba7fc4ed843896f01b26769ece258`.

The former `eex-ch-20260729-v2` selection is historical only. The checkout
audit below remains a developer diagnostic only:

```powershell
python -B -m scripts.run_workspace_local --run-id '<fresh-run-id>' -- `
  python -B -m scripts.audit_ch_eex_current_local_capture `
    --registry C:\Users\jbattaglia\PFC_LT\.planning\phases\14-lt-audit-remediation\CH-EEX-CURRENT-LOCAL-CAPTURE-SELECTION-20260730.json `
    --expected-registry-sha256 5f0b99aa04fabcb8219cfa34f20ea262a705940cbc3db3ab2e114ba99bb4a778
```

Expected checkout-audit status is
`CURRENT_EEX_LOCAL_CAPTURE_VALID_NOT_PIT_NO_GO`, with no price values in
stdout and every authority false. Exit 2, any changed source/manifest/receipt,
parser drift, value-commitment drift or stale selection stops the workflow.
The checkout route imports same-user-writable code before its own checks and
therefore grants no runtime authority. Demonstrated runs `eexsel30v1` and
`eexsel30v2` both exited zero with identical complete stdout SHA-256
`0ff77d96f0e865ae6209ad7147966aed21b69c2a621a1c29b59d49894580c4af`
and empty stderr. Their receipts are respectively
`08cf23396c9ddd3d6ac6a0d2e08c774d37ce57ed3cddbfff946928d42ed6d553`
and
`7d6a42ff81097999c677519833c7f6aa2910ec99352a6300d7c89f2406cacbef`.
Receipts differ by run identity/time as expected; deterministic audit stdout
and source-tree identity are identical.

The current sealed local verifier is v20. It packages the exact two parser
sources bound by the selection, admits a 13-distribution dependency closure,
copies both the zipapp and closure into process-private roots, and only then
imports the EEX audit under `python -I -S -B`. Its reproducible artifacts are:

- `build/provider-verifier-20260730-eex-v20a.pyz` and `...v20b.pyz`;
- byte-identical SHA-256
  `efd896c8c19dc3e4ad1cb04270c09605d86a83e4a126386db4ab8084053a153c`,
  111,858 bytes and 26 exact members;
- source revision
  `6e6ac43f935060bc0495de9d5c401f6d2dfdf548e471bf3256b03c2cc216c9c6`;
- closure receipt SHA-256
  `999e5f5a31631a4562f0c415f1f8ee2172a988007fd94609c524e004cd843d38`,
  13 distributions / 4,928 files, tree SHA-256
  `9eb10eecc91ed6e676e5605d77eb50c2c15b5dcb6dba311a3b14ad9cda6f1541`.

Use an existing user-level CPython read-only. All mutable roots and outputs
remain below repo `build/`; never substitute `AppData`, request admin rights,
launch a project `.exe`, use Playwright or request a Defender exception.
Create a fresh scratch and a fresh direct child of the governed output parent:

```powershell
$root='C:\Users\jbattaglia\PFC_LT'
$inputs="$root\build\verifier-eex-runtime-inputs-v2"
$scratch="$root\build\provider-verifier-eex-v20-scratch-<fresh-id>"
$outputParent="$root\build\provider-verifier-eex-results"
$outputRoot="$outputParent\<fresh-id>"
New-Item -ItemType Directory -Path $scratch,$outputRoot | Out-Null

$env:PFC_LT_VERIFIER_DEPENDENCY_ROOT="$inputs\site-packages"
$env:PFC_LT_VERIFIER_SCRATCH_ROOT=$scratch
$env:TEMP="$inputs\temp"
$env:TMP=$env:TEMP

C:\Users\jbattaglia\.conda\ppa_env\python.exe -I -S -B `
  "$root\build\provider-verifier-20260730-eex-v20a.pyz" `
  audit-current-eex `
  --repository-root $root `
  --registry "$root\.planning\phases\14-lt-audit-remediation\CH-EEX-CURRENT-LOCAL-CAPTURE-SELECTION-20260730.json" `
  --expected-registry-sha256 5f0b99aa04fabcb8219cfa34f20ea262a705940cbc3db3ab2e114ba99bb4a778 `
  --output-root $outputRoot `
  --output-json "$outputRoot\current-eex-audit.json" `
  --runtime-receipt-json "$outputRoot\runtime-receipt.json"
```

The output root and filenames are exact. The verifier rejects every output in
`.planning`, either EEX capture, either workspace-run namespace or either
parser path before writing. The v20 demonstration emitted audit SHA-256
`917dfd899e12c2795b5b3546eb785efdbfb4f4e8c20d5e31cf0840ad2a940dbf`
and runtime receipt SHA-256
`a3ed33c946048bd0d742b518dcaa288bad49d133390f5ade21530db75deb36d7`.
Its runtime receipt records path counts captured/source `1/0/1/0`, process-
private import mode, exact dependency tree, CPython executable SHA-256
`50bfb90ee93bb0cb51175b546f133798dfe4b778677d95d81391e7bf6d85e5ac`
and every authority false. Scratch residue was zero.

On failure, preserve the output and scratch namespaces. Never repair or reuse
them. Retry only with a fresh output child and fresh scratch. A missing,
changed or nonempty terminal scratch is incident evidence. The v20 receipt is
an unsigned local runtime observation, not external attestation or production
authority.

The manifest must remain
`LOCAL_UNTRUSTED_CAPTURE_NOT_PIT_AUTHORITY`. Trusted time, independent
signature, builder-inaccessible copy, external CAS, source semantics,
scientific admission, solver input, training, selection, holdout, candidate,
promotion and production must all remain false. Any contrary state stops the
workflow.

## 10) Packaged CH LT preregistration-supersession verification

Use this read-only check only from the exact sealed repo-local runtime selected
by the current handoff. Never invoke it from checkout Python and never use a
relative evidence path. Set the caller-held runtime receipt identity, then run
from a foreign cwd below `build/`:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH='C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260729-v24.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256='2eeb80e212cff3301c4e8a9349cffc2e93a41e20514dd2cd4cf6d95749219c2d'

C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v24-successor-base\python.exe `
  -I -B -m pfc_shaping.cli.verify_ch_lt_preregistration_supersession `
  --evidence-root C:\Users\jbattaglia\PFC_LT `
  --registry C:\Users\jbattaglia\PFC_LT\.planning\phases\14-lt-audit-remediation\CH-LT-PIT-PREREGISTRATION-SUPERSESSION-20260727.json `
  --expected-registry-sha256 76dba0b05948d336268b0a50c16df82ecd1c5138626c3fa87ee219148cb8e5e8
```

Expected exit is 0 with status
`ACTIVE_FAIL_CLOSED_V1_SUPERSEDED_NO_EXECUTABLE_SUCCESSOR`, ten blockers and
every authority false. Exit 2, any changed hash/path, missing runtime seal or
positive authority stops the workflow. A successful check does not authorize
execution, holdout access, publication, promotion or production. Preserve
stdout/stderr and the exact runtime receipt. Do not edit the registry or
repair a failed namespace in place.

## 11) Selected local CH hourly prospective-capture ledger

This workflow is a local structural diagnostic only. It does not establish
trusted `available_at`, provider revision lineage, official product semantics,
scientific truth, rolling-origin evidence, holdout evidence or production
authority. Operators must resolve the current evidence through
`.planning/phases/14-lt-audit-remediation/CH-LT-CURRENT-EVIDENCE-SELECTION-V3-20260731.json`,
SHA-256
`6518f7e876ce1c233fc055d3d20ad213088d361d784afbe5aa16d4f165e744f7`,
registry ID
`748aba2e2f85711d0a5dcdb07e0acacbf8dbce7a76ab4a4b07ef48371ec25488`.
It selects prospective-capture selection v4, SHA-256
`1adcf532d4df2508491dd4a1fb7ed5429d111a8d3721d900eb9529e4594575be`,
and ledger v10 only. V1-v9 are retained historical evidence. The market-time
contract v1 intentionally embeds v5 as its frozen audit-time observation; it
does not control the current prospective ledger selection.

Resolve and validate those links before invoking the ledger CLI:

```powershell
python -B -m scripts.audit_ch_lt_current_evidence_selection `
  --registry C:\Users\jbattaglia\PFC_LT\.planning\phases\14-lt-audit-remediation\CH-LT-CURRENT-EVIDENCE-SELECTION-V3-20260731.json `
  --expected-registry-sha256 6518f7e876ce1c233fc055d3d20ad213088d361d784afbe5aa16d4f165e744f7
```

Expected status is `CURRENT_SELECTION_LINKS_VALID_LOCAL_ONLY_NO_GO`, current
ledger version `v10`, embedded historical version `v5`, independent rolling
origins `0`, and every scientific/publication/promotion/production authority
false. The validator binds the registry ID and exact bytes, recursively
reconstructs all five capture/acquisition/audit chains, exact-compares the
ledger and validates supersession v4 -> v3 -> v2 -> v1. Absolute, traversal,
drive-relative, NTFS ADS and ambiguous Windows path forms fail closed. Exit 2
or any link/replay drift stops the workflow. This checkout audit resolves
local operator ambiguity; it is not an external signature or production
authority.

Run from exactly `C:\Users\jbattaglia\PFC_LT`, with every mutable path below
repo `build/`:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH='C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260730-v40-origin-registry-v2-chain.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256='651c8caa548d2e1fdd874f7173397c6f2a05a5d2f3b01ae4a084fbf49468f561'
$env:TEMP='C:\Users\jbattaglia\PFC_LT\build\ledger-cli-temp-v40-v10-retry'
$env:TMP=$env:TEMP

C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v40-origin-registry-v2-chain\python.exe `
  -I -B -m pfc_shaping.cli.build_ch_lt_prospective_capture_ledger `
  --repo-root C:\Users\jbattaglia\PFC_LT `
  --request C:\Users\jbattaglia\PFC_LT\build\prospective-ledgers\ch-hourly-local-ledger-request-20260731-v5.json `
  --expected-request-sha256 bc79c1f8a793ed3391ae3dd4a7df5b7793b3319144162b7ecaf55d56430fb136 `
  --output C:\Users\jbattaglia\PFC_LT\build\prospective-ledgers\ch-hourly-local-ledger-20260731-v10.json `
  --execution-receipt-output C:\Users\jbattaglia\PFC_LT\build\prospective-ledgers\ch-hourly-local-ledger-execution-receipt-20260731-v10.json
```

Expected terminal identities are ledger SHA-256
`3805b71f1368a0e742c8627d4995a85c791de1360257f97b78be0b1665723140`,
ledger ID
`cd759fb51a042d0a9f7b3a0c67f2badb3c6aadb098b33a37800bc6bb8c4f25df`
and execution-receipt SHA-256
`cd6f4e5134293546e81ae6f4628e9d31211d3598afd3f1bf2829572e75e9f575`.
V10 contains 776 contiguous native-hourly observations through
`2026-07-31T06:00:00Z`; its 3,104 quarter-hour transport rows are stepwise
proxies and add no independent information. V6 timed out without a terminal
artifact. V7 failed closed because the excluded full-day capture ended after
its capture time; that full-day namespace is retained as local archive but is
never prospective-ledger input.
An exact retry must exit zero and retain the same bytes. Any divergent target,
changed runtime receipt, path outside `build/`, positive authority claim,
T057/holdout marker, gap, overlap or replay mismatch must fail closed. Never
delete or relabel v1-v10, never launch a project executable, and never request
admin, Defender or AppData access. Production remains `NO_GO`.

## 12) Swiss auction market-time transition gate

The locally observed Swiss regime remains hourly for both day-ahead and
intraday auction truth. EPEX currently describes 3 November 2026 as the
*planned first trading day* for the 15-minute switch. This is neither proof of
go-live nor an effective first-delivery UTC boundary. The model's 15-minute
valuation grid therefore remains distinct from native market observations.

The governing contract is
`.planning/phases/14-lt-audit-remediation/CH-MARKET-TIME-REGIME-CONTRACT-20260730.json`,
SHA-256
`71711ae80b64556b8deab88e70581e1c7a0ef7c684d672b2cb550f2058c19c25`,
contract ID
`898d35e37a2df9dc9814039698eb605fdf220ece33d2035c7c7a2ea1b7bc9dba`.
It admits no post-transition quarter-hour market truth until all eight
requirements recorded in that contract are independently satisfied.
Its embedded ledger v5 is a frozen observation used by audit receipt v1, not
the current prospective-ledger pointer. Current ledger resolution is governed
only by the current-evidence registry in section 11; mixing these domains or
using embedded v5 as the current ledger fails closed.

Run the local sealed audit from exactly `C:\Users\jbattaglia\PFC_LT`, with no
administrator right, network access, project executable, browser runtime or
mutable path outside repo `build/`:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH='C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260730-v29.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256='f4ff1d309a7800056e254fe506b61cea5a46e691308e9263d1a9ab701825e8c3'
$env:TEMP='C:\Users\jbattaglia\PFC_LT\build\market-time-regime-audit-temp-v29'
$env:TMP=$env:TEMP

C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v29-market-time-base\python.exe `
  -I -B -m pfc_shaping.cli.audit_ch_market_time_regime `
  --repo-root C:\Users\jbattaglia\PFC_LT `
  --contract C:\Users\jbattaglia\PFC_LT\.planning\phases\14-lt-audit-remediation\CH-MARKET-TIME-REGIME-CONTRACT-20260730.json `
  --expected-contract-sha256 71711ae80b64556b8deab88e70581e1c7a0ef7c684d672b2cb550f2058c19c25 `
  --runtime-receipt C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260730-v29.json `
  --expected-runtime-receipt-sha256 f4ff1d309a7800056e254fe506b61cea5a46e691308e9263d1a9ab701825e8c3 `
  --output C:\Users\jbattaglia\PFC_LT\build\market-time-regime-audits\ch-market-time-regime-audit-20260730-v1.json
```

The selected runtime receipt SHA-256 is
`f4ff1d309a7800056e254fe506b61cea5a46e691308e9263d1a9ab701825e8c3`.
The expected audit receipt SHA-256 is
`9288b10f535974bb512ff33fabb330d3e86fc23a069c3344397c5874b9fdfa68`;
an exact retry must exit zero and preserve these bytes. Runtime attempts v27
and v28 are retained as non-selectable failure evidence.

Before any future regime admission, operations must collect and hash an
official effective-delivery notice, its publication and revision vintages,
the exact UTC boundary, product/auction scope, delivery-area scope, source
schema and units, DST behavior, and complete native quarter-hour coverage.
The audit must then be rerun against a reviewed successor contract. Monitoring
must separately watch EPEX notices, Swissgrid implementation notices, source
schema changes, missing/duplicate quarter-hours, DST cardinalities
(92/96/100), and revisions published after each forecast origin.

Rollback is fail-closed: retain the last admitted hourly regime, quarantine
unadmitted quarter-hour data, revoke the candidate manifest pointer, and
re-run the last sealed audit. No switch is performed directly on this laptop;
any later production change requires an independently governed atomic
manifest promotion with rollback evidence. Production remains `NO_GO`.

## 13) Official EEX DataSource REST API v2 capture path

This is the successor acquisition path for CH forward levels. The existing
desk workbook remains local quarantine evidence only; it is never silently
relabelled as an official DataSource response. EEX REST API v1 is forbidden.
The pinned public guide is release 006, published 2026-07-16, SHA-256
`d24cc35c7600622cba44d00dd988045be20ec01ad325bbb85552626bfcc7ad81`.
Its local research copy is
`build/eex-datasource-v2-research/EEX_Group_DataSource_REST_API_v2_User_Guide_v006.pdf`.

The implementation is `pfc_shaping.data.eex_datasource_v2_capture`. It accepts
only an exact canonical spec, `GET`, root
`https://api.eex-group.com/v2`, JSON, `POWER`, area `CH`, a literal trade
date/shortcode/maturity/ISIN and a repo-local hash-bound OpenAPI YAML. Rolling
symbols, redirects, credential query parameters, duplicate JSON keys, mixed
product identities, responses at the 60,000-record provider limit, non-EUR/MWh
settlements and any authority claim fail closed. Reference data under `/rd/`
must be captured first. A settlement spec recursively reopens and rehashes its
reference spec, guide, OpenAPI, CA, manifest, headers and raw response before
performing its own request.

EEX does not guarantee a uniform result ordering. Raw provider bytes and order
are therefore preserved exactly. For one exact settlement instrument, the
capture validates a unique chronological lifecycle of one `New` followed by
zero or more `Change` records after sorting by `Tm`; it never collapses or
rewrites revisions.

The access token is read only from
`PFC_EEX_DATASOURCE_ACCESS_TOKEN`, is required to be printable ASCII, and is
never accepted in a spec, query, CLI argument, manifest or stored header. Do
not put the token in PowerShell history or a checked-in `.env` file. Provision
and rotate it through the FMV-approved secret channel. A missing/expired token
or missing entitlement produces HTTP 401/403 and is an external blocker, not a
reason to weaken the contract.

The client also ignores ambient `SSLKEYLOGFILE`: it creates an explicit TLS
client context, requires hostname verification and `CERT_REQUIRED`, disables
TLS key logging, requires TLS 1.2 or later and loads only the independently
allowlisted single root bound by the spec hash. The code allowlist is empty by
default. Consequently the observed FMV interception root below cannot be used
for a real capture until Security/IT approves its exact SHA-256 and the token
handling/egress model in an independently reviewed policy change.

Python cannot rely on the ambient Windows certificate store on this managed
workstation. The observed EEX leaf on 2026-07-31 was issued by
`CN=pa850.net.fmv.ch`. The corresponding public self-issued root was exported
read-only to
`build/eex-datasource-v2-research/pa850-net-fmv-ch-observed-root-20260731.der`,
SHA-256
`7f94c8e3bc552d5196c745be0b19b795319f76d4e6bb0a6a965d095f5c697d32`.
An unauthenticated preflight with that single root negotiated TLS 1.3,
`TLS_AES_256_GCM_SHA384`, leaf SHA-256
`9f12be331fb1ae82a833d3c0168987f22027695cab6407b9b7ae88543af1cd54`
and returned HTTP 403 as expected. These are local observations only. IT must
independently attest the approved interception root and rotation procedure
before a governed capture spec may select it.

Before the first real capture, obtain from the authenticated EEX DataSource
Hub and hash the exact Derivatives OpenAPI YAML, confirm the subscribed Swiss
Power settlement endpoints and fields, obtain the exact `/rd/` identity for
each required BASE/PEAK month/quarter/year contract, and record FMV licensing
and redistribution constraints. Keep the guide, YAML, approved CA and every
mutable response below repo `build/`. Do not copy a token into that directory.
As of 2026-07-31, the token, exact authenticated OpenAPI and Security-approved
CA policy are absent: no real capture is runnable and this is an explicit
external blocker, not a prompt to bypass the contract.

For each exact product, create a fresh canonical reference spec and compute
its caller-held SHA-256. Once all external prerequisites are admitted, execute
the packaged module through the workspace supervisor and one exact admitted
repo-local runtime. The checkout compatibility wrapper is not a supported
operations interface:

```powershell
$runtime='C:\Users\jbattaglia\PFC_LT\build\conda-runtime-<admitted-id>\python.exe'
python -B -m scripts.run_workspace_local `
  --run-id <fresh-supervisor-id> `
  --wall-timeout-seconds 180 `
  -- $runtime -I -B -m pfc_shaping.data.eex_datasource_v2_capture `
  --repo-root C:\Users\jbattaglia\PFC_LT `
  --spec C:\Users\jbattaglia\PFC_LT\build\eex-datasource-v2-specs\<reference-spec>.json `
  --expected-spec-sha256 <caller-held-reference-spec-sha256>
```

The command above is the future target contract, not permission to run a token
today. The supervisor strips ambient secrets and currently rejects any nonempty
EEX token before workspace checks, subprocesses or receipts. Forwarding stays
disabled until an independently admitted runtime path+hash+manifest, immutable
execution bytes and a minimal secret handoff close worker/target TOCTOU. This
prevents a mutable `build/conda-runtime-*` lookalike or mutable worker from ever
receiving the Bearer. Once that external authority exists, the supervisor must
still enforce the global wall timeout, capture stdout/stderr and emit its normal
execution receipt. The client serializes in-flight requests with an OS-released
single-link lock and enforces at least one second between request starts. Then
create a different settlement spec that binds the
exact reference manifest and response SHA-256. Use a new capture ID; never
retry or repair a failed directory in place. Run the same command with the
settlement spec. Success becomes visible only after an atomic directory rename.
A caught failure remains under `_incomplete-<capture-id>` with a redacted
terminal `capture-failure.json`, HTTP class, safe `Retry-After`/correlation ID
when present and a mandatory fresh spec/capture ID disposition. A process kill
may leave attempt-only indeterminate staging, which must be quarantined and
inventoried before a new ID; it is never consumed as a finalized capture.

Even a successful pair remains
`LOCAL_EEX_DATASOURCE_V2_CAPTURE_NOT_PIT_AUTHORITY`: workstation time, the
HTTP `Date` header and the locally observed TLS chain are not trusted-time or
source-authenticity authority. Independent trusted time, acquisition
signature, builder-inaccessible immutable copy, external CAS/WORM receipt and
fresh HEAD, license/entitlement attestation and independent OpenAPI response
conformance are still mandatory before vintage normalization or monthly-solver
hard-level admission. No training, selection, holdout, candidate, publication,
promotion or production use is allowed. Production remains `NO_GO`.

## 14) Packaged CH LT prospective truth and scoring rehearsal V6

The current local rehearsal is
`.planning/phases/14-lt-audit-remediation/CH-LT-LOCAL-FUTURE-ORIGIN-SELECTION-V6-20260731.json`,
SHA-256
`11966d1ee85ace46e97006fa74f8aab4789c71f7438de909ed71541aab480df7`.
It is the same local V7 origin as V5, not an additional origin. It has no
scientific, evaluation, publication, promotion or production authority.

Before every command, require both the PowerShell cwd and Git top-level to be
exactly `C:\Users\jbattaglia\PFC_LT`. Use only repo-local mutable paths and a
fresh run ID. Never request admin, ACL or Defender exceptions.

Audit the sealed V6 chain:

```powershell
$expected='C:\Users\jbattaglia\PFC_LT'
$cwd=(Get-Location).Path
$root=(git rev-parse --show-toplevel).Trim().Replace('/','\')
if ($cwd -cne $expected -or $root -cne $expected) {
  throw "Workspace mismatch: cwd=$cwd root=$root"
}

python -B -m scripts.run_workspace_local `
  --run-id <fresh-v6-audit-run-id> `
  --wall-timeout-seconds 900 `
  -- build\pytest-runtime-v1\python.exe -B `
  -m scripts.audit_ch_lt_local_future_origin_selection `
  --registry C:\Users\jbattaglia\PFC_LT\.planning\phases\14-lt-audit-remediation\CH-LT-LOCAL-FUTURE-ORIGIN-SELECTION-V6-20260731.json `
  --expected-registry-sha256 11966d1ee85ace46e97006fa74f8aab4789c71f7438de909ed71541aab480df7
```

The selected terminal run is `fvaudit6`. It exited zero with status
`VALID_LOCAL_FUTURE_ORIGIN_REHEARSAL_V7_PACKAGED_MODULE_SOURCES_BOUND_NONCOUNTABLE_NO_GO`.
Execution/supervisor receipt SHA-256 values are
`2ebe566e52e25c0cbfcc336400585d7a86f8f4b2cf81b7779b0df1121f33c3a6` /
`28940f7e3f0ba681accb5dc0adb0c105f5f719751c06f3b88638bd1f3855f5ff`.

The selected regression commands are the same supervisor form with these
targets:

```text
build\pytest-runtime-v1\python.exe -B -m pytest tests\test_build_ch_lt_native_hourly_truth_bundle_script.py tests\test_ch_lt_prospective_hourly_scoring.py -q -p no:cacheprovider
build\pytest-runtime-v1\python.exe -B -m pytest tests\test_lt_package_contract.py -q -p no:cacheprovider
```

Selected results are `ftruth36` (`17 passed in 12.28s`) and `fpack38`
(`26 passed in 2.69s`). Do not call the broad matrices green: `fpub38` has one
ACL-affected failure after 235 passes and two skips; `fpack36` has six
repo-local Windows temp access/cleanup failures after 69 passes, 12 skips and
one deselection. Route those environment-sensitive branches to the governed
standard-user CI runner.

After any source or wrapper edit, first require a stable V6 source hash window.
The final current-byte local replay used
`build/pytest-runtime-v2-final/python.exe` and returned V6 audit exit zero,
`17 passed` and `26 passed`. Its captured five-entry `sys.path` is entirely
repo-local and contains the canonical checkout root exactly once. Do not launch
the supervisor from an external parent while naming a relative target: it will
correctly bind the parent interpreter and that attempt is non-selectable.

Truth opening procedure for a delivery month:

1. Keep the month closed until its complete native-hourly delivery interval is
   mature. Partial-month truth is forbidden.
2. Re-audit the exact V6 chain and rehash the commitment, predictions, target
   inventory, scoring contract and selected ledger. Any drift fails closed.
3. Require the actual local wall clock and declared open time to be at or after
   the month end; the declared time must not exceed the actual wall clock.
   This remains untrusted local-time evidence and cannot make the origin
   countable.
4. Run `pfc_shaping.validation.ch_lt_native_hourly_truth_bundle` from the
   admitted package namespace with exact caller-held ledger/commitment hashes,
   delivery month, declared open time and a fresh output directory below
   `build/`. The module must publish the truth bundle and post-read publication
   receipt atomically.
5. Only after exact receipt hashing, run
   `pfc_shaping.cli.score_ch_lt_structural_prediction_commitment` with the exact
   commitment, predictions, scoring-contract and truth-publication-receipt
   hashes. Never pass the truth bundle directly around the receipt boundary.
6. Quarantine any divergent staging, maturity, grid, fixture/real, hash or
   recursive-ledger failure. Do not repair published bytes in place and do not
   retune this origin after truth.

A production-capable schedule still requires an assigned FMV owner, measured
provider-lag SLA, UTC cadence, lease, retries, watermark, completeness and DST
alerts, immutable event IDs, missed-run reconciliation, retention, independent
trusted time/signature, linearizable registration and builder-inaccessible
external CAS/WORM. Until then independent countable origins remain zero.

The first quality programme must accumulate preregistered independent rolling
origins and separately report level error, within-month shape error, tails,
calibration, scenario coherence and capture/economic metrics. The current
0/108 negative predictions are a support/calibration risk to test. They are not
permission to force negative forecasts. T057 remains sealed. Monthly solver
level authority, LT/CT separation, OMPEX benchmark-only policy and strict
production `NO_GO` remain unchanged.
