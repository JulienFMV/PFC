# Provider-evidence verifier operator runbook

This zipapp verifies local provider acquisition evidence under an exact,
content-addressed Python dependency closure. It is read-only with respect to
the acquisition and emits two separate append-only files:

- a business audit whose runtime status is always
  `DIRECT_FUNCTION_CALL_NOT_RUNTIME_AUTHORITY`;
- a hash-bound runtime observation whose status is
  `LOCAL_RUNTIME_OBSERVATION_NOT_AUTHORITY`.

Both files set `production_authorization=false`. The verifier cannot sign,
publish, promote or authorize production.

## Current local selection (2026-07-29)

The sole selectable local verifier is
`build/provider-verifier-20260729-v18.pyz`, SHA-256
`7f17be8de8e78ba5a063903c7ea459baed0372b70a128ffab3fb8b17f69b19c5`,
64,877 bytes, 17 members, source revision
`99be2ce84325789aeacda69a41997766f1f365abe6747fa04b7d21bdad6b9a34`
and dependency tree
`0ecb7997997cc124375e92614ca08d9c5274c683c6738448b9bd3c5eafaf78f1`.
V17 has identical bytes and is retained only as the same-host reproducibility
witness. It is not the operator-selected name.

Provider verifier v14 and v15/v16 are historical evidence and are explicitly
non-selectable. Likewise, the CH audit without a version suffix (v2) and the
v3 audit are superseded. Do not delete, repair, relabel or execute them as
current evidence. The selected CH local-quarantine observation is audit v4,
SHA-256
`638a6fc8887b867957fb8cb0ba2cafcf07c00c33fd2620c51e3bca366c2bfc02`,
with runtime receipt SHA-256
`f28bd6d4e93235467ad4a8c4a2102d6aee5ef7231a96ae3f3a512ccbcb12d82e`.
These pins grant no scientific, candidate, runtime or production authority.

## Preconditions

Run only from `C:\Users\jbattaglia\PFC_LT` and verify the canonical root first:

```powershell
$expected = 'C:\Users\jbattaglia\PFC_LT'
$cwd = (Get-Location).Path
$root = (git rev-parse --show-toplevel).Trim().Replace('/','\')
if ($cwd -cne $expected -or $root -cne $expected) {
  throw "Workspace mismatch: cwd=$cwd root=$root"
}
```

IT must provision:

- an approved wheelhouse and a dependency closure generated from `uv.lock`;
- an existing dedicated scratch directory with at least 512 MiB free;
- a dedicated verifier service identity with no untrusted same-token process;
- read-only access to the dependency closure and acquisition evidence;
- a separate audit-output namespace whose DACL lets the service enumerate the
  directory and create, write, flush, read and hardlink files; the writer is
  programmed to remove its temporary name, but the DACL cannot restrict delete
  authority to that name; this namespace is explicitly mutable quarantine
  under the verifier identity and is not an immutable evidence store;
- effective network denial and no signing/mTLS private-key capability.

The supervisor runtime is a private direct child of the governed scratch. The
acquisition, prior audit, dependency closure and both output paths must be
disjoint from that scratch tree and from the supervisor child. The business
audit and runtime receipt must also be distinct from one another and from every
protected input. The CLI checks lexical paths, ancestry and existing physical
aliases before business work.

The output rights above are required by `write_durable_exact_bytes`: it creates
and fsyncs a temporary file beside the target, hardlinks that file to the new
target name, then removes the temporary name. On NTFS the two names share one
file object and DACL; on POSIX unlink authority is directory-based. Therefore
the same writer identity cannot both remove the temporary name and prove it
lacks authority to remove the published name. Do not claim that impossible
local immutability property. IT must instead prove the positive
create/write/read-list/hardlink/delete-temporary operations under the exact
service identity and classify both JSON files as mutable local observations.

Durable retention requires a separate post-run handoff. Ideally the verifier
identity never receives write authority in the retention namespace. An
independent retention identity must, in this exact order:

1. ingest the two exact local byte strings into approved WORM or external CAS;
2. reread and hash the retained bytes;
3. seal WORM/CAS or remove verifier write/delete authority;
4. verify final owner/DACL and run negative overwrite/delete probes under the
   verifier identity;
5. reread and hash the retained bytes again, proving stable identity; and
6. only then emit a signed receipt binding both hashes, retained object
   identities, owner/DACL or WORM/CAS state and all control results.

Negative probes apply to the retained copy, never to the writer namespace.
Until all six steps and the final receipt exist, neither local JSON file is
durable authority. A leftover temporary is an incident, not permission to
broaden the DACL or relabel local output as immutable.

Required environment:

```powershell
$env:PFC_LT_VERIFIER_DEPENDENCY_ROOT = '<exact-content-addressed-site-packages>'
$env:PFC_LT_VERIFIER_SCRATCH_ROOT = '<dedicated-existing-scratch>'
```

The worker receives only an allow-listed host environment plus these roots.
Caller API keys, tokens, passwords and secrets are not inherited.

## Build and reproducibility audit

Use two new output names. Never overwrite or delete an earlier verifier:

```powershell
python -m scripts.build_lt_provider_verifier_zipapp `
  'C:\absolute\build\provider-verifier-a.pyz' `
  --dependency-root $env:PFC_LT_VERIFIER_DEPENDENCY_ROOT `
  --dependency-receipt '<absolute-dependency-closure-receipt.json>' `
  --wheel-directory '<absolute-approved-wheelhouse>'

python -m scripts.build_lt_provider_verifier_zipapp `
  'C:\absolute\build\provider-verifier-b.pyz' `
  --dependency-root $env:PFC_LT_VERIFIER_DEPENDENCY_ROOT `
  --dependency-receipt '<absolute-dependency-closure-receipt.json>' `
  --wheel-directory '<absolute-approved-wheelhouse>'

python -m scripts.build_lt_provider_verifier_zipapp `
  'C:\absolute\build\provider-verifier-a.pyz' --audit
python -m scripts.build_lt_provider_verifier_zipapp `
  'C:\absolute\build\provider-verifier-b.pyz' --audit
Get-FileHash -Algorithm SHA256 `
  'C:\absolute\build\provider-verifier-a.pyz', `
  'C:\absolute\build\provider-verifier-b.pyz'
```

Both artifact hashes, sizes, member counts, source revisions and dependency-tree
hashes must match exactly. Each audit JSON field must match after excluding the
`artifact` field; that field must instead equal the separately audited path for
its own artifact. A build failure or any other mismatch is non-retryable until
the input closure or source difference is explained.

## Runtime admission check

The only supported launcher is CPython 3.11 with all required isolation flags:

```powershell
python -I -S -B '<absolute-provider-verifier.pyz>' runtime-check
```

Exit `0` requires:

- `status=PASS`;
- `captured_artifact_sys_path_count=1`;
- `source_artifact_sys_path_count=0`;
- `captured_dependency_root_sys_path_count=1`;
- `source_dependency_root_sys_path_count=0`;
- exact locked dependency versions and tree hash;
- `runtime_authority=false` and `production_authorization=false`.

The supervisor creates the worker suspended on Windows, assigns it to a
kill-on-close Job Object, then resumes it. Timeout is fixed at 900 seconds.
Timeout, interruption, Job assignment failure or cleanup failure returns exit
`50`. Alert on every exit `50` and every wall time reaching the deadline.

## Audit a v2 acquisition

Hold the manifest hash independently before invocation. Both destination
parents must already exist and must be outside every protected root:

```powershell
$acquisition = '<absolute-v2-acquisition-directory>'
$manifest = Join-Path $acquisition 'acquisition-build-manifest.json'
$manifestSha256 = (Get-FileHash -Algorithm SHA256 $manifest).Hash.ToLower()

python -I -S -B '<absolute-provider-verifier.pyz>' audit-acquisition `
  --acquisition-directory $acquisition `
  --expected-manifest-sha256 $manifestSha256 `
  --output-json '<absolute-new-business-audit.json>' `
  --runtime-receipt-json '<absolute-new-runtime-receipt.json>'
```

On exit `0`, independently hash both outputs and verify that
`runtime-receipt.audit_result.sha256` equals the business-audit hash. The
business audit remains local unsigned quarantine evidence. The runtime receipt
is an unsigned observation and never upgrades its authority.

## Audit a legacy v1 resolution

Hold both legacy hashes independently:

```powershell
$acquisition = '<absolute-v1-acquisition-directory>'
$manifest = Join-Path $acquisition 'acquisition-build-manifest.json'
$priorAudit = '<absolute-prior-audit.json>'
$manifestSha256 = (Get-FileHash -Algorithm SHA256 $manifest).Hash.ToLower()
$priorSha256 = (Get-FileHash -Algorithm SHA256 $priorAudit).Hash.ToLower()

python -I -S -B '<absolute-provider-verifier.pyz>' audit-legacy-resolution `
  --acquisition-directory $acquisition `
  --expected-manifest-sha256 $manifestSha256 `
  --prior-audit $priorAudit `
  --expected-prior-audit-sha256 $priorSha256 `
  --output-json '<absolute-new-legacy-result.json>' `
  --runtime-receipt-json '<absolute-new-legacy-runtime-receipt.json>'
```

An exit `50` with `provider transform runtime fingerprint mismatch` is the
required fail-closed result for an old acquisition that cannot be exactly
replayed under the locked closure. Do not patch or relabel the legacy evidence.
No business output or runtime receipt may exist after this rejection.

## Retry, residue and incident handling

- Exit `0`: retain command, wall time, stdout, both output hashes, verifier
  hash, source revision and dependency-tree hash.
- Exit `50`: preserve stderr and inputs; never weaken the closure or rewrite
  evidence. Retry only the exact command after an explained infrastructure
  repair. A divergent existing output is never replaced.
- Inspect scratch after every run:

  ```powershell
  Get-ChildItem -LiteralPath $env:PFC_LT_VERIFIER_SCRATCH_ROOT `
    -Directory -Filter 'vv-*'
  ```

- A new `vv-*` or nested `vd-*` residue is an incident. Stop scheduling new
  verifier work, capture process tree, owner/DACL, timestamps and hashes, and
  quarantine the residue through the IT incident procedure. Do not delete it
  automatically.
- Historical residues predating the current run are not evidence of a current
  cleanup failure. Record and investigate them separately.

Before any production enablement, IT must retain signed DACL/owner evidence,
wheel/Python/SBOM/release attestations, network-denial evidence and real drills
for parent kill, descendant kill, timeout, locked-file cleanup, concurrent
writes, SMB behavior and power loss. Local PASS results do not replace those
external attestations.
