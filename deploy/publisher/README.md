# LT snapshot publisher delivery contract

The publisher never builds provider captures. Exact provider bodies must first
pass through the separately admitted
`python.exe -I -B -m pfc_shaping.cli.governed_acquisition_builder` module under
a keyless quarantine-only identity, following section 1.2 of
`pfc_shaping/tools/OPERATIONS.md`. Only a bundle subsequently verified and signed
by the independent acquisition/timestamp/journal authorities is eligible as a
publisher `--source-bundle`; an unsigned builder manifest remains
`NOT_PUBLISHED` and confers no production authorization.

First build a dedicated dependency closure directly from the eleven retained
wheel archives in the offline FMV wheelhouse. The builder selects only the
CPython 3.11 / Windows x86-64 wheels named by `uv.lock`, verifies each archive
SHA-256 and size, replays its complete `RECORD`, validates `METADATA`, `WHEEL`
tags and installation paths, and emits a canonical closure receipt:

```powershell
C:\runtime-build\python.exe -m scripts.build_snapshot_publisher_runtime_closure `
  --wheel-directory C:\wheelhouse\publisher-cp311-win_amd64 `
  --output C:\staging\publisher-site-packages `
  --receipt-output C:\staging\publisher-dependency-closure.json
```

Recorded bytecode and nested-wheel residue are hash-verified and then excluded
from the installed closure; `.pth` files remain forbidden. The publisher is
then delivered as the audited deterministic `.pyz` bound to the exact closure,
receipt, wheel evidence and build interpreter:

```powershell
C:\runtime-build\python.exe -m scripts.build_snapshot_publisher_zipapp build `
  --dependency-root C:\staging\publisher-site-packages `
  --dependency-receipt C:\staging\publisher-dependency-closure.json `
  --wheel-directory C:\wheelhouse\publisher-cp311-win_amd64 `
  --output C:\staging\fmv-lt-snapshot-publisher.pyz
```

Audit the finished artifact before deployment:

```powershell
C:\runtime-build\python.exe -m scripts.build_snapshot_publisher_zipapp audit `
  --artifact C:\staging\fmv-lt-snapshot-publisher.pyz
```

The auditor captures one stable, mono-linked byte image and computes the
reported artifact SHA-256 from those exact audited bytes. A path replacement
or in-place change during capture fails closed.

The
positive inventory excludes the bootstrap signer, acquisition signer and
test-only SQLite anchor. The runtime rejects foreign private-key variables in
all phases and rejects the request signing key in `cas` and `finalize`.
The embedded source-inventory digest binds exact application bytes but does not
claim a Git revision. CI must bind the audited zipapp hash, curated clean commit,
dependency lock/SBOM and environment-contract hash in a separately signed FMV
release-attestation manifest before deployment.

The zipapp build independently replays the receipt against the wheel archives
and every extracted byte, then requires the bind-time tree hash and file count
to remain identical to that validated receipt; a standalone, stale or forged
receipt is not trusted. The
`.pyz` is deliberately not a self-contained Python runtime. Its embedded
`FMV-SNAPSHOT-PUBLISHER-RUNTIME-CONTRACT.json` binds the exact CPython version,
platform, launcher and base-launcher hashes, ABI-library hashes, all closure
paths/bytes, the exact distribution inventory and the repository `uv.lock`
hash plus every selected wheel and `RECORD` digest. Any extra or modified file
changes the admitted tree hash. IT must still prove wheel approval, the
Python/standard-library image, SBOM and read-only
mount in the external signed release attestation; application self-checks do
not establish host integrity.

Invoke the artifact through the admitted interpreter; a `.pyz` is not a native
Windows executable:

```powershell
$env:PFC_SNAPSHOT_PUBLISHER_DEPENDENCY_ROOT = 'C:\runtime\publisher-site-packages'
$env:PFC_SNAPSHOT_PUBLISHER_SCRATCH_ROOT = 'C:\pfc-publisher-scratch'
$env:PFC_SNAPSHOT_PUBLISHER_WORKER_TIMEOUT_SECONDS = '900'
$env:PFC_SNAPSHOT_PUBLISHER_ADMISSION_SLO_SECONDS = '300'
C:\runtime\python.exe -I -S -B C:\runtime\fmv-lt-snapshot-publisher.pyz --help
```

The scratch parent must already exist, be writable only by the dedicated
publisher service identity and administrators, contain no reparse points, and
use a short absolute path. Admission reserves a 240-character Windows capture
budget; a dependency destination beyond it fails before the first copied byte.
Capacity must cover the admitted closure plus concurrent workers and residue,
not merely the built-in 512 MiB safety floor.

`-I -S` prevents the current directory, user site, `site` initialization and
`.pth` execution before admission; `-B` prevents runtime bytecode mutation. The
zipapp hashes the complete dependency closure and verifies the exact interpreter
and distribution inventory, then copies every admitted byte into a random
process-private root. It revalidates the complete source tree, captured tree and
captured distribution inventory before making only that private root importable.
The environment-provided source root is never added to `sys.path`. The capture is
made read-only and rehashed before publisher business modules load. A minimal
parent process owns one random supervisor scratch, creates a consumed one-shot
capability bound to its PID, the artifact path and that scratch, then starts the
admitted worker suspended. It assigns that worker to a parent-owned Windows
`KILL_ON_JOB_CLOSE` Job Object before resuming it. The three
reserved `_PFC_SNAPSHOT_PUBLISHER_*` worker variables are forbidden deployment
inputs; an unbound or forged worker context exits `50`. The parent removes the
complete scratch only after the worker exits and Windows has unloaded its
`.pyd` and `.dll` files. A transient cleanup error is cleared after a successful
retry. A successful worker followed by persistent cleanup failure exits `50`;
a failed admitted worker retains its original exit and sole machine-readable
failure document on stderr, while the independently sealed admission metric is
emitted as one JSON line on stdout. Monitor both structured streams, free space,
exit `50`, and supervisor residue. Legacy `fmv-pfc-publisher-runtime-*` residue from runtimes older than
schema v6, and `fmv-pfc-publisher-supervisor-*` residue after an abnormal parent
death, must be removed only after IT proves that no publisher Python process is
using it. Never delete either class from a live-service glob.

The zipapp also replays the embedded positive manifest and member hashes. The
external signed release attestation remains responsible for the initial
`__main__`/admission bootstrap, host image and storage-enforced read-only
isolation. That attestation must cover the exact artifact, source closure,
supervisor scratch parent and captured tree for the whole worker lifetime, and
must require a dedicated service identity with no untrusted co-process under
the same token. Ordinary application `chmod` is not this control. Any admission
mismatch exits `50` with `RUNTIME_ADMISSION_FAILED`.

Deploy `prepare`, `cas` and `finalize` as separate workload identities using
the exact environment contract in `environment-contract.json`. Secrets must
be mounted as read-only files; they must not be baked into the zipapp or image.
Docker mounts are physically present when the container starts. The deferred
capability delays only the paths and activation of permitted authorities; it
does not hide a mounted directory from hostile code already running under the
same token. Production therefore still requires a broker/HSM or an equivalent
separate-identity boundary for authority isolation.
The supervisor constructs the worker environment from the contract allowlist;
unknown variables are dropped and uncontracted `PFC_*`/`FMV_*` secret-shaped
names fail before worker creation. The initial one-shot worker capability carries
only a boolean indicating whether post-admission authority is expected; it never
contains private-key paths. Permitted request-signing and mTLS private-key paths
are published in a separate PID/token-bound capability only after dependency
verification/import and admission-metric sealing, then consumed and activated
once. Metric and capability names become visible only after complete write,
flush and `fsync`; the reader tolerates only the transient two-hardlink publish
window and rejects mono-linked divergent JSON immediately. Temporary PEM copies
used by `SSLContext.load_cert_chain` are created
inside the governed per-worker supervisor directory and removed before worker
exit; abnormal-parent residue remains an IT incident and must be handled under
the no-live-process rule above.

The exact manifest-locked dependency closure is part of the local runtime TCB.
A dependency actively hostile inside the admitted interpreter shares the worker
PID, token and privileges and cannot be isolated by another in-process Python
handshake. Removing dependencies from that TCB requires an external crypto
broker/HSM, dedicated service identity and signed artifact/SBOM attestation;
until those controls exist, production remains `NO_GO`.
`prepare` and `finalize` require a non-empty absolute `FMV_DATA_ROOT`; an
explicit `--data-root` must resolve to the same path or argument admission
fails before publication I/O.
The bootstrap authorization is produced offline by an independent IT identity
and is only consumed as a signed public document by `prepare`.

At startup the process copies every active and historical public trust file to
a private process directory, validates global identity disjointness, and then
registers the exact bytes and each historical-directory inventory as immutable
process-lifetime state. All downstream key loaders resolve those in-memory
bytes without reopening the filesystem; post-start replacement, deletion or
addition cannot alter trust. The mandatory registry includes publication,
acquisition, trusted-time, journal, promotion, rollback, model-governance,
release-request, quote-policy and all five Tier2 authorities. Active keys alone
authorize a new request or bootstrap; historical keyrings authenticate only an
exact already-committed replay. The reference authority freezes its active
signer and active/historical trust sets under the same rule.

`cas` archives only the exact external receipt. `finalize` performs an exact
operation lookup, requests a new nonce-bound HEAD observation on every retry,
archives it as `<observation_id>.json`, and only then projects `current.json`.
The scheduler uses `restartPolicy=Never`; retries are new one-shot jobs under
the exact policy below. Only `41` and `52` permit automatic retries, using
backoff `5, 15, 30, 60, 120` seconds and the same `operation_id`. This means
five retries after the initial run, six total runs at most. For `52`, exact
operation lookup precedes every retry.

| Exit | Meaning | Required action |
|---:|---|---|
| `0` | success | no retry |
| `2` | usage/configuration | correct configuration; no automatic retry |
| `30` | governance pending | obtain new authority evidence; no automatic retry |
| `40` | CAS conflict | no blind retry; reconcile fresh HEAD and create a new intent |
| `41` | transition busy | bounded retry of the exact operation and command |
| `50` | integrity/admission/authentication | quarantine evidence, open incident, no automatic retry |
| `51` | CAS committed, local projection repair required | perform the documented repair, then rerun `finalize` for the same operation; never issue a new CAS |
| `52` | indeterminate authority state | lookup the exact operation, then bounded same-operation retry only |

Every post-commit `51` result includes `commit_status=COMMITTED`, operation,
intent and receipt identities, receipt path/hash, projection state, retry
disposition and the structured local error.
When the underlying failure is a durable-writer residue, the result additionally
retains the complete `durable_artifact_repair.v2` document as `local_repair`.
Before an external commit, the same residue exits `50` with
`commit_status=NOT_COMMITTED` for PREPARE and the complete operator action.

Local evidence writes are process-crash recoverable. A canonical divergent
document is never replaced. Noncanonical or truncated bytes also remain
untouched: after exact authority lookup, the authoritative payload is written
only as a content-addressed repair candidate and
`LOCAL_ARTIFACT_REPAIR_REQUIRED` is returned. Promotion of that candidate
requires an external operator/CAS decision. Unowned temporary, legacy repair
lock or pending-candidate residue is never deleted automatically.

On Windows the reported durability classification is
`PROCESS_CRASH_RECOVERABLE_POWER_LOSS_UNPROVEN_ON_WINDOWS`. Python does not
provide the directory flush guarantee used by the POSIX path. Production
therefore remains blocked until IT proves power-loss behavior on the exact
target NTFS/SMB volume and service identities.

## Governed Windows container lane

`Dockerfile` is the only container lane for runtime v6. It deliberately keeps
the Windows execution model because the supervisor's suspended-start plus
`KILL_ON_JOB_CLOSE` proof is not equivalent to a generic Linux subprocess. The
base image has no default and must be an approved, preloaded, fully qualified
lowercase reference ending in `@sha256:<64 hex>`. A tag alone, `latest`, an
implicit Docker Hub name or a non-SHA-256 digest fails the build contract.
The governed lane fixes Windows isolation to `process`; host and base image
major/minor/build versions must match. Bind-mount access is therefore attested
for `ContainerUser`, not the Hyper-V `LocalSystem` principal. Changing isolation
mode is a contract change requiring a new ACL and compatibility proof.

The Dockerfile has no network acquisition step. The root `.dockerignore`
positive-lists only `uv.lock`, the publisher builders/contracts and the exact
runtime Python source inventory; CT, Power BI, data, outputs, local environment
files and private keys never enter the ordinary build context. Approved wheels
are copied into the single ignored and positive-listed
`build\publisher-container-wheelhouse` staging directory before the build. The
root ignore file names all eleven lock-selected archives explicitly; no `*.whl`
wildcard enters the context. The stager and an immediate pre-build re-audit
admit only exact lock-bound stable bytes, so an extra, torn or substituted
archive fails. The protected runner also binds the host build interpreter to
`PFC_PUBLISHER_BUILD_PYTHON_SHA256`. Same-token mutation after re-audit remains
part of the mandatory JIT-runner/isolated-daemon proof. This intentionally
avoids BuildKit named contexts:
Docker's documented Windows path can still use the legacy backend, for which
BuildKit does not yet have full feature parity. Both builder and final stages
use the same digest-pinned CPython image so the artifact's executable and
standard-library fingerprints match the runtime that executes it:

```powershell
$base = 'registry.fmv.local/pfc/python:3.11-windows@sha256:<approved-digest>'
docker image inspect $base | Out-Null  # it must already be present
$buildPython = 'C:\runtime-build\python.exe'
$buildPythonSha256 = '<protected-lowercase-sha256>'
if ((Get-FileHash $buildPython -Algorithm SHA256).Hash.ToLowerInvariant() -cne $buildPythonSha256) {
  throw 'build Python differs from protected digest'
}
$buildRoot = "$PWD\build"
if (Test-Path -LiteralPath $buildRoot) { throw 'unowned build residue exists' }
$null = [IO.Directory]::CreateDirectory($buildRoot)
& $buildPython -m scripts.stage_snapshot_publisher_container_wheelhouse `
  --source 'C:\approved\publisher-wheelhouse' `
  --output "$buildRoot\publisher-container-wheelhouse" `
  --manifest-output "$buildRoot\publisher-container-wheelhouse-manifest.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $buildPython -m scripts.stage_snapshot_publisher_container_wheelhouse `
  --audit `
  --source "$buildRoot\publisher-container-wheelhouse" `
  --manifest-output "$buildRoot\publisher-container-wheelhouse-manifest.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
docker build --network none --pull=false --isolation process `
  --build-arg "PFC_PUBLISHER_BASE_IMAGE=$base" `
  --file deploy\publisher\Dockerfile `
  --iidfile build\publisher-container-image-id.txt .
$imageId = (Get-Content build\publisher-container-image-id.txt -Raw).Trim()
if ($imageId -cnotmatch '^sha256:[0-9a-f]{64}$') {
  throw 'Docker did not return an immutable image ID'
}
```

The build regenerates the closure and zipapp twice inside that exact base,
requires byte-identical closure receipts and zipapps, replays
wheel/RECORD/lock evidence, runs all four admitted help surfaces as
`ContainerUser`, and embeds `container-build-manifest.json`, the exact wheel
staging manifest and `operations-contract.json`. The manifest binds the base
digest, Dockerfile, artifact, interpreter, receipt, closure, wheel staging and
operations contract. This is an internal build reproducibility check, not proof
that two separately provisioned builders emit the same final image. It explicitly
records missing SBOM/release attestation and
`production_authorization=false`; a successful local build is not a release.

The image deliberately declares no Docker `VOLUME`: every writable path must be
an explicit named or bind mount, so omission fails under `--read-only` instead
of silently creating an anonymous persistence surface. Production execution
must additionally use the registry image by digest, `--read-only`, the exact
phase mounts below and phase-specific network policy:

| Phase | `C:\pfc\scratch` | `C:\pfc\input` | `C:\pfc\data` | `C:\pfc\evidence` | Network |
|---|---|---|---|---|---|
| `prepare` | explicit dedicated RW | source bundle/bootstrap and other arguments RO | explicit FMV data root RW | absent unless an independently specified output requires it | none by default; anchor allowlist only for automatic HEAD discovery |
| `cas` | explicit dedicated RW | signed intent RO | absent | receipt output RW | external CAS/mTLS endpoint allowlist only |
| `finalize` | explicit dedicated RW | intent and anchor receipt RO | explicit FMV data root RW | observation directory RW | external CAS/mTLS endpoint allowlist only |

All public trust material and private key/certificate material is supplied via
separate read-only directory mounts, with environment variables pointing to
exact files below those mounts. On Windows, do not rely on single-file bind
mount semantics. Never mount `FMV_DATA_ROOT` read-only for `prepare` or
`finalize`: both phases perform governed writes. Do not expose a shell, Docker
socket or host root. This minimal command is only the non-authorizing read-only
smoke used by CI:

```powershell
docker run --rm --read-only --user ContainerUser `
  --isolation process `
  --mount 'type=volume,source=pfc-publisher-scratch,target=C:\pfc\scratch' `
  --network none `
  registry.fmv.local/pfc/publisher@sha256:<attested-image-digest> --help
```

Docker documents the Windows legacy-builder exception, the `--read-only`
runtime flag, and Windows persistent-storage semantics separately:
[image build](https://docs.docker.com/reference/cli/docker/image/build/),
[container run](https://docs.docker.com/reference/cli/docker/container/run),
and [Windows persistent storage](https://learn.microsoft.com/en-us/virtualization/windowscontainers/manage-containers/persistent-storage).

The publisher is a one-shot job, so `HEALTHCHECK NONE` is intentional. Health
is the terminal exit plus JSON-lines on both stdout and stderr. External collection must implement
the exact metric/alert/cardinality rules in `operations-contract.json`; IDs and
hashes remain log fields, never metric labels. Exit `50` or `52`, repair exit
`51`, admission-SLO breach, supervisor residue, low scratch capacity and anchor
conflict bursts require alerts. A green build does not prove alert delivery.

Image rollback and data correction are separate. An image rollback may select
only a previous independently attested digest under signed two-person
authorization while preserving both deployment evidences. That authorization
must bind the target digest to compatible publisher/runtime/Operations schemas,
the current publication domain, exact trust-registry inventory and a fresh
observation of the current CAS HEAD. A merely older signed image is not an
eligible rollback target. External CAS HEAD is
never rewound. A business/data correction is a new signed compensating
generation with fresh manifests and HEAD observation. Automatic rollback is
forbidden.

`.github/workflows/publisher-runtime-v6.yml` runs contract tests with read-only
repository permissions. Its Docker job is manual, uses the dedicated
`pfc-publisher-windows-docker` self-hosted runner, requires a preloaded base and
approved wheelhouse, disables build network and mutable pulls, runs a read-only
smoke, never pushes, and retains only a short-lived non-authorizing CI proof.
Registry push, image signing, SBOM/provenance attestation and deployment remain
outside that workflow and remain production blockers.
