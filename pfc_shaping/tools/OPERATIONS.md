# PFC Operations Runbook

## 1) Preconditions
- The reviewed `fmv-pfc-lt` wheel is installed under each phase-specific
  service identity, and `pfc-lt --version` reports the approved source revision.
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
5. install only from the retained offline wheelhouse with hash verification;
6. attest under every Builder, Finalizer, Registrar, Auditor, Promoter and
   Rollback identity that `pfc-lt --version` reports the admitted embedded source
   revision and that the installed distribution hash equals the admitted wheel.

The wheel auditor's `promotion_eligible=false` is intentional: package
structure alone never authorizes a candidate or production deployment.

## 2) Governed LT release

The former `python -m pfc_shaping.pipeline.rolling_update` command is disabled
because it read mutable repo caches and published directly. The canonical
command surface is:

```powershell
pfc-lt --help
```

One invocation performs exactly one phase. There is no `all`, `run`,
`release`, `auto-promote` or failed-gate override command.

```powershell
# Builder identity: public trust anchors only. Exit 30 is the expected policy handshake.
pfc-lt build `
  --run-id <run-id> --release-root <release-root> --failure-root <failure-root> `
  --source-revision <source-revision> `
  --reference-timestamp <ISO-8601-UTC> --build-timestamp <ISO-8601-UTC> `
  --config <lt-config.yaml> --config-sha256 <config-sha256> `
  --input-snapshot-contract <lt_input_snapshot.json> `
  --input-snapshot-sha256 <input-snapshot-sha256> `
  --input-pointer-contract <pfc-lt-current.json> `
  --input-pointer-sha256 <input-pointer-sha256> `
  --input-generation-id <generation-id> --data-root <fmv-data-root> `
  --historical-thresholds <thresholds.csv> `
  --historical-thresholds-receipt <thresholds-receipt.json> `
  --selected-lambda-decision <selected-lambda.json> `
  --selected-lambda-decision-receipt <selected-lambda-receipt.json> `
  --eex-report-path <Price_Report_EEX.xlsx> `
  --eex-acquisition-contract <signed-acquisition.json> `
  --peak-source-policy same_first `
  --use-seasonal-hourly-shape

# Independent policy/finalizer identity.
pfc-lt finalize `
  --run-id <run-id> --release-root <release-root> --failure-root <failure-root> `
  --source-hierarchy-policy <signed-source-policy.json>

# Registration identity: REGISTER private key only.
pfc-lt register `
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
pfc-lt audit `
  --run-id <run-id> --release-root <release-root> --failure-root <failure-root> `
  --workflow-root <workflow-root> --evidence-root <evidence-root> `
  --request-id <request-id> `
  --data-root <fmv-data-root> --signing-private-key <receipt-private.pem>

# Promoter identity: event private key only; receipt private key must be absent.
pfc-lt promote `
  --run-id <run-id> --release-root <release-root> --failure-root <failure-root> `
  --workflow-root <workflow-root> --evidence-root <evidence-root> `
  --request-id <request-id>

# Rollback is never a promote shortcut. It requires a fresh, expiring,
# independently signed authorization bound to the current and target event IDs.
pfc-lt rollback `
  --release-root <release-root> --failure-root <failure-root> `
  --rollback-authorization <signed-rollback-authorization.json>

# Read-only status.
pfc-lt status `
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
  On Windows, file contents are flushed and same-volume `os.replace` is used;
  storage durability and write-through guarantees remain an IT mount contract.
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
