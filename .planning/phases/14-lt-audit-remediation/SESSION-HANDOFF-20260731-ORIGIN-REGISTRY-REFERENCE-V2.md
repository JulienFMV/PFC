# Session handoff - origin registry non-countable reference v2 - 2026-07-31

## Executive status

The source-only CH LT origin-registry reference is accepted for local quality
on its exact current bytes. It is intrinsically non-countable: receipt schema,
registry domain and receipt-ID domain are incompatible with production, and
signed raw/effective `countable_prospective_origin` are both `false`. The
reference is excluded from the governed LT wheel and from all consumers.

Independent terminal read-only roasts report P0/P1 `0/0` for
Security/Governance, IT/Operations and Quant/Data on the selected source/test
bytes and `orgrf31v4` receipts. This closes the demonstrated local P1 defects;
it does not implement or authorize the external registry or scientific
admission. Protocol v2 still has zero countable origins and eleven external
requirements missing. Production remains strict `NO_GO`.

The historical launcherless v9 command naming an `AppData` publisher
wheelhouse is revoked. Current code rejects it both in
`scripts.run_workspace_local` and in the direct runtime-builder CLI. The exact
qualified source-test launcher is
`C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe`; do not
substitute generic `python`, a user `.conda` interpreter or any path outside
canonical repo `build/` for selected evidence. No administrator/elevation,
ACL takeover, Defender exception, project executable or Playwright route is
needed or permitted.

## Governing decision and documents

- D198 in `DECISION-LOG.md` selects only the non-countable reference semantics
  and `orgrf31v4` for local quality.
- `LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md` contains the
  2026-07-31 non-countable-reference amendment.
- `pfc_shaping/tools/OPERATIONS.md` records the exact absolute repo-local
  launcher and the authority boundary.
- D197 and `SESSION-HANDOFF-20260731-WORKSPACE-SUPERVISOR-V6.md` continue to
  govern the workspace supervisor; D196 continues to govern protocol v2 and
  launcherless runtime v40.

## Files changed in this slice

- new `pfc_shaping/data/ch_lt_origin_registry_reference.py`;
- new `tests/test_ch_lt_origin_registry_reference.py`;
- `tests/test_lt_package_contract.py` (explicit wheel exclusion);
- `pfc_shaping/tools/OPERATIONS.md`;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`;
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`;
- this handoff;
- `.planning/HANDOFF.md`.

The worktree was already intentionally very dirty. Existing unrelated changes
were preserved. Nothing was reset, cleaned, restored, staged or committed.
No `pfc_shaping/ct/*`, Power BI or heavy desk data file was intentionally
modified. Protected `data/eex_forwards_history.parquet` remains at SHA-256
`21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

## Implementation closure

### Intrinsic non-authority

- `REFERENCE_AUTHORITY_CLASSIFICATION` is
  `NON_PRODUCTION_TEST_ONLY_NEVER_COUNTABLE`.
- Reference state schema is `ch_lt_origin_registry_reference.v2`; receipt
  schema is
  `ch_lt_origin_registration_receipt.non_production_reference.v1`.
- Registry logical domain is
  `FMV_CH_LT_CONFIRMATORY_ORIGINS_V2_NON_PRODUCTION_REFERENCE_V1`; receipt IDs
  use a separate reference-only domain.
- The authority emits `countable_prospective_origin=false`; the signer rejects
  production receipt schema or `true`, and the verifier independently requires
  the reference schema/domain and `false`.
- Every external/scientific/evaluation/runtime/publication/promotion/
  production authority is false.

### Scientific and temporal boundary

- The request binds exact protocol/schedule bytes, the signed schedule entry,
  origin/commit/deadline chronology, artifact hashes and trusted-time receipt.
- The structural target-mask inventory is loaded by exact hash and assessed
  through the canonical schema-v2 validator: 36 M01..M36 targets, canonical
  ID, origin binding, outcome-blind state and all truth/T057 flags false.
- The first target delivery is derived from the minimum target timestamp in
  that inventory; a caller declaration cannot move the boundary.
- Other EEX/solver/baseline/prediction/scenario/runtime commitments are only
  exact opaque hashes here. Their semantic validation remains external P1
  scientific work before any real countable service.

### Persistence and audit boundary

- Four Ed25519 roles are disjoint: request, exact schedule, trusted time and
  registry. Their key IDs, protocol, schedule, classification and reference
  domain are pinned in SQLite identity metadata.
- SQLite `user_version=2`, WAL, `synchronous=FULL`, foreign keys and bounded
  busy timeout are verified. Existing DB/WAL/SHM paths must be mono-link
  regular non-reparse files below repo `build/`.
- Compare-and-append uses `BEGIN IMMEDIATE`; sequence/predecessor and global
  slot/origin/operation/request uniqueness fail closed. Exact retry and exact
  operation lookup are supported.
- Clean startup re-verifies every exact request, signed receipt, row binding
  and chain predecessor.
- Rejection records retain only request hash, size, bounded error type/code
  and signed IDs after exact validation. Raw payload bytes and free-form error
  text are never persisted; the secret-sentinel regression checks DB/WAL/SHM.
- HEAD observations are signed, exact-receipt-bound, one-nonce-only and have a
  positive TTL capped at five minutes. All returned authorities remain false.

## Exact commands and results

Every shell action first asserted exact cwd and Git root
`C:\Users\jbattaglia\PFC_LT`. Mutable environment and pytest paths were under
repo `build/`.

### Permission/AppData closure

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe -B -m pytest `
  tests\test_launcherless_local_runtime.py `
  tests\test_run_workspace_local_script.py `
  -q -p no:cacheprovider `
  -k "appdata or launcherless_runtime_paths_must_remain_below_repo_build or launcherless_runtime_repo_local_paths_are_allowed" `
  --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-permission-closure-20260731-v1
```

Result: `4 passed, 148 deselected`. Direct builder and supervisor both reject
an AppData wheelhouse before assembly. This is the operational answer to the
repeated permission prompts: never issue the obsolete command and never ask
for elevation; use only repo-local paths.

### Ruff

The repo-local pytest runtime correctly reported `No module named ruff`; no
installation was attempted. Existing workspace `.venv` Python was then used
read-only with caches below `build/`:

```powershell
C:\Users\jbattaglia\PFC_LT\.venv\Scripts\python.exe -B -m ruff check `
  --no-cache --fix `
  pfc_shaping\data\ch_lt_origin_registry_reference.py `
  tests\test_ch_lt_origin_registry_reference.py
C:\Users\jbattaglia\PFC_LT\.venv\Scripts\python.exe -B -m ruff check `
  --no-cache `
  pfc_shaping\data\ch_lt_origin_registry_reference.py `
  tests\test_ch_lt_origin_registry_reference.py
```

Result: four mechanical import/unused-import findings fixed, then
`All checks passed!`.

### Direct focused matrix

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe -B -m pytest `
  tests\test_ch_lt_origin_registry_reference.py `
  tests\test_ch_lt_origin_registry_protocol.py `
  tests\test_ch_lt_origin_target_mask_inventory.py `
  tests\test_lt_package_contract.py `
  tests\test_launcherless_local_runtime.py `
  tests\test_run_workspace_local_script.py `
  -q -p no:cacheprovider -m "not slow" `
  --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-origin-reference-p1-20260731-v2
```

Result: `215 passed in 24.08s`, zero skip/failure/error.

### Import-root probe

Both `-I -B` and source-test mode resolved the reference module from the exact
canonical checkout. Exact `sys.path` was runtime `Lib`, runtime `DLLs`,
`build/pytest-runtime-v1/test-site-packages`, the single canonical checkout
root, and runtime `governed-site-packages`. The checkout root occurs exactly
once.

### Selected supervised proof

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe -B `
  -m scripts.run_workspace_local --run-id orgrf31v4 `
  --wall-timeout-seconds 900 -- `
  C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe -I -B `
  -m pytest `
  tests\test_ch_lt_origin_registry_reference.py `
  tests\test_ch_lt_origin_registry_protocol.py `
  tests\test_ch_lt_origin_target_mask_inventory.py `
  tests\test_lt_package_contract.py `
  tests\test_launcherless_local_runtime.py `
  tests\test_launcherless_runtime_admission.py `
  tests\test_lt_ct_imports.py `
  -q -p no:cacheprovider -m "not slow"
```

The outer tool caller reached its 180-second capture limit during source-tree
preflight; it did not terminate the supervisor Job. The governed supervisor
continued, moved from `PREFLIGHT_PENDING` to `EXECUTION_STARTED`, and wrote
terminal positive receipts. Do not classify the caller timeout as a test
failure or reuse the namespace.

Selected terminal result:

- supervisor `WORKER_EXIT_ZERO_NOT_AUTHORITY`, execution
  `TARGET_EXIT_ZERO_NOT_AUTHORITY`, both exit 0;
- `113 passed, 4 skipped in 9.71s`, zero failure/error/deselect;
- skips only for optional CT imports: two `lightgbm`, one `torch`, one
  `tensorflow`; no CT qualification is claimed;
- supervisor receipt SHA-256
  `fbf85f5e9b3e9a8214691d30b09df0664b21b3f14037704e60627544081a5161`;
- execution receipt SHA-256
  `bd39f487fbda91e4c4bbdc89038948ebcef5b9bb707ea9cc190efbef88fa3831`;
- source tree SHA-256
  `89778238fbd2173d582afe45c545c5a7aa39f2ac559c47e10157fac8635e1ab9`,
  538 files / 9,614,514 bytes;
- import closure `BOUND_REPO_LOCAL_PTH`; repo-local target interpreter SHA-256
  `50bfb90ee93bb0cb51175b546f133798dfe4b778677d95d81391e7bf6d85e5ac`;
- capability consumed then absent, zero active processes, complete stdout/
  stderr and all authorities false.

## Independent terminal roasts

- Security/Governance: P0/P1 `0/0`. P2: same-user local path/sidecar TOCTOU,
  startup-only full-chain validation and missing adversarial corruption/path/
  PRAGMA/predecessor/TTL tests.
- IT/Operations: P0/P1 `0/0`. P2: cold/warm SLO and retention; crash,
  disk-full, WAL recovery, backup/restore and RPO/RTO; multiprocess contention
  and response-loss lookup; key rotation/keyrings; rejection metrics;
  migration/PITR and final exact production-wheel audit.
- Quant/Data: P0/P1 inside the non-countable reference `0/0`. The structural
  inventory and first-target boundary are executable. Opaque scientific
  commitments remain an external P1 before any real registry can make an
  origin countable.

No reviewer read a T057 outcome.

## Residual risks and next work

1. Never extend this same-user SQLite reference into a production authority.
   The real external service needs independent ownership, WORM/CAS, mTLS/ACL,
   trusted time, approved UTC cadence, historical keyrings, fresh HEAD,
   multiprocess/crash recovery, SLOs, backup/PITR, observability and rollback.
2. Add the P2 adversarial regression matrix for corrupted rows/metadata,
   paths outside `build/`, hardlinks/sidecars, PRAGMA failure, forged
   sequence/predecessor, excessive HEAD TTL, crash/response-loss and
   multiprocess contention.
3. Return to outcome-blind fresh prospective CH/EEX acquisition and accumulate
   genuinely independent rolling origins. Do not count this local reference,
   the one-day hourly capture, proxy-expanded quarter-hours or old T057.
4. Before opening a new future holdout or proposing a CH candidate, close the
   semantic commitments for exact EEX final products, solver-level authority,
   same-input baselines, predictions/scenarios and outcome-blind provenance.
5. The current historical local candidate is not production quality evidence:
   challenger rolling-origin was 6 wins / 6 losses / 4 ties, P10/P90 were
   absent and inputs included fixture/proxy data. A new auditable candidate
   requires fresh independent origins and probabilistic/scenario gates.
6. Do not promote production. Monthly solver remains level authority; OMPEX
   remains benchmark-only; LT/CT separation and native CH hourly truth remain
   unchanged pending separately admitted 15-minute go-live evidence.

