# Session Handoff - Durable EEX signing request and explicit runtime admission

Date: 2026-07-29  
Branch: `fix/lt-audit-remediation`  
HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Production: strict `NO_GO`  
Promotion performed: none

## Outcome

The local EEX prospective workflow can now produce a deterministic, resumable
and immutable unsigned signing-request bundle on the managed standard-user
laptop. It neither needs nor requests administrator rights, mutable `AppData`,
an ASR exception, a project `.exe` or Playwright. All mutable inputs and
outputs are confined below `C:\Users\jbattaglia\PFC_LT\build`.

Two demonstrated Security P1 defects were closed: runtime admission is bound
by explicit receipt path plus caller-held SHA-256 and executed with `-I -B`;
trusted-time/acquisition anchors are explicit verifier inputs and no longer
flow through temporary process-global environment mutation. Three subsequent
P2 findings were also closed: partial explicit trust sets fail closed, a
concurrent ABA/rebind test proves immutable anchor bytes, and failure telemetry
shows the exact isolated command. The output remains unsigned and
non-authoritative until independently signed and admitted by an external
builder-inaccessible CAS/WORM/HEAD authority.

## Changed files in this slice

New:

- `pfc_shaping/cli/eex_forward_vintage_builder.py`
- `pfc_shaping/data/eex_forward_vintage_intake.py`
- `scripts/build_eex_forward_vintage.py`
- `tests/test_eex_forward_vintage_intake.py`

Modified:

- `pfc_shaping/cli/governed_acquisition_builder.py`
- `pfc_shaping/data/acquisition_contract.py`
- `pfc_shaping/data/eex_historical_vintage.py`
- `pfc_shaping/package_contract.py`
- `pfc_shaping/pipeline/governed_release_cli_contract.py`
- `scripts/run_workspace_local.py`
- `tests/test_launcherless_runtime_admission.py`
- `tests/test_lt_package_contract.py`
- `tests/test_run_workspace_local_script.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`

The worktree was already intentionally very dirty. Do not infer ownership of
other changes, reset, clean or restore anything.

## Implemented invariants

- Exact source/spec/trusted-time/parser/public-key hashes are bound.
- A cumulative verified bitemporal history rebases archived prior sources and
  parsers; exact retry and crash resume are deterministic.
- The signing request contains no private key and cannot authorize signing,
  CAS, calibration, candidate construction, promotion or production.
- Runtime receipt path/hash and every trust anchor are explicit CLI inputs.
- Partial explicit trust arguments never fall back to ambient state.
- Public key bytes are registered process-immutably; verifier threads do not
  share temporary trust environment state and divergent concurrent rebinds
  fail closed.
- The standard-user runner requires `python -I -B -m`, scrubs ambient
  authority variables and confines all explicit paths plus caches/TEMP/TMP/
  pytest roots below repo `build/`.
- Signing-request source keys are exact; boolean row counts fail closed.
- Package contract and wheel smoke import cover the EEX builder.

## Commands and terminal evidence

Every shell command was preceded by an exact cwd/Git-root guard for
`C:\Users\jbattaglia\PFC_LT`.

Final focused matrix through `scripts.run_workspace_local`, run `p2fix2`:

```text
109 passed in 18.18s
receipt SHA-256 f8e0c308414f67f70fbed10331ca3630b35e5cd2f16988a222a8973bf3ecdf42
```

Final integrated EEX/workflow matrix, run `ewflow3`:

```text
286 passed, 1 deselected, 1 known warning in 27.38s
receipt SHA-256 2502c8fb0f41b94f51644cd1225d2122e4aaec0898100d73c6cd069f60cd00ca
```

Runtime/packaging matrix before the final narrow trust-API P2 delta, run
`runtime4`:

```text
180 passed, 12 skipped, 2 deselected
receipt SHA-256 bd79f485424f1f9c9d3bbf49f1a1d0e15dff49bf7825f4fa4b42384560089ecf
```

Candidate/publication matrix, sequential run `pubcand5`:

```text
181 passed, 2 skipped in 671.99s
receipt SHA-256 eea2868aedd53506fd2fa2b08e11a7e56a409d1c20cf57c1ab559a627b6a77f6
```

External-CAS/publication matrix, sequential run `pubcas5`:

```text
200 passed in 159.01s
receipt SHA-256 235fb34b4e315cb5f4bdadd3f4c82ecfdcdfda4416b1094329756297f6dabc7a
```

Every workspace runner receipt terminates as
`TARGET_EXIT_ZERO_NOT_AUTHORITY` with scientific, runtime, evaluation,
promotion and production authority false. The additional post-roast targeted
run `p2fix1` passed 83 tests and has receipt SHA-256
`6dc94bf09ec1b5654478ceccf6fc19676dada4af3c318a075c1362644ac748b2`.

Targeted Ruff:

```text
All checks passed!
```

Final reproducible wheel builds used `pip wheel --no-deps
--no-build-isolation`; `TEMP`, `TMP`, pip cache and bytecode cache were all
below repo `build/`:

```text
build/wheel-dist-w/fmv_pfc_lt-0.14.0-py3-none-any.whl
build/wheel-dist-x/fmv_pfc_lt-0.14.0-py3-none-any.whl
size 483369 bytes; 86 members; identical SHA-256
07b8228426c2857b30682228181245a7d2367cb31add87a1580f54388ce3b136
embedded source revision
691139df0d2b941823d9c80c3825440a28d1af1d3095ae50f6330a23c130f15e
both audits PASS; promotion_eligible=false
```

The initial `python -m build` attempt exited immediately because the repo-local
`build/` directory shadows the optional Python package of the same name. It
made no wheel and is retained as negative command evidence; it was not a
permission, network or ASR failure.

## Retained negative evidence

- `p1fix1`: `106 passed`, then one Windows `WinError 5` rename failure in the
  pytest basetemp. The exact failed test passed in fresh `p1fix2`; the entire
  focused matrix then passed in fresh `p1fix3`, and final `p2fix2` passed 109.
- The first pre-P2 dual wheel command timed out after wheel U completed. No
  child was left; V was built separately and byte-identical. Final W/X builds
  supersede U/V as current wheel evidence.
- `pubcand4` and `pubcas4` were launched concurrently. Candidate exited 1 and
  CAS exposed a no-argument monkeypatch compatibility defect. The defect was
  corrected; fresh sequential `pubcand5` and `pubcas5` are the terminal green
  evidence. The old receipts remain negative.

## Independent roasts

Security final delta, read-only:

- P0=0, P1=0, P2=0 on current bytes;
- confirms explicit runtime binding, isolated command grammar, explicit trust
  propagation without environment mutation, exact schemas/bool rejection,
  package smoke import, partial-explicit fail-closed behavior and concurrent
  ABA/rebind rejection;
- local GO only for a sealed, isolated, unsigned and quarantined EEX handoff;
  production remains strict `NO_GO` pending independent authorities.

IT/Operations delta, read-only:

- local P0/P1: none;
- confirms `-I -B`, repo-local paths/caches/TEMP/TMP/basetemp, project `.exe`
  and Playwright rejection, explicit builder trust/runtime inputs and current
  operations runbook;
- production P1: import-before-admission, no fresh installed runtime for these
  exact bytes, no Job Object/timeout/memory/descendant supervision or complete
  logs, external signer/CAS/WORM consumer absent, Windows CI/ASR/SBOM and
  active-runtime rollback open.

Quant/Data final, read-only:

- local P0/P1: none;
- external P1: official EEX product/semantic evidence, independent signature
  plus CAS/WORM/HEAD, complete multi-origin rolling-origin and sealed future
  T057 evidence.

## Residual blockers and next direction

Local workflow quality is `GO`; production remains strict `NO_GO`.

Before production authority, obtain:

1. an independently admitted fresh installed runtime for the exact wheel bytes
   with import-before-admission closed;
2. independent acquisition signature and builder-inaccessible external
   CAS/WORM/monotone HEAD consumption;
3. supervised Windows execution, complete logs, CI/ASR evidence, SBOM,
   observability and rollback;
4. fresh official prospective EEX product bytes/semantics and trusted time;
5. multi-origin rolling-origin validation, sealed future T057, probabilistic
   calibration and a new auditable CH candidate.

Do not promote any existing data, holdout result, candidate or production
flag. Monthly solver remains the level authority; OMPEX remains benchmark-only;
LT/CT separation remains mandatory.
