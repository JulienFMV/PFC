# Session handoff - 2026-07-29 - Standard-user workspace runner

> Superseded for current execution by
> `SESSION-HANDOFF-20260729-FRESH-CH-HOURLY-CAPTURE-RUNNER-V3.md` and D175.
> This file remains the immutable historical v2 closure; do not use its
> one-root/run-ID conventions for new tests.

## Status and scope

- Repo/root: `C:\Users\jbattaglia\PFC_LT` only.
- Branch: `fix/lt-audit-remediation`.
- Session HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree remains intentionally very dirty; no reset, clean, restore, stage,
  commit or promotion occurred.
- No admin, ACL takeover, Defender/ASR exception, project EXE, Playwright,
  `H:` repo, CT or Power BI action occurred.
- Protected `data/eex_forwards_history.parquet` was not touched by this slice;
  observed SHA-256 remained
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- Local standard-user harness: `GO_LOCAL_NON_AUTHORITATIVE`.
- Production: strict `NO_GO`.

## Problem and decision

Permission prompts were caused by mutable command paths outside the governed
workspace, notably an obsolete v9 command using an `AppData` wheelhouse. They
were not evidence that the laptop needed administrator rights. V19 was already
the current local-quality runtime, so v9 was not rebuilt.

`AGENTS.md` now makes the standard-user boundary permanent. Every mutable
prefix, wheelhouse, cache, temporary directory and pytest namespace must stay
below the canonical repo, normally `build/`. Existing external interpreters,
Conda tools and archives may be read without mutation only for verified copy or
capture below `build/`. External authority requirements are blockers, not a
reason to request elevation repeatedly. The old generic worktree recommendation
is superseded on the FMV laptop by the literal canonical root.

The new laptop-only module `scripts.run_workspace_local` handles explicitly
allowlisted Python build/audit/test modules. It is not a generic shell, a
filesystem sandbox, a production admission runtime or the CI runner. Conda
exact-prefix creation, wheel construction and installed v19 admission retain
their dedicated governed recipes.

## Runner v2 contract

- Literal source, cwd and Git root must be
  `C:\Users\jbattaglia\PFC_LT`.
- A fresh portable run ID creates one single-use namespace under
  `build/workspace-local-runs/`.
- A receipt is written as `PREFLIGHT_PENDING` before directory probes; probe,
  environment and identity failures are terminalized.
- Known mutable destinations for Conda, pip, uv, Python, pytest, joblib,
  matplotlib, numba and XDG are repo-local; Conda/pip/uv are offline.
- Caller pytest basetemps are rejected; the runner injects a fresh child of an
  identity-bound `pytest-root`.
- CPython grammar is fail-closed: only safe no-value flags followed
  immediately by `-m <allowlisted-module>` are accepted. `-c`, stdin, script
  paths, shells, npx, direct project EXEs and Playwright are rejected.
- A relative Python command resolves to the current interpreter. An absolute
  interpreter must be the current one or `build/conda-runtime-*/python.exe`.
- Python/pytest injection variables, all known FMV/PFC data-root aliases,
  credentials, mTLS material and runtime/promotion/production authority
  variables are removed from the child environment.
- Only a redacted command and hash of the redacted command are persisted.
- Receipt schema is `pfc_lt_workspace_local_execution.v2`; production,
  promotion, scientific, evaluation and runtime authorities are always false.
- Exit zero is `TARGET_EXIT_ZERO_NOT_AUTHORITY`; nonzero is
  `TARGET_EXIT_NONZERO`. It is never relabelled scientific or production PASS.
- Physical containment, reparse absence and directory identities are checked
  before and after the child. A malicious same-user junction swap during the
  child is explicitly outside this local harness trust model and remains
  forbidden for production evidence.

## Final source and evidence hashes

- `scripts/run_workspace_local.py`: 19,836 bytes, SHA-256
  `9b2ceecfc6d903650ffd90871e3ec93cbd616329fb41fd5c0ae281e79bee7878`.
- `tests/test_run_workspace_local_script.py`: 9,025 bytes, SHA-256
  `d1d525ed50192b32c677d6feb4709221b833ba08b5847618dd0d7abe5c0cfc87`.
- final pytest receipt:
  `build/workspace-local-runs/d173-v2-tests-20260729-aa/execution-receipt.json`,
  SHA-256
  `27f93feb9c420210f84ed0ee9ba1bae4f4ac50b3d676f5e0bbf5a36a35782299`.
- final Ruff receipt:
  `build/workspace-local-runs/d173-v2-ruff-20260729-ab/execution-receipt.json`,
  SHA-256
  `9f113f7b69ef6f8895fcb4e91ba1fda79c3743de0eade631b2e8b0e2e532fd07`.
- rejected `-c` bypass receipt:
  `build/workspace-local-runs/d173-v2-reject-c-bypass-20260729-w/execution-receipt.json`,
  SHA-256
  `357b59d3f8a3e2a4d190d58aca2e55ae922da0f5b1691513f20dd8f9832310e0`.
- rejected Playwright receipt:
  `build/workspace-local-runs/d173-v2-reject-playwright-20260729-x/execution-receipt.json`,
  SHA-256
  `4d9544dd3902188f4e44bebe6f700c8e8b3ad17d69c1b58362957e01775e4076`.

The earlier `d173-v2-tests-20260729-t` run is retained as negative evidence:
the target returned nonzero because one test expected the wrong lexicographic
order. It was not reused or relabelled; the corrected test ran in fresh
namespaces.

## Exact final commands and results

Each shell action was preceded by a separate literal cwd/Git-root guard.

```powershell
python -B -m scripts.run_workspace_local --run-id d173-v2-tests-20260729-aa -- `
  python -B -m pytest tests\test_run_workspace_local_script.py -q -p no:cacheprovider
```

Result: exit 0, `14 passed in 0.84s`, receipt status
`TARGET_EXIT_ZERO_NOT_AUTHORITY`.

```powershell
python -B -m scripts.run_workspace_local --run-id d173-v2-ruff-20260729-ab -- `
  python -B -m ruff check scripts\run_workspace_local.py `
    tests\test_run_workspace_local_script.py
```

Result: exit 0, `All checks passed!`, non-authoritative receipt.

```powershell
python -B -m scripts.run_workspace_local `
  --run-id d173-v2-reject-c-bypass-20260729-w -- `
  python -c "import playwright" -m pytest
```

Result: rejected before execution because `-c` is not in the closed CPython
grammar.

```powershell
python -B -m scripts.run_workspace_local `
  --run-id d173-v2-reject-playwright-20260729-x -- `
  python -B -m playwright --version
```

Result: rejected before execution because `playwright` is not allowlisted.

## Independent read-only roasts

Security final verdict: P0 0, P1 0, `GO` local non-authoritative. The residual
same-user junction/TOCTOU race is P2, explicitly outside the harness trust
model and incompatible with production evidence. Documentation of the final
receipts was the other P2 and is closed by this handoff/decision/RFC update.

IT/Operations final verdict after the data-root micro-fix: P0 0, P1 0, P2 0
for that micro-scope and `GO` local. Harness, dedicated Conda/wheel/admission
recipes and independent CI are separated.

Quant/Data final verdict: P0 0, P1 0. The last P2 test-coverage omission for
`PFC_SHARED_DATA_ROOT` was corrected; final pytest/Ruff reruns are aa/ab. No
model, solver, shaping, data, CT, OMPEX or T057 byte changed in this slice.

## Scientific state and next actions

The permission closure does not change scientific quality. Run15 remains a
fixture-backed structural diagnostic only. Corrected T057 remains one distinct
historical fold plus one future episode and cannot be reused for selection.
The captured CH day-ahead Energy Charts track remains verified local quarantine
only, native hourly and not native 15-minute truth. The explicit Swiss
15-minute auction remains planned/not confirmed until official product bytes,
identity, licensed schema and sufficient post-go-live history are admitted.

Next work returns to the actual product frontier:

1. refresh/inventory prospective CH hourly evidence without promoting it;
2. obtain governed fresh CH EEX forward point-in-time quotes and independent
   external CAS/trusted-time/signature authorities;
3. create the executable successor preregistration only after its required
   origin/fold, dependence/power, scenario/MC and FMV materiality inputs exist;
4. build a new hash-bound CH candidate without T057/pilot reuse;
5. run rolling-origin/future holdout quality, probabilistic calibration and
   independent Security, IT/Operations and Quant/Data roasts;
6. never promote before all evidence is complete.
