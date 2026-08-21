# Session handoff - EEX DataSource v2 fail-closed readiness

Date: 2026-07-31  
Branch: `fix/lt-audit-remediation`  
Starting HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Production: strict `NO_GO`

## Outcome

The repo now contains one packaged local EEX Group DataSource REST API v2
capture implementation. It is hardened as quarantine-only evidence and cannot
receive a bearer through the workspace supervisor. No real EEX request, quote,
T057 outcome, candidate, publication or production promotion occurred.

Readiness artifact:

- `.planning/phases/14-lt-audit-remediation/CH-EEX-DATASOURCE-V2-ACQUISITION-READINESS-20260731.json`
- readiness ID
  `7ff67541f66b91ff414d8f269d1e8fc76e645bd5572b8dd9b2d268dc50ade2df`
- readiness file SHA-256
  `5f0ea5f1f6e053400efe71c161e327d6ac9fa8c68d43a071ae5451020425fb0b`

## Changed scope

- `pfc_shaping/data/eex_datasource_v2_capture.py`
- `scripts/capture_eex_datasource_v2.py`
- `tests/test_eex_datasource_v2_capture.py`
- `scripts/run_workspace_local.py`
- `tests/test_run_workspace_local_script.py`
- `pfc_shaping/package_contract.py`
- `tests/test_lt_package_contract.py`
- `pfc_shaping/tools/OPERATIONS.md`
- this readiness artifact, RFC amendment, decision D-20260731-203 and handoffs.

Do not touch or stage `data/eex_forwards_history.parquet`; verified SHA-256:
`21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

## Closed demonstrated findings

- Exact endpoint path segments replace substring identity checks.
- Public capture no longer accepts an injectable transport.
- Raw and decoded token reflection is rejected, including JSON escapes,
  nested percent encoding, Base64 with/without padding and whitespace.
- Spec/request target is re-read, re-hashed and scanned before any output.
- Explicit TLS client disables `SSLKEYLOGFILE`, enforces hostname,
  `CERT_REQUIRED`, TLS 1.2+, one hash-bound independently approved CA.
- CA self-signature, BasicConstraints and KeyUsage are checked; policy allowlist
  is empty by default.
- Reference manifests exact-check top-level and nested request, docs, TLS,
  evidence and authority claims against the reopened spec.
- Windows reserved names, aliases, dot/empty components and trailing dot/space
  fail closed.
- Request starts are serialized at >=1 second with an OS crash-released,
  single-link/non-reparse lock held through the GET.
- Success uses incomplete staging then final rename. Caught failures emit a
  redacted terminal receipt with phase/class, safe Retry-After/correlation,
  fresh-ID disposition and all authorities false.
- The module has `__main__` and is in the package allowlist.
- The supervisor understands only exact `-I -B` packaged-module commands under
  a repo runtime namespace, but removes and refuses every EEX token before cwd,
  Git, subprocess, namespace or receipt. `forwarded=false` is factual.

## Commands and terminal results

Every shell action first verified cwd and Git root exactly
`C:\Users\jbattaglia\PFC_LT`.

- Ruff on changed implementation/tests/package/runner: `PASS`.
- Governed EEX/runner/package matrix:
  `221 passed, 1 skipped, 1 deselected in 38.17s`.
- Supervisor E2E after aligning the checkout supervisor/plugin invocation:
  `4 passed, 141 deselected in 8.46s`.
- Adjacent EEX/data/acquisition/ledger matrix:
  `267 passed, 3 skipped, 1 deselected`, one known pandas timezone warning.
- Packaging/runtime matrix:
  `110 passed, 12 skipped, 2 deselected in 132.04s`.
- Publication splits:
  - governed snapshot: `48 passed in 221.11s`;
  - publisher/container/external CAS/bootstrap/anchors:
    `147 passed, 12 skipped, 1 deselected in 71.87s`;
  - candidate evidence/bundle: `65 passed in 268.86s`.
- The combined publication command timed out at 300s, then the aggregate
  release/quality/promotion command timed out at 360s and its stdout flush
  failed. Neither is counted as evidence. No residual Python process remained.
- Final targeted EEX/runner/package reconciliation:
  `222 passed, 1 skipped in 29.59s`.
- The four release/promotion files were then run independently and completed:
  - atomic promotion: `116 passed, 2 skipped in 130.41s`;
  - governed release: `37 passed in 27.21s`;
  - quality gate: `27 passed in 0.22s`;
  - promotion contract: `18 passed in 0.30s`.
- Full `git diff --check` passed with line-ending warnings only.

Final reconciliation commands (each prefixed by the mandatory exact cwd/Git
root guard; pytest `TEMP`, `TMP` and `--basetemp` were under `build/`):

```text
python -B -m ruff check pfc_shaping\data\eex_datasource_v2_capture.py scripts\capture_eex_datasource_v2.py scripts\run_workspace_local.py tests\test_eex_datasource_v2_capture.py tests\test_run_workspace_local_script.py pfc_shaping\package_contract.py tests\test_lt_package_contract.py
python -B -m pytest tests\test_eex_datasource_v2_capture.py tests\test_run_workspace_local_script.py tests\test_lt_package_contract.py -q -p no:cacheprovider -m "not slow" --basetemp build\pytest\eex-final-20260731
python -B -m pytest tests\test_atomic_promotion.py -q -p no:cacheprovider --basetemp build\pytest\atomic-promotion-20260731
python -B -m pytest tests\test_governed_release.py -q -p no:cacheprovider --basetemp build\pytest\governed-release-20260731
python -B -m pytest tests\test_quality_gate.py -q -p no:cacheprovider --basetemp build\pytest\quality-gate-20260731
python -B -m pytest tests\test_promotion_contract.py -q -p no:cacheprovider --basetemp build\pytest\promotion-contract-20260731
git diff --check
```

Final evidence identities:

- readiness ID `7ff67541f66b91ff414d8f269d1e8fc76e645bd5572b8dd9b2d268dc50ade2df`;
- readiness file SHA-256 `5f0ea5f1f6e053400efe71c161e327d6ac9fa8c68d43a071ae5451020425fb0b`;
- capture source SHA-256 `bd7a1fb87815c2813285eef7d6dfbd378525c291bdf506aa7b02163188d5db62`;
- runner SHA-256 `db1622c7df84dcacb591c9b539c6ba555e14ee8e06967d54e5520bd4b98e9117`;
- no scoped file is staged; branch/HEAD remain
  `fix/lt-audit-remediation` / `2f68125bff869ccb21c1e20df0201ad024ed27d3`.

## Independent roasts

- Security/Governance final: P0=0, P1=0 on the token-refusal design. Residual
  P2: portable no-clobber/power-loss, direct-module capability before future
  activation, and missing external authorities.
- IT/Operations final before the deliberate refusal: P0=0, P1=0, P2=3. The
  forwarding closure was then withdrawn because immutable runtime/worker
  execution was not proven; this is now an explicit blocker, not an open path.
- Quant/Data: P0=0, P1=0. Exact identity/revision/unit negative regressions are
  now present; the capture remains scientifically non-admitted.

## External blockers and next work

1. Obtain the exact authenticated EEX Derivatives OpenAPI, CH subscription and
   licensing evidence through approved FMV channels.
2. Security/IT must choose approved direct egress or explicitly approve the
   interception root/token handling. Do not locally add the observed root hash.
3. Design an immutable runtime path+hash+manifest and secret capability that
   cannot execute mutable worker/module bytes before verification.
4. Add native no-replace finalization for Linux/Docker, Windows power-loss
   drill, orphan inventory/reconcile receipt, singleton quota policy and SLOs.
5. Return to prospective fresh CH evidence, direct-CH losses, rolling
   origin, the sealed structural commitment and a new independent holdout.
   T057 remains unread and cannot be reused confirmatorily.

Monthly solver remains sole level authority. OMPEX remains benchmark-only.
Swiss operational truth remains hourly until native 15-minute go-live is
independently verified and admitted.
