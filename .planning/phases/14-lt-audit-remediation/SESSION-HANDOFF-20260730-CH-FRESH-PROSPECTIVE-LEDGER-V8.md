# Session handoff - fresh CH prospective prefix and ledger v8

Date: 2026-07-30 (Europe/Zurich)

## Outcome

The current local CH hourly prefix is extended through
`2026-07-30T02:00:00Z` without admitting future delivery leakage. Ledger v8
contains 748 contiguous native hourly observations and 2,992 explicitly
stepwise quarter-hour transport proxies. Independent rolling-origin count is
still zero. This is local continuity/replay evidence only.

The full delivery-day v2 capture was rejected from the prospective ledger
because its window ended roughly 19.5 hours after capture. V3 captures only
the four elapsed hours and closes the prefix conservatively. A new
machine-readable current-evidence registry resolves the former v8-current /
v5-market-time-snapshot ambiguity and a strict validator checks every linked
artifact before the ledger command.

No training, model selection, T057 access, candidate creation, publication,
promotion or production transition occurred. Monthly solver level authority,
LT/CT separation and OMPEX benchmark-only status are unchanged. Production is
strict `NO_GO`.

## Workstation and read order

Use only `C:\Users\jbattaglia\PFC_LT`. Before every shell action verify exact
cwd and Git top-level. Every mutable environment, cache, temp, basetemp,
wheelhouse, runtime and output remains under repo `build/`. Never use `H:`,
admin/elevation, ACL takeover, ASR/Defender exceptions, project `.exe` files,
Playwright, system installs or mutable AppData paths. The preinstalled user
Conda interpreter and `C:\certs\git-ca-plus-fmv.crt` were used read-only.

Read next:

1. `AGENTS.md`
2. `.planning/HANDOFF.md`
3. `DECISION-LOG.md` D186
4. this handoff
5. `CH-LT-CURRENT-EVIDENCE-SELECTION-20260730.json`
6. `CH-LT-PROSPECTIVE-CAPTURE-LEDGER-SELECTION-V2-20260730.json`
7. `pfc_shaping/tools/OPERATIONS.md` section 11

## Changed files in this closure

- `.planning/phases/14-lt-audit-remediation/CH-LT-PROSPECTIVE-CAPTURE-LEDGER-SELECTION-V2-20260730.json` (new)
- `.planning/phases/14-lt-audit-remediation/CH-LT-CURRENT-EVIDENCE-SELECTION-20260730.json` (new)
- `scripts/audit_ch_lt_current_evidence_selection.py` (new)
- `tests/test_audit_ch_lt_current_evidence_selection_script.py` (new)
- `tests/test_ch_lt_prospective_capture_ledger.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/HANDOFF.md`
- this handoff

The worktree was already intentionally very dirty and remains so. Nothing was
reset, cleaned, restored, staged or committed. `pfc_shaping/ct/*`, Power BI
and protected heavy desk data were not intentionally touched.

## Capture attempts and exact identities

Full-day v2, retained local archive and never ledger input:

- window `[2026-07-29T22:00:00Z, 2026-07-30T22:00:00Z)`;
- received `2026-07-30T02:29:12.648104Z`;
- capture summary SHA-256
  `9d3201775bd01d41e1bbb170240dc796d2ebc2b68b0dfa0be0bebc3b03639cb8`;
- provider body SHA-256
  `acb8867ed9c1cf52ad86d3c37ecdf240e7841ca90d8d605da800e0f6be385f00`;
- workspace receipt SHA-256
  `f4766730f1addf85830f54c41c96cc43e84ed06fce53afdc4a55bbbcffcbd75d`;
- acquisition manifest SHA-256
  `7cdd72209fe9655c8c3b516ba0f18c7f3866ee8bd4299645ebbfe376b93cd0d5`;
- audit SHA-256
  `13b506bcf95fe622ab3ad858ef9a4494454bee3a802cb59605d6495e2f4c28ea`;
- verifier runtime-receipt SHA-256
  `8a30bcafad8b7c6d55f41ebc6d6587ffdcf5dbb2417d2e2bd974084af09fa580`.

Elapsed-window v3, selected third capture:

- window `[2026-07-29T22:00:00Z, 2026-07-30T02:00:00Z)`;
- received `2026-07-30T02:50:44.544684Z`;
- 4 native hourly observations and 16 QH stepwise proxy rows;
- capture summary SHA-256
  `b8d6a4187e55165b80d608290ef1fdcf7a0725f8bfea1dfabf557d6ce2f9a7db`;
- provider body SHA-256
  `296f5281b20e37aa79e402e3038a85ff67bf64b804dfbab15b08c0a9723d011a`;
- workspace receipt SHA-256
  `ab014aaed30596f36535f430f378fdffef9696d0be58fbc9369fe4fb6aa13531`;
- acquisition manifest SHA-256
  `992b22f24be5acb98671ae4168aab29001876f6760a64e7e97fb5e83a900165d`;
- isolated audit SHA-256
  `30376809b7d3dd5bd68c3384b0b91862cc4478a3696836ae6af5b6c4af9d94b4`;
- verifier runtime-receipt SHA-256
  `a7bc7cfe471dcf096ab9ec8d74a05a015c0dd260e14c6961e7bdf2955eb14c8b`;
- audit status `VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION`, all training,
  model-selection, rolling-origin, T057, candidate, promotion and production
  gates false.

The capture command, wrapped by the workspace-local runner, was:

```powershell
python -B -m scripts.run_workspace_local --run-id chcap30v3 -- python -B -m scripts.capture_public_energy_charts_lt --role epex_ch --start-utc 2026-07-29T22:00:00Z --end-utc 2026-07-30T02:00:00Z --raw-cadence-minutes 60 --acquisition-id ch-da-hourly-20260729t2200z-20260730t0200z-v1 --output-directory C:\Users\jbattaglia\PFC_LT\build\prospective-captures\ch-da-hourly-20260730-v3 --ca-bundle C:\certs\git-ca-plus-fmv.crt
```

Target exit was 0. The runner receipt records source tree SHA-256
`5083774ac00b349fa53111d3d7d4f4e74515b90215e541cba65ba4fff31a849a`,
the read-only interpreter identity and every mutable path under repo `build/`.

The installed runtime v29 replay used:

```powershell
C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v29-market-time-base\python.exe -I -B -m pfc_shaping.cli.governed_acquisition_builder --capture-spec C:\Users\jbattaglia\PFC_LT\build\prospective-captures\ch-da-hourly-20260730-v3\capture-spec.json --output-directory C:\Users\jbattaglia\PFC_LT\build\prospective-acquisitions\ch-da-hourly-20260730-v3
```

The selected verifier was
`build/provider-verifier-20260729-v18.pyz`, SHA-256
`7f17be8de8e78ba5a063903c7ea459baed0372b70a128ffab3fb8b17f69b19c5`,
with dependency tree SHA-256
`0ecb7997997cc124375e92614ca08d9c5274c683c6738448b9bd3c5eafaf78f1`.
The successful isolated audit used a fresh repo-local scratch root and:

```powershell
python.exe -I -S -B build\provider-verifier-20260729-v18.pyz audit-acquisition --acquisition-directory build\prospective-acquisitions\ch-da-hourly-20260730-v3 --expected-manifest-sha256 992b22f24be5acb98671ae4168aab29001876f6760a64e7e97fb5e83a900165d --output-json build\prospective-audits\ch-da-hourly-20260730-v3-audit-v10.json --runtime-receipt-json build\prospective-audits\ch-da-hourly-20260730-v3-audit-v10-runtime-receipt.json
```

One earlier audit invocation omitted `PFC_LT_VERIFIER_SCRATCH_ROOT` and failed
before output. Audit v9 then rejected a pre-existing scratch path. V10 used a
fresh path, exited 0 and left zero scratch residue. These are fail-closed
operator errors, not authority.

## Selected ledger and current resolver

Request v3 SHA-256 is
`d62ae24b5612da8a13b7b56102e9f77ce358c8b507e8dd8be613e82bc67f5f80`.
Selected ledger v8:

- path `build/prospective-ledgers/ch-hourly-local-ledger-20260730-v8.json`;
- SHA-256
  `ac347cb709fc1ae7c75bf516621901851c08656921e3b4d960eb695a8ad32433`;
- ledger ID
  `2921ae31951aed2e9918811c511b88020ee5e6e882ae38902949da851410e1d0`;
- receipt SHA-256
  `06cd8e34297e9f29b7a8b8ac65ea0e350d9cd6245393088efc4d22e1b2e63c49`;
- window `[2026-06-28T22:00:00Z, 2026-07-30T02:00:00Z)`;
- 3 capture episodes, 748 native hourly observations, 2,992 QH proxies;
- independent rolling-origin count `0` and every authority false.

The exact sealed command and retry are recorded in Operations section 11. The
retry exited 0 and preserved ledger and receipt bytes. Its `sys.path` contains
only runtime `Lib`, `DLLs` and `governed-site-packages`.

V6's parent timed out and its child was later terminated; it produced no
ledger/receipt. V7 failed closed on the full-day v2 chronology and produced no
ledger/receipt. Both are non-selectable. Because neither retained a durable
failure receipt with command, runtime, exit, stdout/stderr and terminal cause,
this remains an explicit IT/Operations P2. Do not fabricate retroactive
evidence. A future supervisor must emit a failure sidecar for every attempt.

Prospective selection v2 SHA-256 is
`e22cf06793ccd866e0b37577e76bd03b26ae7873c12889c6e672b1ddb730194a`.
Current-evidence registry SHA-256 is
`644a6c436fa426376f4d2f3e9e0f28a309264b428e0283bc219e9213a4a40ffb`,
ID `2a05f414b713ebda5a6489370be3eddfd08377ff3a63780ac7744fc45bf1b9fd`.
It makes v8 current, v5 historical and the market-time contract's embedded v5
a frozen audit snapshot only.

Validator command:

```powershell
python -B -m scripts.audit_ch_lt_current_evidence_selection --registry C:\Users\jbattaglia\PFC_LT\.planning\phases\14-lt-audit-remediation\CH-LT-CURRENT-EVIDENCE-SELECTION-20260730.json --expected-registry-sha256 644a6c436fa426376f4d2f3e9e0f28a309264b428e0283bc219e9213a4a40ffb
```

It exited 0 with `CURRENT_SELECTION_LINKS_VALID_LOCAL_ONLY_NO_GO`, v8 current,
v5 embedded historical, independent origins zero and all authorities false.

## Tests and independent roasts

Intermediate observations before documentation close:

- ledger chronology regression: `18 passed`;
- scientific/prospective matrix: `54 passed`;
- runtime/packaging matrix: `66 passed, 1 skipped`;
- current-registry and ledger matrix: `23 passed`;
- scoped Ruff: PASS.

Final exact-tree matrices, each with a distinct repo-local `TEMP`, `TMP` and
pytest basetemp:

- prospective chronology, capture plan and market-time: `59 passed` in 7.80 s;
- runtime/packaging/verifier closure: `105 passed, 12 skipped, 2 deselected`
  in 32.60 s;
- publication/external-CAS/anchors/candidate evidence: `343 passed, 2 skipped`
  in 341.13 s.

The first parallel aggregate attempt hit its 300-second orchestration timeout
and returned no attributable summary. It left no matching Python process.
Each matrix was therefore rerun separately as listed above; only those
terminal exit-zero runs count as final evidence.

Terminal read-only roasts:

- Security: P0=0, P1=0, P2=0;
- Quant/Data: P0=0, P1=0, P2=0;
- IT/Operations: P0=0, P1=0, P2=1. The demonstrated v5/v8 split-brain P1
  and missing-registry-validator P2 are closed. Only v6/v7 failure
  observability remains.

## Final scope audit

- exact workspace and Git top-level:
  `C:\Users\jbattaglia\PFC_LT`;
- branch `fix/lt-audit-remediation`;
- HEAD `2f68125bff869ccb21c1e20df0201ad024ed27d3`;
- `git diff --check`: PASS;
- `git diff --cached --check`: PASS;
- staged file count: `0`;
- protected `data/eex_forwards_history.parquet` SHA-256 remains exactly
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- residual matching Python/Playwright/project process count: `0`.

The worktree remains intentionally dirty. Git emitted existing LF-to-CRLF
checkout warnings while inspecting the broad dirty tree; no line-ending
rewrite was requested or performed. No commit or production promotion was
performed.

## Open work in priority order

1. Add prospective fresh CH EEX hard-level observations with trusted
   `available_at`, revisions, official product semantics, independent
   signature/time and external CAS/WORM/fresh monotone HEAD.
2. Accumulate real pre-sealed predictions and post-origin outcomes over
   multiple seasons/regimes; do not count QH proxies or capture episodes as
   independent origins.
3. Reach the preregistered dependence-aware power and calibration gates before
   model selection or candidate assembly.
4. Keep T057 sealed until all predecessor gates pass, then execute it exactly
   once under the governed contract.
5. Only after those proofs, produce a new auditable CH candidate; retain
   monthly solver level authority and exact EEX repricing.
6. Close external publication, service identity, CI/ASR, SBOM/scans,
   observability, rollback and disaster-recovery gates before any production
   promotion.

Production remains strict `NO_GO`.
