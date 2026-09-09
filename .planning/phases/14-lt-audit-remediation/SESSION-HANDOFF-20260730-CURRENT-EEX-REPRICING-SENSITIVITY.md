# Session handoff — 2026-07-30 current CH EEX repricing and sensitivity

## Outcome

The selected 2026-07-29 CH EEX quarantine row is now evaluated by a
non-disclosing, fail-closed current-price BASE audit. The local monthly-solver
mechanics pass on the independent quote basis, but the source quote book and
the full delivered-product claim do not pass:

- 20 BASE and 20 PEAK source identities are present;
- the already-started 2026-07 BASE and PEAK products are excluded;
- 19 wholly undelivered BASE products remain over 2026-08..2032-12;
- 17 form the active independent hierarchy and reprice with maximum error
  `7.105427357601002e-14` EUR/MWh against tolerance `1e-9`;
- `2026-Q4` and `2027` are redundant consistency-only source points;
- all 19 displayed BASE points cannot be repriced simultaneously at `1e-9`:
  maximum residual is `0.0017745586238220312` EUR/MWh;
- no tick, rounding, session or settlement semantics is attested, so the
  residual is not reclassified as an interval/tick pass;
- deterministic quote-to-month sensitivity passes at symmetric shocks
  `0.1` and `1.0` EUR/MWh, with cross-scale maximum derivative difference
  `5.329198415893188e-09` against `1e-7`;
- 19 future PEAK products remain open and final hourly BASE/PEAK repricing was
  not evaluated.

This is local diagnostic evidence only. The capture is not trusted PIT, the
workspace runner is post-import observation rather than pre-import admission,
and every scientific/solver/candidate/publication/promotion/production
authority remains false. T057 was not read. Production remains strict
`NO_GO`.

## Workspace and protected state

- cwd and Git root: `C:\Users\jbattaglia\PFC_LT`;
- branch: `fix/lt-audit-remediation`;
- HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`;
- intentionally dirty worktree preserved; no reset, clean, restore, stage,
  commit or promotion;
- protected `data/eex_forwards_history.parquet` final observed SHA-256:
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- no LT-to-CT import, CT edit, Power BI edit, project executable, Playwright,
  AppData/ProgramData mutation, administrator right, ACL takeover or
  Defender/ASR exception;
- every mutable test/run path remained below repo `build/`.

## Current inputs and output evidence

Selected current registry:

- `.planning/phases/14-lt-audit-remediation/CH-EEX-CURRENT-LOCAL-CAPTURE-SELECTION-20260730.json`;
- SHA-256
  `5f0b99aa04fabcb8219cfa34f20ea262a705940cbc3db3ab2e114ba99bb4a778`;
- selection ID
  `3be51903d7ed2774d464f8bfd49b20fe283fb5d1b2c0bb93f677e99fe4884667`;
- selected capture `eex-ch-20260730-v1` source SHA-256
  `fb71338f51334128878526877b802e48819b555639913a786a35d710a6b151e5`;
- quote identity commitment
  `74ba58f6d00c8734ea668d487c8fb48d6e12e35642045be7a2c834daddbdfc95`;
- quote value commitment
  `399d524c2c002b55bf844a5410791d0fa33ba7fc4ed843896f01b26769ece258`.

Fresh real audit:

- run ID `eexrp30v2`;
- report ID
  `c7fa4d7cca2fe5dd20d4884249a13201d340b67502fc4c1c6ec6d5da49f46446`;
- target stdout
  `build/workspace-local-runs/eexrp30v2/target-stdout.log`, SHA-256
  `f410f12104186aa5d537dd3eb8317336e6618f9746cfb56017e673f5765255a8`;
- execution receipt
  `build/workspace-local-runs/eexrp30v2/execution-receipt.json`, SHA-256
  `b0f368443a474bdd77233e2adcd7b5084d412aba49e9ab9ce57eab9fdf97dcec`;
- status `TARGET_EXIT_ZERO_NOT_AUTHORITY`, target exit `0`, complete output,
  empty stderr;
- bound source-tree SHA-256
  `f37bb4ce5acac4e1c76065579f32e80c0e7f908e6a88932571eb27ee97fca114`.

Fresh matrix:

- run ID `eexrp30mx2`;
- execution receipt SHA-256
  `54eacbfd21c2a44a489a1b25371dd0f5d4fadaea5edbb29ac511ee471b0cc4ce`;
- target stdout SHA-256
  `125f0bffbdb6e19972b8b2a340ef214050f328f40976078be87e80e861f3d8cb`;
- same source-tree SHA-256 `f37bb4ce...ca114`;
- result `263 passed, 1 skipped in 28.36s`; the skip is the existing
  TensorFlow-dependent CT isolation case, not a current-audit failure.

Earlier `eexrp30v1` and `eexrp30mx1` receipts bind superseded source-tree
bytes and must not be selected as current evidence.

## Code and tests

Added:

- `scripts/audit_ch_eex_current_repricing_sensitivity.py`;
- `tests/test_audit_ch_eex_current_repricing_sensitivity_script.py`.

Changed:

- `scripts/run_workspace_local.py`: exact allowlist route for the new module,
  reusing the canonical registry path and frozen SHA argument grammar;
- `tests/test_run_workspace_local_script.py`: positive route plus inherited
  shadow-path, wrong-hash, duplicate/extra-argument rejection.

The target audit:

- replays the existing exact current-capture audit first;
- stable-reads and hashes the registry and selected source again;
- consumes exact in-memory workbook bytes;
- binds non-disclosing identity/value commitments;
- derives the first wholly undelivered month from the selected snapshot;
- emits no quote price or monthly curve level;
- reports active-point exactness separately from all-source-point exactness;
- cannot claim tick/rounding semantics because the executable route hardcodes
  `source_semantics_attested=false`;
- records PEAK/final hourly repricing as failed/open;
- keeps monthly solver as level authority and all authorization flags false.

## Exact commands and results

Every shell action started with the exact cwd/Git-root guard. The final real
audit was:

```powershell
python -B -m scripts.run_workspace_local --run-id eexrp30v2 -- `
  python -B -m scripts.audit_ch_eex_current_repricing_sensitivity `
  --registry .planning/phases/14-lt-audit-remediation/CH-EEX-CURRENT-LOCAL-CAPTURE-SELECTION-20260730.json `
  --expected-registry-sha256 5f0b99aa04fabcb8219cfa34f20ea262a705940cbc3db3ab2e114ba99bb4a778
```

Result: exit `0`, local BASE mechanics PASS, all-source exact repricing FAIL,
semantics FAIL, final PEAK repricing FAIL, production `NO_GO`.

The final matrix was:

```powershell
python -B -m scripts.run_workspace_local --run-id eexrp30mx2 -- `
  python -B -m pytest `
  tests/test_audit_ch_eex_current_repricing_sensitivity_script.py `
  tests/test_audit_ch_eex_current_local_capture_script.py `
  tests/test_monthly_curve_sensitivity.py `
  tests/test_monthly_forward_curve_constraints.py `
  tests/test_monthly_forward_curve_solver.py `
  tests/test_monthly_forward_curve_priors.py `
  tests/test_monthly_forward_curve_integration.py `
  tests/test_monthly_forward_curve_audit.py `
  tests/test_run_workspace_local_script.py `
  tests/test_lt_package_contract.py tests/test_lt_ct_imports.py `
  -q -p no:cacheprovider
```

Result: `263 passed, 1 skipped`.

Scoped Ruff passed. A focused pre-matrix reported `110 passed`, and the final
post-hardening focused run reported `96 passed`; only `eexrp30mx2` is the
selected current matrix receipt.

## Independent read-only roasts

Final current-byte verdicts:

- Security/Governance: P0/P1/P2 `0/0/0`;
- IT/Operations: P0/P1/P2 `0/0/0`;
- Quant/Data: P0/P1/P2 `0/0/0`.

The first Security and IT/Operations responses rejected the v1/mx1 receipts
after the code changed. Fresh v2/mx2 receipts were required and verified.
This stale-evidence rejection is retained as positive fail-closed evidence.
No roast granted production authority or modified files.

## Durable decision and next work

The exact production claim is now decomposed as follows:

1. active independent BASE constraints: locally exact;
2. every displayed BASE source point: not exact and mathematically
   contradictory at `1e-9` under the current hierarchy;
3. interval/tick/rounding acceptance: forbidden until provider semantics and
   FMV Risk policy are independently attested;
4. final delivered BASE/PEAK products: not yet evaluated on current admitted
   bytes;
5. source freshness/authenticity: still not independently admitted.

Next work must obtain provider-authenticated session/settlement/tick/rounding
semantics and an FMV-approved conflict hierarchy or interval policy. Then it
must run final hourly BASE/PEAK/OFFPEAK repricing on independently admitted
PIT bytes. Only after those deterministic gates close may the current evidence
enter a candidate. Rolling-origin power, probabilistic scenarios, a new sealed
future holdout, external CAS/WORM/fresh HEAD, CI/SBOM/observability/rollback
and service identity remain open. T057 remains unconsumed and production
remains strict `NO_GO`.
