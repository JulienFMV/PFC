# Session handoff — CH LT contrast-aware power core v3 and runtime v33

Date: 2026-07-30 (Europe/Zurich)

Branch: `fix/lt-audit-remediation`

Session-start HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`

Production status: strict `NO_GO`. No candidate, holdout, publication or
production promotion occurred.

## Outcome

The outcome-blind CH LT successor policy is now contrast-aware and closes the
Quant/Data findings about delivery-month overlap, strong FWER, conjunctive
power and sample-size-grid ambiguity. The current local package is installed
in a fresh archive-audited launcherless runtime v33. It runs both packaged V3
audit CLIs from a foreign cwd with an exact three-entry `sys.path`.

This is local-quality execution evidence only. The core retains ten `MISSING`
evidence slots; the dependence/power design retains six `MISSING` inputs. No
scientific, execution, promotion or production authority was granted.

## Scientific identities and decisions

- Core:
  `.planning/phases/14-lt-audit-remediation/CH-LT-SUCCESSOR-CANDIDATE-CORE-V3-20260730.json`
  - SHA-256: `e2c2a6f9d3ca677991f3f76f03dec1328492e2f60a4015b7796a2ff6aa6f905a`
  - core ID: `d86dab183dd95a2ca3cda0862f2c9f50cec01716a37c6cdc1c3a15763254dcc3`
- Dependence/power draft:
  `.planning/phases/14-lt-audit-remediation/CH-LT-DEPENDENCE-POWER-DESIGN-DRAFT-V1-20260730.json`
  - SHA-256: `005b8655b817db10e7f3c227b1c5912d545305b68e262ab82d9aa0b5817a6a91`
  - design ID: `2147af9d2fd277a43df6f88f10dfa6c4061a7184c7b84ca0b09af5e66fa120c9`
- Readiness update:
  `.planning/phases/14-lt-audit-remediation/CH-LT-PIT-SUCCESSOR-READINESS-UPDATE-V3-20260730.json`
  - SHA-256: `07abfbe22f1211049d95333b7af5c987983230325d112f14c4299f234d6caadb`
  - readiness ID: `eb69c29f1f3744c4d62d4ea2ecdb2532a8039148e1daba1dcf1f56a25c807f11`

Monthly origins imply exact mechanical overlap/block lower bounds by
contrast: fixed lead month `0/1`, M01–M06 `5/6`, M07–M12 `5/6`, M13–M24
`11/12`, M25–M36 `11/12`. The full M01–M36 aggregate is `35/36` and remains
diagnostic, not primary-confirmatory.

The stationary-bootstrap parameter is the expected mean geometric block
length. Each exact hypothesis must freeze its direction-oriented
least-favourable null boundary, a distinct FMV Risk-approved alternative
effect and a marginal power floor. Strong FWER is calibrated over all
attainable partial-null configurations, unless an external proof establishes
marginal p-value super-uniformity under every configuration. Required power is
the one-sided 95% lower bound for the complete FMV gatekeeping decision at
least 0.80, plus every pre-frozen marginal floor. Candidate sample sizes form
one common ascending grid of unique origin counts; each contrast keeps its own
block length, and the smallest common N must pass size, conjunctive power,
marginal power, effective-cluster and coverage gates.

The read set remains outcome-blind. The V3→V2 chain reads exactly one T057
artifact, `T057-OUTCOME-BLIND-TOMBSTONE-20260730.json`, and never the
outcome-bearing supersession registry or referenced results. Legacy T057 is
permanently excluded and a new independent future holdout remains required.

## Swiss market-time invariant

Current CH day-ahead truth remains native hourly. Native 15-minute CH truth is
allowed only after a separately admitted transition/go-live; hourly values
must never be duplicated as quarter-hour truth. The current local contract
retains the planned first trading date 2026-11-03 with
`confirmed_go_live=false`. Official context checked in this slice:

- Swissgrid balancing roadmap:
  `https://www.swissgrid.ch/dam/jcr:eaa2aa50-deb1-4579-92c6-dacb66429480/balancing-roadmap-en.pdf`
- EPEX 15-minute products:
  `https://www.epexspot.com/en/new-15-minute-products-market-coupling`

## Standard-user and permission closure

All commands ran from literal cwd/Git root
`C:\Users\jbattaglia\PFC_LT`. No `H:` checkout, admin/elevation, ACL takeover,
Defender/ASR exception, project executable or Playwright runtime was used.
Mutable Conda, wheel, pytest, temp and cache state remained below `build/`.
`scripts.run_workspace_local` rejects every explicit launcherless-runtime path
outside repo `build/` (`uv.lock` is the sole declared root exception).

The RFC now marks every retained `AppData\Local\pfc-lt-build` path as
historical-only and forbids resuming the obsolete v9 assembly route. The
current publisher inputs are:

- `build/runtime-inputs-20260728-repolocal-v1/publisher-wheelhouse`;
- `build/runtime-inputs-20260728-repolocal-v1/publisher-site-packages`;
- `build/runtime-inputs-20260728-repolocal-v1/publisher-dependency-closure-receipt.json`.

Confinement tests: `perm30v2`, `128 passed`.

## Reproducible wheels and runtime v33

Wheels G/H:

- `build/wheel-dist-core30v3-g/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- `build/wheel-dist-core30v3-h/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- 99 members, 547,694 bytes, byte-identical SHA-256
  `7fa4e36160717bbd535f717b2f62d95098fd547f8286c6085a64c88ab8969de7`;
- source revision
  `11696869165c70e4b3d98f8d13ca4e38ff74fb38df2fecb77ec792cc3d1baee6`;
- both wheel-contract audits report `PASS`, `promotion_eligible=false`.

Runtime v33:

- prefix: `build/conda-runtime-v33-core-v3-base`;
- Python manifest SHA-256
  `42b102047dbc2a4bcee189c8d2e3f81ec8a63df6f1f9ff8ed55ade38526c16e8`,
  6,285 files, tree
  `edac59a20873d8bc029a8ba267d7b556ef2b113411190c546c08c7eb20c738a9`;
- Conda prefix receipt SHA-256
  `f6d69286061362ed8d73ace3ae8b2caac1d9128887a248fc34e6a1505175256a`,
  prefix receipt ID
  `07098c1078fb56d0669abd1555f03bfa35afa3a8612631b0dfa653c437e166cd`;
- runtime receipt SHA-256
  `3e9af2e9b3ccbb9092e84aa26beaa7b499ba7397e1b1c1b5eaf0d4d0bbd7475d`;
- closure: 8,503 files / 19 distributions, tree
  `a702f4c15fc33b6982f43565d44f0f24db61a4328146e01a22dd4f38e484410a`;
- exact `sys.path`: runtime `Lib`, runtime `DLLs`, then exactly one
  `governed-site-packages`; no checkout, prefix root, user or system site;
- `local_quality_authorization=true`, `production_authorization=false`.

The exact Conda creation shape was:

```powershell
C:\ProgramData\anaconda3\python.exe -B -m conda create --offline --copy `
  --prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v33-core-v3-base `
  --file C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-explicit-20260729-v22.txt `
  --yes --json
```

`CONDARC`, package cache, envs, `TEMP` and `TMP` were repo-local. Crucially,
`PYTHONPYCACHEPREFIX` was absent during this Conda step.

The exact assembly arguments are retained in
`build/workspace-local-runs/asm33v3/execution-receipt.json`: wheel G, the three
repo-local publisher inputs above, `build/launcherless-wheelhouse-20260727-v1`,
manifest/hash, prefix receipt/hash, `uv.lock`, prefix V33 and runtime receipt
V33. Target exit is zero and status is `TARGET_EXIT_ZERO_NOT_AUTHORITY`.

## Negative evidence retained

- Initial wheel namespace A failed before compilation because its `egg-base`
  did not exist. Intermediate C/E/F wheels are superseded and not installed.
- `man32v3` produced a manifest but its runner receipt is `LAUNCH_FAILED` after
  a parent timeout; it is not accepted as green evidence.
- V32 is rejected. `PYTHONPYCACHEPREFIX` was incorrectly redirected during
  `conda create`, leaving 402 generated Conda bytecode files outside the
  prefix. `aud32v3b` failed on missing
  `Lib/site-packages/pip/__pycache__/__init__.cpython-311.pyc`. V32 was never
  assembled or used. It was superseded by fresh V33, not repaired.
- Long parent timeouts for `aud33v3` and `asm33v3` did not trigger retries.
  Their original child processes remained active, were waited to termination,
  and produced terminal exit-zero receipts. No overlap occurred.

## Commands, results and receipts

All runner commands used
`python -B -m scripts.run_workspace_local --run-id <id> -- <target>` (or the
same module under the existing read-only ProgramData Anaconda interpreter for
archive audit/runtime assembly). Exact argv and mutable paths are preserved in
each receipt.

| Run | Result | execution receipt SHA-256 |
|---|---:|---|
| `fix30ruff1` | Ruff pass | `d531b88455e4d2a99cebaca4e76964995915d6c0ca546b4fd56ee92aaed0e563` |
| `fix30test1` | 177 passed | `3c2f5747d699b20cc4fb2eda5cb772e67a388cc31e53e1f895226a64e39669e7` |
| `fix30audit1` | power design valid incomplete `NO_GO` | `449ebfe03be116431125b8dec49990381f99d2c3c088ea23305932dad2492fe4` |
| `whl30g` | wheel contract pass | `20cf1f9b3f4ced2cd3548953153d0d8c1aa6cb9a25c0c81253b0b052eaaa5d88` |
| `whl30h` | wheel contract pass | `b751adec4b769792065591e06333804058104d9e4d2c2af01236956712160b60` |
| `man33v3` | manifest pass | `aaa50776a829825e95022847ebb5b81929acafbad027ee1b9efcbc163f215522` |
| `aud33v3` | prefix audit pass | `802a659ed67881fcf9a0fd49a00f9dc69d74440f612275eddfb89c9f9d6676b1` |
| `asm33v3` | runtime assembly pass | `630d8470803b731df1acab777d74eba01cbf7ce98faf80300a9895b416f2db70` |
| `v33ruff1` | installed-test Ruff pass | `a32b2a18fe5d069b48f24b7c5e37f90e11147ade9d49742a8177ffafc50ac753` |
| `pkg33smk` | installed foreign-cwd smoke: 1 passed | `59052c6d0dfb5cbe46ebd1a653f84cfcf2d800434ea9ba9ce3e03d8fb536f726` |
| `sci33v3` | scientific/current: 268 passed | `2806c3b03d56ed9e437363d25bb00e1fad3699778040a7be0f00877945399088` |
| `pkg33v3` | runtime/packaging: 133 passed, 12 skipped, 1 deselected | `90d3d239acd2a4364464956169f50736b81e96393cf917a7a4ee1df32caf7d5d` |
| `pub33v3` | publication/candidate: 88 passed | `de29f27ab09c44808ed84017c973909d932ad0234c538b5d6cf0f0c99c8d7414` |
| `cas33v3` | external CAS/anchor: 193 passed, 2 skipped | `df7fe22583f000a7b055f669a7c04c425a66b7a558087d2bb50e0538abbcdb7e` |

The final four matrices share source-tree digest
`fb93a34b56491b2c29592332440fdefd75be4b9a3c47d22e300e643cd24583f0`.
All receipts keep scientific, evaluation, promotion, runtime and production
authority false.

## Independent read-only re-roasts

- Quant/Data final targeted verdict: P0/P1/P2 `0/0/0`. It confirms the exact
  geometry, direction-oriented null boundaries, strong-FWER scope,
  conjunctive-power gate, marginal floors, common-N rule and outcome-blind
  tombstone-only chain.
- IT/Operations final verdict: P0/P1/P2 `0/0/0`. It confirms V32 was rejected
  as negative evidence; V33 manifest, prefix audit and assembly are terminal;
  wheels are reproducible; the foreign-cwd installed smoke and all fresh
  matrices pass; every mutable path is repo-local; no process residue or
  overlap, admin, AppData, project executable or Playwright use remains.
- Security/packaging final verdict: P0/P1/P2 `0/0/0`. It independently
  rehashed both wheels, runtime receipt, current source revision and installed
  source bytes; verified the exact three-entry `sys.path`, foreign-cwd smoke,
  complete current receipts and TOCTOU/import tests; and found no AppData,
  admin/elevation, ACL takeover, Defender/ASR, Playwright or project launcher
  use. Neither review is a production authorization.

## Main files changed in this slice

- `pfc_shaping/validation/ch_lt_successor_candidate_core_v2.py`
- `pfc_shaping/validation/ch_lt_successor_candidate_core_v3.py`
- `pfc_shaping/validation/ch_lt_successor_readiness.py`
- `pfc_shaping/validation/ch_lt_preregistration_supersession.py`
- `pfc_shaping/validation/ch_lt_dependence_power_design.py`
- `pfc_shaping/cli/audit_ch_lt_successor_candidate_core.py`
- `pfc_shaping/cli/audit_ch_lt_dependence_power_design.py`
- `scripts/audit_ch_lt_successor_candidate_core.py`
- `scripts/audit_ch_lt_dependence_power_design.py`
- `scripts/run_workspace_local.py`
- `pfc_shaping/package_contract.py`
- `tests/test_ch_lt_successor_candidate_core_v2.py`
- `tests/test_ch_lt_successor_candidate_core_v3.py`
- `tests/test_ch_lt_successor_candidate_core_installed_v33.py`
- `tests/test_ch_lt_successor_readiness.py`
- `tests/test_ch_lt_preregistration_supersession.py`
- `tests/test_ch_lt_dependence_power_design.py`
- `tests/test_lt_package_contract.py`
- `tests/test_run_workspace_local_script.py`
- the three planning JSON artifacts above;
- `LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`;
- this handoff, the Phase 14 decision log and root handoff.

The broader worktree remains intentionally very dirty. No reset, clean,
restore, staging or commit was performed. Protected desk data, LT/CT boundary,
monthly-solver level authority, Power BI exclusion and OMPEX benchmark-only
status remain unchanged.

## Next work

1. Obtain governed fresh point-in-time CH/EEX evidence and an admitted direct-CH
   development paired-loss panel; freeze the exact primary family, margins,
   alternative effects, marginal floors, origin/target mask inventory and MC
   CPU/GPU parity manifest.
2. Externally freeze/sign the complete design with trusted time, independent
   CAS/WORM and fresh head before any future truth is opened.
3. Build a new prospective candidate, evaluate rolling-origin maturity and a
   new independent future holdout. Never reuse T057.
4. Keep current CH scoring hourly until native 15-minute CH go-live is
   independently admitted.
5. Close external production controls: service identity/ACL, signed
   provenance/SBOM, CI, observability, backup/DR and rollback. Production
   remains strict `NO_GO` until all evidence is complete.
