# Session handoff - CH LT successor readiness and packaged runtime v24

Date: 2026-07-29

## Outcome

The superseded CH LT preregistration v1 remains unusable and no successor was
created. A byte-bound structural readiness contract now enumerates ten
distinct evidence families that must exist before an admitted successor can
be created. The checkout-only verifier was also packaged and qualified from a
fresh, repo-local, sealed runtime v24.

This is local quality evidence only. Scientific admission, execution,
publication, promotion and production are all false. T057 outcome/score and
future truth bytes were not read.

## Canonical identities

- readiness contract:
  `.planning/phases/14-lt-audit-remediation/CH-LT-PIT-SUCCESSOR-READINESS-CONTRACT-20260729.json`
  - contract ID
  `5e655cab1ca090cd067100dc8eb06161811b77991132e7c299883a7eb249e706`;
  - document SHA-256
  `734a7824ec747c829526774b346da441b45fff7cff5c9eb79ffc653ac78c7b8e`;
- supersession registry:
  `.planning/phases/14-lt-audit-remediation/CH-LT-PIT-PREREGISTRATION-SUPERSESSION-20260727.json`;
  SHA-256
  `76dba0b05948d336268b0a50c16df82ecd1c5138626c3fa87ee219148cb8e5e8`;
- v1 preregistration remains bound at SHA-256
  `aba798530084b7031a0ac38b1c48b20cff575d6082edbcf37c9a04528900ba61`;
- estimand draft remains bound at SHA-256
  `4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931`;
- compute draft remains bound at SHA-256
  `b231345e96e7664ae02b7dbf3514af87d47ded7783034eaab1f8d449a28fe96f`.

## Ten closed readiness requirements

1. governed point-in-time CH and EEX evidence;
2. native CH layer truth and post-episode outcomes;
3. exact origin, target, mask and inner-fold inventories;
4. frozen deterministic candidate, baselines, complexity and hard market
   gates;
5. frozen dependence, power and multiplicity design;
6. frozen probabilistic, scenario and Monte Carlo design;
7. FMV Risk-approved economic materiality;
8. qualified CPU float64 oracle and CPU/GPU runtime parity;
9. independent Security, IT/Operations and Quant/Data reviews;
10. external admission envelope and monotone one-shot ledger.

All ten evidence entries are `MISSING`. A non-executable candidate core may be
authored and frozen; independent reviews and admission must bind its exact
hash before it can become an admitted successor. Cross-fitting is mandatory
for nuisance/calibration/selection work but cannot replace independent
confirmation or manufacture independent units. Insufficient effective
clusters, power or calibration cells is `UNSUPPORTED_NEVER_PASS`.

The validator requires exact document bytes, verifies mapping-to-bytes
equality and the canonical SHA-256, uses stable mono-link reads and requires
all four bound roles to be physically distinct. It reads only readiness,
registry-bound structural documents and no T057 outcome/score or future truth.

The final contract additionally binds exact native truth layer, market
product, auction/session, unit and admitted product-semantics identities. Its
economic population ID is exactly `HYDRO_DISPATCH`, matching the bound
estimand. CPU float64 is the canonical numerical reference and explicitly
grants no scientific/data authority. Complexity evidence must include
inner-fold results, selected complexity by outer origin and diagnostics;
insufficient or unstable evidence is `UNSUPPORTED_NEVER_PASS`.

## Packaging and runtime v24

Positive wheel inventory now contains:

- `pfc_shaping/cli/verify_ch_lt_preregistration_supersession.py`;
- `pfc_shaping/validation/ch_lt_preregistration_supersession.py`;
- `pfc_shaping/validation/ch_lt_successor_readiness.py`.

The packaged CLI checks sealed installed-runtime identity before parsing
arguments. It requires absolute evidence-root and registry paths and produces
no authority. The workspace runner permits this module only with `-I -B`, the
canonical workspace, canonical registry and exact registry hash.

Reproducible wheel commands, each with separate repo-local `TEMP`, `TMP`, pip
cache, bytecode cache, build and dist directories and
`SOURCE_DATE_EPOCH=1783987200`:

```powershell
python -B setup.py build --build-base build\wheel-build-ac bdist_wheel --dist-dir build\wheel-dist-ac
python -B setup.py build --build-base build\wheel-build-ad bdist_wheel --dist-dir build\wheel-dist-ad
```

Both wheels are byte-identical:

- SHA-256
  `3ebe242cd8b09b5b98c56ffbe4357ef2fccecb5be0d35be5ec9c400cb776597c`;
- 89 members;
- embedded source revision
  `e2b7c89a6413ff6cb84c7b6e2e31e4412fc72fc32157b0fdbffb2b4a39f42cc8`;
- both wheel audits PASS and `promotion_eligible=false`.

The fresh offline Conda prefix was created from the 19 retained repo-local
archives and v22 explicit spec using the preinstalled ProgramData Conda tool
read-only. All mutable paths were under `build/`; there was no network,
installation outside the repo, elevation or Defender exception.

- prefix: `build/conda-runtime-v24-successor-base`;
- pre-first-execution manifest:
  `build/launcherless-python-runtime-manifest-20260729-v24-base.json`, SHA-256
  `c59e0c1af6de5189e1b6915656d25ae8307ee9d16f1f3474f43a1563df1b20d5`;
- prefix receipt:
  `build/launcherless-conda-prefix-receipt-20260729-v24.json`, SHA-256
  `f466b04259881f07f6c7995c32b198da6df0af1af8808f8b021e222fd3405a66`;
- runtime receipt:
  `build/launcherless-runtime-receipt-20260729-v24.json`, SHA-256
  `2eeb80e212cff3301c4e8a9349cffc2e93a41e20514dd2cd4cf6d95749219c2d`;
- runtime closure: 8,493 files / 19 distributions, tree
  `d977fb87e97e2d14b430318a8bd4b1351e39e31ebfe32cb6f40f3e799a872158`;
- `local_quality_authorization=true` and
  `production_authorization=false`.

The installed verifier was run from `build/foreign-cwd-v24`:

```powershell
build\conda-runtime-v24-successor-base\python.exe -I -B -m pfc_shaping.cli.verify_ch_lt_preregistration_supersession --evidence-root C:\Users\jbattaglia\PFC_LT --registry C:\Users\jbattaglia\PFC_LT\.planning\phases\14-lt-audit-remediation\CH-LT-PIT-PREREGISTRATION-SUPERSESSION-20260727.json --expected-registry-sha256 76dba0b05948d336268b0a50c16df82ecd1c5138626c3fa87ee219148cb8e5e8
```

Result: exit 0, status
`ACTIVE_FAIL_CLOSED_V1_SUPERSEDED_NO_EXECUTABLE_SUCCESSOR`, ten blockers and
all authorities false. Output SHA-256:
`69be634e455c385b49b817ffbc198d2adf585078046a83082b0bb49745d46264`.

The live foreign-cwd `python -I -B -c` probe returned exactly, in order:

1. `build/conda-runtime-v24-successor-base/Lib`;
2. `build/conda-runtime-v24-successor-base/DLLs`;
3. `build/conda-runtime-v24-successor-base/governed-site-packages`.

The checkout, foreign cwd, prefix root, user site, system site, AppData and
phantom `python311.zip` are absent.

## Final matrices

All commands ran through `scripts.run_workspace_local` with the exact test
lists retained in their receipts and fresh IDs:

- `srdpro03`: `380 passed, 2 deselected`, known timezone-to-period warning;
  receipt SHA-256
  `80db88233c45bc72e37535c41efeb3113e78adf7395034d24089baffd1e32695`;
- `srdrun03`: `269 passed, 12 skipped, 2 deselected`; receipt SHA-256
  `32c350ae17de647e05d9a6a8f9cc6b8201613d9956a4cbcce8f6cad3f1d9ee2b`;
- `srdcas03`: `200 passed`; receipt SHA-256
  `e8bb7ea0bb235ad03cfcaf0e6942c5d721550d25220be7aa5cc76d721a117a04`;
- `srdcand03`: `181 passed, 2 skipped`; receipt SHA-256
  `4109e523ffb6b792c6e5849be7f739f628db8a4716182ce980d3b267f15938ef`;
- `srdruff03`: Ruff PASS; receipt SHA-256
  `620b0a389afcfdebb6cb5cd9af055dbe5a98a0b3c35b635cfd69826c98fb2405`;
- `git diff --check`: exit 0; only existing LF-to-CRLF checkout warnings.

All terminal runner receipts bind source tree
`d70410c956fdca6841f90a9ee760780c55019a608edf0a57a14821d547299740`,
493 selected Python/config files and 8,890,088 bytes. Runner SHA-256 is
`c0739e8b321fac610fcdfb23f55e874616772fcbab4ed3b36989ed3b4682f49a`.

## Independent terminal re-roasts

- IT/Operations: P0/P1/P2 `0/0/0`, GO for the local non-authoritative slice
  only; production `NO_GO`.
- Quant/Data: P0/P1/P2 `0/0/0`; exact `HYDRO_DISPATCH`, native truth product
  identity, CPU numerical-reference boundary and complexity evidence are
  closed. Ten scientific evidence families remain missing.
- Security: P0/P1/P2 `0/0/0`; byte provenance, exact mapping/bytes binding,
  runtime isolation and final receipts are coherent on the frozen tree.

All three reviews are read-only and accept only the local structural/runtime
slice. They do not close the ten readiness blockers or production controls.

The failed first prefix-audit attempt is retained: the PPA interpreter lacks
`zstandard` and failed before receipt creation. The exact audit then passed
with the preinstalled Anaconda Python used read-only and every mutable path
redirected below `build/`. This was not an admin or security-policy issue.

## Changed source and contract files

- `.planning/phases/14-lt-audit-remediation/CH-LT-PIT-SUCCESSOR-READINESS-CONTRACT-20260729.json`
- `.planning/phases/14-lt-audit-remediation/CH-LT-PIT-PREREGISTRATION-SUPERSESSION-20260727.json`
- `pfc_shaping/validation/ch_lt_successor_readiness.py`
- `pfc_shaping/validation/ch_lt_preregistration_supersession.py`
- `pfc_shaping/cli/verify_ch_lt_preregistration_supersession.py`
- `pfc_shaping/package_contract.py`
- `scripts/run_workspace_local.py`
- `scripts/verify_ch_lt_preregistration_supersession.py`
- `tests/test_ch_lt_successor_readiness.py`
- `tests/test_ch_lt_preregistration_supersession.py`
- `tests/test_run_workspace_local_script.py`
- `tests/test_lt_package_contract.py`

Documentation updates are D183, the RFC amendment, Operations section 10,
root handoff and this handoff.

## Invariants and next work

- No commit, staging, reset, clean, restore or production promotion occurred.
- `data/eex_forwards_history.parquet` was not touched; verify its protected hash
  before the next handoff.
- No `pfc_shaping/ct/*` or Power BI file was touched.
- Monthly solver remains level authority and OMPEX remains benchmark-only.
- V24 supersedes v23 only for current local-quality execution of these source
  bytes. It does not close same-user writable pre-import, external
  signature/CAS/WORM/fresh-HEAD, service identity, CI/ASR, SBOM/scans,
  supervision, observability, rollback or disaster-recovery gates.
- After the final re-roasts are recorded, resume governed prospective CH/EEX
  evidence, rolling-origin design, T057 one-shot admission and a new auditable
  CH candidate. Do not consume T057 or create a successor until the ten
  readiness requirements have independent hash-bound evidence.
