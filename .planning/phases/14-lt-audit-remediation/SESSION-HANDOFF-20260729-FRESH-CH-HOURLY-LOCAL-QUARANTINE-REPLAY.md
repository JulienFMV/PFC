# Session handoff — fresh CH hourly local-quarantine replay

Date: 2026-07-29

Branch: `fix/lt-audit-remediation`

Observed HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`

Production: strict `NO_GO`

## Outcome

One fresh CH day-ahead hourly response was captured through the standard-user
workspace runner, replayed without network access through installed runtime
v22, then audited by the isolated provider verifier. The exact local path is
now technically operational and fail-closed.

This is **local unsigned quarantine evidence only**. It has no trusted
available-at authority, independent product/session identity, signature,
Builder-inaccessible freeze, external CAS/WORM/fresh HEAD, scientific
admission, model-selection, T057, publication, promotion or production
authority.

## Workstation and scope invariants

- Every shell action was guarded by exact cwd and Git root equality with
  `C:\Users\jbattaglia\PFC_LT`; the legacy `H:` checkout was not used.
- No administrator right, elevation, ACL takeover, Defender exception,
  project `.exe` or Playwright runtime was requested or used.
- Every mutable capture, build, temporary, verifier and receipt path is below
  repo `build/`.
- The preinstalled user Python and system Conda were read-only executors only.
- No commit, staging, production promotion, T057 consumption, candidate build,
  CT or Power BI change occurred.
- The monthly solver remains the level authority. OMPEX remains benchmark-only.

## Code and runbook changes

### `scripts/run_workspace_local.py`

- allows `scripts.capture_public_energy_charts_lt` only with required explicit
  `--role`, `--start-utc`, `--end-utc`, `--raw-cadence-minutes`,
  `--acquisition-id`, `--output-directory` and `--ca-bundle`;
- accepts only role `epex_ch` and raw cadence `60`;
- requires an absolute, existing CA bundle on canonical drive `C:`;
- requires capture output below the repo `build/` boundary;
- scrubs ambient `PFC_REQUESTS_CA_BUNDLE` before the child process starts.

### `tests/test_run_workspace_local_script.py`

Tests cover the valid CH-hourly boundary, output below `AppData`, missing
explicit CA, forbidden `epex_de`, forbidden 15-minute raw cadence, and removal
of ambient `PFC_REQUESTS_CA_BUNDLE`.

### `pfc_shaping/tools/OPERATIONS.md`

The standard-user capture command and failure policy are explicit. A partial
capture is non-resumable: preserve/quarantine it and rotate acquisition, run
and output identities. Builder/audit output stays local quarantine and is
forbidden as candidate or T057 evidence.

Durable governance updates are recorded in D-20260729-179 and in the
2026-07-29 fresh CH hourly amendment to the external-CAS RFC.

## Exact fresh capture

Command, after the separate canonical workspace guard:

```powershell
& 'C:\Users\jbattaglia\.conda\ppa_env\python.exe' -B -m scripts.run_workspace_local --run-id chcap01 -- 'C:\Users\jbattaglia\.conda\ppa_env\python.exe' -B -m scripts.capture_public_energy_charts_lt --role epex_ch --start-utc 2026-06-28T22:00:00Z --end-utc 2026-07-28T22:00:00Z --raw-cadence-minutes 60 --acquisition-id ch-da-hourly-20260628t2200z-20260728t2200z-v1 --output-directory 'C:\Users\jbattaglia\PFC_LT\build\prospective-captures\ch-da-hourly-20260729-v1' --ca-bundle 'C:\certs\git-ca-plus-fmv.crt'
```

Result: exit 0, runner status `TARGET_EXIT_ZERO_NOT_AUTHORITY`.

- output: `build/prospective-captures/ch-da-hourly-20260729-v1`;
- capture window: `2026-06-28T22:00:00Z` to
  `2026-07-28T22:00:00Z`;
- acquisition ID: `ch-da-hourly-20260628t2200z-20260728t2200z-v1`;
- local received time: `2026-07-29T15:26:05.738774Z`;
- provider response: 12,632 bytes, SHA-256
  `1ce62eff13e6e596a3a7663349654a32814a63ac2c172d2d56160a6930426537`;
- capture spec SHA-256
  `5bc68818548692e24f1d8a6613d3c4bdad8604ea71c144f966855300b6895405`;
- capture summary SHA-256
  `a03c106c86509888e7bfc9b2ca168b7ba018d09a0cdc90589b5652f327356ae3`;
- capture attempt SHA-256
  `3089251840a9871987eb9e21b0d1f9221e3027698f50660bd248cc8dd8c3edad`;
- `chcap01` runner receipt SHA-256
  `239beec0aff696e3612862698b4e18f27cb2a3b653e16f29a066ca471f9cd6d4`.

The summary reports `COMPLETE_LOCAL_UNTRUSTED_CAPTURE`: 720 native hourly
observations and 2,880 deterministic stepwise 15-minute transport rows.
`native_quarter_hour_truth_eligible=false`. The workstation clock, official
product, auction/session and settlement identity are not independently
trusted.

## Installed v22 networkless replay

The Builder was run with v22 Python using `-I -B`, explicit v22 runtime receipt
and expected receipt hash, repo-local `TEMP`/`TMP`, the capture spec above and
output `build/prospective-acquisitions/ch-da-hourly-20260729-v1`.

Result: exit 0 in approximately 335 seconds.

- Builder manifest SHA-256
  `d1bcddc7d56bfc1c6ad9a2936e6e3b77f1ee662af74df20f63cf22193227f8e0`;
- bronze bytes SHA-256
  `4341865211cbb26f1eceb4fde0212e5504cf4401e71f22f81a20295cc56edc54`;
- bronze semantic-frame SHA-256
  `126cf2037f262a2edfa73ab674ea9b18c9746a691dbf2d4e86662ce25594723e`;
- provider parser SHA-256
  `b6dc574cebd9521c222b1a7022e61aa520f9c575923535950f2e4217c4d39f89`;
- envelope SHA-256
  `b47df5822605f944c0189e13a95afddbfc49706b004ed346590d0a3e6557393d`;
- config SHA-256
  `4373b4c5cb13857e2654d6700639a2e493113e2a2bf6ee9f4047cbb2c44b23d1`.

The manifest remains unsigned, unpublished and Builder-mutable.

## Isolated verifier replay

Inputs:

- artifact: `build/provider-verifier-20260727-v14.pyz`, SHA-256
  `b9afe8358492658214d4bcf01ad1207084ec992df545611c9cb0f02cd0dfa3b5`;
- dependency root:
  `build/runtime-inputs-20260728-repolocal-v1/publisher-site-packages`;
- dependency receipt SHA-256
  `69c0843b4b9a2e202c838ed04027eec1e5008ebec04c0b3c30f37217b0fb45a1`;
- dependency tree SHA-256
  `0ecb7997997cc124375e92614ca08d9c5274c683c6738448b9bd3c5eafaf78f1`;
- scratch: `build/provider-verifier-scratch-20260729-ch-fresh-v1`.

`runtime-check` exited 0 in approximately 245.5 seconds with exactly one
captured artifact root, one captured dependency root and zero source roots.
`audit-acquisition` exited 0 in approximately 106.3 seconds with exact
provider replay.

- audit: `build/prospective-audits/ch-da-hourly-20260729-v1-audit.json`;
- audit SHA-256
  `a6a5ec800e1bd3993aa11ee8c9bf8ec2fa65c3ca384b2df7f6991f4c326fc3fa`;
- verifier runtime receipt SHA-256
  `77b2ce485cf564fa878eddc9f5a46b92827e1053c8289028cf270effcf6fc603`;
- status: `VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION`;
- operational verifier scratch residue: zero.

The `vv-*` names under pytest basetemps are adversarial fixtures, not verifier
runtime residue.

## Terminal matrices

- `captbnd3`: retained negative evidence, exit 1 with one expected-list test
  failure after adding the environment scrub; receipt SHA-256
  `27a33a0e2c8a664d5d9a08b27b40f54408e3c85ffe06f0ecadeb6597dc290d30`.
- `captbnd4`: `28 passed`; receipt SHA-256
  `7f11545e5ee5323bea4beaeb019c4172dacb4c002a8e347e2c35931481a4c69b`.
- `captbnd5`: targeted Ruff pass; receipt SHA-256
  `786710c59ed49dd80df2bbc7ab3c52b648aae499587827b79bc08b22edbb871b`.
- `prospmat2`: `212 passed, 2 deselected` with one pre-existing timezone
  `Period` warning; receipt SHA-256
  `b4ec2e06b5ee952b4fad836640705c1dc8fde483e32b31055dbd3c82079526d0`.
- `runtime8`: `188 passed, 12 skipped, 2 deselected`; receipt SHA-256
  `e72b59911c2489a33ca32b78ad4606204004ad5bd9766b408ea4deddb8cf723b`.

## Independent read-only roasts

### Security

Local P0=0, P1=0, P2=1. The demonstrated ambient-CA bypass is closed. The
remaining local P2 is that CA bytes/hash and the complete TLS policy are not
attested during the request. This is acceptable only for explicit local
untrusted quarantine; it is a production/external-admission P1 together with
trusted time, official product identity, independent signature and external
CAS authority. No claim smuggling or hourly-to-quarter-hour truth confusion
was found.

### IT/Operations

Local P0=0, P1=0; `GO_LOCAL_QUARANTINE_REPLAY_ONLY`. All mutable paths are
inside `build/`, and no admin/AppData/H:/Playwright/project-executable path was
used. Capture retry policy is correctly non-resumable; Builder retry is exact
and idempotent. External signature/freeze/CAS, import-before-admission,
Windows CI/ASR, SBOM, supervision, observability, SLOs and rollback still
block production.

### Quant/Data

Exploratory parser/hourly replay only. The 720 timestamps are exactly hourly;
range is -13.74 to 225.31 EUR/MWh with 28 negative hours. This is one 30-day
summer episode, about 30 observations per hour-of-day, with no DST, winter,
multi-regime, independent rolling origins, sealed forecasts or future
holdout. The 2,880 proxy rows add no information. Scientific automation,
candidate assembly, T057, publication and promotion remain forbidden.

## Governance state and protected evidence

The following evidence remains unchanged and unconsumed:

- T057 supersession registry SHA-256
  `0efec38b768b5e14add6cbc35c9b0cf9f10eb23f2cbf040813e68fe734ca4cf6`;
- PIT supersession SHA-256
  `6bfa49831c91693bd355beace39a6fdd2a74cfa1d25517292634a71dbcfe2282`;
- dependence supersession SHA-256
  `9fd9cf706c768716a42967962527a49a9f92c7a16d1e60de0b3d910565312b72`;
- protected `data/eex_forwards_history.parquet` expected SHA-256
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

## Final scope audit

The terminal read-only audit verified:

- HEAD exactly `2f68125bff869ccb21c1e20df0201ad024ed27d3`;
- protected EEX history hash exactly matched the value above;
- T057, PIT and dependence supersession hashes exactly matched the values
  above;
- staged file count: `0`;
- CT/Power BI status count: `0`;
- targeted trailing-whitespace count: `0`;
- `git diff --check`: exit `0`.

Git emitted only pre-existing Windows LF-to-CRLF working-copy warnings. No
file was staged or rewritten in response.

## Required next work

1. Obtain independently governed trusted available-at/revision evidence and
   official CH product/auction/session semantics.
2. Bind CA/TLS identity and exact transport evidence without granting the
   workstation or Builder external authority.
3. Obtain independently signed, Builder-inaccessible freeze and external
   CAS/WORM with fresh monotone HEAD.
4. Acquire fresh governed EEX PIT forward bytes for exact hard-level repricing.
5. Design and preregister genuinely independent multi-origin rolling-origin
   evaluation across seasons/regimes, then preserve a sealed future holdout.
6. Consume T057 exactly once only after all upstream evidence is admissible;
   build a new CH candidate only if the complete scientific and governance
   gates pass.
7. Keep production strict `NO_GO` until Security, IT/Operations and Quant/Data
   independently accept the complete production evidence chain.
