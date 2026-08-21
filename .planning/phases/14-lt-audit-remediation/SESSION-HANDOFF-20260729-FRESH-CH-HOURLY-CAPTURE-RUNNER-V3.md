# Session handoff - 2026-07-29 - Fresh CH hourly capture and runner v3

## Status and non-negotiable boundaries

- Canonical cwd/Git root: `C:\Users\jbattaglia\PFC_LT` only.
- Branch `fix/lt-audit-remediation`; HEAD remained
  `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree remains intentionally very dirty. No reset, clean, restore, stage,
  commit or production promotion occurred.
- No admin, ACL takeover, Defender/ASR exception, project executable,
  Playwright, legacy `H:` repo, CT or Power BI action occurred.
- Protected `data/eex_forwards_history.parquet` remained at SHA-256
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- Monthly solver level authority, LT/CT independence, OMPEX benchmark-only and
  corrected T057 no-reuse invariants are unchanged.
- Production remains strict `NO_GO`.

## Final local capture status

The previously frozen curl finalizer embedded the 2026-07-27 request window.
The reusable contract validator now binds caller-held contract bytes by
SHA-256 and validates exact source, UTC window, counts, CA bytes, curl argv,
one-shot repo output and negative authority claims. The historical v1 route is
retained unchanged.

The selected final local diagnostic is attempt4:

`output/phase14/ch_da_hourly_capture_20260729_attempt4_curl_v2_local_only`

Contract ID:
`df2b6edfbd3d75c58813345361a9b405ff8de5e5c7718d59f33ae8ee78978e09`.
Contract document SHA-256:
`6542043c198804b46d449051d915eb0f739c969772494adfcd600f649ff1a061`.
Final receipt SHA-256:
`aa4d5a1d37b448a1dc268e271d422082c0260cd2e6a00c612e5420c68461a55c`.

The exact seven-file namespace reports:

- window `[2026-06-28T22:00:00Z, 2026-07-28T22:00:00Z)`;
- 720 unique finite hourly values; provider body 12,632 bytes, SHA-256
  `1ce62eff13e6e596a3a7663349654a32814a63ac2c172d2d56160a6930426537`;
- 2,880 validated in-memory forward-filled proxy rows, not a persisted bronze
  artifact and never 2,880 independent/native quarter-hour observations;
- summary schema `ch_lt_hourly_curl_capture_summary.v2`;
- local transform schema `ch_lt_hourly_local_transform_record.v2`, explicitly
  rejected by the governed acquisition builder and no `capture-spec.json`;
- `builder_input_authorized=false`,
  `external_admission_input_candidate=false`,
  `transport_attestation=false` and
  `certificate_revocation_check_performed=false`;
- hourly truth eligibility blocked pending product/session identity, revision
  policy, trusted availability and external admission;
- trusted time, independent signature, builder-inaccessible immutability,
  external CAS/fresh head, scientific execution/admission, model selection,
  publication, promotion, future holdout, T057 and production all false.

The contract's `as_of_utc` is a contract/window anchor, not a capture or
`available_at` timestamp. Curl uses `--ssl-no-revoke`; this is now explicit in
the contract as local-untrusted-only and forbids any external admission claim.

## Attempt and supersession history

Frozen registry:
`.planning/phases/14-lt-audit-remediation/CH-LT-HOURLY-CAPTURE-SUPERSESSION-20260729.json`.
Registry ID:
`3d66ca81cd535f6b6c94beafbd418b1007502e7ade309ae002f8fd20be5d7a5e`.
Registry document SHA-256:
`ce283678e195a22e50def7d8526f74997669906268c338c836b162daa1d644bf`.
The registry ID is only the canonical internal content identity; neither it nor
the document hash is an independent signature, authority or monotone ledger.

- Attempt1 contract SHA
  `75bd2632e3f53418338f482c255c5196312fb54edfc812ff8311c5ad58770feb`:
  downloaded valid bytes but finalization failed closed on the old hardcoded
  request parameters; no receipt.
- Attempt2 contract SHA
  `1506fb32ebcfa01718b440f9c1e638d516ed10adaf7b3100142f4319b2926659`:
  dynamic window succeeded but the receipt inherited misleading v1 identity
  names and a stale predecessor hash; receipt SHA
  `cdf7b24d1abd3ae490176d0bf84a2ffd11e7ab31d05e5705cceade3c6b85fad7`.
- Attempt3 contract SHA
  `52a74500f9a62c0aff5a793cbb0900f7c975f43b09d8f5cfc02a7717b5134762`:
  clean identity fields but initial Security roast demonstrated prepared-claim
  injection, builder-compatible `capture-spec.v1` without transitive contract
  binding and unqualified external-input wording. Receipt SHA
  `00e4845b9e1ad7471a1b5d17cd64c82321d16f37e413971ba9b77817435e341f`.
- Attempt4 is the only selected local diagnostic. Attempts 1-3 are explicitly
  non-selectable and were not overwritten, repaired or relabelled.

The registry is local documentation, not an independent authority or
production ledger.

## Demonstrated Security/Quant findings and closures

The first read-only Security roast reported P0 0, P1 3, P2 3. Corrections:

1. PREPARE/FINALIZE now enforce exact attempt/command keys and revalidate
   status, paths and every false authority/transport field. Countertests reject
   extra `scientific_execution_authorized`, changed publication and changed
   promotion claims.
2. V2 no longer emits builder-compatible `capture-spec.json`. Its local
   transform record binds contract/body/header records, carries false builder
   and external-input flags, and a test proves the governed builder rejects it.
3. The contract no longer calls the bytes an external-admission candidate. It
   declares curl transport unattested and certificate revocation unchecked.
   External/scientific use therefore remains a blocking future recapture, not
   a claim closed by this local slice.
4. Summary schema is now v2 and explicitly separates the 2,880 in-memory proxy
   rows from materialized/native observations.
5. The supersession registry selects attempt4 locally and rejects attempts
   1-3.

Quant/Data verified the provider payload: 720 unique hourly timestamps at
exact 3,600-second cadence, 720 finite EUR/MWh values, min -13.74, max 225.31,
28 negative hours, no gap or duplicate. It keeps four scientific P1 blockers
open: no point-in-time/available-at/revision evidence, external product/session
identity unverified, quarter-hour rows non-native, and only one 30-day summer
window without sealed predictions or future holdout. Candidate CH and
production remain `NO_GO`.

## Workspace runner v3

Long v2 pytest roots caused Windows false failures, not requests for admin:

- runtime/packaging negative: `103 passed, 12 skipped, 2 deselected`, one
  transient `WinError 5`; receipt SHA
  `5e113a4fc575b0c3067613344f88372f188a5445868dd1a952de11d17f00bcbf`;
- long publication negative: `324 passed, 1 skipped, 175 failed`; receipt SHA
  `f1e345d3efb0470983d62e03176ca4cc3f2ee3bb617b284c422ca03cefb89810`;
- short-id but nested-root negative: `495 passed, 2 skipped, 3 deep-path
  failures`; receipt SHA
  `7aec32acccf09daa74ee3e0e7d88c6b84b2ce0b5037b86835de139dfb597b85d`.

No failed receipt was reused or counted green. Runner v3 caps run IDs at 16
characters, retains descriptive receipt/caches under
`build/workspace-local-runs/<id>` and uses the fresh receipt-bound pytest root
`build/wpt-<id>`. Both roots are preflighted and revalidated. `OPERATIONS.md`
requires coordinated retention of both roots and records that stdout/stderr,
test counts, interpreter/runner identity and descendant supervision are not
embedded in the receipt.

## Exact final commands and results

Final implementation/evidence hashes:

- `scripts/run_workspace_local.py`: 20,412 bytes, SHA-256
  `c01aac7f02fd9b21caf5efe8e840cf04b2b14ce79fd6b2ce762223fa1a1ade0a`;
- `tests/test_run_workspace_local_script.py`: 9,363 bytes, SHA-256
  `46bd0fcdb48a7182f4ef065bdccecca675332bb5e6d00734caa9001a12ecc910`;
- `pfc_shaping/validation/ch_lt_hourly_capture_contract_v2.py`: 16,634
  bytes, SHA-256
  `6f3ab0d31f0568b1cb50172ca4dc8924e6df1a53b53a3b0c1f9d21bf8356f21e`;
- `scripts/run_ch_lt_hourly_curl_retry_from_spec.py`: 28,367 bytes,
  SHA-256
  `1454270be2993fb7666056f9c9de95891838dd0391c2084227719ce1a69f8ad8`;
- `tests/test_ch_lt_hourly_capture_contract_v2.py`: 12,386 bytes,
  SHA-256
  `1c5618e21fa5d2b2e5ada283a9943c6e1aabf69d1ee984270f214dc4d9b2b640`;
- attempt4 contract: 3,211 bytes, SHA-256
  `6542043c198804b46d449051d915eb0f739c969772494adfcd600f649ff1a061`;
- supersession registry: 2,109 bytes, SHA-256
  `ce283678e195a22e50def7d8526f74997669906268c338c836b162daa1d644bf`.

Every shell action was preceded by the separate literal cwd/Git-root guard.

```powershell
python -B -m scripts.run_workspace_local --run-id q12 -- python -B -m pytest `
  tests\test_ch_lt_hourly_capture_contract_v2.py `
  tests\test_ch_lt_hourly_capture_run_spec.py -q -p no:cacheprovider
```

Result: `18 passed in 1.32s`.
Receipt SHA-256:
`f93eec9f0b45bda54521af6c189e7d0d9339c917a9271ce89c0366f0efe84709`.

```powershell
python -B -m scripts.run_workspace_local --run-id q13 -- python -B -m pytest `
  tests\test_ch_lt_hourly_capture_contract_v2.py `
  tests\test_ch_lt_hourly_capture_run_spec.py `
  tests\test_governed_lt_acquisition.py `
  tests\test_audit_provider_acquisition_quarantine_script.py `
  tests\test_audit_legacy_provider_resolution_script.py `
  tests\test_lt_package_contract.py -q -p no:cacheprovider -m "not slow"
```

Result: `126 passed, 1 deselected, 1 known timezone-to-Period warning in
20.66s`.
Receipt SHA-256:
`f31551f1b8f9458b862b68e37dbaa884dac4ba6976f5fdf2e75aa654bad31d8c`.

```powershell
python -B -m scripts.run_workspace_local --run-id q14 -- python -B -m ruff check `
  scripts\run_workspace_local.py tests\test_run_workspace_local_script.py `
  pfc_shaping\validation\ch_lt_hourly_capture_contract_v2.py `
  scripts\run_ch_lt_hourly_curl_retry_from_spec.py `
  tests\test_ch_lt_hourly_capture_contract_v2.py
```

Result: `All checks passed!`.
Receipt SHA-256:
`f63959e8c5cd61fbbeb61495af8f5579d49d708144746b89f15a83b95fc4c214`.

Unaffected runner-v3 terminal baselines retained in the same source session:

- q5: `30 passed`, receipt SHA
  `18fd977b0546839638d701bfe9c4cdba574747f2b5686ab47e6e31929df5f70f`;
- q7 publication/CAS/candidate: `498 passed, 2 skipped`, receipt SHA
  `33ac7406420b79cc4e7b87e6da31964476dec33fbdd4e6b7f42a4112b4172f57`;
- q8 runtime/packaging: `104 passed, 12 skipped, 2 deselected`, receipt SHA
  `b2ba81b6e2de4d6f72173ad74c5d06c916c9e30a664e3dd4939fa1c197507419`.

The runtime/packaging matrix was rerun after the attempt4 fixes under q15:
`104 passed, 12 skipped, 2 deselected in 129.90s`; receipt SHA-256
`75cad264b2ac6270db5b3351e257389c4c57b201ffa2de513cb8e7ed73f29b93`.

Attempt4 prepare, exact curl and finalize exited 0. Final post-documentation
`git diff --check` exited 0; staged and forbidden CT/Power BI counts were both
zero; the protected parquet SHA-256 remained
`21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

## Independent re-roasts

Final read-only Security/Governance re-roast: local P0 0, P1 0, P2 2. The
prepared-claim, builder-boundary and external-transport wording P1 findings are
closed for the untrusted local scope. Residual P2 are same-user path races and
a documentary, non-consumed supersession registry. Verdict: GO only for the
attempt4 parser/hourly diagnostic; external admission/science/publication/
production `NO_GO`.

Final read-only IT/Operations re-roast: local P0 0, P1 0. Runner v3 two-root
operation, retention, latest-handoff pointer, D174 supersession, D175/RFC and
attempt4 seven-file inventory are coherent. Residual local P2 concern receipt
observability/descendant supervision and Windows path-budget generality;
external admission and production remain `NO_GO`.

Final read-only Quant/Data re-roast: local parser/diagnostic P0 0, P1 0, P2 0.
Four scientific blockers remain intentionally open: no PIT/available-at/
revision lineage, product/session identity unverified, no native quarter-hour
information, and only one ex-post 30-day summer window. Candidate CH and
production remain `NO_GO`.

## Next work

1. Do not admit, train, select, backtest prospectively or open a holdout from
   this local hourly diagnostic.
2. Design a separately supervised/hash-bound capture plus product/session,
   revision/available-at, trusted-time, signature, immutable retention and
   external-CAS/fresh-head authorities.
3. Obtain fresh licensed CH EEX forward PIT bytes with exact quote/product
   convention; the monthly solver remains the sole level authority.
4. Freeze the executable successor preregistration only after unique origins,
   sample-size/dependence/power, scenario/MC and FMV materiality values exist.
5. Build a new hash-bound CH candidate without pilot/T057 selection reuse,
   then execute rolling-origin, sealed future holdout and probabilistic
   calibration.
6. Confirm the Swiss explicit 15-minute auction from archived official bytes
   before treating any 15-minute day-ahead series as supported truth.
7. Never promote before all evidence and independent roasts are complete.
