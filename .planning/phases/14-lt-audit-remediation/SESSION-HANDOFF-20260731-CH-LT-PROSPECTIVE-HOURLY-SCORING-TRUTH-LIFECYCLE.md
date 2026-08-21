# Session handoff - CH LT prospective hourly scoring and truth lifecycle

Date: 2026-07-31

Workspace: `C:\Users\jbattaglia\PFC_LT`

Branch: `fix/lt-audit-remediation`

Production: strict `NO_GO`

## Outcome

The existing V7 structural origin now has a locally executable, fail-closed
hourly scoring and truth-publication lifecycle. It remains a non-countable
rehearsal: no future truth and no T057 outcome were opened, independent origin
count is zero, and no scientific, candidate, publication, promotion or
production authority was created.

The selected corrected local registry is V5:

- path:
  `.planning/phases/14-lt-audit-remediation/CH-LT-LOCAL-FUTURE-ORIGIN-SELECTION-V5-20260731.json`;
- SHA-256:
  `77e8480a563cf64d095ade55282af64e9b569fa9e77c003084d639adf94d35f5`;
- selection ID:
  `4f13d9fe274087449cca92a5ed234f997aefb9fb3bc8ac0e799cf42fc0fbf5e7`;
- origin: 36 monthly targets and 108 scenario predictions from the already
  sealed V7 structural commitment;
- first delivery starts `2026-07-31T22:00:00Z`;
- implementation binding was completed at
  `2026-07-31T08:39:01Z` according to the explicitly untrusted local clock.

## Implemented boundaries

Changed implementation files:

- `pfc_shaping/validation/ch_lt_prospective_hourly_scoring.py`;
- `scripts/score_ch_lt_structural_prediction_commitment.py`;
- `scripts/build_ch_lt_native_hourly_truth_bundle.py`;
- `scripts/audit_ch_lt_local_future_origin_selection.py`.

Changed tests:

- `tests/test_ch_lt_prospective_hourly_scoring.py`;
- `tests/test_build_ch_lt_native_hourly_truth_bundle_script.py`;
- `tests/test_audit_ch_lt_local_future_origin_selection_script.py`.

The scorer now:

- recursively audits the selected future-origin chain before reading any
  receipt or truth path;
- re-hashes the exact commitment, predictions, scoring contract and all three
  bound PFCs, reconstructs the sealed hourly functionals and rejects any
  mismatch;
- requires a hash-bound post-read truth publication receipt before reading the
  truth bundle;
- applies the wall-clock maturity gate to fixture and real paths alike;
- rejects any fixture/real receipt-bundle classification mismatch;
- scores monthly level error, centered hourly MAE/RMSE/correlation, centered
  midday/evening premiums and negative-price count/fraction/deficit diagnostics;
- keeps central as the primary scenario, slow/fast as sensitivities, scenario
  pooling forbidden, all authority false;
- publishes one atomic `score.json` directory and resumes either an empty or
  exact-complete staging directory, while rejecting divergent residue.

The truth builder now:

- audits selection, commitment and target inventory before the prospective
  ledger or bronze paths;
- requires actual local wall clock `>= target_end`, declared read time
  `>= target_end`, and declared read time `<= actual wall clock` before any
  outcome-bearing read;
- recursively rebuilds the exact capture ledger and reconstructs native hourly
  values only from exact four-row stepwise quarter-hour transport blocks;
- requires complete exact UTC target coverage, including DST month counts;
- publishes `truth-bundle.json` plus
  `truth-publication-receipt.json` as one staging directory and rename;
- treats that receipt as post-read evidence, never as truth-open authorization.

## Sealed-artifact incident and closure wording

V4 was incorrectly mutated in place while correcting a demonstrated staging
finding. Its original observed SHA-256 was
`13135fffaf7fef4e97f86b182f5d16cbba941a5a9df1739db2636bb7263f1cd7`.
The path was subsequently observed with other bytes. V5 records
`v4_sealed_byte_identity_preserved=false`; V4 is permanently compromised and
non-authoritative. Do not repair or select it again.

Direct binding V1:

- path:
  `.planning/phases/14-lt-audit-remediation/CH-LT-FUTURE-SCORING-IMPLEMENTATION-CLOSURE-V1-20260731.json`;
- SHA-256:
  `4af806154a28f98e0f92f6185d6cd5c80d568150b2c68640d28a95c003ae03b1`;
- ID: `025523e39b36436550fefc0ade910e5f5982ead24a84cfe0096da890d05aa6e3`.

It binds V5, the canonical auditor, scorer core, scorer CLI, truth builder and
scoring contract. Its status wording overstated this as transitive. Closure
correction V2 is therefore authoritative for interpretation:

- SHA-256:
  `fc9d548a17f8f51f3e7b1115677d7505892133a81a0f54e9168e5fee8da8f573`;
- ID: `a24dcb3b89c5804199df5c744bcdc24be39c0d70f6ccbbd74b3b5e1f30f3a0a2`;
- proven scope: six direct bindings only;
- not proven: transitive imports, exact installed wheel/runtime, loaded-code
  hashes, native dependencies or runtime consumption of the closure.

## Exact verification

Every command below began with the literal cwd/Git-root guard required by
`AGENTS.md`; mutable paths were kept below `build/`.

Targeted final matrix:

```text
build\pytest-runtime-v1\python.exe -B -m pytest
  tests\test_ch_lt_prospective_hourly_scoring.py
  tests\test_build_ch_lt_native_hourly_truth_bundle_script.py
  tests\test_build_ch_lt_structural_prediction_commitment_script.py
  tests\test_audit_ch_lt_local_future_origin_selection_script.py
  -q -p no:cacheprovider
```

Result: `37 passed in 63.37s`.

Focused scorer/truth rerun after the final V5 fixture rebinding:

```text
build\pytest-runtime-v1\python.exe -B -m pytest
  tests\test_ch_lt_prospective_hourly_scoring.py
  tests\test_build_ch_lt_native_hourly_truth_bundle_script.py
  -q -p no:cacheprovider
```

Result: `17 passed in 32.87s`.

Canonical V5 audit alone: `9 passed in 19.31s`.

Ruff over all changed implementation and focused tests: `All checks passed!`.

`git diff --check`: exit `0`.

Protected file check:

```text
data/eex_forwards_history.parquet
SHA-256 21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013
```

The protected parquet was not touched by this session and must not be staged.

## Broad-matrix limitation

The first broad scientific/runtime/packaging/publication command reached:

```text
83 passed, 9 skipped, 2 deselected, 120 errors
```

Every reported error was pytest setup failure caused by inaccessible managed
Windows temp/basetemp ACLs. A second invocation with `TEMP`, `TMP` and an exact
repo-local `--basetemp` reproduced the same inability to read the directory
pytest had just created. No elevation, ACL takeover or sandbox override was
requested. These runs are non-conclusive and must not be reported as green or
as product regressions. Use the governed standard-user CI runner or replace
remaining generic `tmp_path` fixtures with governed repo-local fixtures.

## Independent read-only roasts

Security final:

- P0: `0`;
- local P1: direct source binding is not a generated, exhaustive and
  runtime-consumed import closure;
- external P1: no trusted time, independent registry/signature or
  builder-inaccessible CAS/WORM;
- P2: same-user ABA/reparse windows, post-rename verification window and
  Windows power-loss directory durability remain.

IT/Operations final local result: P0/P1/P2 `0/2/4`.

- P1: installed wheel does not package the `scripts.*` auditor dependency;
  collector has no governed scheduler, owner-approved SLA, lease,
  retry/backoff, contiguous watermark, late-revision policy, alerts/SLO or
  retention/runbook.
- P2: V7 runtime remains local/unbound with about 1.02 GiB peak; crash,
  concurrency and power-loss drills are incomplete; closure is not consumed by
  runtime; no real matured truth-to-score E2E receipt exists yet.

The roasts confirmed the maturity ordering, honest post-read receipt semantics,
fixture biconditional, atomic publication and strict `NO_GO` behavior.

## Next work, in order

1. Move the canonical future-origin auditor and prediction-functional verifier
   into `pfc_shaping.*`; expose packaged scorer/truth CLIs and eliminate runtime
   imports from `scripts.*`.
2. Generate and verify an exhaustive import closure; build an exact wheel and
   standard-user runtime; run from foreign cwd with checkout absent from
   `sys.path`; make the supervised receipt consume closure V2 or its successor.
3. Implement the collector service contract: single-writer lease, idempotent
   cadence, retry/backoff/catch-up, contiguous watermark, late/revision policy,
   disk retention, alerts and measurable lag SLO. FMV must name owner and approve
   SLA.
4. Continue fresh CH hourly captures. Do not score August 2026 before exact
   target maturity and complete ledger coverage. At maturity, run the real
   truth-to-score E2E once, emit supervised receipts and mark the prospective
   truth consumed; never use it for retuning this origin.
5. Accumulate the preregistered number of independent externally anchored
   origins and obtain direct-CH same-mask baselines before any quality claim.
6. Keep Swiss truth hourly until the native 15-minute market/data transition is
   independently verified and admitted. T057 remains unread and cannot be
   reused confirmatorily.

Monthly solver authority, LT/CT separation, OMPEX benchmark-only status and
strict production `NO_GO` are invariants.
