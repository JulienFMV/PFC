# Session handoff - real runner external prerequisites

Date: 2026-09-07. Base HEAD: `e4bfd457d3d37d654be54abfbf10d821640c1650`.
Decision: D-20260907-298. Audit only; no runtime/source authority change.

## Outcome and scope

`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS` remains the applicable gate.
No independently admitted origin, PRD model-input package or externally frozen
future holdout was established by the accessible evidence. This is an evidence
availability conclusion, not a claim that no external service or response exists.

Read `AGENTS.md`, `.planning/HANDOFF.md` and the 20260904 common-full-price
assembly handoff first. Then inspected the Phase 14 decision log, origin
transport-gap audit, source outage plan, evaluation protocol v6 and its source,
ENTSO-E model-input contract and July replay receipt. The AFRY agent context,
source registration, semantic and shape-diagnostic contracts were also read;
no restricted scenario values or T057 truth were opened.

Success criterion: distinguish locally proven construction/integrity from
external admission for each prerequisite, identify the exact missing evidence,
and preserve every operational authority at `false`. Existing dirty work was
preserved. No new assembly or transport adapter is justified.

## External prerequisite findings

| Prerequisite | Accessible evidence | Blocker and required external evidence |
|---|---|---|
| Origin registration | D278/D279 and `LT-ORIGIN-REGISTRY-TRANSPORT-GAP-AUDIT-20260903.md`: public-key signature verification, receipt/HEAD binding and synthetic CAS conformance exist. | No admitted service profile, root/key custody and signer-role identities, trusted time, independently operated CAS/WORM durability or independent conformance evidence established. Require the externally owned profile and trust admission, then exact signed schedule/request commitments, trusted-time receipt, remote append receipt and fresh authenticated HEAD. Synthetic validity and a wire `countable_prospective_origin=true` claim confer no authority. |
| PRD EEX | D280 binds the existing 5 August capture to the exact three-table Gold query; snapshot and manifest hashes still match. | `INDEPENDENT_SOURCE_TIME_EVIDENCE`, `SIGNED_ENVELOPES_AND_EXTERNAL_TIME`, and `CONVERSION_TO_EXISTING_SIGNED_EEX_VINTAGE_CATALOG` remain pending in the current outage plan. Reuse captured bytes; local hash integrity is not PIT or model admission. |
| PRD ENTSO-E | July `realized_latest_candidate` capture and unsigned replay remain intact. The recorded LSEG reconciliation covers CH/AT/DE-LU/FR; IT-North remains ENTSO-E-only. | No admitted causal package for the runner established. Require origin-safe availability, hash-bound PRD lineage/export and quality evidence, effective-dated series/zone/cadence semantics and governed consumer/model admission for the actual input roles. Latest-source parity proves neither causal history nor finality. The July backfill cannot become retrospective PIT history. |
| Future independent holdout | Protocol v6 locally schedules `ch-lt-future-cohort-2026-10-v1`: 12 monthly origins from 2026-10-06 12:00 UTC through 2027-09-07 12:00 UTC, lead months 1-36. | `external_registration_status=PENDING`, `countable_origin_count=0`, `truth_open_authorized=false`, `holdout_consumed=false`. A local freeze is not independent registration. Require prospective external registration and sealed commitments before the governed windows; no missed-slot shift, backfill or reweighting. FMV risk margins/MDE and outcome-blind power calibration remain separate requirements; 12 slots do not establish sufficient power. |

ENTSO-E finality is optional for the existing authority-negative local replay
and reconciliation. Promotion of these same bytes to `realized_final` would
require external finality evidence binding the semantic hash, exact July window
and selected SeriesKeys. This audit does not reopen finality as a blocker to
already completed local construction. September CH recovery is separate from
July coverage and from origin/source admission.

The old July future-origin v6 selection explicitly describes a noncountable
local rehearsal. The historical T061 handoff is for a different T060 line;
neither it nor an elapsed date substitutes for the new cohort. T057 stays sealed.

## Live external visibility limit

Two read-only connector calls were attempted:

- `github_fetch_issue(repository_full_name="FMVSA/opendata-lakehouse", issue_number=4)`;
- `github_fetch_issue_comments(repo_full_name="FMVSA/opendata-lakehouse", issue_number=4)`.

Both returned GitHub API `404 Not Found`. The current connector cannot establish
the issue's current state or later owner responses. The outage plan's OPEN,
unassigned, three-comment state is historical evidence from 4 September only.
Do not interpret the 404 as proof of no response or reassert current Warehouse
state. No Databricks API, SQL, Warehouse action or external write was performed.

## Smallest next authorized act

Receive or read, through an already authorized channel, the external owner's
existing origin-service profile/trust-admission evidence and source attestations
bound to the captured EEX/ENTSO-E hashes. For issue 4, the next observation is a
read-only retrieval with access that can see the private issue, or an owner
response supplied for inspection. No message or access-change request was sent.

Audit that evidence against the existing contracts without starting the runner.
The origin profile must identify the owner, endpoint/protocol, admitted public
trust material, client identity/credential custody, deadlines/status/recovery,
trusted clock/nonce, retention/WORM, concurrency and independent conformance.
Do not invent or implement these properties locally. Evidence receipt alone
does not authorize registration writes, source promotion, fitting or truth opening.

If the evidence remains unavailable, retain this blocker. Do not repeat the
July query/reconciliation, add an assembly adapter, run a solver for evaluation,
fit, score, rank, start Warehouse/GPU, substitute legacy/synthetic inputs or
unseal a previous holdout to create apparent progress.

## Integrity evidence rechecked on 7 September

Opaque file hashing only; no Parquet/NDJSON records decoded.

| Path | SHA-256 |
|---|---|
| `build/databricks-eex-daily/2026-08-05/eex_ch_power.ndjson` (29,763,661 bytes) | `593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812` |
| `build/databricks-eex-daily/2026-08-05/manifest.json` | `f8ec096be43851d85b16ec2b678d4a695fb0521c2c651e8bcf7c2491a29b50c1` |
| `build/entsoe-july-candidate-20260904/latest-revision-candidate.parquet` (564,727 bytes) | `046ae86ab84a72c61ea44547cc386f85e49d37d325352f4bfca208a08e5a9baa` |
| `build/entsoe-july-candidate-20260904/latest-revision-candidate-replay/manifest.json` | `c2c733cac80ae87f7b8b56572720ba02447a574d030313530f953c92d4b8b8a2` |
| Same replay: `source/day-ahead-export.parquet` | `a2a759bbadacbd34ae96dcc9fbfb3c8b66ef494eb8c65ac24c5b456a514fc76c` |
| Same replay: `consumer/day-ahead-source.parquet` | `3aadd9e8cbdeeba51d8b8cd56b42c6d9b90f914042d720ced154b80b0da90bda` |
| Same replay: `evidence/export-audit.json` | `937ecb22b58d5c849c61b279e1ec507d05206b7cae5c6c113c60b078df008689` |
| `pfc_shaping/lt/evaluation_curve_assembly.py` | `a34b0324785b036a9b0735335c10392bf9225bd92795c22f033458ed403278f3` |
| `pfc_shaping/lt/evaluation_protocol.py` | `516edb82a74f14289db59839c40147f7c3531845b9725b89a0ad5c1102d2c0ee` |
| `pfc_shaping/lt/model/shape_hourly_mlp.py` | `e11e991f9d1585f214870bcdbc32934485dc7b8cf1b8bce39127b0486f5130d8` |
| `pfc_shaping/pipeline/production_phases.py` | `6dcdba561946747dbb8023ff799f72d1188c2898f56a7c47b568b6ac9016d712` |

The replay manifest still declares `UNSIGNED_LOCAL_DAY_AHEAD_EXPORT_REPLAY_NOT_PUBLISHED`
and build ID `b00f2b725c66d91e7b8ec681b578fd080be77837a6e75f9b1de675caade20a26`.
Semantic hashes and previous validation results were read as historical evidence,
not recomputed through a data replay in this session.

## Changes, verification and failures

Only this handoff, `.planning/HANDOFF.md` and the append-only
`.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` are changed by this audit.
No runtime artifact or configuration was created or changed.

Every executable shell action used the canonical workspace and began with
`Get-Location` and `git rev-parse --show-toplevel` guards. Read-only commands:
`Get-Content -Encoding utf8`, `rg --files`, bounded `rg -n`, `Get-ChildItem`,
`Get-FileHash -Algorithm SHA256`, `git status --short`, `git rev-parse HEAD`.
An in-memory SHA-256 inventory recorded all 50 pre-existing dirty/untracked files
before documentation edits, for preservation verification. Documentation edits
used `apply_patch`; final checks use `git diff --check` and the hash inventory.
Observed result: `git diff --check` passed. The before/after inventory identified
exactly the three documentation paths above, no missing pre-existing file and
no other dirty-file byte change. Assembly and protocol source hashes were also
rechecked unchanged. No baseline/source configuration was edited.

Non-substantive failures: an initial file search named nonexistent `config/`;
one PowerShell hash command failed parsing before execution and was corrected;
one in-memory JSON parse encountered Git line-ending warnings and was repeated
with warnings suppressed. The two remote 404 responses remain the external
visibility limitation. One documentation patch had no matching context and
made no change; the corrected append succeeded. No approval, elevation or
alternate runtime was requested.

No tests, training, scoring, ranking, solver, GPU, browser or Warehouse runtime
was launched. This documentation-only audit does not claim a new test pass.
All trust/admission, registration/countability, truth-opening, training,
selection, monthly-level-change, publication, production and trading authorities
remain `false`. The CH monthly BASE solver remains sole level authority;
LT/CT separation and T057 sealing remain unchanged.
