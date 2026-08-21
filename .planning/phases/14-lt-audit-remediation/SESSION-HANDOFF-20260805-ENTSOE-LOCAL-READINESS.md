# Session handoff - ENTSO-E local readiness and Databricks target contract

Date: 2026-08-05

## Outcome

The preferred reusable local ENTSO-E import is captured, hash-verified,
profiled and explicitly rejected as empirical/model evidence. It remains useful
for schema and quality-tool development only. The normalized Databricks target
contract is now machine-readable and a short data-engineer gap note is ready.

No Databricks request, Warehouse start or Databricks write occurred.

## Source capture

External source was read without mutation from:

`C:\Users\jbattaglia\pfc_local_data\entsoe\imports\pfc-ct-data-20260522-v3-inventory`

The selected repo-local captured import is:

`build/entsoe-local-readiness/2026-08-05/source/f07dc8a1bbcff4d296f2e17fc45d8c97c9ec2850f0a829e1914b92eb98f13040/pfc-ct-data-20260522-v3-inventory`

It contains nine archive members / 18,892,656 bytes. Source manifest SHA-256:
`f07dc8a1bbcff4d296f2e17fc45d8c97c9ec2850f0a829e1914b92eb98f13040`.

The first flat capture below the manifest-hash directory was correctly refused
by the existing archive verifier because its directory name did not match the
manifest `import_id`. It is unselected negative build evidence. The selected
nested copy preserves the exact import name and passes the existing verifier.

## Changed files

- `pfc_shaping/validation/entsoe_local_readiness.py`
  - validates the legacy import authority boundary;
  - profiles files and series at UTC timestamp/derived-series grain;
  - measures global and within-span coverage;
  - checks 15-minute grid alignment without inferring native cadence;
  - compares dedicated fundamentals/border views with the combined view;
  - emits explicit PIT, lineage, history and consistency findings.
- `tests/test_entsoe_local_readiness.py`
  - eight focused tests including authority escalation, index/cadence,
    missingness, projection divergence and target-contract invariants.
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATABRICKS-INTAKE-CONTRACT-V1.json`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-GAPS-20260805.md`
- `docs/data/SHARED-DATA-PLATFORM.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` (D214)
- `.planning/HANDOFF.md`
- build-only producer:
  `build/entsoe-local-readiness/materialize_readiness.py`

## Selected evidence

Readiness bundle:

`build/entsoe-local-readiness/2026-08-05/audits/1bc0d85177e8f98d2703d98ac9d37d3a063a4f262eb5fa74325ae0d5a22a8e77`

- content ID:
  `1bc0d85177e8f98d2703d98ac9d37d3a063a4f262eb5fa74325ae0d5a22a8e77`;
- manifest SHA-256:
  `1272dc25bd0e0f97b8adfe00295875c47230e6f7d62867c417a796cbc9c31cb6`;
- summary SHA-256:
  `3480c8a1ec386405337774b0ed9a60f37d8734faec6afa23e2924b4756bbc46c`;
- target contract SHA-256:
  `7ede1698099390babfa1d130bfecae61fd1e090888a3d6e6e4f892119db52b87`;
- seven file profiles, 159 series profiles and 67 projection checks;
- two independent materializations returned the same content ID.

## Findings

Verdict: `FAIL_LOCAL_SCHEMA_OR_CONSISTENCY_NO_GO_EMPIRICAL_USE`.

- Critical: archive authority forbids empirical/model use.
- Critical: all three forecast files lack `as_of_utc`.
- High: document, series, zone, unit, native-resolution, revision and quality
  lineage cannot be replayed from the wide Parquets.
- High: neighbour actuals, physical flows, scheduled exchanges and raw NTC span
  only about 55 days versus the explicit 730-day minimum for a seasonal
  cross-market diagnostic.
- High: `entso_15min` versus `entso_fundamentals_15min` diverges on 11 of 27
  projected series and 490,560 cells. Seven raw CH physical series disagree
  during 2026-05-08 through 2026-05-21; derived signals have broader drift.
- Pass: `entso_15min` exactly matches all 40 dedicated border series.
- Medium: the DE renewable-forecast file has one discontinuity in its index.

Local CH actuals are nearly complete from 2021 through May 2026, but this does
not override the import-level authority, PIT and coherence failures.

## Verification

- Focused: `8 passed in 0.26s`.
- Two consecutive offline materializations returned the same selected content
  ID and exact existing bytes.
- Final archive/shared/replay/LT-input matrix, including
  `test_governed_lt_input_snapshot_v2.py`: `132 passed, 4 skipped in 104.31s`.
- All commands used the repo-local Python runtime and mutable paths below
  `build/`. No network or Databricks connector was invoked.

## Next safe step

Give the short gap note and machine contract to data engineering. When one
bounded immutable offline export is available, verify its hashes, dimension
joins, native cadences, units, per-series gaps and vintage history locally.
Only then can governed EEX/ENTSO-E empirical comparisons or AFRY Batch 4 begin.

The monthly solver remains sole level authority. AFRY and OMPEX remain
benchmark-only, T057 remains sealed, local/synthetic substitution remains
forbidden and production remains strict `NO_GO`.
