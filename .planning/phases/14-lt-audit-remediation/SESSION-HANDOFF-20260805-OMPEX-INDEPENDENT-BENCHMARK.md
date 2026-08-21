# Session handoff - OMPEX independent benchmark

Date: 2026-08-05

## Outcome

D218 admits only a local structural audit and freezes the scientific comparison
contract. Selected audit content ID:
`228deb1fb1adca0e7a4e6cb64406d2f33621c8dc1e2a98c21294fd68354fb3bf`.
OMPEX remains read-only, post-freeze and benchmark-only. It has no model,
training, tuning, selection, monthly-level, promotion or production authority.
Scientific scoring is still `NO_GO`.

## Exact files added or changed in this batch

- `pfc_shaping/validation/ompex_benchmark.py`
- `tests/test_ompex_benchmark.py`
- `.planning/phases/14-lt-audit-remediation/OMPEX-INDEPENDENT-BENCHMARK-CONTRACT-V1.json`
- `docs/research/OMPEX-INDEPENDENT-BENCHMARK.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

Local build evidence, intentionally outside the production/source contract:

- `build/ompex-benchmark/materialize_audit.py`
- `build/ompex-benchmark/2026-08-05/source-samples/`
- selected bundles
  `build/ompex-benchmark/2026-08-05/audit-v4-a/` and `audit-v4-b/`

No existing user change was reverted. No CT, Power BI or protected heavy data
file was edited by this batch.

## Frozen source captures

Seven selected workbooks total 4,926,902 bytes. Exact SHA-256 values are bound
in `audit-v4-a/manifest.json`. The six curve vintages span the initial archive,
two schema/size transitions, an adjacent-vintage revision pair, the year-end
horizon extension and the latest observed vintage. The seventh file is the
empty template. The captures were copied only after before/copy/after hashes
matched their read-only source bytes.

## Structural findings

- read-only inventory: 353 XLSX files, 351 dated curves, two templates;
- filename dates: 2025-07-02 through 2026-08-05; 49 missing calendar dates;
  zero duplicate date;
- every selected curve has one visible `HFC` sheet, exact `Date`/`EUR/MWh`
  headers, no formulas/active content and finite numeric prices;
- native cadence is hourly, not 15 minutes;
- Europe/Zurich hour-ending interpretation gives exact contiguous unique UTC
  hours through both DST transitions, but remains vendor/desk unauthenticated;
- adjacent 2025-09-10/2025-09-11 vintages change 37,753 of 43,824 common rows,
  including 11 rows earlier than the later filename timestamp;
- the 2025-12-31/2026-01-01 common 43,824 rows are identical and the later
  file adds 8,760 rows;
- no price values or price statistics are emitted by the audit bundle.

## Contract and decision rule

Candidate and OMPEX are both predictions and must be scored against one
independent truth. Candidate bytes/config/inputs are frozen before OMPEX access.
Exact OMPEX vintage bytes and authenticated availability at origin are
mandatory. Pre-origin delivery intervals are excluded. Error-minimizing
automatic timestamp alignment is forbidden for scientific scoring.

Primary estimands are hourly MAE delta, monthly solver/CH EEX level integrity,
and native-quarter-hour MAE delta only where independent price truth is itself
native 15 minutes. Hourly truth is never counted four times. The final rule is
conjunctive: all integrity gates, a preregistered primary improvement,
non-inferiority across mandatory subgroups with multiple-testing control, and
strict improvement in at least one shape/tail/ramp/uncertainty dimension.
Numeric margins and alpha must be FMV-approved before outcomes.

## Commands and outputs

All shell commands verified canonical cwd and Git root first and ran from
`C:\Users\jbattaglia\PFC_LT`. Mutable temporary and pytest paths stayed below
`build/`.

- focused parser/contract tests: `7 passed`;
- JSON validation and Python byte-compilation: pass;
- Ruff over the parser, tests and local producer: pass;
- adjacent OMPEX/PIT/scoring/import matrix: `52 passed, 1 skipped`, with 13
  existing Matplotlib/PyParsing deprecation warnings;
- two real materializations: identical content ID
  `228deb1fb1adca0e7a4e6cb64406d2f33621c8dc1e2a98c21294fd68354fb3bf`;
- byte-identical replay members:
  - `manifest.json` SHA-256
    `33966d3a7e5724ee3152ae966d647234218aa1047cc1cb1067e738d6d8bf1aac`;
  - `summary.json` SHA-256
    `fb2cbc684979507044b5b765f6e97b010e7904a16b5242d4c144226a2b25e82a`;
  - `workbook-profile.json` SHA-256
    `bb358c3247a1257151907112dbbebc9a4a0bc24f0a4ffcedcf87a4db3ed1c5e5`.

One first test attempt failed only on a DatetimeIndex name assertion (`2
failed, 5 passed`) and was corrected without changing product logic. One first
materializer attempt failed before output because a directly launched build
script lacked the repo root on `sys.path`; the producer was fixed and both
subsequent replays passed.

## Open blockers and next batch

- obtain a desk/vendor statement for OMPEX timestamp and availability
  semantics;
- build a full hash-bound archive inventory instead of relying on the current
  read-only filename inventory observation;
- freeze FMV-approved numeric superiority/non-inferiority margins before
  outcomes;
- wait for governed ENTSO-E PIT evidence, signed EEX vintages, independent
  realised truth and multiple countable future origins;
- then implement the outcome-blind scorer from the frozen contract. Do not use
  the existing advisory script's minimum-error automatic alignment for
  scientific scoring.

No Databricks request, Warehouse start, write or network call occurred.
Production remains strict `NO_GO`; T057 stays sealed; AFRY and OMPEX stay
benchmark-only; LT/CT separation and monthly solver authority are unchanged.
