# OMPEX independent benchmark contract

## Outcome

OMPEX can be used as a frozen external benchmark after a PFC candidate is
frozen. It cannot be used as truth, a model input, a tuning target, a selection
criterion or a promotion gate. The PFC and OMPEX must both be scored against
the same independently frozen realised truth.

## Archive findings on 2026-08-05

- The read-only folder contains 353 XLSX files: 351 dated curve vintages and
  two templates. Filename dates run from 2025-07-02 through 2026-08-05, with
  49 missing calendar dates and no duplicate date.
- Six hash-frozen curve samples contain one visible `HFC` sheet, exactly the
  `Date` and `EUR/MWh` columns, finite numeric prices and no formulas, macros,
  external links, connections or embedded objects.
- Despite the folder and template name, the curve is native hourly, not native
  15 minutes. A four-row expansion is only a stepwise transport proxy and
  retains one effective hourly observation.
- The local labels are structurally consistent with Europe/Zurich hour-ending
  intervals. Subtracting one hour and applying Swiss DST yields a unique,
  contiguous UTC hourly grid across every selected curve, including both DST
  transitions. Desk or vendor authentication of this convention is still
  missing.
- The older vintages contain five delivery years; vintages from 2026 add a
  sixth year. A year-end adjacent pair is identical over its common horizon
  and adds one full delivery year.
- Historical delivery values can change between vintages. In the adjacent
  2025-09-10/2025-09-11 pair, 37,753 of 43,824 common delivery rows change,
  including 11 intervals earlier than the later filename timestamp. No price
  value or price statistic is stored in the audit bundle.

## Scientific comparison

The authoritative contract is
`OMPEX-INDEPENDENT-BENCHMARK-CONTRACT-V1.json`. It requires:

1. candidate bytes, configuration and input manifests frozen before OMPEX is
   opened;
2. exact OMPEX filename, size and SHA-256 plus authenticated availability at
   the forecast origin;
3. the same origin, target window, UTC calendar, EUR/MWh unit and missingness
   mask for candidate and OMPEX;
4. exclusion of every delivery interval already started before the origin;
5. comparison of both forecasts against independent realised truth;
6. dependence-aware inference by independent origin and native hour;
7. numeric improvement and non-inferiority margins approved before outcomes.

“Superior from every point of view” does not mean winning at every timestamp.
That criterion is statistically unstable and encourages overfitting. It means
strict improvement on preregistered primary estimands, non-inferiority on every
mandatory protected subgroup, and no failure of PIT, calendar, monthly-solver,
provenance or operational guardrails.

Periods with hourly Swiss price truth are scored hourly. A native 15-minute
accuracy claim remains `UNSUPPORTED` until the corresponding delivery regime
has independently governed native 15-minute price truth. Hourly observations
must never be counted four times.

## Current status

The local structural audit passes and is exactly replayable under content ID
`228deb1fb1adca0e7a4e6cb64406d2f33621c8dc1e2a98c21294fd68354fb3bf`.
The complete price-free archive inventory and readiness findings are documented
in `OMPEX-ARCHIVE-DATA-QUALITY-REPORT-20260805.md`; its independently replayed
content ID is
`336700af0b38324bbfc99c5332b5f360a01e00f2fd14baab090ebcb8e087a57a`.
Scientific scoring remains `NO_GO` until timestamp/availability semantics,
governed realised truth, countable future origins and preregistered numeric
margins are available. AFRY remains a separate descriptive benchmark and the
monthly solver remains the sole CH level authority.

## Corrected local comparison entry point

The historical direct HPFC-to-OMPEX comparator is superseded. It previously
selected the shift with the smallest candidate/OMPEX distance and did not
require realised truth; those behaviours cannot support an accuracy claim.

The current `scripts/compare_hpfc_ompex_benchmark.py` requires a hash-bound
local candidate, local OMPEX vintage and realised-truth file on one exact UTC
hourly grid. It has no automatic alignment mode, rejects missing or duplicate
hours, and reports only paired descriptive errors against truth. Details and
the remaining scientific blockers are in
`OMPEX-PAIRED-TRUTH-DIAGNOSTIC-REPORT-20260805.md`.

## Price-free receipt-chain reference

The repository also contains a non-production reference for the evidence
sequence that must precede any scientific comparison:

1. freeze and register the candidate before benchmark access;
2. bind the exact OMPEX vintage and its asserted availability at the origin;
3. bind independent realised truth published only after delivery.

The reference verifies canonical signed bytes, distinct role keys,
cross-receipt hashes and strict chronology without opening price values. It is
deliberately not an external trust system: caller-supplied local public keys do
not authenticate their organizational owners, and locally asserted timestamps
are not trusted time. Consequently, every passing reference assessment retains
zero countable origins and no scoring, selection, superiority, promotion or
production authority.

The exact trust boundary, 15-test adversarial result and production gaps are
documented in
`OMPEX-BENCHMARK-EVIDENCE-CHAIN-REFERENCE-REPORT-20260805.md`; the machine
contract is
`OMPEX-BENCHMARK-EVIDENCE-CHAIN-REFERENCE-CONTRACT-V1.json`.
