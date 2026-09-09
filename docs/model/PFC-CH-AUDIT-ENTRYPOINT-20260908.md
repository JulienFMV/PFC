# PFC CH audit entry point — 8 September 2026

Read AGENTS.md and .planning/HANDOFF.md first. This checkpoint is provided for
the user-requested independent Claude audit. It preserves the accumulated LT
implementation, comparisons and tests without granting production or model
admission. The GitHub repository is public: source data, fitted artifacts,
credentials and granular client portfolio diagnostics remain local.

## The actual objective

Improve the national Swiss hourly PFC. The monthly BASE solver remains the
sole monthly-level authority and its EEX projection is reused. Historical
price levels and within-month hourly shape must be assessed separately.

Client consumption/production budgets are optional economic exposure profiles.
They do not represent Swiss national load/generation and are not prerequisites
for the next CH hourly model experiment. BLOC13 is not a required product unless
an actual use case is confirmed. The D313–D314 business-source investigations
must not redirect the core PFC roadmap toward client portfolio modelling.

## Current reference and comparisons

- D304 signed calendar reference retained. D305 composition, D306 conditional
  intraday, D307 recency, D308 fixed global blends and D310 maturity comparisons
  have not justified replacement under the declared gates.
- No per-month winner selection, post-solver monthly patch, clipping of negative
  prices or external OMPEX/LSEG price fitting. Preserve price-zero, ramps,
  hourly shape extremes, delivery year, season and horizon diagnostics.
- D309 seam/revision diagnostics do not establish archived-vintage forecast
  accuracy. D311 external distances are descriptive. D312–D314 daily pilot has
  one actual day; a same-day replay is not a second origin. No scheduler exists.
- Solver/assembler/EEX invariants and all six authorities remain unchanged:
  production, promotion, scientific_admission, trading, externally_registered,
  countable_origin are false.

## Audit the implementation and claims

1. Trace signed hourly construction through `signed_benchmark.py`,
   `evaluation_curve_assembly.py` and the existing `lt/model/assembler.py`.
   Check monthly-neutral inputs, finite full-grid output, DST/leap handling,
   monthly solver conservation and final BASE/PEAK projection.
2. Review D307–D310 experiment scripts for frozen global configurations,
   pre-origin thresholds, fair common populations, level/shape separation and
   faithful reporting of failed stability gates. Exposed historical development
   results do not become independent holdout evidence.
3. Review `databricks_lt_materialization.py` for exact national source semantics,
   timestamp units, interval expansion/revision ties and local latest-observed
   versus causal historical inputs. Do not relabel revised history as PIT.
4. Review the daily collector for actual date, duplicate-day guard, bounded
   Warehouse/query behavior, immutable attempts and preservation of historical
   EEX dates absent from a new response. Inspect all nine quote conflicts;
   a rounding hypothesis and unsigned policy do not accept any conflict.
5. Identify defects and unsupported claims with file/line evidence. Distinguish
   production blockers from useful locally authorized exploratory work. National
   source qualification can continue while future truth and custody mature.

Minimum LT tests are in AGENTS.md. Focused checkpoint tests and their exact
results are recorded in the checkpoint handoff. Do not run production release,
AFRY, T057, protected-data mutation or model promotion as part of this audit.

## Existing national evidence to reuse

D300 captured national ENTSO-E inputs and SFOE OGD17, with exact receipts in
`build/local-pfc-source-preflight-20260907/`. It prepared a full September2025–
August2026 physical window, native CH historical prices and national weekly
hydro. These are real latest-observed sources, not retrospectively independent
PIT archives. D304–D310 already use the retained common inputs appropriately
within their explicitly local development scope.

The next bounded task checks the broader national dictionary and coverage for
load, generation by technology, forecasts, hydro, outages and CH interfaces.
It must distinguish usable training history, origin-frozen future physical
assumptions and short operational forecasts. A day-ahead forecast cannot be
extended silently to delivery years2027–2029. Existing scientific feature
inventories remain frozen; a new experiment needs its own explicit protocol.

## Evidence access

Public Git contains code, tests, protocols and summaries. Paths under build/
refer to local immutable evidence; their presence in a document is not proof
that an auditor has those bytes. Report any unavailable artifact as not
independently replayed. Local full business notes are intentionally more
detailed than their public Git summaries; do not publish those local additions.
