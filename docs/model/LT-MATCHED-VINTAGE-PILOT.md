# D312 local matched-cutoff benchmark pilot

Twenty distinct Swiss capture days, one immutable local record per day. The
first record is produced at an actual current valuation; future dates remain
pending. No simulated backfill, repeated same-day model run or reused valuation
counts as another snapshot. Daily records are not independent accuracy trials.

D304 signed equal-history recipe, existing monthly BASE solver and EEX projection
remain fixed. Current EEX may refresh; closed-month CH history updates only at
the next complete month. No external curve enters solver priors, shape fitting,
model selection or tuning. Benchmark sources are selected by actual recorded
availability at the shared cutoff, not by closeness to D304. Their nominal issue
times can differ. Local observed-before-cutoff is weaker than independently
authenticated vendor publication or point-in-time historical qualification.

Bind all inputs and outputs by hashes; preserve source issue, capture, valuation,
candidate commitment and registration times separately. Sources captured earlier
in the same session may be reused by exact hash and recorded observation time.
Never relabel their issue date as the common cutoff. D311 prices are already
exposed; fixed-recipe prospective monitoring does not erase that exposure.

Generate D304 first from candidate inputs alone. Monthly-neutral signed shape,
unchanged solver levels, same assembler and final projection. No monthly patches.
Compare full common Swiss months, level and shape separately, and retain
uncovered months as UNSUPPORTED. Compute both pairwise full coverage and the
common coverage of all three curves. Check DST, years, FMV seasons, horizons,
negative/near-zero prevalence, pre-origin shape/ramp thresholds and month seams.
Forecast distances are descriptive; no superiority threshold is inferred.

Revision series use unchanged delivery timestamps across genuinely later daily
snapshots. Decompose D304 changes into solver level, pre-projection shape and
projection. Preserve each external curve's original issue and source generation.
With one pilot day, revision stability is UNSUPPORTED. D311 versus D312 is a
separate transition diagnostic, not two prospective pilot days.

Future truth: separately capture native CH hourly realised prices after each
Swiss delivery month closes. Preserve first observation, revisions, publication
and finalization evidence. A complete exact common hourly grid is required;
repeated quarter-hour transport is not native QH truth. Register before scores
the truth provider/series, settlement calendar and finalization contract. Missing
independent custodian/finalization authority blocks scientific admission, not
local formula tests. Freeze thresholds from each candidate's pre-origin closed
history. Report full error, within-month shape error and monthly level error
separately, with ramp/tail/regime diagnostics. Do not use vendor forecasts as truth.

The registry verifies local hash continuity, chronology and duplicate-day rules;
it is not an authenticated external ledger, signer or production admission path.
All six authorities remain false. No system scheduled task, background daemon,
share default or autonomous Warehouse start is installed by this pilot. The
daily receipt command is explicit and local; source capture remains governed.
GPU and Warehouse use are user-authorized when useful. The current lane uses
CPU and, if necessary, an already-active Warehouse with bounded query duration;
manual stop403 is not retried and existing auto-stop is recorded.
