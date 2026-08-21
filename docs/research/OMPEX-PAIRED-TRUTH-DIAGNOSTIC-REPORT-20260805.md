# OMPEX paired-truth diagnostic: methodological correction

## Technical summary

The historical `compare_hpfc_ompex_benchmark.py` workflow was not suitable for
proving that the FMV PFC outperforms OMPEX. It selected the timestamp shift
that minimized the direct distance between the two forecasts and did not
require realised truth. That is outcome-seeking alignment and measures
agreement with OMPEX, not forecast accuracy.

The entry point now requires three exact, local, SHA-256-bound inputs:
candidate PFC, one frozen OMPEX vintage and realised truth. Both forecasts are
evaluated on the same complete hourly UTC grid against the same truth. Inner
joins, missing-row deletion, duplicate aggregation and automatic alignment are
forbidden. The current result is deliberately
`DESCRIPTIVE_ONLY_NO_SUPERIORITY_DECISION`.

No real price outcome was opened in this batch. Tests use repo-local synthetic
fixtures only. Therefore this correction improves the validity of the future
comparison but supplies no evidence yet that either curve is more accurate.

## The old comparison could reward an arbitrary timestamp shift

| Property | Historical entry point | Corrected entry point |
|---|---|---|
| Accuracy reference | OMPEX itself | Same realised truth for both forecasts |
| Timestamp alignment | Best of three shifts by lowest error | Fixed UTC delivery-start semantics only |
| Missing observations | Inner join and row deletion | Exact complete common grid or failure |
| Duplicate timestamps | Averaged | Rejected |
| Input identity | Paths only | Local path plus exact SHA-256 |
| OMPEX access | Could read a live external path | Local frozen file only; no live `H:` dependency |
| Output claim | Advisory distance | Descriptive paired forecast errors only |
| Superiority decision | Not statistically valid | Explicitly unavailable |

This correction follows the central forecast-evaluation principle that
competing forecasts must be compared through paired losses against observed
outcomes, rather than by their distance from each other. Diebold and Mariano
formalize paired predictive-accuracy comparison, while Giacomini and White
extend the framing to out-of-sample and conditionally varying predictive
ability. [Diebold and Mariano (1995)](https://doi.org/10.3386/t0169),
[Giacomini and White (2006)](https://doi.org/10.1111/j.1468-0262.2006.00718.x)

## Scope and metric definitions

The cohort is one explicitly bounded half-open delivery interval
`[target_start_utc, target_end_utc)` at native hourly resolution. Every hour
must exist exactly once in candidate, OMPEX and truth. Prices are EUR/MWh.

The primary descriptive quantity is:

`MAE delta = MAE(candidate, truth) - MAE(OMPEX, truth)`

A negative value means lower candidate MAE on that exact window. It is not a
superiority decision. The diagnostic also computes RMSE, bias, median, p95 and
maximum absolute errors for both forecasts, ramp-error MAE, and paired
subgroups for Swiss delivery block, season, realised-price sign and tails,
forecast horizon and DST-adjacent hours.

The subgroup rows are overlapping diagnostics, not a partition and not
independent tests. Empty groups remain `UNSUPPORTED_EMPTY`; they are never
silently removed from the interpretation. Hourly OMPEX values are not counted
as four independent quarter-hours.

## Method and implementation boundary

`pfc_shaping.validation.ompex_truth_comparison` creates an exact expected UTC
hourly index from explicit target bounds and requires each input to match it.
It computes paired losses only after all completeness, finiteness, uniqueness,
ordering and post-origin checks pass.

The script captures each local input as bytes, checks stable file identity
before and after the read, verifies the caller-supplied SHA-256 and parses only
those captured bytes. OMPEX uses the strict OOXML parser from D218. Outputs are
new files below `build/`:

- `paired_hourly_against_truth.csv`;
- `subgroup_metrics.csv`;
- `benchmark_metrics.json`.

The machine contract is
`OMPEX-PAIRED-TRUTH-DIAGNOSTIC-CONTRACT-V1.json`. It keeps model selection,
promotion and production authority false and records zero countable origins.

## Limitations and robustness checks

The current tool is a descriptive primitive, not the final scientific
scorecard. It does not authenticate that the candidate was frozen before
OMPEX access, that the selected OMPEX file was available at the claimed
origin, or that the truth source is independent and was published only after
delivery. OMPEX hour-ending semantics also remain unauthenticated.

Even with those receipts, one vintage is not enough. Hourly losses within a
forecast origin are dependent, long-horizon windows overlap, and checking many
subgroups inflates false discoveries. The final design therefore needs several
independently frozen origins, dependence-aware inference and family-wise
multiplicity control. Romano and Wolf provide a stepwise framework that
accounts for joint dependence; Lago et al. emphasize long, rigorous,
multi-period electricity-price forecast evaluation against meaningful
baselines. [Romano and Wolf (2005)](https://doi.org/10.1111/j.1468-0262.2005.00615.x),
[Lago et al. (2021)](https://doi.org/10.1016/j.apenergy.2021.116983)

Adversarial tests prove that a missing truth hour fails instead of improving
one forecast's denominator, a wrong input hash fails before parsing, a target
starting before the origin fails, the CLI has no automatic alignment option,
and a numerically negative MAE delta still cannot authorize superiority.

No quantitative visual is included because no real comparison outcome was
opened; a chart over synthetic prices would be misleading. The standardized
interactive HTML report surface is also unavailable on this managed laptop
because the project contract forbids launching browser/Playwright runtimes.
This technical Markdown report is the governed local documentation surface.

## Recommended next steps

1. Obtain desk or vendor confirmation of OMPEX filename availability and
   hour-ending semantics.
2. Freeze a new candidate and its full input/configuration manifest before
   opening the corresponding OMPEX vintage.
3. Register independent realised truth after delivery with its publication
   receipt and native hourly/15-minute regime.
4. Approve numeric primary-superiority and protected-subgroup
   non-inferiority margins before outcomes are opened.
5. Extend the scorer across multiple origins using origin-aware dependence
   handling and preregistered family-wise correction.
6. Keep the monthly solver integrity gate separate and hard; forecast accuracy
   cannot compensate for a rewritten EEX monthly mean.

## Further questions

- Which desk system can attest the first availability time of each OMPEX
  vintage independently of its filename?
- Which governed source will provide the final Swiss realised price truth for
  the hourly and native 15-minute market regimes?
- What minimum economically material improvement and subgroup
  non-inferiority margins should FMV approve before the future holdout opens?

