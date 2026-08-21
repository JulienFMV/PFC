# OMPEX archive readiness for an independent PFC benchmark

## Technical summary

The complete current OMPEX archive is byte-inventoried and structurally
consistent enough to support occasional, frozen benchmark refreshes. It is not
yet valid rolling-origin evidence.

- Two independent read-only scans of all 353 workbooks produced the same
  content ID:
  `336700af0b38324bbfc99c5332b5f360a01e00f2fd14baab090ebcb8e087a57a`.
- The archive contains 351 dated curves and two non-curve workbooks. All 351
  curves have the same sheet/header/container signature, literal formula-free
  content and no macro, external-link, connection or embedded-object member.
- The dated archive covers 351 of 400 calendar dates, leaving 49 missing
  origins (12.25%). A single early gap accounts for 47 consecutive days; two
  later dates are individually absent.
- The filenames expose 351 candidate origins, but zero origins are currently
  countable for scientific comparison. Neither the filename clock nor actual
  desk availability has been authenticated, and historical curve values can be
  revised between vintages.
- `H:` is an exceptional, explicit, read-only benchmark source. Routine code,
  scoring, manifests, tests, reports and mutable artifacts run under
  `C:\Users\jbattaglia\PFC_LT`. Scoring must consume a locally frozen vintage
  or local hash-bound evidence, never a live `H:` path.

The archive therefore passes current-byte inventory quality and remains
`NO_GO` for rolling-origin claims, model selection or production promotion.

## The archive is structurally uniform but temporally incomplete

The intended unit is one frozen OMPEX curve per filename date. Every dated
curve has one `HFC` sheet, the exact `Date` and `EUR/MWh` headers and one of two
expected delivery-row counts. The row-count change is a horizon extension, not
a column-schema change.

| Curve regime | Dated workbooks | Delivery rows per workbook | Interpretation |
|---|---:|---:|---|
| Earlier horizon | 136 | 43,824 | Five complete delivery years |
| Extended horizon | 215 | 52,584 | Six complete delivery years |
| Total | 351 | — | One common curve schema signature |

No two complete workbook files have the same SHA-256. This does not prove that
every hourly value changed: workbook metadata can differ even when some curve
segments are identical. Value-level revision analysis must use the exact
post-freeze benchmark protocol, not workbook hashes as an error metric.

## Three gaps prevent a nominal daily origin panel

| Missing run | Missing days | Analytical consequence |
|---|---:|---|
| 10 July–25 August 2025 | 47 | No daily benchmark origins across most of this interval |
| 6 January 2026 | 1 | Isolated origin absent |
| 9 February 2026 | 1 | Isolated origin absent |

The missing rate is 49 / 400 = 12.25%. The 47-day interruption dominates the
coverage loss and must not be imputed or reconstructed from a later workbook.
The two isolated dates may have operational explanations, but no explanation is
present in the archive itself.

## Filename clocks are consistent labels, not availability proof

Of the 351 dated curves, 309 use `10:17:00`, 35 use `10:17:01`, four use
`10:17:02`, and three are isolated clock variants. This regularity is useful
for identifying the apparent publication process, but it does not establish
when a forecaster could access a file.

The earlier sample audit also proved that common historical delivery rows can
change between adjacent vintages, including rows before the later filename
timestamp. Consequently:

- the latest workbook cannot recreate an earlier information set;
- a filename timestamp cannot be treated as an authenticated `as_of` field;
- every comparison requires the exact frozen workbook hash and an independently
  authenticated availability time;
- pre-origin delivery rows must be excluded from forecast scoring.

## Quality findings and impact

| Severity | Finding | Evidence | Impact |
|---|---|---|---|
| Critical | Availability at origin is unauthenticated | 351 filename candidates, zero countable scientific origins | Rolling-origin superiority cannot yet be claimed |
| Critical | Latest-vintage time travel is possible | Historical revisions observed across adjacent vintages | A latest-file backtest can leak future revisions |
| High | Filename-date coverage is incomplete | 49 missing dates, 12.25%; maximum run 47 days | A daily panel has material gaps and non-random coverage |
| Pass | Current bytes and curve schema are stable | 353 hashes replayed twice; one curve schema; zero curve formulas/active content | Exact frozen benchmark refreshes are technically feasible |

These findings concern benchmark fitness. They do not say whether OMPEX prices
are economically good or bad and do not expose any price value or statistic.

## Scope, definitions and comparison basis

- **Archive member:** one regular `.xlsx` file below the benchmark root,
  including the nested template directory.
- **Curve vintage:** a root workbook named
  `HFC_Ompex_YYYYMMDD_HHMMSS.xlsx` with the exact `HFC` curve schema.
- **Candidate origin:** the date/time encoded in a curve filename. This is a
  label only until availability is authenticated.
- **Countable scientific origin:** a candidate origin whose exact candidate,
  OMPEX vintage, information set and availability are frozen before realised
  truth is consumed. Current count: zero.
- **Completeness denominator:** every calendar date from 2 July 2025 through
  5 August 2026 inclusive, or 400 dates.
- **Native cadence:** hourly for the observed OMPEX curves. Stepwise expansion
  to four rows is transport only and never native 15-minute evidence.

OMPEX remains an imperfect external prediction benchmark. Candidate and OMPEX
must both be scored against the same independent realised truth; distance to
OMPEX is not a quality metric.

## Methodology and reproducibility

Each scan:

1. recursively enumerated regular `.xlsx` files and rejected links, reparse
   points, unsafe depth and non-XLSX members;
2. captured every file with descriptor-level before/after identity checks;
3. hashed the exact bytes and inspected bounded OOXML structure;
4. checked all 351 curve sheets, headers, row counts, formula tags and active or
   linked content;
5. re-enumerated the directory tree and rechecked every source identity before
   publication;
6. emitted only filenames, hashes, filesystem metadata and structural counts —
   never prices or price statistics.

The two complete scans independently read 297,244,351 compressed source bytes
and emitted byte-identical `inventory.json`, `summary.json` and `manifest.json`
members. Runtime cost is accepted for this occasional refresh. Routine scoring
must use the local manifest/capture and must not repeat the full `H:` scan.

## Limitations and robustness

- The complete inventory verifies bytes and schema, not the economic quality of
  the curves.
- Native hourly and hour-ending semantics were fully parsed on six frozen
  representative vintages, including both DST transitions; the full 351-file
  inventory checks schema and row counts rather than reparsing every timestamp.
- Filesystem timestamps are not independent provider or desk attestations.
- Missing dates may be expected operational closures or true archive loss; no
  reason-coded lineage currently distinguishes them.
- The archive ends on the audit date and cannot provide a new independent
  future holdout by itself.
- No portable HTML report was generated: the managed-workstation contract
  forbids launching browser/Playwright runtimes. This Markdown report is the
  canonical local technical surface requested for `docs/`.

## Recommended next steps

1. Obtain a short desk/vendor statement defining the OMPEX filename clock,
   timestamp convention and actual availability semantics.
2. Ask whether the 47-day run and the two isolated dates are expected omissions
   and capture reason-coded lineage without filling them.
3. For each future comparison, freeze the PFC candidate first, then explicitly
   read one chosen OMPEX vintage from `H:` and capture/hash it under the
   canonical `C:` workspace.
4. Score only post-origin delivery intervals against independently governed
   truth and apply the preregistered superiority/non-inferiority contract.
5. Keep full `H:` inventory refreshes occasional. The absence of `H:` must not
   block routine PFC construction, validation or production work.

## Further questions

- Is the filename time the generation start, completion time, publication time
  or a scheduler label?
- Were the 49 absent dates expected, and can an immutable archive receipt prove
  that answer?
- Which exact realised-price source and native cadence will be the independent
  truth for each Swiss market regime?
- What FMV-approved effect-size and subgroup non-inferiority margins will be
  frozen before the first countable comparison?
