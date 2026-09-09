# Session handoff - ENTSO-E PFC data coverage audit

Date: 2026-08-06  
Decision: `D-20260806-252`  
Status: `PARTIAL_DEV_MACRO_COVERAGE_EXACT_CURRENT_INVENTORY_UNPROVEN_NO_MODEL_AUTHORITY`

## Outcome

D252 answers the user's question without querying Databricks: all ENTSO-E data
types needed by the CH LT PFC are not yet proven in `dev.gold`. The real three
table schemas are known and the 2026-08-03 audit observed 11 macro-signals, but
the exact current `GroupName` / `FieldName` inventory remains uncaptured because
D247 stopped before SQL when the Warehouse was stopped.

The main specification correction is coupled-zone coverage. The first baseline
must include day-ahead prices for logical zones CH, DE_LU, FR, IT_NORTH and AT.
Neighbouring load/forecasts, generation/forecasts, renewable horizons and
regional Alpine reservoir storage are high-value evidence for a superior Swiss
hourly shape. Exact EIC identifiers and configuration-validity windows must be
owner mapped; logical labels alone are not source authority.

The official ENTSO-E review also adds or makes explicit nine useful families:
intraday prices, energy-storage actual/capacity, FCR capacity/prices,
transmission-unavailability impact on net positions, long-term flow-based
parameters, short-term adequacy forecasts, consumption-unit unavailability,
bidding-zone/EIC history, and balancing elastic-demand/product-selection
detail. Only bidding-zone/EIC history is a new baseline metadata blocker. The
others remain non-blocking candidate features.

The data-quality skill materially shaped the verdict: table existence,
macro-family discovery, exact dimension mapping, fact quality and real PIT are
separate evidence levels. The report skill produced an answer-first technical
report with explicit sources, limitations and next actions. The visualization
skill forced the checklist-count chart to state that its categories overlap and
must not be read as a coverage percentage.

## Cost and authority

D252 made zero Databricks connections or statements, zero Warehouse starts,
zero Databricks writes, opened zero real ENTSO-E rows and accessed no `H:` path.
The public ENTSO-E Manual of Procedures, EDI Library and MoP v3.5 consultation
were reviewed over the web; those documentation reads have no Databricks cost.

Current exact inventory, all baseline groups, coupled-zone coverage, additional
families, PIT, training, selection, model input, candidate assembly, promotion
and production authorities remain false.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-PFC-DATA-COVERAGE-AUDIT-V1.json`
- `pfc_shaping/validation/entsoe_pfc_data_coverage.py`
- `tests/test_entsoe_pfc_data_coverage.py`
- `docs/research/ENTSOE-DEV-PFC-DATA-COVERAGE-REPORT-20260806.md`
- `docs/research/entsoe-dev-pfc-data-coverage-report-20260806/artifact.json`
- `docs/research/entsoe-dev-pfc-data-coverage-report-20260806/report.html`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY, OMPEX, T057, heavy desk data or `H:` content was opened
or changed. Existing unrelated and concurrent changes were preserved.

## Canonical identities

- audit raw SHA-256 / canonical content ID:
  `a6697680dda8164f9dc7e96c65c79ee637ab483fe70cf1587acb10b80f6d3cd7` /
  `a032c630091b491adc2925b7432ed6a916080f10f21ba631975291b7f8b1470c`
- validator SHA-256:
  `23261b557234db509c46b8ae305908a9deb6cb695f0bb88bf15a0deeacccbeac`
- tests SHA-256:
  `c793883c4159de30093452416f3f9b959c6fb720c13bd6b45ff83591d1595e83`
- Markdown report SHA-256:
  `8d13d9f7a037a9087174817a919d4a65260c808b991f10c100367d7bd69423b0`
- canonical HTML artifact source SHA-256:
  `7dd5e8aa069eaea67c5db461ecc95d0830a76dfbe8e7be62e3a0297eb88672f5`
- generated self-contained HTML SHA-256:
  `2802ec79d5f3dfc7ebf88d25cc6da777ad135fdf3fe3d020c23fc5e52ad67b81`

Bound local evidence:

- D247 family status SHA-256:
  `c5232db312cd89d469b714b66d63cb0b6e6960ff53c00ec8c2f7ac4a2ca60c10`
- D251 exact-mapping handoff SHA-256:
  `e331e746ec78f837811a46306fdc15e9520486147763895893abeeae252c96e2`
- D243 inventory contract SHA-256:
  `ba8f6945b4a43b54762fa475228edabea4018bfea5871f2581dc53536c0743a1`
- D250 real-mapping contract SHA-256:
  `4a0d69a770fe0b45063bc4e5b4be4b9aca1e86c546b40d2bf8816a001cd06b8f`
- D241 schema proof content ID / manifest SHA-256:
  `d6c006609d881b51f08be6d60e01f68b59a40be8bdf2898ef0a98491f5771544` /
  `1835f93a517e9c6769079984a376fb9879b22d7c8ce2922aa42b0dd646627ada`

## Findings

Observed at macro level on 2026-08-03:

- actual load;
- day-ahead prices;
- actual and forecast generation;
- renewable forecasts without exact DA/ID split;
- hydro reservoir storage;
- physical flows and scheduled exchanges;
- day-, month- and year-ahead NTC.

Still unproven for the baseline:

- distinct load forecast;
- renewable DA versus intraday horizon;
- six generation technologies;
- all four CH borders and directions;
- native resolution and historical regime;
- canonical EUR/MWh metadata;
- sign, lineage, quality and revision semantics;
- owner-approved PIT rule.

Coupled-zone requirements now explicit:

- day-ahead prices: CH, DE_LU, FR, IT_NORTH, AT — baseline required;
- actual/forecast load and generation plus renewable horizons for all five
  logical zones — high value for superiority;
- reservoir storage for CH, AT, FR and IT_NORTH — high value.

## Roast and validation

The first focused run correctly failed `2/10` tests because D247 had been
updated concurrently after its first hash was captured. D252 was rebound to the
current D247 bytes, the audit identities were regenerated, and the final run is
green.

- Ruff: passed.
- focused D252: `10 passed in 0.11s`.
- adjacent D243/D250/D251/D252: `56 passed in 0.38s`.
- portable artifact validation: passed.
- structural HTML verification: `13` blocks, `1` chart, `2` tables, title
  exact, `ok=true`.
- all three local SQLite queries exactly reproduce their bounded snapshot:
  chart `5` rows, baseline table `10` rows, new-family table `9` rows.
- browser verification was intentionally not launched because AGENTS.md
  forbids browser runtimes on this managed workstation. The HTML is therefore
  structurally verified, not browser-E2E qualified.

## Report artifacts

- readable Markdown:
  `docs/research/ENTSOE-DEV-PFC-DATA-COVERAGE-REPORT-20260806.md`
- canonical report payload:
  `docs/research/entsoe-dev-pfc-data-coverage-report-20260806/artifact.json`
- self-contained HTML:
  `docs/research/entsoe-dev-pfc-data-coverage-report-20260806/report.html`

The chart shows counts of overlapping checklist classes. It is explicitly not
a coverage ratio and its bars must not be summed.

## Next permitted action

Do not query Databricks merely to continue D252. On a future allowed
Europe/Zurich day, run D243 at most once and only when the Warehouse is already
running. Capture only the bounded dimension inventory, then obtain exact D251
owner mapping for family, zone/EIC validity, product, horizon, technology,
direction, unit, resolution and sign. D250 cadence/PIT/lineage/quality/revision
gaps must close before any real local value export can be admitted.

The monthly solver remains sole level authority. All ENTSO-E fundamentals are
zero-mean shape evidence only. LT remains independent from CT, T057 stays
sealed, OMPEX remains post-freeze benchmark-only and AFRY descriptive only.
