# LT source acquisition during an ENTSO-E service incident

## Current disposition

The public ENTSO-E incident does not make historical materialized Silver rows
invalid, and it does not prove that those rows are available. It only blocks a
claim that a fresh source pull is currently complete. The local workstation
must not test source recovery by querying Databricks or starting compute. A
later explicitly authorized, bounded construction comparison used an already
running Warehouse; it did not test or establish source recovery or freshness.

The machine-readable disposition is
`.planning/phases/14-lt-audit-remediation/LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json`.
It grants no execution, model, monthly-level, publication, production or
trading authority.

## Independent lanes

### EEX

Reuse the existing 5 August local capture. Its artifact and manifest hashes
match their recorded identities. Do not issue a replacement statement merely
because ENTSO-E is degraded, and do not switch to the unqualified direct EEX
API route.

The next admissible work is evidence completion around those exact bytes:

1. bind the exact three-table join SQL, predicates and physical table
   identities;
2. obtain independent source-time and signed-envelope evidence;
3. convert the causal daily snapshots into the existing signed EEX vintage
   catalogue rather than creating a second authority.

Local status on 2026-09-03: item 1 is complete under D-20260903-280. The
zero-query validator now binds the exact reviewed SQL bytes and SHA-256 to the
exact historical manifest, three ordered PRD tables, CH/POWER predicates,
12-column result schema and opaque artifact declaration. It does not open the
price artifact. Items 2 and 3 remain external/governed work and all model,
selection and production authorities remain false.

Until then the snapshot is useful local evidence, not a model input.

### ENTSO-E

Use only rows already materialized before the incident. On 2026-09-03 the PBI
SQL Warehouse was independently observed `RUNNING`; one explicitly authorized
bounded aggregate comparison ran without starting, resizing or creating
compute. This plan still authorizes no Warehouse start.

The first request is deliberately a July 2026 `realized_final` smoke export.
It exercises the v2 adapter and the LSEG reconciliation on a complete Swiss
market month without pretending that the August backfill was known in July.
Before values are delivered, the producer or data owner must provide the
effective-dated AT and DE-LU series selections. The export must use the exact
v2 realized SQL hash recorded in the plan and carry value-bound finality
evidence.

Local status on 2026-09-03: the metadata-only owner request is frozen at
`.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-EFFECTIVE-SERIES-SELECTION-REQUEST-V1-20260903.json`
with canonical JSON SHA-256
`6794566035e40b69eb5d104d09e317896b69502ca800513ef094ca46673d6cfb`.
It covers exactly the July Swiss-local window and the two admitted
classification candidates for each of AT and DE-LU. It requests no business
values. It has not been transmitted, no owner response has been received and
the request itself authorizes no selection. Defaults, averaging and consumer
inference remain forbidden. Each selected key must cover the whole request
window; an in-window classification change blocks this request and requires a
separately reviewed segmented-export plan.

The construction-only selection question was subsequently resolved without an
owner response. The deployed producer code proves that sequences 1 and 2 are
distinct A44 auction identities with no implicit default. A bounded comparison
against the independently configured LSEG EPEX day-ahead curves then found:

- AT sequence 1: exact equality for all 2,976 July quarter-hours; sequence 2
  MAE `11.299412 EUR/MWh`;
- DE-LU sequence 1: exact equality for all 2,976 July quarter-hours; sequence 2
  MAE `9.549943 EUR/MWh`.

Therefore the July construction smoke export uses
`day_ahead_prices||at_price||1` and
`day_ahead_prices||de_lu_price||1`. The aggregate query returned no raw price
rows, but it read 9,364,142,086 bytes; do not repeat it. Its evidence is frozen
in
`.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-EFFECTIVE-SERIES-SELECTION-EVIDENCE-V1-20260903.json`.
This parity selects the construction reference only. It does not prove source
publication time, realized finality, causal availability, model input or
production authority.

If the internal materialized Silver snapshot is unavailable, stop. Wait for
producer recovery; do not replace ENTSO-E with legacy local or synthetic data.
If the snapshot is available, the public outage still forbids any claim of
freshness beyond its documented watermark.

### Causal history and future holdout

The July rebuild is a backfill. It cannot become historical causal truth.
`causal_asof` resumes only with a genuinely prospective capture whose
availability is known at each frozen origin. Freeze the origin schedule and a
new independent future holdout before collecting that evidence or retraining
anything.

### LSEG

Do not query LSEG speculatively. First validate the exact ENTSO-E realized
frame, then request a matched CH/AT/DE-LU/FR extract and evidence for that same
window and assessment cutoff. IT-North remains ENTSO-E-only. Any discrepancy
blocks; it never licenses silent source substitution.

## Incident recovery check

On recovery, record the provider notice and first complete post-recovery
watermark separately from the historical materialized snapshot. A public
status page, successful HTTP response or absence of a banner does not by
itself prove source completeness, original publication time or finality.

All later products retain the existing authority boundary: the CH monthly
BASE solver remains the sole monthly-level authority, ENTSO-E/LSEG may only
provide realized truth, controls or zero-mean shape, LT remains independent
from CT, and T057 remains sealed.
