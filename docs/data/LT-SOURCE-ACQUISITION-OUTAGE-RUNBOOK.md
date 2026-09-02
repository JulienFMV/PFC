# LT source acquisition during an ENTSO-E service incident

## Current disposition

The public ENTSO-E incident does not make historical materialized Silver rows
invalid, and it does not prove that those rows are available. It only blocks a
claim that a fresh source pull is currently complete. The local workstation
must not test that distinction by querying Databricks or starting compute.

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

Until then the snapshot is useful local evidence, not a model input.

### ENTSO-E

Use only a platform-owned delivery of rows already materialized before the
incident. The laptop-side cost preflight remains `STOP_NO_ACTIVE_WAREHOUSE`.
This plan neither asks for nor authorizes a Warehouse start.

The first request is deliberately a July 2026 `realized_final` smoke export.
It exercises the v2 adapter and the LSEG reconciliation on a complete Swiss
market month without pretending that the August backfill was known in July.
Before values are delivered, the producer or data owner must provide the
effective-dated AT and DE-LU series selections. The export must use the exact
v2 realized SQL hash recorded in the plan and carry value-bound finality
evidence.

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
