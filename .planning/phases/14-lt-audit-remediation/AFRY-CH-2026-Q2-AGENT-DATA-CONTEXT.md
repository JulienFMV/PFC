# AFRY CH 2026 Q2 restricted scenario data context

Status: local quality evidence only; model input `NO_GO`; production `NO_GO`.

## Purpose

This context tells Codex, local agents and human reviewers how the restricted
AFRY Switzerland 2026 Q2 delivery may be inspected without turning vendor
scenario values into an ungoverned knowledge base or an LT level authority.
The controlled interface is a content-addressed structured catalog, not RAG,
a copied wiki, or committed extracts.

## Registered evidence

The source registration is
`AFRY-CH-2026-Q2-SOURCE-REGISTRATION.json`; its separate semantic contract is
`AFRY-CH-2026-Q2-SEMANTIC-CONTRACT.json`. Both are safe to commit because they
contain schemas, authorities, evidence locations and hashes, but no AFRY
numeric scenario values.

Restricted source bytes remain outside Git:

- annual scenario workbook, SHA-256
  `460c5030b1f2c1b0e281dfd93fa5ef120a0b19c62af8a39d066313b7c7f926ea`;
- hourly price workbook, SHA-256
  `903850cff94738eba2845366b99423bef87c9c59fe122ae82de681d0199318f8`;
- Commodity and Modelling Annex, SHA-256
  `9eaa00233074fac38efe9f9d048f0f9524d833a0ef9313d5acc50773a4acfbe1`;
- Switzerland Quarterly Update Note, SHA-256
  `dc0c0c2c06ddf95d523c418e7ae4899c0aed1ed0f8940566992e4008596769e8`.

The PDF copies and extracted text are restricted build evidence. They must not
be copied into `docs/`, committed, embedded, or sent to an external RAG/vector
service. Their hashes are required supporting evidence for materialization.

## Why the two reports are material

The Quarterly Update Note resolves scenario meaning that column labels alone
cannot provide. In particular, `High` is principally a price/commodity upside
case and shares the Central demand and capacity build-out assumptions. It is
therefore not an FMV “fast transition”, “high Swiss renewables”, legal-target
compliance case, or probability proxy. The note also explains the negative
price treatment, storage/flexibility effects and the use of several weather
patterns.

The Commodity and Modelling Annex describes the BID3 dispatch and investment
framework, hourly chronology, hydro/storage representation and weather input.
It makes the hourly data useful as an externally modelled benchmark, while
also exposing model risk: bidder behaviour, economic build/retire assumptions
and a representative calendar remain vendor-model outputs rather than
observed Swiss truth.

## Authority matrix

| Use | Status | Constraint |
|---|---|---|
| Source discovery and data-quality audit | Allowed | Hash-verified local catalog only |
| Structural scenario comparison | Allowed as benchmark | Preserve vendor scenario label and release provenance |
| Hour-slot shape comparison | Allowed as benchmark/teacher candidate | Additive zero-mean shape only; no level rewrite |
| Scenario probability | Forbidden | Vendor scenarios are not probabilities |
| FMV scenario mapping | `NO_GO` | Requires a separately approved, versioned mapping |
| Monthly LT level | Forbidden | The CH monthly solver remains the sole level authority |
| Timestamp, DST, leap-year or 15-minute inference | Forbidden | Requires a separately governed calendar mapping |
| Training or production model input | `NO_GO` | Rolling-origin, future holdout and policy gates remain open |
| RAG/vector indexing of vendor values | Forbidden | Restricted licence and value-leakage risk |
| Production promotion | `NO_GO` | No catalog artifact is production authority |

## Structured access for agents

The reproducible local bundle is:

`build/afry-catalog-final-v6-a-20260803/a7ca9238ad715e67269c73a862607c3a3301358821818bc5d2fa4f1581c54454`

An independently rebuilt byte-identical copy exists below
`build/afry-catalog-final-v6-b-20260803/` with the same content ID. These are
ephemeral local build artifacts, not Git content or an external publication.

Use `catalog.json` first and require all relevant gates. Read only the
necessary Parquet columns with PyArrow and retain filters in analytical
receipts. Example:

```python
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pfc_shaping.data.afry_scenarios import read_verified_artifact_bytes

bundle = Path("build/afry-catalog-final-v6-a-20260803") / "a7ca9238ad715e67269c73a862607c3a3301358821818bc5d2fa4f1581c54454"
catalog, artifact_bytes = read_verified_artifact_bytes(
    bundle, "hour_slot_shape_benchmark.parquet"
)
assert catalog["integrity_verdict"] == "PASS_LOCAL_INTEGRITY_ONLY"
assert catalog["decision_state"] == "NO_GO_MODEL_AND_PRODUCTION"
assert catalog["admission"]["model_input_authorized"] is False
shape = pq.read_table(
    pa.BufferReader(artifact_bytes),
    filters=[("scenario_vendor", "=", "Central")],
)
```

The catalog contains the restricted numeric artifacts. Agents must summarize
conclusions and provenance, not reproduce vendor tables into prompts,
documentation, logs, or Git. DuckDB is optional only in a separately governed
runtime; PyArrow is the current supported local interface.

## Calendar-free shape diagnostics

The current derived diagnostic bundle is:

`build/afry-shape-diagnostics-final-v3-a/8ef0290502359b6e1e16093bcf34e6400d8e74dad23f9b1baa091e0a81a372f6`

An independently rebuilt byte-identical copy exists below
`build/afry-shape-diagnostics-final-v3-b/` with the same content ID. Verify it
with an explicit caller anchor to one of the canonical v6 source catalogs:

```python
from pathlib import Path

from pfc_shaping.validation.afry_shape_diagnostics import (
    verify_shape_diagnostic_bundle,
)

source = Path("build/afry-catalog-final-v6-a-20260803") / "a7ca9238ad715e67269c73a862607c3a3301358821818bc5d2fa4f1581c54454"
diagnostic = Path("build/afry-shape-diagnostics-final-v3-a") / "8ef0290502359b6e1e16093bcf34e6400d8e74dad23f9b1baa091e0a81a372f6"
manifest = verify_shape_diagnostic_bundle(
    diagnostic,
    source_catalog_bundle=source,
)
assert manifest["status"] == "PASS_LOCAL_DIAGNOSTIC_ONLY"
assert manifest["decision_state"] == "NO_GO_MODEL_AND_PRODUCTION"
assert manifest["admission"]["model_input_authorized"] is False
```

The verifier is side-effect-free. It binds the canonical workspace, Git root,
contract, three producer files and the complete repo-local runtime tree; it
rejects reparse ancestry, requires a disjoint source/output layout, and
recomputes all three Parquet outputs plus the audit in memory from the two
hash-bound source artifacts. The diagnostic uses ordered representative-slot
blocks only. Those blocks are not Swiss clock hours.

Across all four vendor scenarios, the restricted descriptive output exhibits
both a deepening-then-recompression pattern and non-monotone negative-price
share across available delivery years for every weather pattern. Weather
dispersion is non-zero throughout. These are vendor-output diagnostics, not
observed effects, causal evidence, calibrated probabilities or empirical
validation. EEX and ENTSO-E Databricks access remains required before any
rolling-origin/model decision; no legacy or synthetic substitute is allowed.

## Time and shape semantics

The hourly workbook contains four vendor scenarios, several delivery years
and five weather years. Each scenario/delivery/weather group has exactly
8,760 ordered representative-hour slots. Those slots are not admitted UTC or
Europe/Zurich timestamps and must not be used to invent DST, leap-day,
calendar-month or quarter-hour mappings.

`hour_slot_shape_benchmark.parquet` is an annual hour-slot diagnostic. Its
additive component is zero-mean within every scenario/delivery/weather group.
This preserves the monthly solver principle conceptually but does not by
itself prove monthly neutrality: the missing vendor calendar mapping prevents
safe monthly application. Negative-price behaviour must be assessed in
absolute price space, not hidden by positive multiplicative factors.

## Swiss policy interpretation

The Federal Act target for 2035 is a normative Swiss policy target. AFRY
scenarios are descriptive market-model cases. The registered policy audit is
therefore deliberately `NO_GO`: the apparent difference cannot be called
“purely political” until scope is reconciled, including gross versus net
generation, behind-the-meter self-consumption, curtailment and the renewable
fractions of aggregate thermal/waste categories. `High` must not be used as
an automatic policy-compliance proxy.

## Annual July refresh protocol

For each yearly delivery:

1. create a new release registration; never overwrite the prior release;
2. record first observation time, publisher dates, exact hashes and licence;
3. require the matching QUN and modelling annex before semantic admission;
4. capture each workbook once into immutable process memory, hash those exact
   bytes, inspect the captured container, parse only that capture, and rehash
   the same bytes after parsing;
5. fail closed on macros, formulas, active content, schema, style, unit,
   scenario, calendar, continuity or reconciliation drift;
6. materialize twice into separate roots and compare every artifact byte;
7. re-run Security, IT/Operations and Quant/Data review;
8. keep model and production authority false until a distinct governed
   promotion decision supplies all missing evidence.

## Evidence still required before shaping use

- an authoritative mapping from representative hour slots to the delivery
  calendar, including leap years and Swiss DST;
- definitions reconciling vendor production, demand, storage and flow fields;
- a governed FMV scenario mapping distinct from vendor scenario labels;
- rolling-origin comparison against simple CH seasonal and history-based
  shape baselines, with capture-price and tail metrics;
- a new independently frozen future holdout; T057 remains sealed and cannot
  be reused;
- sensitivity and stability evidence across vendor releases and weather
  years;
- approved licences, retention, access control and operational ownership.
- startup reconciliation/quarantine of orphan staging, concurrent-publication
  fault injection and durable structured failure receipts for monitoring.

Until those items close, AFRY improves challenge, diagnostics and candidate
feature design, but it does not change the PFC, the monthly solver, or any
production decision.
