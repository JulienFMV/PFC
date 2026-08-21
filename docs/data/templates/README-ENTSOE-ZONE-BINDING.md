# ENTSO-E semantic and zone binding templates

These templates are intentionally incomplete and contain no real EIC code,
series identifier, signature, market value, Databricks credential or authority.

Use them only after the bounded D243 dimension inventory has been captured on
an allowed day:

1. Before discarding raw dimension fields, build the physical-series semantic
   binding from the exact `SeriesID`, `GroupName`, `FieldName`, `DocumentType`,
   `BusinessType`, `ProcessType`, `PsrType`, `FromZone`, `ToZone` and `Unit`.
   Every normalized base series must occur once, and the number of physical
   series per signature must equal the D243 inventory count.
2. Build the zone registry from a governed official EIC registry snapshot and
   accountable owner review. Add one entry per validity regime. Preserve old,
   deactivated and superseded entries; never overwrite history.
3. If `FromZone` or `ToZone` contains a label rather than an EIC, add a separate
   `OWNER_CANONICAL_LABEL` entry containing that exact raw value and binding it
   to the governed logical zone for an exact interval.
4. Build one or more zone bindings for each non-directional coupled-zone
   signature. A binding interval must fit entirely inside one registry entry;
   split it at every code or configuration change.
5. Keep every authority field false. Validator success proves structure and
   owner assertion only, not official membership, real fact coverage, PIT,
   model usefulness or production readiness.

Place completed evidence below a governed local `build/` intake directory, not
in Git. The PFC pipeline must never write it back to Databricks.
