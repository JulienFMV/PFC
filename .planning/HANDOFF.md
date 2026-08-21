# Current handoff

Read in this order:

1. `AGENTS.md`
2. `.planning/HANDOFF.md`
3. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
4. `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260821-REPO-HYGIENE-AND-DATABRICKS-LAYER-BASELINE.md`

## Current state

- The repository baseline is aligned with the audited EEX, ENTSO-E and LSEG
  source layers.
- ENTSO-E Gold is the current-serving layer; Silver vintages are the PIT
  authority. The Gold resource bridge is optional enrichment.
- LSEG curve `110181967` is an independent benchmark only.
- Local datasets, model weights, copied research PDFs and runtime outputs are
  outside Git. Deterministic test fixtures and governed evidence remain in
  Git.
- Generated caches and output directories were cleaned on 2026-08-21.

## Invariants

- The CH monthly BASE solver is the sole monthly-level authority.
- ENTSO-E, neighboring markets, history, weather, Swissgrid, AFRY and LSEG may
  shape or benchmark only; they cannot rewrite monthly solver means.
- LT code must not import `pfc_shaping.ct.*`.
- T057 remains sealed.
- Model admission remains
  `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS` until independently
  governed local exports and a new future holdout exist.

See durable decisions D-20260821-248 through D-20260821-250 and the linked
session handoff for exact files, cleanup counts, tests and residual risks.
