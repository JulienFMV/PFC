# Data documentation index

## Current contracts

- `DATABRICKS-LT-SNAPSHOT-INTAKE.md`: current mixed-layer Databricks export
  contract.
- `SHARED-DATA-PLATFORM.md`: consumer-neutral local storage and immutable view
  contract.
- `templates/`: schema/binding templates used by current validation.

## Historical replay only

Files and validators whose names contain `GOLD-SNAPSHOT-INTAKE`, D293,
`zero_query`, `metadata_admission` or `physical_mapping_compiler` describe the
pre-rebuild ENTSO-E design. They are retained only where a prior decision or
receipt must remain reproducible. They must not request or admit the retired
Gold ENTSO-E vintage fact.

Time-stamped files under `docs/research/` are observations made at their stated
date. They are evidence, not a current source contract.
