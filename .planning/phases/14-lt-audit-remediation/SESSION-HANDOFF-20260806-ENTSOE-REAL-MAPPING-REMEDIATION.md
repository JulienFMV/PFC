# Session handoff - ENTSO-E real mapping remediation

Date: 2026-08-06
Decision: D-20260806-250
Status: `FAIL_REAL_MAPPING_REMEDIATION_REQUIRED_NO_MODEL_AUTHORITY`

## Outcome

D250 converts the D241 real Unity Catalog schema finding into an exact,
fail-closed normalization contract without querying table values. It closes a
material temporal risk: both real fact-table comments define `DateTimeUtc` as
the UTC right edge of the ENTSO-E interval, while the PFC normalized interface
uses interval starts. Direct aliasing is forbidden; the required mapping is
`target_ts_utc = DateTimeUtc - native_resolution`.

The cadence is not guessed. `native_resolution` must be supplied per
`SeriesID`; hourly history is not treated as defective merely because some
later products use 15-minute delivery periods.

## PIT and lineage conclusion

`PublicationTimestampUtc`, `PullTimestampUtc` and `Meta_Load_Timestamp` are
preserved separately. A conservative candidate
`as_of_utc=max(publication,pull,load)` is represented only when all three are
non-null, but remains inactive until the responsible data owner approves the
semantics. `latest` is forbidden as historical point-in-time evidence.

The contract also forbids:

- `DocumentType` as source document ID;
- `VintageID` as source revision number;
- default quality `OK`, default revision zero or inferred sign;
- silent timestamp coalescing, null dropping or cadence inference.

## Canonical evidence

- contract raw SHA-256:
  `4a0d69a770fe0b45063bc4e5b4be4b9aca1e86c546b40d2bf8816a001cd06b8f`;
- contract canonical content ID:
  `56826ff4d97074a58f961e08f4c0fe2769431ad9ab3d1afebd5d032e635d8c7b`;
- validator SHA-256:
  `406309d4315f4b0cf20a985152201b7f71fbc2caf4dee24989e728e14468cad0`;
- tests SHA-256:
  `66ffecc78e4c44594616f90edb6756d5f0f7a4417b508d53b467b0316806663a`;
- materializer SHA-256:
  `f979a91638fadeeae24e2366e2af7fe98367b65331a44433b60a3a818eeefa7d`;
- deterministic proof/content ID:
  `5269f46ac8bc078eacbd57e41ab6a447f9153ce9662880e99c846bc3774d3576`;
- proof manifest SHA-256:
  `b1e49ebba72da7d3cf0a40ce5b0764ceda32ecfe813b2cc67ed5af057568fd8a`;
- proof path:
  `build/databricks-eex-daily/2026-08-06/entsoe-real-mapping-remediation-proofs/5269f46ac8bc078eacbd57e41ab6a447f9153ce9662880e99c846bc3774d3576/`.

The proof binds D241 proof
`d6c006609d881b51f08be6d60e01f68b59a40be8bdf2898ef0a98491f5771544`
and its exact captured schema, assessment and validator hashes.

## Qualification

- focused logic: `14 passed` in 0.13-0.17 seconds on each final identity;
- adjacent D234/D239/D241/D250 semantic matrix before final identity rename:
  `88 passed`;
- copied repo-local Ruff 0.15.12 SHA-256
  `ccfbe6e11d75c3c2b6b419adf1fd018de519055543d28d261caad3cf78335754`:
  selected validator, tests and materializer pass;
- latest supervised run:
  `build/workspace-local-runs/d250final/execution-receipt.json`;
- latest target result: exit 0 and `14 passed in 0.13s`;
- latest supervisor result: `TARGET_EVIDENCE_INVALID`, because
  `entsoe_series_family_mapping.py` was modified by a concurrent session
  during the run. The same concurrency condition invalidated the prior final
  receipts. These are functional assertion results, not authoritative stable
  execution receipts.

No repeated retry is recommended while the concurrent ENTSO-E chain is active.

## Cost and authority receipt

D250 performed zero Databricks connections/statements, zero control-plane GETs,
zero Warehouse starts, zero opened table-value rows, zero network calls, zero
`H:` accesses, zero remote writes and zero Databricks writes. The D247 daily
reservation remains consumed; no same-day Databricks retry is allowed.

Real schema observation is true. Real-value, PIT, quality, model, candidate,
promotion and production authority all remain false. The monthly solver remains
the sole monthly-level authority, T057 remains sealed, AFRY remains descriptive
only and OMPEX remains post-freeze benchmark only.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-REAL-MAPPING-REMEDIATION-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_real_mapping_remediation.py`
- `tests/test_entsoe_real_mapping_remediation.py`
- `build/databricks-eex-daily/materialize_entsoe_real_mapping_remediation.py`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-REAL-MAPPING-DATA-ENGINEER-REQUEST-20260806.md`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-GAPS-20260805.md`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff.

## Next safe actions

1. Have the data owner approve/correct the timestamp, PIT and lineage rules in
   the short data-engineer request.
2. On a later reserved Europe/Zurich day, run the bounded D243 dimension
   inventory only if the Warehouse is already `RUNNING`; do not start it.
3. After a governed immutable export exists, pass D240/D244 integrity and D239/
   D245 quality gates, then freeze a new independent future holdout.
4. Only then resume empirical rolling-origin selection and construct a
   deterministic research PFC candidate. Calibrated probabilistic intervals
   require their own independent receipt.

Predecessor:
`SESSION-HANDOFF-20260806-ENTSOE-DEV-FAMILY-COVERAGE-STOPPED.md`.
