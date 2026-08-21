# Session handoff - ENTSO-E temporal zone configuration

Date: 2026-08-06  
Decision: D-20260806-254  
Status: offline zone-history gate ready; no real zone registry admitted

## Outcome

D252 established that day-ahead prices for `CH`, `DE_LU`, `FR`, `IT_NORTH`
and `AT` are baseline evidence for Swiss LT shape. D241 and D253 do not yet
bind source zone identifiers to these logical zones through time. D254 adds
that missing temporal metadata gate without opening any ENTSO-E value.

The validator requires, for every mapping:

- exact identifier scheme and source identifier;
- logical zone and domain kind;
- left-closed/right-open UTC validity window;
- active/deactivated/superseded status;
- explicit registry-lineage kind, document ID and HTTPS endpoint;
- owner-confirmed semantics.

EIC area identifiers must have the official fixed 16-character syntax and
object type `Y`. The validator rejects overlapping source mappings and
simultaneous EICs for the same logical bidding zone, reports every temporal gap
without fill, counts contiguous code changes and resolves an identifier only
at an exact instant. A registry published today may legitimately describe a
future configuration window.

Format is not official membership. The check digit, official registry entry,
external owner identity, current dev coverage, PIT, model and production
authorities remain false until independently governed evidence exists.

## Changed files

- `pfc_shaping/validation/entsoe_zone_configuration.py`
  - new temporal registry validator and exact point-in-time resolver;
  - SHA-256
    `a8bf37d373e0711e97daa63a22cd23c927bda64c88fdc7ddbfabe8fb41e06422`.
- `tests/test_entsoe_zone_configuration.py`
  - complete/gapped histories, contiguous code change, overlap, EIC syntax,
    future window, lineage, resolver and authority tests;
  - SHA-256
    `3344f85879bf838eeeadd5c5163b67cb84122b17f3b30009bde2b6d80526a54b`.
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-GAPS-20260805.md`
  - adds the exact zone-registry fields to request.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D254.
- `.planning/HANDOFF.md`
  - routes current work to this handoff.

Concurrent D260 resolution-regime work was preserved. No file belonging to
that batch or to the concurrent D245 staging run was modified or deleted.

## Verification

Every shell action verified exact cwd `C:\Users\jbattaglia\PFC_LT` and Git
root `C:/Users/jbattaglia/PFC_LT`. Mutable test state stayed below `build/`
through `scripts.run_workspace_local`.

- focused D251-D254 matrix:
  `python -B -m scripts.run_workspace_local --run-id entzone254f python -B -m pytest tests/test_entsoe_zone_configuration.py tests/test_entsoe_pfc_data_coverage.py tests/test_entsoe_series_family_mapping.py tests/test_entsoe_series_inventory.py -q`
  -> `58 passed in 0.36s`;
- Ruff:
  `python -B -m scripts.run_workspace_local --run-id entzoneruff254f python -B -m ruff check pfc_shaping/validation/entsoe_zone_configuration.py tests/test_entsoe_zone_configuration.py`
  -> `All checks passed!`;
- final adjacent ENTSO-E matrix:
  `python -B -m scripts.run_workspace_local --run-id entadj254g python -B -m pytest <all tests/*entsoe*.py> -q`
  -> `493 passed in 17.68s`; post-documentation rerun `entadj254z` ->
  `493 passed in 15.80s`.

The earlier `entadj254f` target itself reported `492 passed, 1 failed`: the
D245 cleanup test observed a staging directory created by another concurrent
run. The directory disappeared when its owner finished. It was not deleted or
counted as D254 evidence; the final clean rerun is authoritative.

## Cost and authority

D254 made zero Databricks requests, SQL statements, Warehouse starts, remote
writes and real-row reads. It used official ENTSO-E documentation over the web
only to confirm EIC format and type-Y area semantics. The 2026-08-06 Databricks
reservation remains consumed; no same-day retry is allowed.

## Next safe action

Integrate the temporal zone resolver into the exact D253 series-family mapping
so every non-directional coupled-zone series binds one registry entry at its
delivery time. Then prepare an owner-fillable registry template. Do not insert
real codes from memory or infer them from labels; obtain a governed official
registry snapshot and owner review first.

The monthly solver remains sole level authority. LT/CT separation, historical
hourly CH truth, T057 sealing, OMPEX benchmark-only status, AFRY
descriptive-only status and production `NO_GO` remain unchanged.
