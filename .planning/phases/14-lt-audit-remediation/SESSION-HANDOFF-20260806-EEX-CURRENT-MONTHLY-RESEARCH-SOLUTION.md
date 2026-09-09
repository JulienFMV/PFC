# Session handoff — current CH EEX monthly research solution

Date: 2026-08-06  
Decision: D-20260806-286  
Status: implemented, locally materialized and roasted; research authority only

## Outcome

D286 produces the first actual CH monthly BASE level layer from the exact local
D212 EEX bytes. It covers 76 months from 2026-09 through 2032-12. August is not
fabricated because it is already delivering.

The latest surface contains 19 BASE and 19 PEAK CAL/Q/M quotes. BASE alone
enters one hard monthly solver: 17 constraints are independent and two source
quotes are redundant but consistent. PEAK is persisted separately and has no
monthly-level or shaping-input authority yet. DAY/WEEK/WEEKEND remain excluded
from monthly levels.

Historical CH quotes enter only as a zero-mean within-parent shape prior. No
neighbor, ENTSO-E, AFRY or OMPEX level enters the solution. The active hard
maximum absolute residual is `4.263256414560601e-14 EUR/MWh`; all displayed
BASE quotes remain within the declared `0.01 EUR/MWh` conflict tolerance.

This is not yet a full PFC, PIT evidence, a model input, a candidate, or an
OMPEX superiority result. All such authorities remain false.

## Selected immutable inputs

- D212 normalization content ID:
  `2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f`;
- normalization manifest SHA-256:
  `f91805f3004e746ac588e19aa6745ae0dc0f490a9ef5c49aae0918e7ec3f8f53`;
- solver history: 34,105 rows, SHA-256
  `fc5de85d1870937955dcc93cbf1cea0e1d2f85d892b286f490a6de11650d7a25`;
- latest surface: 38 rows, SHA-256
  `3b8c9a9831a3ff44b8b9880e914fa1bb1e60c4e03d0610d1fd40ab8b83490aaf`;
- D212 surface audit content ID:
  `bb1a09932b4bbff31dfdbb4ada561befb02050413ee819b03bf6c28f4858ab54`;
- D213 horizon audit content ID:
  `435ecbc737f95268f03f7f347dfafc4163f5b5a4cb5b8dc9cec87e05f1645108`.

All selected inputs were read locally below `build/`. No new Databricks,
network or external-volume access occurred.

## Changed files

- `.planning/phases/14-lt-audit-remediation/EEX-CH-CURRENT-MONTHLY-RESEARCH-SOLUTION-CONTRACT-V1.json`
  - raw SHA-256:
    `3faa859d50b136d06df29afe02e635662c1641b7321f1bb105dddc3cf95c691a`;
  - canonical content ID:
    `e93e0c31903c93ad6ba382f9f34b05f5842c6f17d7ea0f8fe4e22eaf9eee2ca3`.
- `pfc_shaping/validation/eex_current_monthly_research_solution.py`
  - SHA-256:
    `2990cbf316fe190192234f4c4e54473b21ce81b80130db8e54be8bbdb445ab6a`.
- `tests/test_eex_current_monthly_research_solution.py`
  - SHA-256:
    `e10410384d3c84bb3b9fd404caddf2fbed1cb0ea6835e42b5489da57f107daed`.
- `build/d286_materialize.py`
  - SHA-256:
    `97f284254da5393968ad67f3005bbeb57c7e96a3de6c01892b48404cba05860b`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `docs/research/forwards_sources.md`
- `.planning/HANDOFF.md`
- this handoff.

No CT, Power BI, AFRY, OMPEX, T057 or heavy desk-data file was opened or
changed by D286.

## Local artifacts

Price-bearing bundle, ignored below `build/`:

`build/databricks-eex-daily/2026-08-06/eex-current-monthly-research-solutions/a2e27c4e78515d2e7473769e60d7fa9767756fd2078ce5698388880e3a329a5b/`

- bundle content ID:
  `a2e27c4e78515d2e7473769e60d7fa9767756fd2078ce5698388880e3a329a5b`;
- `monthly_solution.parquet`: 76 rows;
- `peak_sidecar.parquet`: 19 rows;
- `constraint_diagnostics.parquet`: 19 rows;
- `prior_diagnostics.parquet`: 17 rows.

Price-free proof:

`build/databricks-eex-daily/2026-08-06/eex-current-monthly-research-solution-proofs/5db497336ddc218cb9256809a3f3005fbce7d475cfc2d048004163f461d2c8bd/manifest.json`

- proof content ID:
  `5db497336ddc218cb9256809a3f3005fbce7d475cfc2d048004163f461d2c8bd`;
- proof raw SHA-256:
  `968713ed6446cc9b19844113da14bbd215beccac26f88fe022c001e21dec6de1`;
- assessment content ID:
  `d67c42823fc47242bdadd56dca1c2d6960ddb9c69afbd5f55acb19f5bd8e218b`;
- deterministic replay count: 2;
- price values in proof: false.

## Verification commands and outcomes

All commands ran from the canonical root with `TEMP`/`TMP` and pytest
basetemps below `build/`.

- `python -m ruff check ...` and `python -m py_compile ...`: pass.
- `python -m pytest tests/test_eex_current_monthly_research_solution.py -q`:
  `31 passed`.
- D286 plus D212/D213 EEX matrix:
  `54 passed`.
- monthly constraint/solver/prior/integration/audit plus LT boundaries:
  `153 passed, 1 skipped`.
- `python -m build.d286_materialize` twice: identical bundle, proof and
  assessment IDs.

One initial combined test command hit its 120-second process timeout before a
result. It was split into the two explicit passing matrices above; this was a
runner timeout, not a test failure.

## Invariants and remaining risks

- `monthly_level_authority="solver"`: the one CH BASE solver is the sole level
  authority.
- PEAK and all future hourly/15-minute layers may shape but may not rewrite
  monthly BASE means.
- The current quote surface is local real EEX evidence but not proven PIT;
  rolling-origin selection and source authenticity remain unresolved.
- Spot truth is not admitted. The visible granular spot candidate is currently
  only in `dev` and still needs source/licence, coverage, cadence, completeness
  and PIT profiling when a Warehouse is already running under authorization.
- The actual governed ENTSO-E delivery is still absent. `dev.gold` proves only
  macro-family presence; exact series mappings, cadence history, zones/EIC,
  directions/signs, lineage, revisions, quality, complete vintages and PIT
  semantics remain to be delivered.
- T057 remains sealed; no AFRY rolling-origin work may start; OMPEX remains a
  post-freeze benchmark only.

## Next safe batch

Do not start a Databricks Warehouse. Continue locally with a fail-closed hourly
shaping intake design that consumes the D286 BASE solution but cannot change
its monthly means. Actual calibration must wait for governed CH spot truth and
the ENTSO-E delivery; until then, only structural/calendar mechanics and
synthetic mutation tests are admissible.
