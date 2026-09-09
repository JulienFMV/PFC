# Session handoff — EEX short-tenor solver-neutral shape contract — 2026-08-05

## Outcome

A dormant CH LT boundary now exists for a future EEX DAY/WEEK/WEEKEND shape
signal. It validates a pre-built additive signal at local delivery-day by
08:00-20:00 block grain and projects it into the exact nullspace of active
BASE/PEAK/OFFPEAK constraints.

Selected qualification status:

`PASS_LOCAL_MATHEMATICAL_CONTRACT_ONLY_NO_MODEL_AUTHORITY`

Selected qualification content ID:

`1c3b833e2c7abc8ddc60867c2cdee8b152e2128a9699e4f38958a3b7062b8343`

This proves solver neutrality and native-grain discipline only. It does not
construct a feature from prices, estimate amplitude, prove point-in-time
availability, validate predictive skill, assemble a candidate or authorize
production.

## Changed files

- `.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-SHAPE-CONTRACT-V1.json`
  - SHA-256
    `cddea7b1a47e81bfa3e85a1ed70bfb1adfa39ed56cc6b546bad415f9c9dce9ee`;
  - binds D212 normalization and hardened D216 short-tenor audit;
  - defines permitted same-vintage additive contrast families, native grain,
    projection tolerance and activation prerequisites;
  - keeps model input, assembly and production authorities false.
- `pfc_shaping/lt/model/short_tenor_shape_contract.py`
  - SHA-256
    `951274f220ac7b5d3dc4a992ec83e672cc10e6b0b28036ecfca1336a2cbb981e`;
  - supports regular hourly and 15-minute UTC delivery indexes;
  - requires contiguous complete Europe/Zurich months and pre-delivery
    valuation;
  - rejects within-day/block dispersion, non-finite inputs, incomplete months,
    malformed provenance and unsupported country/cadence;
  - projects additively into active CH BASE/PEAK/OFFPEAK nullspace;
  - returns price-target-free constraint and monthly residual evidence;
  - has no connector, quote-fitting, assembler or production dependency.
- `tests/test_short_tenor_shape_contract.py`
  - SHA-256
    `7f29aaa91982633a60f929cb5e202e90c54cdaa3606934d0de2a4c1a5a38dd97`;
  - 11 tests spanning a complete hourly year, spring and autumn 15-minute DST
    months, monthly and quarterly PEAK constraints, negative signals, zero
    signal, deterministic/idempotent projection and fail-closed inputs.
- `docs/research/forwards_sources.md`
  - records the dormant mathematical boundary and its authority limits.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D-20260805-217.
- build-only materializer:
  - `build/eex-short-tenor-shape-contract/materialize_short_tenor_contract_qualification.py`,
    SHA-256
    `d2758778bcee40d2c74bc7cbd2ac4c2764a7fa9664e4c994d4bb06ea5d5fd0e1`.
- this handoff.

No CT file, Power BI artifact, protected heavy desk data, AFRY numeric artifact,
monthly solver implementation, assembly path or production flag was modified.

## Source and evidence bindings

- D212 all-product normalization content ID:
  `2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f`.
- D216 hardened short-tenor audit content ID:
  `b09eb3250df5a3c0616eb169c512319c514ddf540251b405023d9351bd5d8bde`.
- contract SHA-256:
  `cddea7b1a47e81bfa3e85a1ed70bfb1adfa39ed56cc6b546bad415f9c9dce9ee`.
- selected qualification:
  `build/eex-short-tenor-shape-contract/qualifications/1c3b833e2c7abc8ddc60867c2cdee8b152e2128a9699e4f38958a3b7062b8343/qualification.json`.
- selected qualification file SHA-256:
  `f1cce0e7b762cc248dcb8717fbe6f91999a83626516af6a950013748c4c35304`.

An earlier qualification
`f9237128ebd31488a921cfe1daff5877cea9374ef4bd80c0be44aaf5da16cde5`
was generated while the contract still referenced D215. It is retained under
`build/` as superseded local forensic evidence and is not selected.

## Mathematical contract

- Price space is additive EUR/MWh; zero and negative values are supported.
- The maximum native signal grain is one value per local delivery day and
  short block: 08:00-20:00 versus its complement, on every delivery day.
- No hourly or quarter-hour variation may be inferred inside a quoted block.
- Delivery months must be complete in Europe/Zurich, including 23-hour and
  25-hour DST days.
- Monthly BASE levels must exactly cover all delivered local months. PEAK
  constraints may be monthly or accepted coarser products represented by the
  active solver constraint system.
- Orthogonal projection enforces zero additive residual against every active
  BASE/PEAK/OFFPEAK row.
- Both the maximum hard-constraint residual and the weighted monthly mean must
  remain at or below `1e-9` EUR/MWh.
- Individual month patches after projection are forbidden.

The selected qualification observed a maximum residual below `4e-14` EUR/MWh
and exactly zero introduced within-block dispersion across its three cases.

## Commands and results

Every shell action first verified exact cwd and Git top-level
`C:\Users\jbattaglia\PFC_LT`. Mutable temporary, pytest and bytecode paths
remained below `build/`.

1. Initial focused contract test:

   `python -m pytest tests/test_short_tenor_shape_contract.py -q`

   Result: `10 passed`, with two test-regex deprecation warnings. The regex
   literals were corrected before selection.

2. Final focused contract test after accepted coarser PEAK coverage was added:

   `python -m pytest tests/test_short_tenor_shape_contract.py -q`

   Result: `11 passed in 0.99s`.

3. Local qualification materialization after binding D216:

   `python -m build.eex-short-tenor-shape-contract.materialize_short_tenor_contract_qualification`

   Result: content ID
   `1c3b833e2c7abc8ddc60867c2cdee8b152e2128a9699e4f38958a3b7062b8343`.

4. Adjacent LT regression matrix covered the new contract, hardened EEX
   short-tenor audit, EPEX A/B shape lab, monthly solver/audit/constraints/
   integration/priors, LT package and LT/CT import boundaries, cascading,
   arbitrage-free and water value.

   Result: `232 passed, 1 skipped in 115.55s`.

No Databricks request, SQL Warehouse start, network call or remote write was
performed in this batch.

## Authority and invariants

- The CH monthly solver remains the sole monthly-level authority.
- The new module is dormant and disconnected from the assembler and pipeline.
- DAY/WEEK/WEEKEND settlements remain day/block evidence, not hourly or
  15-minute observed truth.
- Missing quotations remain missing. No fill-forward, synthetic quote or
  mixed-vintage operation is authorized.
- Local settlement chronology does not prove signed provider-time
  availability.
- Signed EEX PIT vintages, governed ENTSO-E, preregistered rolling-origin
  construction and a new independently frozen future holdout remain mandatory
  before fitting or activation.
- AFRY and OMPEX remain benchmark-only, T057 stays sealed, production is strict
  `NO_GO`, and LT/CT separation is unchanged.

## Next safe batch

Stay offline. Specify a pure, deterministic constructor for same-vintage
DAY-versus-WEEK and WEEKEND-versus-WEEK additive contrasts that consumes only
complete non-overlapping quote strips and emits the native-grain signal accepted
by this contract. Keep all amplitudes untrained and activation false. Roast the
constructor against D216 metadata and missing-strip cases without opening T057,
using OMPEX or AFRY values, or querying Databricks.
