# Local structural shaping v1 — 7 September 2026

This first executable milestone inventories the inputs needed for 2030–2035
and verifies signed hourly shape arithmetic. It does not produce a structural
forecast, dispatch assets, infer scenario probabilities, or select a model.
Historical shape alone cannot identify future capacity and flexibility effects.

## Inputs and current evidence

Audit retained public-source annual tables separately, with exact SHA-256 pins
in `scripts/audit_lt_structural_readiness.py`. Preserve their source, edition
and scenario labels. No cross-source join, scenario relabelling, interpolation,
year clamping or missing-value substitution. Read-only local evidence is not
a fresh PRD extraction or an independently admitted scenario release.

The annual diagnostic covers CH/DE/FR/IT/AT, each year 2030–2035. A present
field, a declared zero, a missing value, invalid value, absent annual row and
ambiguous duplicate row have different statuses. Publication after the audit
origin and known later ingestion exclude the row. Missing ingestion remains
visible as descriptive coverage, without a point-in-time availability claim.
Partial/proxy/neutralized quality labels stay in the report; they cannot be
erased by non-null coverage. Country/scenario/year duplicates are not resolved
by choosing the latest row or combining separate releases.

The diagnostic uses the existing scenario field names for demand, renewables,
storage, flexibility, hydro, supply, costs and CH interconnections. Additional
battery operation fields define charge/discharge efficiencies in (0,1], initial
and terminal state-of-charge shares in [0,1]. `dsm_max_shift_hours` defines a
positive shift deadline. Operational parameters are not required when both
battery power and energy, or DSM capacity respectively, are explicitly zero.
One positive battery size and one zero size is an inconsistency. This is a
field inventory, not a complete dispatch admission validator.

Chronological inputs still required for a useful physical model:

- a versioned reconciliation of source scenarios to FMV assumptions;
- consistent hourly demand by use and weather-conditioned solar/wind profiles,
  with capacity/energy calibration and explicit behind-the-meter accounting;
- storage efficiencies, power/energy limits, initial/terminal conditions and
  cycling costs; charging must not invent or destroy energy;
- hydro inflows, turbine/pump constraints and reservoir carry-over across days
  and seasons; aggregate historical fill is not a future inflow trajectory;
- flexible demand availability, deadlines and recovery; annual EV/PAC energy
  must not be counted twice in total demand;
- available firm capacity, operating costs, outage schedules, constrained
  cross-border exchange and price-setting/bidding assumptions;
- several weather chronologies, explicit forecast-information assumptions and
  sensitivities to perfect foresight. Stress labels are not probabilities.

The first physical implementation should use exogenous installed-capacity
trajectories and a reduced chronological system. Endogenous investment,
pan-European plant-level fidelity and asset-specific FMV hydro optimisation
are separate increments, not hidden claims of this initial inventory.

D303 places signed-shape integration and a controlled final-PFC comparison
before an isolated storage kernel, while preparing public scenario assumptions.
The first physical milestone must connect chronology, flexibility, price
formation and final market-constrained shape on a coherent case. See Phase 14
`SESSION-HANDOFF-20260907-PFC-PRIORITY-REVIEW.md` for sequencing and exit criteria.

D304 implements that first integration and local comparison; read
`docs/model/LT-SIGNED-SHAPE-LOCAL-EXPERIMENT.md`. The signed input reuses the
existing assembler and final projection with neutral ancillary layers. Its
hourly transport does not qualify intraday, hydro or uncertainty composition.
The signed seasonal reference is promising on the exposed development years;
neither independent confirmation nor structural2030 accuracy is established.

D305 qualifies explicit additive intrahour and existing water-value composition
in that same assembler; read `LT-SIGNED-COMPOSITION-LOCAL-EXPERIMENT.md` and the
Phase14 signed-composition handoff. Local hydro ablation shows no improvement.
The unconditional additive DE intraday reference improves aggregate conditional
disaggregation but materially worsens negative-parent hours, so it is not
adopted. Complete PFC alternatives are comparison artifacts, with no new
production authority, CH quarter-hour accuracy claim or2030 structural evidence.

## Signed hourly shape

`center_signed_hourly_shape` takes a finite price series in EUR/MWh on complete,
consecutive Swiss months at hourly UTC cadence. It preserves repeated autumn
clock hours, spring-short months and leap years. It returns the price minus
its own Swiss-month average. Its monthly mean is zero to 1e-9 EUR/MWh, without
positive ratios, denominator division, clipping, or removal of negative days.
Finite precision failures reject the output.

This is a numerical target/shape operation, not an assembler or a scenario
calendar mapping. It reuses the existing UTC-index validation boundary.
Historical training targets may use only complete months ending before the
training origin. Realized future monthly means must never become predictors.
Dispatch simulations may provide candidate prices under explicit scenarios;
their normalized shape still has no production authority.

Later integration must retain one monthly BASE solver and reuse the existing
`PFCAssembler` and final BASE/PEAK projection. The frozen v6 multiplicative
candidate seam does not acquire a signed EUR/MWh contract merely by renaming
an array `f_H`. No new adapter or post-solver month patch is allowed. A separate
versioned integration test must show monthly conservation, supported quote
repricing and the effect of market projection on the proposed shape.

## Execution and success criteria

The local audit writes a source manifest, exact-row coverage, field matrix,
mechanism summaries and a French report under a fresh `build/` directory.
It also verifies this arithmetic on complete native CH historical months from
the retained, hash-verified D300 PRD export. It checks that all four transport
quarter-hours agree before counting one hourly observation. Incomplete months
are explicitly excluded. The resulting signed targets are diagnostics, not
forecasts or benchmark wins. No fitting, forecast scoring or Warehouse call.

Positive/negative tests cover explicit zero versus unknown capacity, impossible
physical values, source timing, duplicate releases, no year fabrication,
negative prices, DST/leap months, missing/replicated hours, monthly level-shift
invariance, storage timestamp resolution and unchanged input frames.

The data engineer owns source definitions/availability/revisions. Model and
product teams own the physical formulation, scenario choice, calibration and
FMV loss function. An unavailable real trajectory remains missing while local
engineering proceeds. Restricted AFRY remains diagnostic/teacher-candidate;
its existing model/calendar gate is unchanged and T057 remains sealed.
