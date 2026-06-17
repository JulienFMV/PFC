# Expert task — LT/HPFC remediation: close the defects found by the dual audit

## 0. Who you are / repo orientation
You are a senior energy-finance quant + ML engineer (Alpiq trading-desk pragmatism
× ETH/EPFL rigor) on the **PFC** repo (Swiss/German electricity Price-Forward-Curve;
long-term "LT" HPFC + short-term "CT"). This task closes the **defects found by two
independent audits** (an internal quant deep-dive and an external review) of the LT
model. Read these source files before touching anything:
- `pfc_shaping/lt/model/assembler.py` — `PFCAssembler.build()`: `price = B·f_S·f_W·f_H·f_Q·f_WV` + arbitrage-free calibration.
- `pfc_shaping/lt/model/water_value.py` — f_WV hydro reservoir correction (**two P0 bugs here**).
- `pfc_shaping/lt/model/solar_modulation.py` — solar-aware f_H correction (§4quater).
- `pfc_shaping/lt/model/electrification_shape.py` — Phase 13 scenario shape + fan chart.
- `pfc_shaping/lt/model/shape_hourly.py` — f_H (trend/horizon governance).
- `pfc_shaping/calibration/arbitrage_free.py`, `cascading.py`, `robust_seasonal_ratios.py`, `intraday_amplitude.py`.
- `tests/test_solar_modulation.py`, `tests/test_shape_hourly_infra.py`.

## 1. Sync first (worktree-safe)
The work lives on `feat/lt-next-sota` (NOT merged to `main`); it may already be
checked out in another git **worktree** (a direct checkout from a second worktree
fails — expected). Do NOT touch other worktrees. Create your branch from the
up-to-date remote:
```
git fetch origin
git checkout -b fix/lt-audit-remediation origin/feat/lt-next-sota
git log --oneline -3   # expect 9935c27 at/near HEAD
```
If you don't see the recent commits, STOP and report.

## 2. Environment & how to validate (READ — avoids false conclusions)
- Install deps if missing: `numpy pandas scipy scikit-learn holidays statsmodels openpyxl matplotlib pytest`.
- **Data availability is the #1 source of false test verdicts.** In a fresh clone
  only `pfc_shaping/data/*.parquet` ship (entso_15min, epex_15min, epex_de_15min,
  de_renewable_forecast, hydro_reservoir). `data/epex_hourly.parquet` and
  `data/forwards_history_phase10.parquet` are git-ignored / **absent**, so the
  EPEX/forwards-dependent tests **SKIP** (they do not run). The loader path is
  `DEFAULT_EPEX_PATH = <repo>/data/epex_hourly.parquet` (note: top-level `data/`,
  NOT `pfc_shaping/data/`). The 15-min source already has a UTC `DatetimeIndex`.
  Bootstrap the hourly EPEX with the **repo's canonical builder** (do NOT hand-roll
  a divergent one — the audit's diagnostic numbers were produced with it; it fills
  the ~72 gap-hours by time-interpolation to keep the index gapless, whereas a naive
  `.dropna()` yields a non-contiguous index and may not reproduce the cited figures):
  ```bash
  python -c "from scripts.run_phase10_real import bootstrap_epex_hourly; bootstrap_epex_hourly(force=True)"
  ```
  This writes `data/epex_hourly.parquet`; the `@requires_epex` solar tests then flip
  SKIP→RUN. (`test_phase10_reproducibility.py` double-runs against whatever file is
  present, so either build keeps it self-consistent.)
  `data/forwards_history_phase10.parquet` needs the FMV poste — the `@requires_fwds`
  tests will still SKIP without it; say so. **State clearly which tests RAN vs SKIPPED.**
- **Capture a baseline first**: on the parent commit (with the bootstrapped data +
  statsmodels/openpyxl installed) run the FULL suite and record `X passed / Y skipped
  / Z failed`. Report only the delta you own; never read a pre-existing red as your
  regression. Confirm any failure reproduces on the parent before attributing it.

## 3. Hard contracts (non-negotiable — verify after every change)
- **Reproducibility atol=1e-12**: every new/changed behaviour ships behind a flag
  defaulting OFF; with flags OFF the pipeline is byte-identical
  (`tests/test_phase10_reproducibility.py`; needs the bootstrapped epex_hourly +
  statsmodels). Prove OFF byte-identity (atol=0) after your work.
- **No leakage**: at vintage v, fit only on data `< v`; forward periods projected,
  never realized-future.
- **Energy / arbitrage invariants**: `mean_h(f_H)=1` per local day after every
  f_H transform; the calibrated curve must reprice every traded block (Cal/Q/M,
  BASE+PEAK) to <1e-9.
- **Scientific honesty**: do NOT force a correction that empirically degrades the
  realized target. If a layer's estimator is wrong-signed or hurts the metric on
  real data, FIX the method or NEUTRALIZE it on the prod path — never tune a test
  to green. A null/negative result, reported with numbers, is a valid outcome.

## 4. P0 — BLOCKING quant errors (fix first)

### P0-1 — Water-value SIGN is inverted (`water_value.py` ~l.251 + ~l.401)
The effective coefficient is `beta_wv_ × season_sensitivity_`. `beta_wv_` is clipped
**negative** (`BETA_WV_MIN/MAX = [-0.10,-0.001]`) AND `season_sensitivity_ =
raw_sens/abs(beta_wv_)` is clipped **negative** (`[-2.0, 0.0]`). Product
(neg)×(neg)=**positive** → the **opposite sign of the fitted regression coef**.
- Evidence (verified analytically): low reservoir fill (scarcity) → `f_wv≈0.997<1`
  → price DOWN. Backwards: low fill = high water value = **high** price.
- **Fix**: make the applied sign equal the regression sign (low fill ⇒ f_wv>1 ⇒
  higher price; high fill ⇒ f_wv<1). Likely the double-negative in
  `season_sensitivity_` normalization; re-derive so `effective = raw_sens` sign.
- **Acceptance**: a unit test asserting, on a controlled fill series,
  `f_wv(low_fill) > 1 > f_wv(high_fill)` and that the implied price moves UP in
  scarcity. Magnitude sane (≈ ±1–2 %/σ of fill).

### P0-2 — Water-value delta-additive breaks the annual mean (`water_value.py::compute_delta_wv` ~l.514)
`delta = (f_wv−1)·|B|`. `f_wv−1` is renormed to mean-zero **per year**, but
`(f_wv−1)·|B|` is NOT mean-zero when the correction correlates with `|B|` (winter:
high B + large correction). Measured annual forward drift ≈ **−0.89 €/MWh** in a
realistic seasonal scenario → silent arbitrage violation.
- **Fix**: enforce that the additive WV delta is mean-zero **over each traded
  delivery block** (Cal AND each quoted Q/M), e.g. subtract the B-weighted mean
  per block, or apply WV multiplicatively then re-anchor. Whatever the route,
  prove the **final price contribution per block** is unchanged (not just `f_wv`
  itself — multiplicative geometric-mean-1 ≠ additive arithmetic-demean; they are
  not equivalent, so anchor on the delivered €/MWh, not on the factor).
- **SEQUENCING (critical)**: do P0-1 BEFORE designing P0-2. The sign fix flips
  `f_wv` in winter scarcity (<1 → >1), which CHANGES the `|B|`-correlation that
  drives the drift — so the −0.89 €/MWh figure is a PRE-FIX number. **Re-measure
  the annual/per-block drift AFTER P0-1**, and derive P0-2's fix and thresholds
  from the post-sign-correction values, not from −0.89.
- **Acceptance** (pin the block universe — do NOT let it be self-chosen):
  1. Enumerate the **exact traded block set actually quoted** in the calibration
     instrument inventory used by `arbitrage_free.py` (every quoted Cal/Q/M), and
     assert the set is non-trivial (≥1 Cal, >1 Q, >1 M) so the test can't pass on a
     single self-selected block.
  2. Assert `compute_delta_wv` leaves **every** block in that set mean-unchanged to
     <1e-9, AND the delivered price reprices **each** block (not the aggregate
     `converged` flag — §7 shows it can hide per-block drift) to <1e-9 post-calibration.
  3. **Anti-parking guard**: assert that NO un-quoted (free) calendar month's
     B-weighted WV-delta mean exceeds 1e-9 — so drift cannot be silently absorbed by
     a delivery month outside the asserted set.

## 5. P1 — must close before any "prod-ready" claim

### P1-1 — Full pytest is RED: `588 passed, 2 skipped, 3 failed` (external, Windows+data)
1. `tests/test_shape_hourly_infra.py:1083` — Windows `cp1252` crash on a
   `read_text()` without `encoding`. **Fix**: pass `encoding="utf-8"` on all
   file reads/writes in tests and code (grep for `read_text(`/`write_text(`/`open(`
   without `encoding=`).
2. `tests/test_solar_modulation.py:208` — fitted **night β ≥ midday β** (wrong:
   midday should dominate). 
3. `tests/test_solar_modulation.py:235` — the solar layer **attenuates** the duck
   curve instead of deepening it on the real dataset. **This is the real quant
   signal**, and it independently confirms the internal audit's "2030 bowl
   under-deepened" finding (see P1-2). These tests SKIP without `data/epex_hourly.parquet`
   — bootstrap it (§2) so they RUN, then diagnose.

### P1-2 — Duck curve is structurally UNDER-deepened (the root cause behind P1-1.2/3)
Two compounding causes (internal audit, empirically verified):
- `shape_hourly.py:438-445` — `_flatten_strength`/`_trend_strength` make the
  far-horizon profile **flatten** (a midday factor deepens 0.718→0.673 at y+1 but
  **rebounds to 0.787 at y+6**, shallower than history) — it reverts the deepening
  exactly at the 2030 horizon it should express.
- `solar_modulation.py:~225` — forward solar penetration is **capped at the p99 of
  the TRAINING distribution (~16%)**, so 2030 PV (which exceeds historical p99)
  cannot drive a deeper bowl.
- **Fix (scientific, not cosmetic)**: (a) re-examine the solar β estimator — if
  midday β is not robustly negative & dominant on real data, the method is
  mis-specified (block pooling, ridge λ, or the de-levelled-residual target);
  recalibrate or gate OFF. (b) Reconsider the p99 cap and the far-horizon
  flattening so a *scenario-justified* deeper bowl is allowed for 2030. Do NOT
  hard-code a deepening to pass the test — justify it from the data/scenario.
- **Acceptance**: either (i) on real data, midday β robustly negative &
  |β_mid|>|β_night|, and the layer moves bowl_depth toward realized (tests 208/235
  green for the right reason), OR (ii) the solar layer is explicitly marked
  experimental and OFF on the prod path with a documented rationale. Both are
  acceptable; a green test obtained by weakening the assertion is NOT.

### P1-3 — Structural fan chart is statistically degenerate (`electrification_shape.py:~552`)
`_weighted_quantile` over **3 values** (slow/central/fast): p10=slow, p50=central,
p90=fast — no interpolation, no tail; `structural_width = fast−slow`. It is a
3-scenario bracket mislabelled as probabilistic quantiles.
- **Fix**: either (a) rename outputs to `scenario_low/central/high` + `scenario_spread`
  (honest bracket, no p-label), OR (b) build a genuine distribution (intra-scenario
  perturbation / level-uncertainty) before computing p10/p90. Do not present 3
  points as calibrated quantiles.
- **Acceptance**: the report/columns no longer claim p10/p90 from 3 points without
  a stated interpolation/distribution model; if (b), width widens with horizon.

### P1-4 — solar_modulation stays EXPERIMENTAL until P1-2 is understood
Keep `enable_solar_modulation=False` on the prod path; document in
`10-PERFECT-FORESIGHT-SHAPING.md` §4quater that the empirical estimator currently
degrades the realized bowl on the live dataset and must be recalibrated or stay off.
- **Gate-OFF is NOT a proof.** If you neutralize solar on the prod path,
  flags-OFF byte-identity becomes trivially true and proves nothing about the
  layer. You MUST still report the flag-ON result on real data (midday β sign,
  bowl_depth-vs-realized number) that JUSTIFIES the gate decision. "Byte-identity
  proven" alone does NOT satisfy this item.

### P1-6 — f_H clip applied AFTER per-day renorm breaks mean_h=1 (`shape_hourly.py:533`)
`result = result.clip(0.4, 2.0)` runs AFTER the per-day renormalization in
`apply()`, so on any day that hits a clip bound `mean_h(f_H) ≠ 1` (measured
0.99985). This violates the §3 hard contract directly (energy silently lost before
calibration). `get_for_horizon` already does it correctly (clip THEN renorm).
- **Fix**: invert the order — clip first, then renormalize per local day.
- **Acceptance**: after `apply()`, `mean_h(f_H)=1` per local day to <1e-12 even on
  days where clipping bites; byte-identity OFF still holds for unclipped days.

### P1-5 — Phase 13 scenarios still carry `partial/proxy/..._neutralized` flags
Acceptable for local/test QA, NOT for a governed production run. Keep Phase 13
local-test only; the production gate (`assert_production_scenario_inventory` /
`require_production_scenario_data`) must reject proxy/partial/neutralized rows
(verify it does). Update `VALIDATION.md` to distinguish "local-test validated" vs
"governed-production still blocked by data/proxy".

## 6. P2 — hygiene & robustness (close if time permits, separate commits)
- **Repo hygiene**: two ~11 MB tracked parquets `pfc_shaping/output/pfc_15min_2026-03-15.parquet`
  and `pfc_shaping/output/pfc_de_15min_2026-03-15.parquet`. `git rm --cached` them,
  extend `.gitignore` (already covers `pfc_shaping/output/*.parquet` — confirm and
  ensure these are now ignored), keep local copies. Define the data policy: small
  governed EEX parquet OK; large generated artifacts → storage/DVC/LFS.
- **Windows robustness**: enforce `encoding="utf-8"` everywhere (code + tests).
- **pytest config**: register the `slow` marker (avoid PytestUnknownMarkWarning).

## 7. Lower-priority quant findings (open tickets; fix only if cheap & safe)
Document these in a short `.planning/` note even if not fixed now:
- `arbitrage_free.py:~735` — inconsistent overlapping products (Cal vs its Q/M)
  are least-squares **smeared** by the SVD pseudo-inverse instead of repriced
  exactly (Cal-vs-Q 5€ inconsistency → Cal off 4.0, each Q off ~1.0). And
  `converged` (l.~577) only checks aggregate residual<tol, not per-block exactness.
- `cascading.py` — `fit_peak_spreads` peak mask omits holidays while `count_hours`/
  calibrator exclude them (spread fitted on a different hour set than applied).
- `intraday_amplitude.py:~173` — target is peak−**base** (~0.65·(peak−off)) but
  compared to peak−**offpeak** → ~35% over-compression (flag-gated, default OFF).
- `robust_seasonal_ratios.py:~317` — Kish ESS is scale-invariant, so crisis
  down-weighting does NOT reduce the shrinkage-to-prior weight (crisis years get
  the same shrinkage as clean years); and the in-sample median anchor (l.~258)
  self-anchors to a crisis-dominated window. Exogenous anchor never wired.
- (clip-after-renorm `shape_hourly.py:533` is promoted to **P1-6** above — fix there.)
- Season definitions disagree across modules: calendar `Hiver={11,12,1,2,3}`,
  `Automne={10}` vs electrification `WINTER={12,1,2}`, `SHOULDER=rest` → March &
  November conflict. Unify to one source of truth.

## 8. Order of work (dependencies)
1. §2 env + bootstrap data so tests actually run; reproduce the 3 failures.
2. P0-1 then P0-2 (water value) — biggest economic errors; add tests.
3. P1-1.1 (UTF-8) — unblocks the suite cross-platform.
4. P1-2 → decide recalibrate vs gate-off solar; then P1-1.2/3 resolve correctly.
5. P1-3 fan chart honesty; P1-4 doc; P1-5 prod-gate + VALIDATION.md.
6. P2 hygiene (separate commits).
7. §7 note for the deferred findings.
After each step: run the affected tests + prove flags-OFF byte-identity (atol=0).

## 9. Git workflow
- Branch `fix/lt-audit-remediation`. Descriptive commits, grouped by P0/P1/P2.
- Run the FULL suite; report `X passed / Y skipped / Z failed` and the green delta.
- Push the branch. **Do NOT open a PR** unless explicitly asked.

## 10. Definition of done
- [ ] P0-1 WV sign corrected; test proves scarcity ⇒ higher price.
- [ ] P0-2 WV additive delta: drift RE-MEASURED after P0-1; final price reprices
      EVERY Cal/Q/M block to <1e-9 post-calibration (per-block, not aggregate).
- [ ] P1-6 f_H clip order fixed; mean_h(f_H)=1 per local day <1e-12 incl. clipped days.
- [ ] Full pytest GREEN (0 failed) on Linux AND Windows (UTF-8); state ran-vs-skipped.
- [ ] Solar layer: midday β robustly dominant on real data AND bowl moves toward
      realized — OR explicitly experimental & OFF on prod with documented rationale.
- [ ] Fan chart no longer mislabels 3 points as p10/p90.
- [ ] Phase 13 prod-gate rejects proxy/partial/neutralized; VALIDATION.md clarified.
- [ ] flags-OFF byte-identity proven (atol=0); reproducibility test green.
- [ ] (P2) big output parquets untracked; UTF-8 enforced; `slow` marker registered.
- [ ] §7 deferred-findings note committed.
- [ ] Branch pushed; no PR.

## 11. Confirm understanding, then proceed (no human ack required)
Post a short confirmation, THEN proceed immediately without waiting for a reply
(this is an async agent run): (a) branch + HEAD; (b) baseline `X passed/Y skipped/Z
failed` on the parent commit and which of the 3 cited failures RUN vs SKIP in your
env and why; (c) the WV sign bug in one line; (d) your solar plan (recalibrate vs
gate-off) with the number that motivates it; (e) how you'll prove flags-OFF
byte-identity. Commit per logical step (grouped P0/P1/P2) and **push at the end**
of each priority tier; do not wait for confirmation between tiers.

## 12. If blocked / ambiguous
If a fix would require changing the reproducibility-guarded output, or the solar
recalibration is non-obvious, STOP and report with numbers + your proposed options
— never weaken a test assertion to obtain green, and never introduce leakage or an
arbitrage break to make a metric look better.
