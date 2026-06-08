# Expert LT task — advance the Swiss long-term HPFC to the next SOTA frontier

## 0. Who you are / mandate
You are a senior energy-finance quant + ML engineer (Alpiq-trading-desk pragmatism
× ETH/EPFL rigor) on the **PFC** repo. Mandate (deliberately open-ended):
**absorb ALL the long-term (LT / HPFC) work already shipped, then deliver the next
state-of-the-art improvement** — literature-grounded, evidence-first, measured,
reproducible. You choose the lever, but it must be justified by *our own*
perfect-foresight residuals (numbers, not priors) and approved at the §5
plan-gate before large implementation. Pick **ONE** high-value lever and do it
properly; do not scatter across many.

## 1. Sync first (worktree-safe)
All recent LT work lives on `claude/clean-lt-ct-integration` (NOT merged to
`main`). That branch may already be checked out in another git **worktree** on
this machine (e.g. `PFC_phase10`), in which case a direct `checkout` will fail —
that is expected. Do NOT touch other worktrees or their local changes. Instead,
create your feature branch directly from the up-to-date remote:
```
git fetch origin
git checkout -b feat/lt-next-sota origin/claude/clean-lt-ct-integration
# if it already exists locally: git checkout -B feat/lt-next-sota origin/claude/clean-lt-ct-integration
git log --oneline -6
```
You must see `7446a36 Add agent prompt: next SOTA LT/HPFC improvement` in the
log. This gives you the synced state (all three briefs + §4quater solar) on your
own branch, with no worktree conflict. If you don't see that commit, STOP and
report before coding.

## 2. The LT model (code map — read before editing)
- `pfc_shaping/lt/model/assembler.py` — `PFCAssembler.build()`:
  `price = B × f_S × f_W × f_H × f_Q × f_WV` then arbitrage-free calibration.
- Shape components: `shape_hourly.py` (f_H), `shape_intraday.py` (f_Q),
  `water_value.py` (f_WV), `msfc_spline.py` (backbone B), `block_distribution.py`,
  `uncertainty.py` (p10/p90).
- `solar_modulation.py` — the shipped §4quater solar-aware f_H correction
  (your reference template for a clean flag-gated, leak-free, tested add-on).
- Calibration: `pfc_shaping/calibration/` — `cascading.py` (baseline
  `ContractCascader` seasonal_ratios + peak_spreads), `arbitrage_free.py`, and the
  SOTA drop-ins `robust_seasonal_ratios.py`, `robust_peak_spreads.py`.
- Validation: `pfc_shaping/validation/` — `perfect_foresight.py` (shaping
  diagnostic + estimators baseline/sota/sota_solar), `scorecard.py` (5-pillar,
  `build_one`, `run_scorecard_full`), `structural_tests.py` (Hildmann),
  `dm_test.py`, `christoffersen.py`, `block_masks.py`.
- Runners: `scripts/run_perfect_foresight.py` (`--ab`),
  `scripts/run_phase10_scorecard.py`, `scripts/run_phase10_real.py`.
- Phase docs: `.planning/phases/10-pfc-fmv-quality-scorecard/` — read
  `10-PERFECT-FORESIGHT-SHAPING.md` end-to-end (findings, SOTA rationale, §4ter
  half-life sweep, §4quater solar) and `solar_research/`.

## 3. What is ALREADY done (do NOT redo / contradict)
- **Regime-aware seasonal_ratios**: crisis down-weighting (Marcjasz et al. 2025) +
  Bayesian shrinkage to a CH-physical prior (Bevilacqua 2022) + partial-year obs.
- **Hydro-aware peak_spreads**: holiday-correct mask + regime down-weighting;
  fixed the ~3× peak/off-peak over-statement (model 20.3 → realized 6.4).
- **Intra-day f_H half-life** = 90d (§4ter interior optimum: diurnal cosine
  0.925→0.932, demeaned RMSE 6.81→6.45).
- **Solar-aware f_H correction** (§4quater): block-pooled ridge β on monthly CH
  solar penetration; flag `enable_solar_modulation`; estimator `sota_solar`;
  closes most of the residual peak/off-peak amplitude gap; pf_cal_corr unchanged
  by construction (intra-day, daily-mean-preserving).
- 5-pillar scorecard + perfect-foresight decomposition; the SOTA stack clears the
  0.85 SC#1 gate on all 12 Cal-2025 vintages. **Your job is the NEXT frontier.**

## 4. Evidence-first: run the diagnostic BEFORE choosing a lever
Do not pick a lever from intuition. First reproduce the baseline picture:
```
python scripts/run_perfect_foresight.py --ab        # baseline vs sota vs sota_solar
```
Reference numbers already established (confirm, then look for what's still off):
- pf_cal_corr (monthly-shape) median ≈ 0.92, min ≈ 0.86 across 12 vintages.
- de-levelled diurnal cosine ≈ 0.93; demeaned RMSE ≈ 6.4–6.45 €/MWh.
- CH-physical sub-KPIs (model vs realized Cal-2025): winter/summer ratio;
  solar-bowl depth ≈ 0.448 (sota) vs realized 0.558; peak/off-peak spread
  realized ≈ 6.4 €/MWh. NOTE: the §4quater solar A/B figures (bowl ~0.53, spread
  ~8.7) were ship *targets* not yet measured on real prod data in the cloud clone
  — re-measure them yourself via `--ab`; do not trust them as established.
Read the residuals: where is the model still furthest from realized AND not yet
addressed by §3? That gap — quantified — is your target.

## 5. Candidate levers (choose ONE, justify from §4 residuals)
- **Probabilistic calibration** — p10/p90 are only checked unconditionally
  (Christoffersen). Add CRPS/energy-score + pinball-optimal quantiles
  (Gneiting-Raftery 2007, Lago 2021). Likely high value if coverage is off.
- **f_W weekend/holiday shape** — is the holiday/weekend ratio still mis-stated
  (Pillar 1 gate band)?
- **Far-horizon governance** — `_shape_freedom` damping to backbone B for
  Y+2/Y+3: losing real seasonal structure, or under-damping?
- **Cross-border basis** — CH↔DE/FR/IT coupling in level/backbone (Bevilacqua
  2022); static vs modeled basis.
- **Water-value f_WV** — richer reservoir-fill analogue / snowmelt signal.
- **MSFC backbone** — knot/penalty vs Benth(2007)/Fleten-Lemming(2003) optima.
A genuinely better lever you discover from the residuals is welcome — argue it.

## 6. MANDATORY plan-gate (open-ended task — do not skip)
Before non-trivial code, commit a short proposal in a NEW phase folder under
`.planning/phases/` (e.g. `12-lt-<lever>/RESEARCH.md`) containing:
1. the diagnosed residual you target (with §4 numbers);
2. method + literature justification + rejected alternatives;
3. exact integration point (component + flag) and why it preserves the repro
   contract;
4. validation plan + ship threshold (the effect size that would justify shipping);
5. leakage analysis (what data, which vintage cutoff).
Then implement.

## 7. Hard conventions (non-negotiable)
- **Reproducibility (atol=1e-12)**: new behaviour ships behind a flag defaulting
  OFF; pipeline byte-identical when off (`tests/test_phase10_reproducibility.py`).
  Prove byte-identity OFF (atol=0).
- **No leakage**: at vintage v, train/feature only on data < v; forward periods
  projected, never realized-future; guards intrinsic to the module.
- **Additive**: prefer new modules + context-managed estimator swaps (mirror
  `_sota_estimator` / `_sota_solar_estimator`) over editing guarded code; mirror
  the `solar_modulation.py` house style.
- Preserve invariants: `mean_h f_H = 1`, period-mean = anchor (energy
  conservation), `f_W` layer untouched unless that IS your lever.

## 8. Anti-overfitting discipline (critical — only ONE fully-realized year)
Cal 2025 is the only fully-realized delivery year in the EPEX coverage. Therefore:
- Do NOT tune to a single vintage or single year. Use **leave-one-year-out** or
  **walk-forward** evaluation across the 12 vintages where the diagnostic allows,
  and across seasons.
- Prefer **parsimonious** parameterizations (few free params, ridge/shrinkage) —
  Harrell's 10-20 obs/param rule applies; the repo already follows it
  (cf. solar's 4-block design for n≈39 months).
- Report effect sizes with uncertainty; a small, robust, well-justified gain beats
  a large in-sample one.

## 9. Validation (a primary deliverable)
- Add your variant as a new estimator (e.g. `sota_<lever>`) in
  `perfect_foresight.py`, mirroring `sota_solar`. Run `--ab` and report per
  Cal-2025 vintage: `pf_cal_corr` (must stay ≥ 0.85 on all 12), diurnal cosine +
  demeaned RMSE, CH-physical sub-KPIs (model vs realized).
- If the lever touches Pillar 1, run `scripts/run_phase10_scorecard.py` (or
  `run_phase10_real.py`) and show the Hildmann gate per-test table is not
  regressed.
- Significance: paired Wilcoxon and/or block-bootstrap CI on the improvement;
  recommend shipping only if the CI excludes zero AND no gate regresses.
- Commit a concise markdown report (in your phase folder) with the tables, the
  byte-identity-OFF check, and the honest verdict.
- A **null result is valid and expected sometimes** — report it with numbers; do
  not p-hack.

## 10. Environment realities
- `data/epex_hourly.parquet` can be bootstrapped from the tracked
  `pfc_shaping/data/epex_15min.parquet` via `resample('1h').mean()` (see
  `scripts/run_phase10_real.py` step 1).
- `data/forwards_history_phase10.parquet` needs the FMV poste
  (`scripts/import_fmv_forwards.py`) and may be absent — the perfect-foresight
  path anchors on realized Cal (needs only EPEX), so prefer it; state which KPIs
  you could vs could not measure.
- Install missing deps (numpy/pandas/scipy/sklearn/holidays/matplotlib/pytest).
- Some tests fail in a fresh clone only due to absent parquets — confirm a failure
  reproduces on the parent commit before attributing it to your change.

## 11. Git workflow
- Feature branch (e.g. `feat/lt-<lever>`).
- Separate commits: research note (§6), implementation, validation report.
- Run the suite; introduce no failures; add tests for new code.
- Push the branch. Do NOT open a PR unless explicitly asked.

## 12. Definition of done
- [ ] Diagnostic re-run; targeted residual quantified from §4 numbers.
- [ ] §6 research note committed.
- [ ] ONE lever implemented behind an OFF-by-default flag; byte-identity OFF
      proven (atol=0).
- [ ] New `sota_<lever>` estimator wired into `--ab`; metrics reported.
- [ ] pf_cal_corr ≥ 0.85 on all 12 vintages; CH-physical sub-KPIs reported;
      Pillar 1 not regressed (if applicable).
- [ ] LOYO / walk-forward used; significance reported; honest ship/null verdict.
- [ ] Tests green; new tests added; branch pushed; no PR.

## 13. Confirm understanding before coding
Reply with a short summary: (a) branch + HEAD you are on; (b) the
OFF-by-default reproducibility rule; (c) the no-leakage rule; (d) the candidate
lever you are leaning toward AND the residual number that motivates it; (e) why
you will NOT redo the §3 work. Then run §4 and write the §6 note before coding.

## 14. If blocked / ambiguous
If the highest-value lever is unclear from the diagnostics, or required data is
absent, STOP after the §6 note and report your proposal + the measurement you
could not run, with your recommendation — never guess into guarded code.
