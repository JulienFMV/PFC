# Session handoff - CH LT laptop model qualification v2

Date: 2026-07-31  
Branch: `fix/lt-audit-remediation`  
HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Workspace: `C:\Users\jbattaglia\PFC_LT`  
Production: strict `NO_GO`

## Outcome

The LT model now executes twice to terminal manifests on the managed FMV
laptop as a standard user, without admin, elevation, Defender exception,
project executable, Playwright, network installation or mutable path outside
the canonical repository.  The selected v2 pair uses an explicit opt-in to the
experimental sparse intraday regularization; the default/flag-OFF path retains
the frozen historical baseline byte-for-byte at the numerical contract.

This is local engineering evidence only.  It is source-checkout execution, not
installed-wheel model qualification.  Levels remain `TEST_FIXTURE`; intraday
shape remains a mixed-authority DE-LU proxy; slow/central/fast are structural
paths, not calibrated probabilities; P10/P90 contain zero non-null rows; there
are zero independent prospective origins and no direct governed CH 15-minute
truth.  T057 was not consumed.  No candidate, publication, promotion or
production transition occurred.

## Selected evidence

- Current selection:
  `.planning/phases/14-lt-audit-remediation/CH-LT-LOCAL-QUALITY-CURRENT-SELECTION-V2-20260731.json`
  - physical SHA-256:
    `e9cc9f7db6274df37102e7d72e6fc0596d6067a4209a219cf8bcb0ed1361cd69`
  - selection ID:
    `93504ab8a9c6b1f063fdd31029fae23264a59a85aa1ded291bdabcfd7e40c828`
- Pair qualification:
  `build/local-model-qualification/ch-lt-laptop-model-pair-v2`
  - qualification ID:
    `7bca8042742e3054144bf798b048cae2acce43417cd2d2308389a6f92faedf95`
  - `quality.json` SHA-256:
    `a3cdf2543ff5bc74990d46516df49be5dbf851090d704750ece6c1ca04a2f844`
  - `QUALITY.md` SHA-256:
    `7941598f00a352d4aec69ca9666d153c76fb7d8a780f6a32f51018380ebbce0b`
  - terminal receipt SHA-256:
    `2a254cc4c4305b2a65097c59aaf440098e789cc6178e61555f9adf71bdd0d0d5`
  - supervised audit receipt `qa31pairv2` SHA-256:
    `9e8289710f7935d11bd732b653eb0d0a194311fff7b36b289f103abe1fa56762`
- Current-selection verifier `qsel31v4` execution/supervisor receipt SHA-256:
  `91678e84ee1cacb5c8135b9abd9c859def88556e34de2cd61af783f6955a8ba4` /
  `5154c276bc306588d4703ada07e16161e20bc60bab17c207e03a4fd79bb3c127`.

## Model pair

Run A is `build/local-model-runs/run31e2`, supervised by `mdl31e2`:

- execution receipt SHA-256:
  `563ce1bf9421ecda4957b431548f1bafd8cafa9944f8597695fb1fe7f881af99`;
- outer supervisor receipt SHA-256:
  `636c72af09094700a1e9225adb5ef647adfe3e2c419675bf9d49d8fec8a0aa40`;
- completion manifest SHA-256:
  `e9d486171a2ea85dbd36e046c16231aced2c72b3de77aa2ec7220ec5e7be753c`;
- target wall 46.484 s; peak Job memory 7,105,343,488 bytes;
- zero active processes after wait.

Run B is `build/local-model-runs/run31f2`, supervised by `mdl31f2`:

- execution receipt SHA-256:
  `6c3fb8fa4717446e3ed92c8137f50c739808601d661154a15b20ffd7fcd201ba`;
- outer supervisor receipt SHA-256:
  `ed684d9e5e3e197a4ff109e046809d98fd6801cb0a561a45ede6faf1aeab8275`;
- completion manifest SHA-256:
  `75721ad892fc482d72d9c6cf004bd2fefe1dff10365f6178dae1ea036c51571c`;
- target wall 46.844 s; peak Job memory 7,080,230,912 bytes;
- zero active processes after wait.

Both receipts bind source tree SHA-256
`7a90d8fbec9bb3443b066ba49e914a61c73568e39f9f6fe0060b04eae0e120fa`,
interpreter SHA-256
`50bfb90ee93bb0cb51175b546f133798dfe4b778677d95d81391e7bf6d85e5ac`,
closure `BOUND_REPO_LOCAL_PTH`, five `sys.path` entries and the canonical
checkout root exactly once.  Eight material output roles are byte-identical;
only path-bearing `QUALITY.md` differs and passes bounded path normalization.

The selected material hashes are:

| Role | SHA-256 |
|---|---|
| expanded scenarios | `088ec78771e2348a54e338c9da453b5ccfccddcefb304c5511c5a5271a4dceab` |
| governance report | `9f18bcd2f01592f3bd7b4180813fd3e19beaff18d7030b724f71d2265848b719` |
| monthly manifest | `866f3469d58b2f18503c398a8ff917b074927502dadd130b1af7e4bc5961fab2` |
| PFC central | `b109acc1173ae4effdba8d70ac38700cbea5e7802cd405858f172d89935685a3` |
| PFC fast | `083ba5aa6fd64a6eec520cb4bfcd64f50db9ee492e10e88f69e10f866a9919af` |
| PFC slow | `04c05490e0fe61a5f839887377080e053a31297e10fd0796ed7962996cff386f` |
| scenario features | `f82da93a30ddefee5aeb2b570665ed5fde90ce9263e01b9a8b4d9da3ea7c1f10` |
| structural fan | `e8d58af0dd637365d656e7363c2f0b658bc0810cd1c00a635210a1d711cd0484` |

## Structural results and quality boundary

- 155,616 quarter-hours per scenario, from
  `2026-07-24T22:00:00Z` through `2030-12-31T21:45:00Z`.
- Monthly level authority: `solver`.
- Solver maximum absolute constraint residual:
  `2.8421709430404007e-14` EUR/MWh.
- Maximum monthly mean residual:
  `5.4924953474255744e-11` EUR/MWh.
- Maximum parent-hour quarter-factor neutrality residual:
  `2.220446049250313e-16`.
- Solver stationarity residual: `1.5719360463372172e-13`.
- Weighted mean: `85.48362297717911` EUR/MWh.
- Mean structural spread: `0.5603787968565872` EUR/MWh; maximum:
  `3.033323074306338` EUR/MWh.
- P10/P90: zero non-null rows in every scenario.

The linked historical diagnostic is not a CH LT validation.  Its target is
DE-LU quarter-hour price conditional on the realized parent-hour mean.  It has
16 historical origins, candidate/incumbent MAE 5.6818/5.7831 EUR/MWh, 6 wins,
6 losses, 4 ties and 1.752% relative gain; the latest fold is not non-inferior
and `candidate_accepted=false`.  The current CH ledger still has zero
independent origins and explicitly forbids price, shaping and economic
inference.

## Demonstrated failures and fixes

1. `mdl31b` reached the target and failed because the historical intraday
   manifest is schema v1 while the builder required v3.  Receipt SHA-256:
   `c97474e41da4524891f84938af39f1f73fbe777e3d17733a88e0a27fb837fe5c`.
   The fix accepts v1 only when the complete manifest has the exact pinned
   SHA-256, exact field set and exact panel hash; altered bytes/newline fail.
2. The first integrated matrix found three frozen-baseline failures.  The
   causal change rounded maturity to the parent hour even with sparse
   regularization OFF, changing exactly three of four quarter-hours.  The fix
   restores timestamp maturity when OFF and uses parent-hour maturity only
   when the experimental flag is ON.  The flag is now mandatory and exact in
   the supervised local-model route.
3. `mdl31e` was rejected before execution because PowerShell split the CSV
   weights.  Receipt SHA-256:
   `06a96a3cdf7b75ab5de9328553541e0bf54e0851ad2a4f4b91c46af21911f296`.
   Runs `e2/f2` use quoted CSV arguments.
4. `lmqmat31b` correctly failed closed after the source fix invalidated the
   old v1 qualification hash: 675 passed, 16 skipped, 2 deselected, one
   failure.  Receipt SHA-256:
   `182cae7bbf6ea27bb7b1ee47b39c29fcce07760215227eb06f560a153822a622`.
   No old evidence was rebound; the model pair and qualification were rerun
   as v2.
5. `qsel31v3` was deliberately left as negative timeout evidence after the
   calling tool stopped waiting while the source tree was still changing.
   The supervisor reached its 300-second wall budget, terminated the tree and
   reported zero active processes.  Execution/supervisor receipt SHA-256:
   `8e1fbc5ea95c4964129c198c949e52b20b6870a0daf1fb7269d7198e9b7d3b43` /
   `3e3dab2c36e08f35faa75cfa015feeea2f6e8740597654ae337d1cf61b11ea29`.
   It is not a positive verifier receipt; `qsel31v4` supersedes it.
6. The first final IT and Quant roasts found that the current selection did
   not bind the outer A/B supervisor receipts and that upstream ledger,
   registry and EEX documents were mainly hash-pinned rather than
   semantically revalidated.  The selected instance now binds both supervisor
   receipts and checks outer deadline/capability/cleanup/authority, plus exact
   upstream schemas, statuses, zero-origin/T057/training/science/candidate/
   publication/promotion boundaries and forbidden claims.  `qsel31v4` and
   `lmqmat31d` are the post-fix positive receipts.

## Exact command pattern

Every shell invocation began with the literal cwd/Git-root guard required by
`AGENTS.md`.  The model target command recorded verbatim in each execution
receipt was:

```powershell
build\pytest-runtime-v1\python.exe -B -m scripts.build_local_test_ch_pfc --inventory C:\Users\jbattaglia\PFC_LT\data\electrification_scenarios_prod_candidate_neutralized_2030.parquet --manifest C:\Users\jbattaglia\PFC_LT\.planning\phases\13-lt-electrification-scenario-shape\SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml --governance-report <run-root>\LOCAL-TEST-GOVERNANCE-GATE.md --vintage 2026-06-12 --scenarios "slow,central,fast" --weights "0.25,0.50,0.25" --market CH --start-date 2026-07-24T22:00:00 --horizon-days 1621 --epex-hourly C:\Users\jbattaglia\PFC_LT\data\epex_hourly.parquet --intraday-epex C:\Users\jbattaglia\PFC_LT\output\phase14\prospective_intraday_panel_20260724\epex_de_15min_complete.parquet --intraday-market DE --intraday-cutoff 2025-10-01T00:00:00Z --intraday-provenance-manifest C:\Users\jbattaglia\PFC_LT\output\phase14\prospective_intraday_panel_20260724\panel-manifest.json --expected-intraday-provenance-sha256 af8232e7045eb9aa685d946770a56a108e438a6a71e9cbdab47614abcc5a9e4f --require-nonflat-intraday-shape --require-direct-intraday-seasons "Hiver,Printemps,Ete,Automne" --forwards C:\Users\jbattaglia\PFC_LT\data\eex_forwards_history.parquet --expanded-output <run-root>\scenario_expanded.parquet --features-output <run-root>\hpfc_scenario_features.parquet --output-dir <run-root> --output-prefix local-pfc-smoke --fan-chart-output <run-root>\structural_fan_chart.parquet --summary <run-root>\QUALITY.md --completion-manifest <run-root>\completion_manifest.json --enable-monthly-forward-curve-solver --monthly-solver-constraint-tolerance 1e-9 --monthly-solver-lambda-smooth-month 1.0 --monthly-solver-lambda-smooth-yoy 0.25 --monthly-solver-lambda-shape 1.0 --monthly-solver-neighbor-shrinkage 0.5 --monthly-solver-allow-template-structural-fallback --monthly-solver-structural-amplitude 110.0 --disable-cascade-trend-for-annual-only --enable-experimental-sparse-intraday-regularization
```

It was supervised with `--run-id mdl31e2` / run root `run31e2`, then
`--run-id mdl31f2` / run root `run31f2`.

Pair publication and current verification:

```powershell
build\pytest-runtime-v1\python.exe -I -B -m scripts.run_workspace_local --run-id qa31pairv2 --wall-timeout-seconds 300 -- build\pytest-runtime-v1\python.exe -B -m scripts.audit_ch_lt_laptop_model_pair --output-directory C:\Users\jbattaglia\PFC_LT\build\local-model-qualification\ch-lt-laptop-model-pair-v2
build\pytest-runtime-v1\python.exe -I -B -m scripts.run_workspace_local --run-id qsel31v4 --wall-timeout-seconds 600 -- build\pytest-runtime-v1\python.exe -B -m scripts.audit_ch_lt_local_quality_current_selection
```

The final integrated command used `run_workspace_local --run-id lmqmat31d`
and pytest over the runner/model/auditors, monthly constraints/solver/priors/
integration/audit, arbitrage/cascading/shaping/intraday/solar/water/uncertainty,
LT/CT imports, runtime/packaging, publisher closure, external publication and
anchor reference tests with `-m "not slow"`.

Result: `676 passed, 16 skipped, 2 deselected`, 3 existing solar timezone
warnings, zero failures in 130.48 seconds target pytest time.  Receipt SHA-256:
`ee443bc43def6dd916770858fa969714005778ebb4bfae7a041a3a78eaa182d7`;
target Job wall 112.875 seconds, peak 5,327,126,528 bytes, zero active members,
no timeout/interruption and tree termination confirmed.  Import closure is
`BOUND_REPO_LOCAL_PTH`; workspace root count is exactly one of five entries.

Focused tests: `165 passed`, then pair/runner `141 passed`, then current-chain
`143 passed`.  Ruff passes on the changed Python files via repo `.venv`.
`git diff --check` passes.  Nothing is staged.  Protected
`data/eex_forwards_history.parquet` remains SHA-256
`21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

## Files changed for this slice

- `AGENTS.md`
- `pfc_shaping/lt/model/shape_intraday.py`
- `scripts/run_workspace_local.py`
- `scripts/build_local_test_ch_pfc.py`
- `scripts/audit_ch_lt_laptop_model_pair.py`
- `scripts/audit_ch_lt_local_quality_current_selection.py`
- `tests/test_run_workspace_local_script.py`
- `tests/test_build_local_test_ch_pfc_script.py`
- `tests/test_audit_ch_lt_laptop_model_pair_script.py`
- `tests/test_audit_ch_lt_local_quality_current_selection_script.py`
- `tests/test_intraday_amplitude.py`
- `CH-LT-LOCAL-QUALITY-CURRENT-SELECTION-V1-20260731.json` retained as
  superseded local history
- `CH-LT-LOCAL-QUALITY-CURRENT-SELECTION-V2-20260731.json` current
- root handoff, decision log, external-CAS RFC and this handoff

The worktree remains intentionally very dirty; unrelated user changes were not
reset, cleaned, restored, staged or committed.  No CT or Power BI file was
touched by this slice.

## Independent roasts

- Security/Governance: P0/P1/P2 `0/0/3`; live
  checkout/ABA TOCTOU, unsigned same-user evidence without external CAS/time,
  and skipped installed/publisher/symlink cases remain non-authoritative.
- IT/Operations: P0/P1/P2 `0/0/4` after the outer-supervisor binding closed
  the fifth P2.  Remaining debt is installed-wheel execution, prefrozen SLO/
  capacity, resumable power-loss recovery/CAS, and exhaustive filesystem
  confinement/telemetry.
- Quant/Data: P0/P1/P2 `0/0/3` after semantic upstream validation closed one
  P2.  The exact `2.22e-16` result is factor neutrality, not full price
  neutrality: the independent counterfactual found maximum/p99 pre-calibration
  hourly residual `0.001613` / `0.000117` EUR/MWh.  Direct support is 192/480
  cells with 60% fallback, median 88 QH = 22 parent hours, nine contractions
  and envelope 0.3403.  The v2 instance intentionally still uses stable
  schema `v1`; this instance/schema distinction needs clearer naming.

All roasts accept only local structural engineering evidence.  Scientific
admission, model selection, publication, promotion and production remain
strictly `NO_GO`.

## Next work

1. Keep runtime v40 as the selected installed packaging runtime and the v2
   pair as source-checkout laptop evidence; do not claim installed-wheel model
   execution.
2. Define and register a new outcome-blind future prediction before truth,
   with external trusted time/signature, independent registry/CAS/WORM and a
   builder-inaccessible commitment.
3. Accumulate fresh direct-CH point-in-time origins; retain hourly truth until
   the independently verified Swiss 15-minute transition is admitted.
4. Run pre-registered rolling-origin validation and a new holdout successor;
   old T057 is never confirmatory again.
5. Calibrate probabilistic P10/P50/P90/scenarios and validate coverage,
   sharpness, dependence and capture-price/product metrics.
6. Only after scientific evidence, close installed-wheel model execution,
   cold/warm capacity SLOs, CI/ASR/SBOM, external CAS publication, recovery,
   observability and rollback.  Production remains strict `NO_GO`.
