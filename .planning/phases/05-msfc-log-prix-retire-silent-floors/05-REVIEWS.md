---
phase: 5
reviewers: [codex]
reviewed_at: 2026-05-19T20:14:25Z
plans_reviewed: [05-01-PLAN.md, 05-02-PLAN.md, 05-03-PLAN.md]
skipped:
  - reviewer: claude
    reason: self-CLI (running inside Claude Code)
  - reviewer: gemini
    reason: user request (output quality judged medium)
---

# Cross-AI Plan Review — Phase 5 (MSFC retire silent floors + PFC peut être négative)

## Codex Review

### Plan 05-01 — Summary
Plan 05-01 is strong on scoping and sequencing: it isolates MSFC/clamp + arbitrage floor behavior + NEG-05 wording reformulation before touching WV/cascading math. The explicit focus on the **two** MSFC floors and `converged=False` propagation addresses the main hidden failure modes. Biggest risk is test scaffolding complexity (10-test scaffold in wave 1) and potential brittleness in the new signed clamp edge cases if knot cardinality/range degenerates.

### Plan 05-02 — Summary
Plan 05-02 correctly tackles the core semantic fix (`delta_wv = (f_wv - 1) * |B|`) and explicitly handles legacy rollback via `enforce_floor=True`. It also anticipates double-damping pitfalls in `assembler.py`. Main risk is behavioral drift from the additive refactor interacting with existing `shape_freedom` logic and downstream assumptions on `f_WV` semantics/columns.

### Plan 05-03 — Summary
Plan 05-03 is comprehensive and closes the phase with cascading spread-additive migration, master-flag audit trail, fixture/baseline generation, and production callsite updates. It is well thought out but dense; the main risk is integration fragility from many coupled edits in one wave (cascading semantics + assembler ctor surface + pipeline callsites + binary fixtures + tests).

---

### Strengths
- Clear wave decomposition with dependency order aligned to math risk.
- Good identification of hidden floors:
  - `pfc_shaping/lt/model/msfc_spline.py:131` and `:203`
  - `pfc_shaping/calibration/arbitrage_free.py:517`
  - `pfc_shaping/lt/model/water_value.py:394,407`
- Strong backward-compat intent via explicit rollback knobs (`enforce_*`, `allow_negative_peak`) and dedicated regression test.
- Correct recognition that WV multiplicative→additive is non-no-op; new baseline strategy is justified.
- Good acceptance gate for SC #2 dependent on 5bis-B marker instead of false negatives.
- Explicit production callsite audit and migration off `fit_peak_ratios` is pragmatic.

---

### Concerns
- **[HIGH][Plan 05-02]** `compute_delta_wv(..., fill_df, calendar_df)` parameter naming/order is easy to misuse (existing `apply(timestamps, calendar_df, hydro_forecast)` is different). Risk of swapped args in `assembler.build()`.
- **[HIGH][Plan 05-03]** `ContractCascader.fit_peak_ratios` deprecation shim may break callers that read `peak_base_ratios_` directly if shim no longer populates it (stated behavior is inconsistent across sections).
- **[MEDIUM][cross-cutting]** Clamp formula `margin = 0.5 * np.ptp(y_knots)` needs explicit guard for pathological knot sets:
  - all-equal knots (`ptp=0`)
  - too few knots for PCHIP extrapolation assumptions.
- **[MEDIUM][Plan 05-01]** `converged=False` propagation when clip hits is correct for NEG-02, but may increase false non-convergence in noisy positive regimes unless logging/reporting clarifies "floor-induced".
- **[MEDIUM][Plan 05-03]** Scope is broad (code + pipeline + fixtures + docs + 6 tests + baseline binaries); merge conflict/integration risk is high.
- **[MEDIUM][cross-cutting]** UTC vs local-time assumptions for monthly means and Sunday h13 checks need explicit normalization in tests (especially July mask and `type_jour` derivation).
- **[LOW][Plan 05-01]** Creating 10-test scaffold early is useful for visibility but adds maintenance overhead and noisy skips.
- **[LOW][Plan 05-03]** Master flag is audit-only while adding ctor overrides at assembler level; this is good, but docs must avoid ambiguity about precedence vs component ctor args.

---

### Suggestions
- Add explicit edge tests in `tests/test_phase05_negative_prices.py`:
  - `test_msfc_clamp_all_equal_knots` for `pfc_shaping/lt/model/msfc_spline.py:120`.
  - `test_msfc_clamp_all_negative_knots_no_inverted_bounds`.
- In `pfc_shaping/lt/model/water_value.py`, enforce keyword-only args for `compute_delta_wv` to prevent swapped `fill_df/calendar_df`.
- In `pfc_shaping/lt/model/assembler.py` (WV branch), assert index alignment before adding:
  - `assert delta_wv.index.equals(B.index)`.
- In `pfc_shaping/calibration/cascading.py`, keep transitional compatibility by populating both:
  - `peak_base_spreads_` (new)
  - `peak_base_ratios_` (derived) during deprecation window.
- In `tests/test_phase05_negative_prices.py::test_phase05_summer_bowl_negative_acceptance`, derive Sunday/h13 from index timezone explicitly (UTC->local conversion) before filtering.
- Split Plan 05-03 execution into two commits:
  1. cascading + pipeline migrations
  2. master flag + fixtures/baselines/tests/docs
  This reduces rollback/debug cost.
- For legacy regression (`baseline_pfc_seed42.parquet`), pin all four rollback kwargs at `PFCAssembler(...)` and assert each component state in test preconditions.
- Add a test for empty/insufficient `spot_history` in `fit_peak_spreads` with deterministic fallback spread and warning assertion.

---

### Risk Assessment
**Overall risk: MEDIUM-HIGH**

Justification: architecture and math direction are solid, but Plan 05-03 bundles many interdependent changes, and WV/cascading semantic flips can create subtle regressions despite tests. Risk is manageable if you harden edge-case tests (knot degeneracy, timezone filtering, empty spot history) and reduce integration batch size in wave 3.

---

## Consensus Summary

Single-reviewer pass (Codex). Synthesized below as actionable themes rather than cross-reviewer consensus, since `claude` was skipped (self-CLI) and `gemini` was skipped per user request.

### Strengths (highlighted by reviewer)
- Wave decomposition aligned to math risk (clamp → additive WV → cascading + flag).
- Both MSFC floors (lignes 131 et 203) explicitly conditioned on `enforce_positivity`.
- `converged=False` propagation when m_factor clip fires (NEG-02 littéral).
- Explicit rollback knobs at all 4 callsites + dedicated baseline regression test.
- New baseline `baseline_pfc_seed42_phase05.parquet` justified by D-A3-3 dry-run.
- SC #2 acceptance gated on 5bis-B bowl marker (no false negatives).

### Top concerns (priority order for `--reviews` ingestion)
1. **[HIGH][Plan 05-02] `compute_delta_wv` signature ergonomics** — argument order `(B_smooth, fill_df, calendar_df)` differs from `apply(timestamps, calendar_df, hydro_forecast)`. Easy to swap args at the `assembler.build()` callsite. Mitigation: keyword-only args + a docstring example.
2. **[HIGH][Plan 05-03] `fit_peak_ratios` shim contract** — the plan mentions both "transparent redirection to `fit_peak_spreads`" and "raise NotImplementedError if no spot_history". Need a single authoritative behavior, ideally: shim ALWAYS calls `fit_peak_spreads` AND populates `peak_base_ratios_` derived during the deprecation window, so legacy attribute readers don't break.
3. **[MEDIUM][cross-cutting] Clamp degenerate cases** — `margin = 0.5 * np.ptp(y_knots)` is 0 when all knots are equal; this leaves `np.clip(x, k, k)` (pinning to a constant). Add an explicit floor on margin (e.g. `max(0.5*ptp, 1.0)` or `max(0.5*ptp, 0.5*abs(median(y_knots)))`) and cover with `test_msfc_clamp_all_equal_knots`.
4. **[MEDIUM][cross-cutting] Timezone handling in acceptance test** — `test_phase05_summer_bowl_negative_acceptance` needs explicit UTC→Europe/Zurich conversion before applying `type_jour=='Dimanche'` and `heure==13` masks. The PFC index is UTC; `h13 Sunday` is local-time semantics.
5. **[MEDIUM][Plan 05-03] Wave 3 scope** — bundling cascading + pipeline migrations + master flag + binary fixtures + 6 tests + docs into one wave is dense. Suggest splitting into two commits within the same wave (no plan restructuring).
6. **[MEDIUM][Plan 05-01] `converged=False` semantics** — propagating `converged=False` when the m_factor floor fires is correct for NEG-02, but logging should distinguish "floor-induced" from "true non-convergence" to avoid confusing operators in positive-only regimes.

### Suggested edge tests to add to `tests/test_phase05_negative_prices.py`
- `test_msfc_clamp_all_equal_knots` (knots constant → margin=0, clamp ≠ pinning).
- `test_msfc_clamp_all_negative_knots_no_inverted_bounds`.
- `test_fit_peak_spreads_empty_spot_history` (fallback spread + warning).
- `test_compute_delta_wv_index_alignment` (assertion on `B.index.equals(delta_wv.index)`).
- Optional: `test_fit_peak_ratios_deprecation_populates_ratios_attribute` (verifies legacy attribute survives the shim).

### Divergent views
N/A (single reviewer).

### Action items for `/gsd:plan-phase 5 --reviews`
| # | Priority | Plan | Action |
|---|----------|------|--------|
| 1 | HIGH | 05-02 | Make `compute_delta_wv` keyword-only after `B_smooth`; add `assert delta_wv.index.equals(B.index)` in assembler. |
| 2 | HIGH | 05-03 | Unify `fit_peak_ratios` shim contract: always redirect + populate `peak_base_ratios_` derived during deprecation window. Update RESEARCH.md §Pattern 6 + 05-03 plan tasks accordingly. |
| 3 | MEDIUM | 05-01 | Add `margin` floor for degenerate knot sets in `msfc_spline.py:120` clamp; cover with two new edge tests. |
| 4 | MEDIUM | 05-03 | Document explicit UTC→local conversion in `test_phase05_summer_bowl_negative_acceptance` before applying Sunday/h13 masks. |
| 5 | MEDIUM | 05-03 | Split wave 3 into two commits (cascading+pipeline / flag+fixtures+tests+docs). No plan restructuring — commit boundary only. |
| 6 | MEDIUM | 05-01 | Differentiate "floor-induced" vs "true non-convergence" in `converged=False` logging (NEG-02). |
| 7 | LOW | 05-03 | Add `test_fit_peak_spreads_empty_spot_history` for fallback spread + warning. |

---

*Reviewers actually invoked: codex.*
*Skipped: claude (self-CLI inside Claude Code), gemini (user opted out).*
