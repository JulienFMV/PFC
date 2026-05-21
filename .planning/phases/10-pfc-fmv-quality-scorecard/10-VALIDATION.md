---
phase: 10
slug: backtest-par-bloc-vs-hfc-ompex
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-20
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Extracted from `10-RESEARCH.md` §Validation Architecture (lignes 917-957).
> Phase 10 = PFC FMV Quality Scorecard (5-pillar SOTA replication). SC#1 Hildmann 4/4 PASS sur Config 4 = UNIQUE GATE.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.2 (pinned in `pfc_shaping/requirements.txt:34`) |
| **Config file** | None (pytest defaults) — `tests/conftest.py` provides autouse env hygiene (`PFC_LT_*` snapshot/restore per test, héritage 5bis-A D-12) |
| **Quick run command** | `pytest tests/test_phase10_hildmann.py -x` |
| **Full Phase 10 suite** | `pytest tests/test_phase10_*.py -x -v` |
| **Full repo suite** | `pytest tests/ -x` (must keep ≥ 272 passing post-Phase 5) |
| **Estimated quick runtime** | ~5-15s (unit) / ~30-60s (SC#1 integration mock) |
| **Estimated full runtime CI mock** | ~30-90s |
| **Estimated full runtime real-run** | ~1.5-2h Mac Mini (96-build ablation grid, per RESEARCH Pitfall 7 mitigation : `with_uncertainty=True` Config 4 only) |

---

## Sampling Rate

- **After every task commit (atomic, gsd-executor mode no-worktree) :** Run `pytest tests/test_phase10_<pillar>.py -x` (~5-15s unit ; ~30-60s SC#1 integration with mock data)
- **After every plan wave (Plans 10-01..10-04) :** Run `pytest tests/test_phase10_*.py -x -v` (~30-90s CI mock)
- **Before `/gsd:verify-work` (phase gate) :** Full suite `pytest tests/ -x` (≥ 272 passing) + dedicated real-run scorecard Mac Mini (~1.5-2h)
- **Max feedback latency :** ~60s pour les tests unitaires ; ~2h pour le real-run scorecard final (gated)

---

## Per-Task Verification Map

| Req ID (post-reformulation Plan 10-01) | Behavior | Plan | Wave | Test Type | Automated Command | File Exists | Status |
|--------|----------|------|------|-----------|-------------------|-------------|--------|
| BT-01 (preserved) | Block schema {hours, dow, months} accepted by harness | 10-01 | 1 | unit | `pytest tests/test_phase10_empirical.py::test_block_mask_schema -x` | ❌ Wave 0 | ⬜ pending |
| BT-02 (preserved, blocs renamed per D-A2-2) | 5 blocs renamés tz Europe/Zurich → boolean mask correct | 10-01 | 1 | unit | `pytest tests/test_phase10_empirical.py::test_5_blocks_mask_correct -x` | ❌ Wave 0 | ⬜ pending |
| BT-04 (reformulated DM vs 3 naive baselines) | DM stat + p-value computed on synth (PFC better than baseline) | 10-03 | 3 | unit | `pytest tests/test_phase10_dm.py::test_dm_basic_synthetic -x` | ❌ Wave 0 | ⬜ pending |
| BT-06 (NEW Pillar 1.1) | Hildmann arb-free `\|mean(PFC) - forward\|` < 0.01 €/MWh PASS Config 4 | 10-02 | 2 | integration (= SC#1 gate) | `pytest tests/test_phase10_hildmann.py::test_phase10_sc1_arb_free -x` | ❌ Wave 0 | ⬜ pending |
| BT-07 (NEW Pillar 1.2) | Hildmann holiday-weekend ratio ∈ threshold (résolu Plan 10-01) PASS Config 4 | 10-02 | 2 | integration (= SC#1 gate) | `pytest tests/test_phase10_hildmann.py::test_phase10_sc1_holiday_weekend -x` | ❌ Wave 0 | ⬜ pending |
| BT-08 (NEW Pillar 1.3) | Hildmann seasonal corr > 0.85 PASS Config 4 | 10-02 | 2 | integration (= SC#1 gate) | `pytest tests/test_phase10_hildmann.py::test_phase10_sc1_seasonal_profile -x` | ❌ Wave 0 | ⬜ pending |
| BT-09 (NEW Pillar 1.4) | Hildmann continuity max jump < 2 €/MWh PASS Config 4 | 10-02 | 2 | integration (= SC#1 gate) | `pytest tests/test_phase10_hildmann.py::test_phase10_sc1_continuity -x` | ❌ Wave 0 | ⬜ pending |
| BT-10 (NEW Pillar 3 sanity) | Christoffersen LR_uc binomial test correctness on synth | 10-03 | 3 | unit | `pytest tests/test_phase10_probabilistic.py::test_lr_uc_synth -x` | ❌ Wave 0 | ⬜ pending |

> **Note BT-04** : reformulé per CONTEXT D-A0-2 (était "DM test markdown table vs HFC OMPEX" → devient "DM test vs 3 naive baselines"). BT-03 (HFC OMPEX path) et BT-05 (Δ MAE ≤ -1.5 €/MWh vs OMPEX) sont migrés vers BT-10B-* group (Phase 10B deferred, requires FMV poste H:\).

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_phase10_hildmann.py` — covers BT-06..BT-09 (4 SC#1 gate tests + module-scoped fixture qui build une PFC Config 4 seed=42 une fois et la partage)
- [ ] `tests/test_phase10_empirical.py` — covers BT-01, BT-02 (block masks schema + 5-blocs correctness) + Pillar 2 KPIs sanity (MAE/RMSE/bias/MZ unit)
- [ ] `tests/test_phase10_probabilistic.py` — covers BT-10 (Christoffersen LR_uc + degenerate edge cases — zero violations, all violations)
- [ ] `tests/test_phase10_dm.py` — covers BT-04 (DM stat sanity on known-better baseline synth fixture) + variance≤0 fallback (Pitfall 3) + Newey-West HAC HLN correction asserted
- [ ] No new shared fixtures needed — re-use `tests/fixtures/baseline_pfc_seed42*.parquet` for synth PFC + add mocked `epex_hist` via pure-pandas generator (no parquet on disk for CI)
- [ ] CI-mock vs real-run toggle : `@pytest.mark.slow` marker per `pytest.ini` (déféré post-Plan 10-04 si needed ; CI mock = default, real-run via `pytest -m slow`)
- [ ] Framework install : `pip install statsmodels==0.14.6 matplotlib>=3.7` — `checkpoint:human-verify` en Plan 10-01 Task 1 (slopsquat audit pré-install)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| **96-build ablation grid real-run produces a valid `10-VERIFICATION.md`** | BT-10 + SC#1 gate close-out | Real-run Mac Mini ~1.5-2h, dépendant de cache EPEX local + `energy_charts.info` connectivity ; pas testable en CI mock |  Plan 10-04 Task 2 (`python scripts/run_phase10_scorecard.py`) → Plan 10-04 Task 4 `checkpoint:human-verify` user review du scorecard markdown + figures PNG. |
| **D-FLIP-1 flip decision** (set `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` default True dans code + PROJECT.md log entry) | Phase close-out | Decision gated par SC#1 Hildmann 4/4 PASS (automated) ; le flip lui-même est un manual code edit + commit dans Plan 10-04 Task 5 | IF SC#1 PASS → operator flips + commits + PROJECT.md entry ; IF SC#1 FAIL → operator documents blocker in 10-VERIFICATION.md + next steps. |
| **Pillar 5 peer review editorial accuracy** (Table 9×6 + gap analysis 3 paragraphes) | BT-10 | Markdown editorial content sourced from `reference_pfc_state_of_art.md` user-memory + vendor pages (KYOS/Volue/EULER) ; nécessite human review pour exactitude factuelle | Plan 10-04 Task 4 `checkpoint:human-verify` covers cette review. |
| **Pitfall 1 threshold decision** (Hildmann holiday-weekend ratio sur Config 4) | BT-07 | Mesure empirique 2019-2023 EPEX puis décision technique (keep [0.65, 0.95] vs re-calibrate vs exclude PV hours) | Plan 10-01 Task 3b documente la mesure + décision dans `10-01-NOTES.md` ; threshold passé en constante au structural_tests.py (Plan 10-02). |
| **Forwards historical Mac Mini access** (Open Q2) | BT-01 infra | Probe `forwards.eex_report_path` local availability ; fallback derive depuis EPEX-hist Cal/Q/M means si pas accessible | Plan 10-01 Task 3c documente l'accès + fallback dans `10-01-NOTES.md`. |

---

## Validation Sign-Off

- [ ] All tasks have `<acceptance_criteria>` automated verify OR Wave 0 dependencies
- [ ] Sampling continuity : no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (4 new test files + 1 framework install + 1 conftest extension if needed)
- [ ] No watch-mode flags (`pytest` runs one-shot, exit-code-driven)
- [ ] Feedback latency < 60s pour les unit tests ; documenté pour le real-run scorecard (~2h, gated)
- [ ] `nyquist_compliant: true` set in frontmatter after Plan 10-01 Wave 0 tests scaffolded
- [ ] D-A6-3 reproducibility contract enforced via dedicated assert_frame_equal test (re-run sample ≥ 4 builds)
- [ ] D-A3-1 IC95 unconditional coverage : Uncertainty API verified Plan 10-01 expose IC95 OR explicit user-accept defer to Phase 5ter

**Approval :** pending
