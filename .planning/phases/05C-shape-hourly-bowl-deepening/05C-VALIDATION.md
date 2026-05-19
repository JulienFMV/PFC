---
phase: 5bis-B
slug: shape-hourly-bowl-deepening
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-19
---

# Phase 5bis-B — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Full Validation Architecture lives in `05C-RESEARCH.md`. This file is the
> per-task verification map filled in by the planner.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (déjà en place via 5bis-A) |
| **Config file** | `pytest.ini` (déjà fonctionnel) |
| **Quick run command** | `pytest tests/test_shape_hourly_bowl.py -x -q` |
| **Full suite command** | `pytest tests/ -q` |
| **Estimated runtime** | ~45 secondes (suite 247 tests + 7 nouveaux 5bis-B) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_shape_hourly_bowl.py -x -q`
- **After every plan wave:** Run `pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green (atol=1e-12 contract)
- **Max feedback latency:** ~10 secondes (quick run)

---

## Per-Task Verification Map

> Filled by gsd-planner during plan generation.
> Each task in a PLAN.md MUST reference a row here OR declare `<automated>` inline.

| Task ID | Plan | Wave | Requirement | Test | Type | Automated Command | Status |
|---------|------|------|-------------|------|------|-------------------|--------|
| TBD-by-planner | 01-03 | 1-3 | D-A1..A4 / SC#1..5 | see RESEARCH §Validation Architecture | unit/regression | `pytest tests/test_shape_hourly_bowl.py::<name>` | ⬜ pending |

---

## Wave 0 Requirements

- [ ] `tests/test_shape_hourly_bowl.py` — empty test file scaffolded in Plan 05C-01 Task 1
- [ ] `tests/fixtures/_generate_bowl_fixture.py` — deterministic synth duck-curve generator (seed=42)
- [ ] `tests/fixtures/bowl_seed42.parquet` — output of `_generate_bowl_fixture.py` (~50KB)
- [ ] `tests/fixtures/baseline_pfc_seed42_bowl.parquet` — generated in Plan 05C-03 Wave 0 (frozen flag=ON output)
- [ ] `np.ptp` threshold CAL — Plan 05C-01 Wave 0 dry-run + commit measured threshold (plancher 1.05)
- [ ] `M+30 amplitude` threshold CAL — Plan 05C-02 Wave 0 dry-run (plancher 0.50)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real HFC OMPEX block-MAE delta | Phase 10 (deferred) | Requires `H:\` access (FMV station / Databricks Gold) | Phase 10 plan — out of scope here |

*All in-scope 5bis-B behaviors have automated verification on synthetic fixture.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (fixture generator, baseline_pfc_seed42_bowl, thresholds)
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s (quick run)
- [ ] `nyquist_compliant: true` set in frontmatter at plan-checker pass

**Approval:** pending
