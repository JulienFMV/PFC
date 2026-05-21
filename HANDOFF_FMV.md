# Handoff for FMV-side agent — Phase 10B unblock via real EEX forwards

**Created**: 2026-05-21 by Mac Mini agent (Claude Code Opus 4.7, branch `claude/clean-lt-ct-integration`).
**Target audience**: Coding agent (Claude Code / Codex / equivalent) running on Julien's FMV office machine where the real EEX forwards data is downloaded.
**Status**: Phase 10 PFC FMV Quality Scorecard delivered and code-reviewed; verdict currently **DIAGNOSTIC-ONLY** because the only forwards available on the Mac Mini are synthetic (`fallback_diagnostic`). This handoff explains exactly what to do to upgrade the scorecard to **gate-eligible** using real EEX forwards.

---

## 0. Read this first — environment context

- Repo: `https://github.com/JulienFMV/PFC.git`
- Branch: **`claude/clean-lt-ct-integration`** (HEAD = `f0c7a6b` at handoff time). **Do all work on this branch.** Pull latest before starting: `git pull origin claude/clean-lt-ct-integration`.
- Python: **3.12.12** with `statsmodels==0.14.6` required. The Mac Mini system pytest (3.9) does **not** have statsmodels — all commands below assume Python 3.12. On the FMV machine, use whichever Python has the project's `pfc_shaping/requirements.txt` deps installed.
- Test runner: `python -m pytest tests/test_phase10_*.py --tb=short` should report ≥87 passed, 0 failed.
- Project context: this is a long-term Power Forward Curve (PFC) model for FMV SA (Sion, Switzerland) hydroelectric trader/quant. See `.planning/PROJECT.md` for full context and `CLAUDE.md` if present.

---

## 1. The single thing to do

Replace `data/forwards_history_phase10.parquet` (currently 100% synthetic `fallback_diagnostic` rows) with real EEX forward curves snapshotted at the end of each of the **24 vintage timestamps below**, then re-run the scorecard and audit the new verdict.

### 1.1 Required schema

The replacement parquet must have **exactly these 4 columns** (the runner reads them by name):

```python
vintage          datetime64[ns, UTC]   # vintage snapshot timestamp (see § 1.2)
key              object (str)          # delivery period identifier (see § 1.3)
price            float64               # forward price in EUR/MWh
forwards_source  object (str)          # provenance tag, e.g. "eex_xlsx" — anything except "fallback_diagnostic"
```

Index must be `RangeIndex` (no MultiIndex). Save with `df.to_parquet(path, index=False)`.

### 1.2 The 24 expected vintage timestamps (UTC)

Each vintage corresponds to the last EEX trading session of a calendar month. **Must match these exactly** — they are hard-coded in the scorecard's vintage list (`pfc_shaping.validation.scorecard.list_vintages_2024_2025`). Stored as tz-aware UTC `datetime64`:

```
2024-01-31 17:00:00+00:00     2024-07-31 16:00:00+00:00     2025-02-28 17:00:00+00:00     2025-08-29 16:00:00+00:00
2024-02-29 17:00:00+00:00     2024-08-30 16:00:00+00:00     2025-03-31 16:00:00+00:00     2025-09-30 16:00:00+00:00
2024-03-29 17:00:00+00:00     2024-09-30 16:00:00+00:00     2025-04-30 16:00:00+00:00     2025-10-31 17:00:00+00:00
2024-04-30 16:00:00+00:00     2024-10-31 17:00:00+00:00     2025-05-30 16:00:00+00:00     2025-11-28 17:00:00+00:00
2024-05-31 16:00:00+00:00     2024-11-29 17:00:00+00:00     2025-06-30 16:00:00+00:00     2025-12-31 17:00:00+00:00
2024-06-28 16:00:00+00:00     2024-12-31 17:00:00+00:00     2025-07-31 16:00:00+00:00
```

The seasonal hour shift (16:00 vs 17:00 UTC) reflects CET/CEST end-of-day settlement; preserve it. If the real EEX dataset has the snapshot at a slightly different time of day, pin it to these UTC timestamps before saving (vintage equality is checked by exact `Timestamp ==`).

### 1.3 The 48 expected key strings per vintage (144 = 3+9+36 actually checked)

For each vintage you need at minimum the keys the scorecard reads. The current synthetic parquet uses 48 keys per vintage:

- **3 Cal-Y keys**: `"2025"`, `"2026"`, `"2027"` (use the relevant Cal-Y for each vintage horizon)
- **9 Quarter keys**: `"2025-Q1"` … `"2027-Q1"` (Q-keys 1..4 for each Cal-Y, truncated to ≤ vintage+3y)
- **36 Month keys**: `"2024-01"` … `"2027-01"` (M-keys for each month within ≤ 3y horizon from vintage)

The exact set per vintage depends on `derive_forwards_from_epex_hist` in `pfc_shaping/validation/scorecard.py:280-380` (now fixed by WR-03, see commit `4d0c8f0`). The simplest approach: **mirror the key set of the current synthetic parquet vintage-by-vintage** (just replace the `price` column with real EEX values for the same keys).

To verify the key set per vintage:
```python
import pandas as pd
df = pd.read_parquet("data/forwards_history_phase10.parquet")
for v in sorted(df.vintage.unique()):
    sub = df[df.vintage == v]
    print(v, sorted(sub.key.unique()))
```

### 1.4 Source: which EEX curve to use

Per `.planning/PROJECT.md` and Phase 10 RESEARCH, the model targets **EEX Phelix DE-AT Base**. Use the **Base** (24h average) Cal/Quarter/Month curve, not Peak/Off-Peak. The current Phase 10 scorecard's Pillar 1 (Hildmann SC#1 arb-free) and Pillar 3 (Christoffersen IC80) tests will only validate against Base products — Peak/Offpeak keys are silently skipped by `_period_mask`.

---

## 2. Suggested script

Create `scripts/import_fmv_forwards.py` to convert the FMV export into the target schema. Skeleton:

```python
"""scripts/import_fmv_forwards.py — convert FMV EEX export to Phase 10 schema."""
import argparse
import pandas as pd
from pathlib import Path

EXPECTED_VINTAGES = [pd.Timestamp(s, tz="UTC") for s in [
    "2024-01-31 17:00:00", "2024-02-29 17:00:00", "2024-03-29 17:00:00",
    "2024-04-30 16:00:00", "2024-05-31 16:00:00", "2024-06-28 16:00:00",
    "2024-07-31 16:00:00", "2024-08-30 16:00:00", "2024-09-30 16:00:00",
    "2024-10-31 17:00:00", "2024-11-29 17:00:00", "2024-12-31 17:00:00",
    "2025-01-31 17:00:00", "2025-02-28 17:00:00", "2025-03-31 16:00:00",
    "2025-04-30 16:00:00", "2025-05-30 16:00:00", "2025-06-30 16:00:00",
    "2025-07-31 16:00:00", "2025-08-29 16:00:00", "2025-09-30 16:00:00",
    "2025-10-31 17:00:00", "2025-11-28 17:00:00", "2025-12-31 17:00:00",
]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="FMV EEX export (xlsx/csv/parquet — adapt parser below)")
    ap.add_argument("--output", default="data/forwards_history_phase10.parquet")
    args = ap.parse_args()

    # TODO: parse the FMV export format. Expected output rows have:
    #   vintage (UTC timestamp), product_type ('Cal'|'Q'|'M'),
    #   delivery_period (int year for Cal, 'YYYY-QN' for Q, 'YYYY-MM' for M),
    #   price (€/MWh, EEX Base)
    raw = pd.read_excel(args.input) if args.input.endswith((".xlsx", ".xls")) \
          else pd.read_csv(args.input) if args.input.endswith(".csv") \
          else pd.read_parquet(args.input)

    # Map to canonical key strings:
    def to_key(row):
        if row["product_type"] == "Cal":
            return f"{int(row['delivery_period'])}"
        return str(row["delivery_period"])  # "2025-Q1" or "2025-01"

    out = pd.DataFrame({
        "vintage": pd.to_datetime(raw["vintage"], utc=True),
        "key": raw.apply(to_key, axis=1),
        "price": raw["price"].astype("float64"),
        "forwards_source": "eex_xlsx",   # or whatever tag identifies the source
    })

    # Sanity: vintage timestamps must match the expected set exactly
    found_vintages = set(out.vintage.unique())
    expected = set(EXPECTED_VINTAGES)
    missing = expected - found_vintages
    extra = found_vintages - expected
    if missing:
        raise ValueError(f"Missing vintages: {sorted(missing)}")
    if extra:
        print(f"Warning: extra vintages not in scorecard list, will be ignored: {sorted(extra)}")
        out = out[out.vintage.isin(expected)]

    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print(f"  Vintages: {out.vintage.nunique()}, Keys/vintage avg: {len(out)/out.vintage.nunique():.1f}")
    print(f"  forwards_source: {out.forwards_source.value_counts().to_dict()}")

if __name__ == "__main__":
    main()
```

Adapt the parser to the actual FMV export format. Commit the script + the regenerated parquet:
```bash
git add scripts/import_fmv_forwards.py data/forwards_history_phase10.parquet
git commit -m "feat(10B): real EEX forwards from FMV export (replaces fallback_diagnostic)"
```

---

## 3. Re-run the scorecard and audit

### 3.1 Invalidate caches that depend on forwards

The PFC build chain anchors on forwards via `ArbitrageFreeCalibrator` → calibrated `price_shape` per vintage. Replacing forwards requires invalidating **all 96** cached PFCs (4 configs × 24 vintages), not just Config 4:

```bash
rm -rf .planning/phases/10-pfc-fmv-quality-scorecard/cache/
```

This will force a full rebuild (~30 min cold cache on Mac Mini class hardware; FMV machine may be faster).

### 3.2 Re-run

```bash
python scripts/run_phase10_scorecard.py \
    --epex-source parquet \
    --output-dir .planning/phases/10-pfc-fmv-quality-scorecard
```

Outputs to inspect after completion:
- `.planning/phases/10-pfc-fmv-quality-scorecard/10-VERIFICATION.md` — headline verdict + per-pillar tables
- `.planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar{2,3,4}.parquet`
- `.planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar*.png`

### 3.3 Re-run the test suite to confirm no regression

```bash
python -m pytest tests/test_phase10_*.py tests/test_uncertainty_calibration.py --tb=short -q
```

Expected: ≥87 passed (was 87/0 fail before the forwards swap; should stay green since the swap only changes data not code).

### 3.4 IMPORTANT — restore the hand-written Pillar 5 section

`render_markdown_report` overwrites the **Pillar 5 section** of `10-VERIFICATION.md` with a 2-line PLACEHOLDER each time the scorecard runs. The real 87-line Pillar 5 (peer-review SOTA comparative table from Plan 10-04 Task 3) lives in git HEAD. After every re-run, splice it back:

```bash
python -c "
import subprocess
from pathlib import Path
vfile = Path('.planning/phases/10-pfc-fmv-quality-scorecard/10-VERIFICATION.md')
current = vfile.read_text()
head = subprocess.check_output(['git', 'show', f'HEAD:{vfile.as_posix()}'], text=True)
def slc(t, a, b):
    L = t.split('\n'); i = next(i for i,l in enumerate(L) if l.startswith(a))
    j = next(i for i,l in enumerate(L) if i > L.index(L[i]) - 1 + 1 and l.startswith(b))
    return '\n'.join(L[i:j]).rstrip()
head_p5 = slc(head, '## Pillar 5', '## Annexes')
cur_p5 = slc(current, '## Pillar 5', '## Annexes')
vfile.write_text(current.replace(cur_p5, head_p5, 1))
print('Pillar 5 restored.')
"
```

(This is a known wart documented in commit `9418e3a`; a permanent fix is on the post-Phase-10 todo list.)

---

## 4. Audit checkpoints — what to look for

After re-run, the headline change should be:

### 4.1 Pillar 1 (Hildmann SC#1 arb-free) — currently 2/4 PASS DIAGNOSTIC-ONLY

- **`forwards_source` column in VERIFICATION.md Pillar 1 table**: should now show `eex_xlsx` (or your tag) instead of `fallback_diagnostic`.
- **Gate eligibility callout**: the auto-generated text "Diagnostic only — not gate-eligible (forwards derived from EPEX-history fallback)" should be replaced or removed (search `forwards_source` logic in `render_markdown_report` around line 1957 of `pfc_shaping/validation/scorecard.py`).
- **`arb_free` deviation**: currently 22.64 €/MWh (vs 0.01 tol). With real forwards, expect a much smaller value — probably still > 0.01 but should drop to single-digit €/MWh range. If still > 5 €/MWh, that's a real PFC quality issue worth investigating (likely in `ArbitrageFreeCalibrator` convergence on real forwards vs. synthetic).
- **Verdict**: if 4/4 PASS, the SC#1 gate is officially achieved. Update `.planning/STATE.md` and `.planning/ROADMAP.md` to reflect.

### 4.2 Pillar 3 (Christoffersen IC80) — currently 1/5 PASS DIAGNOSTIC-ONLY

The Uncertainty v2 method (commit `c545d4c`) was validated by-construction on synthetic data (in-sample coverage = nominal 20% by empirical quantile property). On real forwards, expect:
- Per-bloc observed_freq to move closer to nominal 0.20 from the current 0.025–0.291 range.
- If 3-5/5 blocs pass LR_uc (p > 0.05) after the swap → v2 calibration is validated end-to-end.
- If still <3/5 pass → real fat-tail problem (winter peaks, gas crisis residue). Then consider Phase 11 EVT-based tail estimation (Peaks-Over-Threshold with Generalized Pareto Distribution fit on the upper/lower tail residuals).
- **Update DIAGNOSTIC-ONLY callout**: section "Pillar 3" in VERIFICATION.md currently has a paragraph starting "Gate eligibility : ⚠ **DIAGNOSTIC-ONLY**". Rewrite or remove once gate-eligible.

### 4.3 Pillar 4 (Diebold-Mariano vs 3 baselines)

- `forwards_flat` baseline currently uses fake forwards as its prediction, so DM comparisons vs PFC are circular (both anchored on same fakes). With real forwards, the `forwards_flat` baseline becomes meaningful.
- Expect `better_than_baseline` ratios to shift; the more skill the real PFC has vs the flat real-forwards baseline, the better.

### 4.4 Pillar 2 (KYOS empirical MAE/RMSE/MZ)

Should be largely unchanged by the forwards swap — Pillar 2 measures realised - PFC residuals where PFC is the same shape pipeline; only the calibrated level differs.

---

## 5. Context: what was just done on the Mac Mini side

The 4 most recent commits (all on the branch you just pulled):

| Hash | Type | What |
|---|---|---|
| `f0c7a6b` | docs | Mark Pillar 3 as DIAGNOSTIC-ONLY (inherits fake forwards) |
| `3fd5abc` | chore | Re-run scorecard post Uncertainty v2 |
| `c545d4c` | fix | Uncertainty v2 — empirical residual quantile method (EPFL rewrite) |
| `9418e3a` | chore | Re-run scorecard post code-review fixes (13 atomic fixes) |

Phase 10 was code-reviewed in depth (Claude + Codex independent passes); 13 fixes were applied (commits `9a20784` → `8d7bc39`). The Uncertainty rewrite (v2) addressed a major bug — v1's `HORIZON_WIDENING ×1.5–4.2` + multiplicative ratio bootstrap produced IC80 bandwidths 10× too wide (1204 €/MWh vs true 115 €/MWh per `data/epex_hourly.parquet` vintage 2024-06-28). v2 uses empirical residual quantiles per (saison, type_jour, heure) cell and is validated by 6 unit tests in `tests/test_uncertainty_calibration.py`.

**Key files modified by this work**:
- `pfc_shaping/lt/model/uncertainty.py` (full rewrite, v2)
- `pfc_shaping/validation/{scorecard.py, structural_tests.py, christoffersen.py, dm_test.py}` (13 code-review fixes)
- `tests/test_phase10_*.py` + `tests/test_uncertainty_calibration.py` (regression + calibration tests)
- `.planning/phases/10-pfc-fmv-quality-scorecard/{10-VERIFICATION.md, 10-REVIEW.md, 10-REVIEW-CODEX.md}`

**Audit finding that motivates this handoff** (from commit `f0c7a6b`):

> All 4 configs show identical bias structure (-24.7 / -23.3 / -13.9 / -27.0 €/MWh on M+1..M+3 / M+4..M+6 / M+7..M+12 / Y+1..Y+2) → bias is in the **level** (MSFC anchor), not in shape/uncertainty modules. EPEX 2024 realised mean = 76 €/MWh, EPEX 2025 = 102, PFC Config 4 mean = 119 — a 17–43 €/MWh structural gap inherited from `fallback_diagnostic` forwards. Uncertainty v2 method validated by construction on synthetic Gaussian EPEX — observed_p = nominal 20% to within 2% sampling noise.

This is **why** the Mac Mini agent stopped: further code work on Uncertainty/Pillar-3 calibration would be overfit to the fake forwards. The only meaningful next step is exactly this handoff — get real EEX forwards in, then re-evaluate.

---

## 6. Deliverables expected back

After running the workflow in §§ 2–4, commit and push back to `claude/clean-lt-ct-integration`:

1. `scripts/import_fmv_forwards.py` — the parser tailored to your FMV export format.
2. `data/forwards_history_phase10.parquet` — replaced with real EEX (NOT `fallback_diagnostic`).
3. `.planning/phases/10-pfc-fmv-quality-scorecard/{10-VERIFICATION.md, scorecard_kpis_pillar{2,3,4}.parquet, figures/*}` — refreshed.
4. **Brief audit report** in `.planning/phases/10-pfc-fmv-quality-scorecard/POST_REAL_FORWARDS_AUDIT.md` documenting:
   - Before/after table for each Pillar (verdict + observed_freq / arb_free deviation / DM ratios)
   - Whether the SC#1 gate is now PASS (4/4) or still has FAIL items (and why)
   - Whether Pillar 3 Uncertainty v2 is validated end-to-end (≥3/5 blocs PASS LR_uc on real data) or whether Phase 11 EVT-based tail estimation is needed
   - Any new findings (PFC level bias, calibration convergence issues, etc.)

If any pillar still rejects after real forwards, the audit report should propose the next phase (Phase 10C / Phase 11 / etc.) with one-paragraph problem statement.

---

## 7. Out of scope for this handoff

Don't touch on the FMV side:

- **Pillar 5 D-FLIP-1 audit-trail** — BLOCKED separately, requires investigation of what flip-flag mechanism was missing in Plan 10-04 Task 3 (see `.planning/phases/10-pfc-fmv-quality-scorecard/10-04-SUMMARY.md` for context). Out of scope here.
- **6 INFO findings** from the code review (IN-01..IN-06 in `.planning/phases/10-pfc-fmv-quality-scorecard/10-REVIEW.md`) — cosmetic, low-value, can be batched later with `/gsd:code-review 10 --fix --all`.
- **Phase 10B block-MAE vs HFC OMPEX** — separate phase requiring OMPEX data in addition to forwards. This handoff unblocks Phase 10 gate-eligibility first; 10B can follow.
- **Phase 11 EVT-Uncertainty** — only relevant if §4.2 confirms real fat-tail problem after the forwards swap. Defer until measured.

---

## 8. If stuck — escalation

If the FMV export schema is complex (e.g., wide-format with one column per delivery period, EEX-specific date encoding, missing months in shoulder seasons, etc.) and the §2 skeleton needs non-trivial adaptation, **save the raw export to `data/_raw_fmv_forwards_<date>.xlsx`** (gitignored — do NOT commit raw FMV files if confidential) and commit a markdown note `data/FMV_FORWARDS_SCHEMA_NOTES.md` describing what you found. The Mac Mini agent can iterate on the parser via a follow-up handoff if needed.

If a test in `tests/test_phase10_*.py` fails after the swap, **do not** silently update the test to match — that's how regressions hide. Investigate, document root cause, and either fix the underlying code or call it out explicitly in the audit report.

If pillar verdicts get **worse** (e.g., Pillar 1 SC#1 deviation increases vs fallback_diagnostic baseline), that's a genuine finding: probably means the real EEX forwards reveal a calibration issue in `ArbitrageFreeCalibrator` that the synthetic-flat forwards were masking. Document it; don't paper over it.

---

**End of handoff.** Pull, run, audit, commit, push — this is the unlock.
