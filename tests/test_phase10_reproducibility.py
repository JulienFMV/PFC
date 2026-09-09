"""D-A6-3 Reproducibility contract test (Plan 10-04 Task 5).

Per CONTEXT.md §D-A6-3 : ``assert_frame_equal(scorecard_kpis_run1, scorecard_kpis_run2,
check_exact=False, atol=1e-12, rtol=0)`` — convention 5bis-A/B/5 tolerance préservée.

Subset = 4 builds (Config 4 × 4 vintages Q1/Q2/Q3/Q4 2024) — full 96 grid réexécutée
Plan 10-04 Task 2. Justification subset : 4 builds × ~10s = ~40s total compute (vs
96 builds × ~6.6s = ~10.6 min). Subset suffit pour prouver determinism (même chaîne
``seed=42`` + ``np.random.default_rng`` / ``np.random.RandomState``).

[C5 REVIEWS] Tolerance double-niveau :
- Primary : ``atol=1e-12 rtol=0`` (target SOTA-grade D-A6-3).
- Fallback : ``atol=1e-10`` autorisé UNIQUEMENT via entrée correspondante dans
  ``.planning/phases/10-pfc-fmv-quality-scorecard/REPRODUCIBILITY-EXCEPTIONS.md``.
- Aucune relaxation silencieuse — le test FAIL avec message explicit si 1e-12 échoue
  sans entrée correspondante.

NB : ce test fixe `vintages_limit` à 4 mais utilise les vintages canoniques 1-4
(jan/fév/mar/avr 2024) — le set Q1/Q2/Q3/Q4 stricto-sensu (mars/juin/sept/déc) n'est
pas accessible via vintages_limit sans changement d'API. Les 4 premiers vintages
suffisent pour prouver le determinism (le numéro/identité du vintage n'affecte pas
la propriété testée).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


REPRODUCIBILITY_TOLERANCE_PRIMARY = {"check_exact": False, "atol": 1e-12, "rtol": 0}
REPRODUCIBILITY_TOLERANCE_FALLBACK = {"check_exact": False, "atol": 1e-10, "rtol": 0}
EXCEPTIONS_FILE = (
    ".planning/phases/10-pfc-fmv-quality-scorecard/REPRODUCIBILITY-EXCEPTIONS.md"
)


def _load_documented_exceptions(exceptions_path: str) -> set[str]:
    """[C5 REVIEWS] Parse REPRODUCIBILITY-EXCEPTIONS.md, return set of authorized
    pillar_keys for the 1e-10 fallback.

    Each row of the markdown table must contain 6 cells:
        pillar_key | machine | libversion | observed_delta | suspected_root_cause | signoff

    Returns set of pillar_key strings for which the fallback is documented and
    authorized. Empty set means "no exceptions documented" — the primary
    1e-12 contract is strict.
    """
    p = Path(exceptions_path)
    if not p.exists():
        return set()
    text = p.read_text(encoding="utf-8")
    documented: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        # Skip header / separator rows
        if "---" in line:
            continue
        if "pillar_key" in line.lower() and "machine" in line.lower():
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 6:
            continue
        # Only count rows whose first cell looks like a real pillar key
        first = cells[0]
        if any(k in first for k in ("pillar1", "pillar2_df", "pillar3_df", "pillar4_df")):
            documented.add(first)
    return documented


@pytest.fixture(scope="module")
def epex_parquet_available() -> bool:
    """Skip if ``data/epex_hourly.parquet`` absent (Plan 10-01 Task 3 required)."""
    if not Path("data/epex_hourly.parquet").exists():
        pytest.skip(
            "data/epex_hourly.parquet not bootstrapped (Plan 10-01 Task 3 required)"
        )
    return True


def _scalar_diff(a: object, b: object) -> float:
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return 0.0 if a == b else float("inf")


def test_reproducibility_subset_4_builds_config_4(
    epex_parquet_available, tmp_path_factory
) -> None:
    """D-A6-3 contract — primary atol=1e-12 rtol=0 ; [C5 REVIEWS] fallback
    atol=1e-10 ONLY via documented entry."""
    from pfc_shaping.validation.scorecard import run_scorecard_full

    out_dir_1 = tmp_path_factory.mktemp("phase10_repro_run1")
    result_1 = run_scorecard_full(
        epex_source="parquet",
        output_dir=out_dir_1,
        vintages_limit=4,
        cache_intermediate=False,  # ensure each call recomputes
    )
    out_dir_2 = tmp_path_factory.mktemp("phase10_repro_run2")
    result_2 = run_scorecard_full(
        epex_source="parquet",
        output_dir=out_dir_2,
        vintages_limit=4,
        cache_intermediate=False,
    )
    documented_exceptions = _load_documented_exceptions(EXCEPTIONS_FILE)

    # ---- Pillar 1 : TestResult.observed scalars identical ----
    for key in ("arb_free", "holiday_weekend", "seasonal_profile", "continuity"):
        obs_1 = result_1["pillar1"][key].observed
        obs_2 = result_2["pillar1"][key].observed
        # Handle NaN==NaN as a pass (both degenerate the same way)
        if pd.isna(obs_1) and pd.isna(obs_2):
            continue
        diff = _scalar_diff(obs_1, obs_2)
        pillar_key = f"pillar1.{key}"
        if diff >= 1e-12:
            if pillar_key in documented_exceptions and diff < 1e-10:
                print(
                    f"INFO [C5 REVIEWS] reproducibility relaxed for {pillar_key}: "
                    f"diff={diff} < 1e-10 per {EXCEPTIONS_FILE}"
                )
            else:
                raise AssertionError(
                    f"D-A6-3 violation Pillar 1 {key}: "
                    f"obs_1={obs_1}, obs_2={obs_2}, diff={diff}. "
                    f"Primary atol=1e-12 failed. To allow 1e-10 fallback, add entry "
                    f"to {EXCEPTIONS_FILE} with pillar_key='{pillar_key}', "
                    f"machine, libversion, observed_delta, suspected_root_cause, signoff."
                )

    # ---- Pillar 2/3/4 : KPIs DataFrames identical ----
    for pillar_key in ("pillar2_df", "pillar3_df", "pillar4_df"):
        df_1 = result_1[pillar_key]
        df_2 = result_2[pillar_key]
        if df_1.empty and df_2.empty:
            continue
        df_1 = df_1.sort_index(axis=0).sort_index(axis=1).reset_index(drop=True)
        df_2 = df_2.sort_index(axis=0).sort_index(axis=1).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                df_1, df_2, **REPRODUCIBILITY_TOLERANCE_PRIMARY
            )
        except AssertionError as primary_err:
            numeric_cols = df_1.select_dtypes(include="number").columns
            max_diffs = {
                c: float((df_1[c] - df_2[c]).abs().max(skipna=True))
                for c in numeric_cols
            }
            # Filter NaN values out of the max calc
            valid_diffs = {
                c: v
                for c, v in max_diffs.items()
                if not (v != v)  # NaN-safe
            }
            max_overall = max(valid_diffs.values()) if valid_diffs else 0.0
            if pillar_key in documented_exceptions and max_overall < 1e-10:
                try:
                    pd.testing.assert_frame_equal(
                        df_1, df_2, **REPRODUCIBILITY_TOLERANCE_FALLBACK
                    )
                    print(
                        f"INFO [C5 REVIEWS] reproducibility relaxed for {pillar_key}: "
                        f"max_diff={max_overall} < 1e-10 per {EXCEPTIONS_FILE}"
                    )
                except AssertionError as fallback_err:
                    raise AssertionError(
                        f"D-A6-3 violation {pillar_key}: BOTH 1e-12 AND 1e-10 "
                        f"fallback failed. Max abs diffs: {max_diffs}. "
                        f"Fix determinism — fallback exhausted."
                    ) from fallback_err
            else:
                raise AssertionError(
                    f"D-A6-3 violation {pillar_key}: assert_frame_equal failed at "
                    f"primary atol=1e-12.\n"
                    f"Max abs diffs per column: {max_diffs}\n"
                    f"To allow 1e-10 fallback, add entry to {EXCEPTIONS_FILE} with "
                    f"pillar_key='{pillar_key}', machine, libversion, "
                    f"observed_delta, suspected_root_cause, signoff.\n"
                    f"Original error: {primary_err}"
                ) from primary_err
