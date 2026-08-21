"""
test_sidecar_compat.py
----------------------
M4 cross-AI review fix (05C-REVIEWS.md mandatory #4): parametrized backward-compat matrix
for ShapeHourly.load() across three historical sidecar formats.

CONTEXT
-------
Plan 05C-03 Task 7 (M4 / REVIEWS.md mandatory fix).

The backward-compat surface for Plan 05C-03's sigma_off/sigma_on breaking change is
ShapeHourly.load() — the cross-plan fallback in load() handles three sidecar schemas:

    pre_5bisA — written before 5bis-A landed (no use_seasonal_hourly, no _off/_on keys)
    5bisA     — written after 5bis-A (has use_seasonal_hourly, single sigma/hydro_weight_sigma)
    5bisB     — written by this plan (all 10 hyperparams JSON keys)

The M4-mandated invariant: for ALL THREE formats, ShapeHourly.load() MUST produce a model
where the legacy single-σ caller invariants hold:
    sh.sigma == sh._sigma_off                   (flag=OFF: resolved σ is the flag=OFF value)
    sh.hydro_weight_sigma == sh._hydro_weight_sigma_off
    sh._sigma_on         == expected_sigma_on   (legacy formats → single-σ fallback)
    sh._hydro_weight_sigma_on == expected_hws_on (same)

Per M4 / 05C-REVIEWS.md guidance: prefer fixture-factory over committed binaries when
generation is < 100ms. Sidecars are constructed in-memory via the fixture factory below.

LEGACY CALLSITES (manual audit, Plan 05C-03 Task 1):
    autoresearch.py:234             — ShapeHourly(sigma=0.3)
    rolling_update.py:365           — ShapeHourly(sigma=0.3, ...)
    tests/test_shape_hourly_infra.py:239  — ShapeHourly(sigma=0.3, halflife_days=90.0, ...)
    tests/test_shape_hourly_infra.py:250  — ShapeHourly(sigma=0.5, halflife_days=180.0, ...)

This matrix test complements the manual audit by parametrizing across HISTORICAL sidecar
formats. External scripts that load sidecars (the most common backward-compat surface) are
covered here. External scripts that bypass sidecars entirely are documented as the residual
breaking-change surface via the `sigma` parameter docstring (Plan 05C-03 Task 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.lt.model.shape_hourly import ShapeHourly


# ---------------------------------------------------------------------------
# Fixture-factory: build minimal sidecar parquets in-memory at the requested
# schema version.  Generation is ~5ms per invocation (pandas to_parquet + json).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _make_sidecar(tmp_path_factory):
    """Generate a synthetic ShapeHourly sidecar parquet at the requested schema version.

    Per M4 / 05C-REVIEWS.md: fixture-factory preferred over committed binaries when
    generation is < 100ms.

    The main parquet carries the minimal schema required by ShapeHourly.load():
        saison, type_jour, heure (0..23), f_H (float), n_obs (int)
    One cell (Hiver, Ouvrable) with 24 hours of f_H = 1.0 is sufficient to pass the
    groupby loop in load() and populate factors_.

    The meta sidecar carries only the `hyperparams` row; all other fitted-state rows
    (factors_by_year_, trend_per_hour_, etc.) are absent — load() handles them gracefully
    with `if len(rows) > 0` guards.
    """

    def _factory(version: str) -> Path:
        out_dir = tmp_path_factory.mktemp(f"sidecar_{version}")
        main_path = out_dir / "shape_hourly.parquet"
        meta_path = out_dir / "shape_hourly.meta.parquet"

        # Minimal main parquet — one cell (Hiver, Ouvrable) with flat f_H = 1.0
        main_records = [
            {"saison": "Hiver", "type_jour": "Ouvrable", "heure": h, "f_H": 1.0, "n_obs": 10}
            for h in range(24)
        ]
        pd.DataFrame(main_records).to_parquet(main_path, index=False)

        # Build hyperparams JSON at the requested schema version
        if version == "pre_5bisA":
            # No use_seasonal_hourly, no _off/_on/_resolved triplets
            hp = {
                "halflife_days": 180.0,
                "hydro_weight_sigma": 0.25,
                "sigma": 0.5,
            }
        elif version == "5bisA":
            # Added use_seasonal_hourly (5bis-A D-07), single sigma/hydro_weight_sigma
            hp = {
                "halflife_days": 180.0,
                "hydro_weight_sigma": 0.25,
                "sigma": 0.5,
                "use_seasonal_hourly": False,
            }
        elif version == "5bisB":
            # Full 10-key schema (Plan 05C-03, D-A3-3)
            hp = {
                "halflife_days": 180.0,
                "hydro_weight_sigma": 0.25,
                "hydro_weight_sigma_off": 0.25,
                "hydro_weight_sigma_on": 0.08,
                "hydro_weight_sigma_resolved": 0.25,
                "sigma": 0.5,
                "sigma_off": 0.5,
                "sigma_on": 0.25,
                "sigma_resolved": 0.5,
                "use_seasonal_hourly": False,
            }
        else:
            raise ValueError(f"unknown version: {version}")

        meta_records = [{"attr": "hyperparams", "value": json.dumps(hp, sort_keys=True)}]
        pd.DataFrame(meta_records).to_parquet(meta_path, index=False)
        return main_path

    return _factory


# ---------------------------------------------------------------------------
# M4 parametrized sidecar load matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sidecar_version,expected_use_seasonal,expected_sigma,expected_hydro_sigma,expected_sigma_on,expected_hws_on",
    [
        # pre_5bisA: legacy single-σ fallback → _sigma_on == _sigma_off == 0.5
        ("pre_5bisA", False, 0.5, 0.25, 0.5,  0.25),
        # 5bisA: same single-σ key → _sigma_on == _sigma_off == 0.5
        ("5bisA",     False, 0.5, 0.25, 0.5,  0.25),
        # 5bisB: full triplet persisted → _sigma_on == 0.25 (from sigma_on key)
        ("5bisB",     False, 0.5, 0.25, 0.25, 0.08),
    ],
)
def test_sidecar_load_matrix(
    _make_sidecar,
    sidecar_version,
    expected_use_seasonal,
    expected_sigma,
    expected_hydro_sigma,
    expected_sigma_on,
    expected_hws_on,
) -> None:
    """Verify ShapeHourly.load() satisfies legacy caller invariants for all 3 sidecar formats.

    M4 cross-AI review fix (05C-REVIEWS.md mandatory #4):
    For ALL THREE historical sidecar formats (pre_5bisA / 5bisA / 5bisB), ShapeHourly.load()
    must produce a model where:
        sh.sigma == sh._sigma_off  (resolved σ == flag=OFF σ for default flag=OFF)
        sh.hydro_weight_sigma == sh._hydro_weight_sigma_off
        sh._sigma_on == expected  (legacy formats: single-σ fallback; 5bisB: persisted value)
        sh._hydro_weight_sigma_on == expected  (same)
        sh._use_seasonal_hourly == expected_use_seasonal
    """
    sidecar_path = _make_sidecar(sidecar_version)
    sh = ShapeHourly.load(sidecar_path)

    # --- use_seasonal_hourly ---
    assert sh._use_seasonal_hourly == expected_use_seasonal, (
        f"[{sidecar_version}] _use_seasonal_hourly mismatch: "
        f"expected {expected_use_seasonal}, got {sh._use_seasonal_hourly}"
    )

    # --- sigma invariants ---
    assert sh.sigma == expected_sigma, (
        f"[{sidecar_version}] sigma mismatch: expected {expected_sigma}, got {sh.sigma}"
    )
    assert sh._sigma_off == expected_sigma, (
        f"[{sidecar_version}] _sigma_off mismatch: expected {expected_sigma}, got {sh._sigma_off}"
    )
    assert sh._sigma_on == expected_sigma_on, (
        f"[{sidecar_version}] _sigma_on mismatch: "
        f"expected {expected_sigma_on}, got {sh._sigma_on} "
        f"(pre/5bisA: single-σ fallback should set _sigma_on = legacy_sigma; "
        f"5bisB: persisted sigma_on)"
    )

    # --- hydro_weight_sigma invariants ---
    assert sh.hydro_weight_sigma == expected_hydro_sigma, (
        f"[{sidecar_version}] hydro_weight_sigma mismatch: "
        f"expected {expected_hydro_sigma}, got {sh.hydro_weight_sigma}"
    )
    assert sh._hydro_weight_sigma_off == expected_hydro_sigma, (
        f"[{sidecar_version}] _hydro_weight_sigma_off mismatch: "
        f"expected {expected_hydro_sigma}, got {sh._hydro_weight_sigma_off}"
    )
    assert sh._hydro_weight_sigma_on == expected_hws_on, (
        f"[{sidecar_version}] _hydro_weight_sigma_on mismatch: "
        f"expected {expected_hws_on}, got {sh._hydro_weight_sigma_on} "
        f"(pre/5bisA: single-σ fallback; 5bisB: persisted hydro_weight_sigma_on=0.08)"
    )

    # --- Legacy single-σ invariant: sigma == _sigma_off for flag=OFF (key M4 contract) ---
    assert sh.sigma == sh._sigma_off, (
        f"[{sidecar_version}] Legacy single-σ invariant violated: "
        f"sh.sigma ({sh.sigma}) != sh._sigma_off ({sh._sigma_off})"
    )
    assert sh.hydro_weight_sigma == sh._hydro_weight_sigma_off, (
        f"[{sidecar_version}] Hydro single-σ invariant violated: "
        f"sh.hydro_weight_sigma ({sh.hydro_weight_sigma}) "
        f"!= sh._hydro_weight_sigma_off ({sh._hydro_weight_sigma_off})"
    )

    # --- factors_ populated from main parquet ---
    assert ("Hiver", "Ouvrable") in sh.factors_, (
        f"[{sidecar_version}] factors_ not populated from main parquet"
    )
    assert len(sh.factors_[("Hiver", "Ouvrable")]) == 24, (
        f"[{sidecar_version}] factors_ wrong length: expected 24, got "
        f"{len(sh.factors_[('Hiver', 'Ouvrable')])}"
    )
    # Note: apply() requires a fitted ShapeIntraday context and calendar enrichment;
    # the fixture-factory writes hyperparams-only sidecar (no factors_by_year_ etc.),
    # so full apply() is not exercised here. The M4 contract focuses on sigma attribute
    # correctness post-load(), which is fully covered by the assertions above.
