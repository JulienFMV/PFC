"""
_generate_legacy_fixture.py
---------------------------
This script generates a FORMAT-COMPATIBLE MOCK of a pre-5bis-A ``ShapeHourly.save()``
output.  It is NOT a verbatim artifact from ``main@28dfd65`` — it is hand-crafted to
match the schema emitted by the save() block at ``shape_hourly.py:308-325`` as observed
at HEAD before plan 05B-02 added the meta sidecar.  The mock is sufficient for
exercising the legacy-compat warning path in ``load()``; it is NOT a regression artifact
for upstream behavior changes.  If a true verbatim legacy artifact is needed in the
future, regenerate by checking out ``main@28dfd65``, fitting a ``ShapeHourly()`` on the
same synthetic data as ``_generate_baseline.py``, and calling ``.save()``.

PURPOSE
-------
Produces two files consumed by ``test_save_load_legacy_compat`` (plan 05B-05):
  - ``tests/fixtures/shape_hourly_legacy.parquet``  — long-format main file (no sidecar)
  - ``tests/fixtures/f_W_legacy.parquet``           — f_W companion file

The absence of a ``*.meta.parquet`` sidecar is intentional: these fixtures represent
pre-5bis-A artifacts where the meta sidecar did not exist.  ``ShapeHourly.load()``
is expected to emit one ``logger.warning`` and fall back to constructor defaults.

SCHEMA (verified against save() at shape_hourly.py lines 443-455)
------
shape_hourly_legacy.parquet:
  saison   : object (str)
  type_jour: object (str)
  heure    : int64
  f_H      : float64
  n_obs    : int64

f_W_legacy.parquet:
  type_jour: object (str)
  f_W      : float64

REGENERATION
------------
CLI: ``python tests/fixtures/_generate_legacy_fixture.py``
Uses ``np.random.seed(42)`` for determinism; re-running produces byte-stable output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CELLS = [
    ("Hiver", "Ouvrable"),
    ("Ete", "Samedi"),
]

_TYPES_JOUR_ALL = ["Ouvrable", "Samedi", "Dimanche", "Ferie_CH", "Ferie_DE"]


def _normalized_f_H() -> np.ndarray:
    """Return a 24-element f_H array with mean ≈ 1.0 (per constraint)."""
    arr = np.linspace(0.8, 1.2, 24)
    return arr / arr.mean()


def main() -> None:
    np.random.seed(42)

    # ── shape_hourly_legacy.parquet ────────────────────────────────────────
    records = []
    for saison, type_jour in _CELLS:
        f_h_arr = _normalized_f_H()
        for h, v in enumerate(f_h_arr):
            records.append({
                "saison": saison,
                "type_jour": type_jour,
                "heure": h,           # int64
                "f_H": float(v),      # float64
                "n_obs": 100,         # int64
            })

    main_df = pd.DataFrame(records)
    # Enforce exact dtypes matching save() output
    main_df = main_df.astype({
        "saison": "object",
        "type_jour": "object",
        "heure": "int64",
        "f_H": "float64",
        "n_obs": "int64",
    })

    out_main = _DIR / "shape_hourly_legacy.parquet"
    main_df.to_parquet(out_main, index=False)
    print(f"Wrote: {out_main}  ({len(main_df)} rows, cols={list(main_df.columns)})")

    # ── f_W_legacy.parquet ────────────────────────────────────────────────
    fw_records = []
    base_ratios = {"Ouvrable": 1.05, "Samedi": 0.97, "Dimanche": 0.90,
                   "Ferie_CH": 0.85, "Ferie_DE": 0.83}
    for tj in _TYPES_JOUR_ALL:
        fw_records.append({"type_jour": tj, "f_W": base_ratios[tj]})

    fw_df = pd.DataFrame(fw_records)
    fw_df = fw_df.astype({"type_jour": "object", "f_W": "float64"})

    out_fw = _DIR / "f_W_legacy.parquet"
    fw_df.to_parquet(out_fw, index=False)
    print(f"Wrote: {out_fw}  ({len(fw_df)} rows, cols={list(fw_df.columns)})")

    # ── Confirm no sidecar was produced ──────────────────────────────────
    assert not (_DIR / "shape_hourly_legacy.meta.parquet").exists(), \
        "FATAL: meta sidecar must NOT exist for legacy fixtures"
    print("OK — no .meta.parquet sidecar present (legacy fixture confirmed)")


if __name__ == "__main__":
    main()
