"""
block_masks.py
--------------
5 blocs client (D-A2-2 CONTEXT Phase 10) implémentés comme classes uniformes
avec une méthode `apply(idx_utc: pd.DatetimeIndex) -> np.ndarray[bool]`.

Convention
----------
- L'index d'entrée DOIT être tz-aware (UTC attendu en pratique car c'est la
  convention canonique du modèle PFC LT — cf. pfc_shaping/data/calendar_ch.py).
- La sémantique des blocs est exprimée en heures LOCALES Europe/Zurich
  (convention trader). La conversion `tz_convert("Europe/Zurich")` est faite
  en interne ; le mask retourné est aligné sur l'index UTC original.
- DST-safe : `tz_convert` depuis UTC est non-ambigu. Les transitions DST
  Europe/Zurich (dernier dim mars à 02:00→03:00, dernier dim oct à 03:00→02:00)
  ne tombent dans aucun des 5 ranges horaires définis, donc aucune mitigation
  spéciale requise (cf. RESEARCH §Pattern 1 Critical DST note).

Les 5 blocs
-----------
- `block_overnight_weekday`   : 18h-09h Lun-Ven, all months (crosses midnight)
- `block_midday_weekday`      : 10h-15h Lun-Ven, all months
- `block_weekend_midday`      : 11h-15h Sam-Dim, all months
- `block_summer_solar_bowl`   : 11h-14h all DOW, mai-août (test 5bis-B duck curve)
- `block_winter_evening_peak` : 17h-21h Lun-Ven, nov-fév

`ALL_BLOCKS` est la liste canonique à itérer pour le scorecard Pillar 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BlockMask:
    """Base class : prend un DatetimeIndex tz-aware, retourne un mask booléen.

    Subclasses surchargent `_mask_local(idx_local: DatetimeIndex) -> np.ndarray[bool]`
    où `idx_local` est l'index converti en Europe/Zurich pour appliquer la
    définition bloc en heures locales suisses.
    """

    name: str = "base"

    def apply(self, idx_utc: pd.DatetimeIndex) -> np.ndarray:
        """Retourne un array booléen aligné sur `idx_utc` (True = dans le bloc).

        Raises
        ------
        ValueError
            Si l'index est tz-naive (UTC attendu).
        """
        if idx_utc.tz is None:
            raise ValueError("Index must be tz-aware (UTC expected).")
        idx_local = idx_utc.tz_convert("Europe/Zurich")
        return self._mask_local(idx_local)

    def _mask_local(self, idx_local: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError


class BlockOvernightWeekday(BlockMask):
    """18h-09h Lun-Ven, all months (crosses midnight)."""

    name = "block_overnight_weekday"

    def _mask_local(self, idx_local: pd.DatetimeIndex) -> np.ndarray:
        is_weekday = idx_local.weekday < 5
        hour_mask = (idx_local.hour >= 18) | (idx_local.hour < 9)
        return np.asarray(is_weekday & hour_mask, dtype=bool)


class BlockMiddayWeekday(BlockMask):
    """10h-15h Lun-Ven, all months."""

    name = "block_midday_weekday"

    def _mask_local(self, idx_local: pd.DatetimeIndex) -> np.ndarray:
        is_weekday = idx_local.weekday < 5
        hour_mask = (idx_local.hour >= 10) & (idx_local.hour < 15)
        return np.asarray(is_weekday & hour_mask, dtype=bool)


class BlockWeekendMidday(BlockMask):
    """11h-15h Sam-Dim, all months."""

    name = "block_weekend_midday"

    def _mask_local(self, idx_local: pd.DatetimeIndex) -> np.ndarray:
        is_weekend = idx_local.weekday >= 5
        hour_mask = (idx_local.hour >= 11) & (idx_local.hour < 15)
        return np.asarray(is_weekend & hour_mask, dtype=bool)


class BlockSummerSolarBowl(BlockMask):
    """11h-14h all DOW, mai-août. Test 5bis-B duck curve."""

    name = "block_summer_solar_bowl"

    def _mask_local(self, idx_local: pd.DatetimeIndex) -> np.ndarray:
        hour_mask = (idx_local.hour >= 11) & (idx_local.hour < 14)
        month_mask = (idx_local.month >= 5) & (idx_local.month <= 8)
        return np.asarray(hour_mask & month_mask, dtype=bool)


class BlockWinterEveningPeak(BlockMask):
    """17h-21h Lun-Ven, nov-fév."""

    name = "block_winter_evening_peak"

    def _mask_local(self, idx_local: pd.DatetimeIndex) -> np.ndarray:
        is_weekday = idx_local.weekday < 5
        hour_mask = (idx_local.hour >= 17) & (idx_local.hour < 21)
        month_mask = (idx_local.month >= 11) | (idx_local.month <= 2)
        return np.asarray(is_weekday & hour_mask & month_mask, dtype=bool)


# Liste canonique itérée par le scorecard Pillar 2 (et par les tests).
ALL_BLOCKS: list[BlockMask] = [
    BlockOvernightWeekday(),
    BlockMiddayWeekday(),
    BlockWeekendMidday(),
    BlockSummerSolarBowl(),
    BlockWinterEveningPeak(),
]

__all__ = [
    "BlockMask",
    "BlockOvernightWeekday",
    "BlockMiddayWeekday",
    "BlockWeekendMidday",
    "BlockSummerSolarBowl",
    "BlockWinterEveningPeak",
    "ALL_BLOCKS",
]
