"""
calendar_ch.py
--------------
Classification des jours selon la grille FMV :
    Ouvrable | Samedi | Dimanche | Ferie_CH | Ferie_DE

Les fériés DE sont inclus car EPEX Spot réagit fortement aux fériés allemands
(volume réduit côté DE → distorsion de prix).

Saisons calendaires :
    Hiver    : nov, déc, jan, fév, mar
    Printemps: avr, mai
    Eté      : jun, jul, aoû, sep
    Automne  : oct
"""

from __future__ import annotations

import pandas as pd
import holidays


SAISONS = {
    1: "Hiver",
    2: "Hiver",
    3: "Hiver",
    4: "Printemps",
    5: "Printemps",
    6: "Ete",
    7: "Ete",
    8: "Ete",
    9: "Ete",
    10: "Automne",
    11: "Hiver",
    12: "Hiver",
}


def get_day_type(date: pd.Timestamp, ch_holidays: set, de_holidays: set) -> str:
    """
    Retourne le type de jour pour une date donnée.

    Priorité : Ferie_CH > Ferie_DE > Dimanche > Samedi > Ouvrable
    """
    d = date.date()
    if d in ch_holidays:
        return "Ferie_CH"
    if d in de_holidays:
        return "Ferie_DE"
    dow = date.dayofweek  # 0=lundi, 6=dimanche
    if dow == 6:
        return "Dimanche"
    if dow == 5:
        return "Samedi"
    return "Ouvrable"


def build_calendar(start: str, end: str) -> pd.DataFrame:
    """
    Construit un DataFrame avec type_jour et saison pour chaque jour
    dans l'intervalle [start, end].

    Args:
        start: date de début au format 'YYYY-MM-DD'
        end:   date de fin au format 'YYYY-MM-DD'

    Returns:
        DataFrame indexé par date avec colonnes [type_jour, saison]
    """
    years = range(
        pd.Timestamp(start).year,
        pd.Timestamp(end).year + 1,
    )

    ch_hols: set = set()
    de_hols: set = set()
    for y in years:
        ch_hols |= set(holidays.Switzerland(years=y, subdiv="VS").keys())
        de_hols |= set(holidays.Germany(years=y).keys())

    dates = pd.date_range(start=start, end=end, freq="D")
    records = []
    for d in dates:
        records.append(
            {
                "date": d.date(),
                "type_jour": get_day_type(d, ch_hols, de_hols),
                "saison": SAISONS[d.month],
            }
        )

    return pd.DataFrame(records).set_index("date")


def enrich_15min_index(index_15min: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Ajoute type_jour, saison, heure_hce et quart à un index 15min.

    Args:
        index_15min: DatetimeIndex en timezone Europe/Zurich (ou UTC)

    Returns:
        DataFrame avec colonnes [type_jour, saison, heure_hce, quart]
        indexé sur index_15min
    """
    if index_15min.tz is None:
        raise ValueError("L'index doit être localisé (timezone-aware). Utilisez Europe/Zurich ou UTC.")

    idx_zurich = index_15min.tz_convert("Europe/Zurich")

    start = idx_zurich.min().strftime("%Y-%m-%d")
    end = idx_zurich.max().strftime("%Y-%m-%d")
    cal = build_calendar(start, end)

    df = pd.DataFrame(index=index_15min)
    df["date"] = idx_zurich.date
    df = df.join(cal, on="date")
    df["heure_hce"] = idx_zurich.hour
    # quart : 1 = :00, 2 = :15, 3 = :30, 4 = :45
    df["quart"] = (idx_zurich.minute // 15) + 1
    df.drop(columns=["date"], inplace=True)

    return df
