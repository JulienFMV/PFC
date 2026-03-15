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


def get_day_type(
    date: pd.Timestamp,
    primary_holidays: set,
    secondary_holidays: set,
    primary_label: str = "Ferie_CH",
    secondary_label: str = "Ferie_DE",
) -> str:
    """
    Retourne le type de jour pour une date donnée.

    Priorité : primary_holidays > secondary_holidays > Dimanche > Samedi > Ouvrable
    """
    d = date.date()
    if d in primary_holidays:
        return primary_label
    if d in secondary_holidays:
        return secondary_label
    dow = date.dayofweek  # 0=lundi, 6=dimanche
    if dow == 6:
        return "Dimanche"
    if dow == 5:
        return "Samedi"
    return "Ouvrable"


def build_calendar(start: str, end: str, country: str = "CH") -> pd.DataFrame:
    """
    Construit un DataFrame avec type_jour et saison pour chaque jour
    dans l'intervalle [start, end].

    Args:
        start: date de début au format 'YYYY-MM-DD'
        end:   date de fin au format 'YYYY-MM-DD'
        country: 'CH' (default) or 'DE'. Controls holiday priority:
                 CH: Ferie_CH > Ferie_DE
                 DE: Ferie_DE > Ferie_CH

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

    if country == "DE":
        primary_hols, secondary_hols = de_hols, ch_hols
        primary_label, secondary_label = "Ferie_DE", "Ferie_CH"
    else:
        primary_hols, secondary_hols = ch_hols, de_hols
        primary_label, secondary_label = "Ferie_CH", "Ferie_DE"

    dates = pd.date_range(start=start, end=end, freq="D")
    records = []
    for d in dates:
        records.append(
            {
                "date": d.date(),
                "type_jour": get_day_type(
                    d, primary_hols, secondary_hols,
                    primary_label, secondary_label,
                ),
                "saison": SAISONS[d.month],
            }
        )

    return pd.DataFrame(records).set_index("date")


def enrich_15min_index(index_15min: pd.DatetimeIndex, country: str = "CH") -> pd.DataFrame:
    """
    Ajoute type_jour, saison, heure_hce et quart à un index 15min.

    Args:
        index_15min: DatetimeIndex en timezone Europe/Zurich (ou UTC)
        country: 'CH' (default) or 'DE'. Controls holiday priority.

    Returns:
        DataFrame avec colonnes [type_jour, saison, heure_hce, quart]
        indexé sur index_15min
    """
    if index_15min.tz is None:
        raise ValueError("L'index doit être localisé (timezone-aware). Utilisez Europe/Zurich ou UTC.")

    tz = "Europe/Berlin" if country == "DE" else "Europe/Zurich"
    idx_local = index_15min.tz_convert(tz)

    start = idx_local.min().strftime("%Y-%m-%d")
    end = idx_local.max().strftime("%Y-%m-%d")
    cal = build_calendar(start, end, country=country)

    df = pd.DataFrame(index=index_15min)
    df["date"] = idx_local.date
    df = df.join(cal, on="date")
    df["heure_hce"] = idx_local.hour
    # quart : 1 = :00, 2 = :15, 3 = :30, 4 = :45
    df["quart"] = (idx_local.minute // 15) + 1
    df.drop(columns=["date"], inplace=True)

    return df
