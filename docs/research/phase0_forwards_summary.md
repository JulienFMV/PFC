# Phase 0 — snapshot des sources EEX (auto-généré)

_Généré par `scripts/phase0_sniff_forwards.py`_

## Price_Report_EEX_Yearly.xlsx

| Feuille | Format | Date min | Date max | Lignes valides | Produits |
|---|---|---|---|---|---|
| DE | legacy | 2024-05-01T00:00:00 | 2026-05-31T00:00:00 | 761 | {'Cal_BASE': 9, 'Cal_PEAK': 5, 'Q_BASE': 11, 'Q_PEAK': 10, 'M_BASE': 25, 'M_PEAK': 9, 'other': 12, 'empty': 4} |
| CH | legacy | 2024-05-01T00:00:00 | 2026-05-31T00:00:00 | 761 | {'Cal_BASE': 6, 'Cal_PEAK': 6, 'Q_BASE': 7, 'Q_PEAK': 7, 'M_BASE': 7, 'M_PEAK': 7, 'other': 12} |
| AT | legacy | 2024-05-01T00:00:00 | 2026-05-31T00:00:00 | 761 | {'Cal_BASE': 6, 'Cal_PEAK': 6, 'Q_BASE': 11, 'Q_PEAK': 11, 'M_BASE': 10, 'M_PEAK': 10, 'other': 12} |
| IT | legacy | 2024-05-01T00:00:00 | 2026-05-31T00:00:00 | 761 | {'Cal_BASE': 8, 'Cal_PEAK': 6, 'Q_BASE': 10, 'Q_PEAK': 11, 'M_BASE': 7, 'M_PEAK': 7, 'other': 32, 'empty': 4} |
| FR | legacy | 2024-05-01T00:00:00 | 2026-05-31T00:00:00 | 761 | {'Cal_BASE': 10, 'Cal_PEAK': 6, 'Q_BASE': 11, 'Q_PEAK': 11, 'M_BASE': 10, 'M_PEAK': 10, 'other': 12} |
| FX | unknown | 2024-05-03T00:00:00 | 2026-05-31T00:00:00 | 759 | {'other': 1} |
| Produits | unknown | None | None | 0 | {'other': 1} |
| HFC | unknown | None | None | 0 | {'empty': 1} |

### Échantillon 3×10 — feuille `DE`
```
 | M02_2027_BASE | M09_2027_BASE | M10_2026_BASE | M11_2026_BASE | M12_2026_BASE | 205_2026_PEAK | 305_2026_PEAK | Y01_2027_BASE | Y01_2030_BASE
 | DE000EEX0DEBM_M022027_BASE0000000 | DE000EEX0DEBM_M092027_BASE0000000 | DE000EEX0DEBM_M102026_BASE0000000 | DE000EEX0DEBM_M112026_BASE0000000 | DE000EEX0DEBM_M122026_BASE0000000 | DE000EEX0DEP2_2052026_PEAK0000000 | DE000EEX0DEP3_3052026_PEAK0000000 | DE000EEX0DEBY_Y012027_BASE0000000 | DE000EEX0DEBY_Y012030_BASE0000000
Date | Fév 27 BASE | Sep 27 BASE | Oct 26 BASE | Nov 26 BASE | Déc 26 BASE | Week19 26 PEAK | Week20 26 PEAK | Cal 27 BASE | Cal 30 BASE
```

## Price_Report_EEX_CH_DE_Historique2019.xlsx

| Feuille | Format | Date min | Date max | Lignes valides | Produits |
|---|---|---|---|---|---|
| DE | unknown | 2019-08-01T00:00:00 | 2025-08-31T00:00:00 | 2223 | {'Cal_BASE': 5, 'Cal_PEAK': 2, 'Q_BASE': 7, 'Q_PEAK': 6, 'M_BASE': 28, 'M_PEAK': 17, 'other': 134} |
| CH | unknown | 2019-08-01T00:00:00 | 2025-08-31T00:00:00 | 2223 | {'Q_BASE': 4, 'M_BASE': 15, 'M_PEAK': 4, 'other': 176} |
| AT | unknown | 2019-08-01T00:00:00 | 2025-08-31T00:00:00 | 2223 | {} |
| IT | unknown | 2019-08-01T00:00:00 | 2025-08-31T00:00:00 | 2223 | {} |
| FR | unknown | 2019-08-01T00:00:00 | 2025-08-31T00:00:00 | 2223 | {} |
| FX | unknown | None | None | 0 | {'other': 1} |
| Produits | unknown | None | None | 0 | {'other': 1} |
| HFC | unknown | 2019-08-09T03:00:00 | 2025-08-08T00:00:00 | 52582 | {'other': 3} |

### Échantillon 3×10 — feuille `DE`
```
 | 504_2018_BASE | M07_2018_BASE | M09_2018_BASE | M10_2018_BASE | Y01_2020_BASE | Q01_2019_PEAK | 305_2018_BASE | 207_2018_PEAK | 307_2018_BASE
 | DE000EEX0DEB5_5042018_BASE0000000 | DE000EEX0DEBM_M072018_BASE0000000 | DE000EEX0DEBM_M092018_BASE0000000 | DE000EEX0DEBM_M102018_BASE0000000 | DE000EEX0DEBY_Y012020_BASE0000000 | DE000EEX0DEPQ_Q012019_PEAK0000000 | DE000EEX0DEB3_3052018_BASE0000000 | DE000EEX0DEP2_2072018_PEAK0000000 | DE000EEX0DEB3_3072018_BASE0000000
Date | Week17 18 BASE | Juil 18 BASE | Sep 18 BASE | Oct 18 BASE | Cal 20 BASE | Q1 19 PEAK | Week20 18 BASE | Week27 18 PEAK | Week28 18 BASE
```
