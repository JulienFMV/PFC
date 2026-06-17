# Power BI - CH HFC/PFC validation

Ce dossier contient le pack Power BI local pour auditer la derniere HFC/PFC CH.

Le refresh Power BI prend automatiquement le plus recent CSV horaire matching:

```text
output\ch_hfc_hourly*.csv
```

Le script filtre les candidats sur le schema horaire attendu afin d'ignorer les
CSV auxiliaires de diagnostic. Il genere ensuite des tables CSV pre-agregees
dans:

```text
powerbi\data\
```

## Regeneration des donnees Power BI

Depuis la racine du repo:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\powerbi\build_and_open_pfc_qa.ps1
```

Pour forcer un fichier precis:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\powerbi\build_and_open_pfc_qa.ps1 -Csv output\ch_hfc_hourly_YYYYMMDD_....csv
```

Tables produites:

| fichier | role |
|---|---|
| `hfc_hourly_powerbi.csv` | table horaire enrichie avec annee, mois, heure, saison, EEX peak |
| `period_means.csv` | controles annee / trimestre / mois |
| `duck_month_hour.csv` | duck curves mois x heure |
| `duck_season_hour.csv` | duck curves saison x heure |
| `heatmap_month_hour.csv` | matrice mois x heure |
| `negative_low_hours.csv` | heures negatives / tres basses |
| `annual_shape.csv` | KPI annuels de shape |
| `eex_residuals.csv` | residus EEX BASE et PEAK |
| `summary_metrics.csv` | score et gates principaux |

## Creation manuelle du rapport Power BI Desktop

La voie normale est d'utiliser `build_and_open_pfc_qa.ps1`, qui regenere les
exports CSV, le modele TMDL et le layout PBIR avec le chemin local du repo.

Les fichiers `queries/*.pq` sont des templates manuels. Ils utilisent la
requete/parametre `RepoRoot.pq`; remplacer le placeholder
`<SET_REPO_ROOT_HERE>\` dans ce seul fichier si les requetes sont importees
manuellement.

1. Ouvrir Power BI Desktop.
2. `Get data` > `Blank query`.
3. Ouvrir `Advanced Editor`.
4. Creer d'abord la requete `RepoRoot` avec `queries/RepoRoot.pq`.
5. Copier ensuite le contenu d'un fichier `queries/*.pq`.
6. Renommer la requete avec le nom du fichier, par exemple `HFC_Hourly`.
7. Repeter pour les tables utiles: `HFC_Hourly`, `EEX_Residuals`,
   `Duck_Month_Hour`, `Duck_Season_Hour`, `Annual_Shape`,
   `Negative_Low_Hours`, `Summary_Metrics`.
8. Fermer et appliquer.
9. Coller les mesures DAX de `measures/PFC_Measures.dax`.

Alternative rapide: utiliser directement `Get data > Text/CSV` sur les fichiers
dans `powerbi\data\`. Les scripts `.pq` servent surtout a figer les types et a
eviter les problemes de parsing de dates.

## Pages recommandees

| page | visuels |
|---|---|
| Executive | cards score, EEX date, max residus BASE/PEAK, min price, negative hours |
| EEX Calibration | table `eex_residuals`, barres residus par produit/load_type |
| Duck Curves | line chart heure vs prix, legend mois/saison, slicers annee/serie |
| Seasonality | monthly average, winter/summer, Jan/Oct |
| Negative Tail | heatmap mois/heure des heures negatives fast/P10 |
| HFC vs Spot | KPI correlation shape, peak/offpeak, evening-midday |

## Note gouvernance

Ce pack est un outil de validation local/test. Il ne change pas la PFC et ne
constitue pas une approbation production.
