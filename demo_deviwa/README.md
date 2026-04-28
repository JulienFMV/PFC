# FMV Deviwa Demo

POC client pour présentation Deviwa (mardi). Tourne en local uniquement.

## Lancement

```bash
# depuis la racine du repo
pip install -r dashboard/requirements.txt  # si pas déjà fait
streamlit run demo_deviwa/Cockpit.py
```

Ouvre http://localhost:8501.

## Pages

1. **Marktüberblick** (home) — spot CH, prévision D+1, forwards EEX, TTF/EUA
2. **Kurzfristprognose** — LEAR 10 jours avec bande P10-P90, MAE 30j
3. **Lastprofil-Pricing** — upload CSV/xlsx, pricing contre la PFC
4. **Ihre Transaktionen** — deals Deviwa par acteur (RELL/EDSH/EW Binn/EVTL)
5. **Marktdaten** — ENTSO-E (charge, solaire/éolien, flux frontaliers), hydro

## Données attendues

Les chemins sont relatifs à la racine du repo (`/home/user/PFC` ici).

| Fichier | Source | Utilisé dans |
|---|---|---|
| `pfc_shaping/data/epex_15min.parquet` | ingestion EPEX | pages 1, 2 |
| `pfc_shaping/data/entso_15min.parquet` | ingest_entso | pages 1, 5 |
| `pfc_shaping/data/hydro_reservoir.parquet` | BFE | page 5 |
| `pfc_shaping/output/lear_forecast_latest.parquet` | LEAR run | pages 1, 2 |
| `pfc_shaping/output/lear_backtest_latest.parquet` | LEAR backtest | page 2 |
| `pfc_shaping/output/pfc_15min_*.parquet` | PFC shaping | page 3 |
| `data/eex_forwards_history.parquet` | ingest_forwards | page 1 |
| `data/commodities_cache.parquet` | commodities ingestion | page 1 |
| `data/Deviwa.xlsx` (ou `.csv`) | **à fournir** | page 4 |

Le parseur `deviwa_parser.py` attend un fichier Excel avec un sheet par
acteur (RELL, EDSH, EW Binn, EVTL) + HPFC en option. Les noms de colonnes
sont normalisés (tolérant aux variantes Volume (Sum)/Volume_Sum/Volumen).

## Avant la démo

Rafraîchir les données:

```bash
python -c "from pfc_shaping.data.ingest_epex import update_cache; update_cache()"
python -c "from pfc_shaping.data.ingest_entso import update_cache; update_cache()"
# Générer une PFC et un forecast récents
python run_pfc_production.py --short-horizon 10
```

## Parcours de démo suggéré (20 min)

1. Ouvrir `Marktüberblick` — planter le décor (spot, forecast, forwards, commodities).
2. Cliquer sur `Kurzfristprognose` — montrer la prévision 10j + bande d'incertitude + MAE 30j.
3. Cliquer sur `Lastprofil-Pricing` — utiliser le profil synthétique puis, si possible, uploader un vrai profil client Deviwa.
4. Cliquer sur `Ihre Transaktionen` — sélectionner RELL, puis EDSH, montrer volume/PnL par mois.
5. Finir sur `Marktdaten` — ancrer l'expertise ENTSO + hydro.
