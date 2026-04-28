# Handoff — FMV Deviwa Demo · Agent local (VSCode)

> **Copie-colle ce fichier dans la session Claude Code locale comme premier
> message. Il est auto-suffisant.**

---

## Contexte

Tu prends la suite d'un agent distant qui a construit une démo Streamlit
pour une présentation client le **mardi 28 avril 2026** à FMV. Public:
**4 distributeurs du Haut-Valais** (RELL, EDSH, EW Binn, EVTL), qui forment
l'**Energiepool Deviwa**. Présentation **en allemand simple**, durée ~20 min,
sur **localhost** uniquement.

L'objectif est un POC qui « met plein la vue » au client, pas une mise en
production. Le but commercial: démontrer que FMV peut leur proposer un portail
SaaS dans 6-8 semaines.

## Repo et branche

- Repo: `JulienFMV/PFC` (clone local existant)
- Branche de travail: `claude/review-repo-state-ftYuW`
- Pousse sur cette branche uniquement, jamais sur `main`
- Avant toute chose: `git fetch origin && git checkout claude/review-repo-state-ftYuW && git pull --ff-only`

## Ce qui existe déjà (livré dans la branche)

```
demo_deviwa/
├── Cockpit.py                      → Portfolio-Cockpit (Home) (home)
├── pages/
│   ├── 1_Marktueberblick.py        → spot/forecast/forwards/commodities
│   ├── 2_Kurzfristprognose.py      → LEAR 10j + bandes P10/P90
│   ├── 3_Lastprofil_Pricing.py     → upload CSV → pricing PFC
│   ├── 4_Ihre_Transaktionen.py     → deals par acteur
│   ├── 5_Programmqualitaet.py      → Programm vs Real vs Budget
│   ├── 6_VaR_Stresstest.py         → VaR paramétrique + historique + 6 stress
│   └── 7_Marktdaten.py             → ENTSO + hydro
├── utils.py                        → loaders, formatage CH, thème
├── pricing.py                      → load × PFC pricing
├── deviwa_parser.py                → parser xlsx multi-sheet
├── portfolio_analytics.py          → hedge-quote, MtM, P&L, tenor
├── PRAESENTATION_DE.md             → script allemand simple par page
├── README.md
└── .streamlit/config.toml          → thème FMV (#0F52CC)
```

**Important:** ne touche pas au `dashboard/` interne (analystes FMV) ni au
package `pfc_shaping/`. Tout se passe dans `demo_deviwa/`.

## Données attendues (déjà placées par le client)

| Fichier | Format | Contenu | Statut |
|---|---|---|---|
| `data/Deviwa.xlsx` | xlsx multi-sheets | Voir schéma ci-dessous | À vérifier |
| `pfc_shaping/data/epex_15min.parquet` | parquet | Spot CH 15min | Présent, à rafraîchir avant démo |
| `pfc_shaping/data/epex_de_15min.parquet` | parquet | Spot DE 15min | Présent |
| `pfc_shaping/data/entso_15min.parquet` | parquet | Charge/gen/flux ENTSO | Présent |
| `pfc_shaping/data/hydro_reservoir.parquet` | parquet | Stocks hydro CH | Présent |
| `pfc_shaping/output/lear_forecast_latest.parquet` | parquet | Prévision LEAR 10j | À régénérer avant démo |
| `pfc_shaping/output/lear_backtest_latest.parquet` | parquet | Backtest pour MAE 30j | Présent |
| `pfc_shaping/output/pfc_15min_*.parquet` | parquet | PFC dernière génération | Présent (15.03.2026) |
| `data/eex_forwards_history.parquet` | parquet | Forwards EEX historique | Présent |
| `data/commodities_cache.parquet` | parquet | TTF, Brent, EUA | Présent |

### Schéma Deviwa.xlsx (confirmé par le client)

**Sheets `<Akteur> - Deals`** (4 sheets, un par acteur, agrégation **mensuelle**):

| Colonne | Type | Notes |
|---|---|---|
| Counterparty | str | "ENRELL" |
| Asset Type | str | "EPA" |
| Custom | str | Vide la plupart du temps |
| Deal | str | Identifiant deal long (ex `ReLL_50%_Vaparoid_Annee2026_A85216`) |
| Deal Trade Date | date DD.MM.YYYY | Date d'exécution du deal |
| Deal Delivery Start | date | Début livraison du contrat |
| Deal Delivery End | date | Fin livraison |
| Product | str | "Custom" majoritairement |
| Scope | str | "Intake" (achat) ou "Withdrawal" (vente) |
| Date | str ou date | Mois de livraison "2026-01" → "2026-12" |
| Volume (Sum) | float | **Volume mensuel en MWh, signe positif** |
| Volume (Mean) | float | Moyenne MW sur le mois |
| Volume (Net) | float | **Volume signé** (négatif pour Withdrawal) |
| Volume (Net Mean) | float | |
| Market Value (Sum) | float | EUR |
| Market Value (Mean) | float | EUR |
| Notional (Sum) | float | **Notional EUR du deal sur ce mois** |
| Notional (Mean) | float | |
| PNL (Sum) | float | PnL EUR cumulé |
| PNL (Mean) | float | |

⚠️ **Convention de signe à valider en chargeant**:
- `Volume (Sum)` est toujours positif (8767.53 même pour Withdrawal)
- `Volume (Net)` est signé (-8767.53 pour Withdrawal)
- Le parser utilise `abs(volume_sum)` partout — c'est cohérent.
- Mais vérifie ce que vaut `Notional (Sum)` pour un Withdrawal. Si négatif,
  les calculs `avg_hedge_price` et `MtM` doivent garder la valeur absolue.
  Le parser actuel le fait (`notional.abs()` partout dans
  `portfolio_analytics.py`).

**Sheets `<Akteur> - ProgrammeReal`** (4 sheets, un par acteur, **horaire**):

| Colonne | Type | Notes |
|---|---|---|
| Date | timestamp DD.MM.YYYY HH:MM | UTC ou Europe/Zurich? À confirmer |
| `<Akteur>_Programm` | float | Plan en MW (ex `EW_Binn_Programm`) |
| `<Akteur>_Real` | float | Réalisé en MW |
| `<Akteur>_Budget` | float | Budget annuel/forecast en MW |

⚠️ Les valeurs sont très petites (~0.18 MW pour EW Binn) — c'est normal,
les distributeurs Haut-Valais sont petits. Ne pas confondre avec GW.

**Sheet `HPFC`**: courbe PFC du pool. Pas encore utilisé dans la démo.
Possible bonus: comparer HPFC du pool avec la PFC FMV.

## Ta mission

### Phase 1 — Setup et démarrage (30 min)

1. **Pull et installer**:
   ```bash
   git checkout claude/review-repo-state-ftYuW
   git pull --ff-only
   python -m venv .venv  # si pas déjà fait
   source .venv/bin/activate  # ou .venv\Scripts\activate sur Windows
   pip install -r dashboard/requirements.txt
   pip install scipy plotly openpyxl  # spécifiques à la démo
   ```

2. **Placer la xlsx**: l'utilisateur doit avoir mis `Deviwa.xlsx` dans
   `data/Deviwa.xlsx`. Vérifie:
   ```bash
   ls -lh data/Deviwa.xlsx
   ```

3. **Lancer la démo**:
   ```bash
   streamlit run demo_deviwa/Cockpit.py
   ```
   Ouvre http://localhost:8501

### Phase 2 — Validation page par page (1-2 h)

Pour chaque page, charge-la dans le navigateur et **valide à l'œil**:

| Page | Critère de validation | Si KO |
|---|---|---|
| 🎯 Portfolio-Cockpit | 8 KPI cards remplis, jauge Hedge-Quote affichée, tenor-buckets non vides, time series mensuelle visible, P&L waterfall et cumulatif | Vérifier `data_by_actor` non vide, ajuster `portfolio_analytics.compute_portfolio_metrics` si signes incohérents |
| 📊 Marktüberblick | 5 KPI, chart spot 14j + forecast 10j avec band, forward curve EEX, commodities normalisées | Si pas de forecast: lancer `python run_pfc_production.py --short-horizon 10` |
| 📈 Kurzfristprognose | MAE 30j affiché, chart 10 jours avec P10/P90, table D+1, chart realisé vs forecast | Vérifier `lear_backtest_latest.parquet` présent |
| 💡 Lastprofil-Pricing | Bouton "Beispielprofil verwenden" coché par défaut, KPIs remplis, chart mensuel | Tester aussi avec un upload de vrai profil client |
| 📒 Ihre Transaktionen | Sélecteur 4+1 acteurs, KPIs, chart Vol/PnL mensuel, tableau deals filtrable, export CSV | Vérifier la détection des sheets "<Acteur> - Deals" |
| 🎚️ Programmqualität | 5 KPIs, pool comparison cards, **calendar heatmap** (le wow-effect), bias heatmap stunde × wochentag, time series triple, top 10 worst days, monthly accuracy | Si Budget manquant: les KPIs Budget sont absents — c'est OK |
| 🛡️ VaR & Stresstest | VaR/ES affichés, chart position mensuelle avec bande, **6 stress** dans le tableau, distribution historique | Si σ daily non finie: pas assez de spot, étendre le cache EPEX |
| 🌍 Marktdaten | KPIs, chart load + gen stack, flux frontaliers, hydro fill | OK si ENTSO récent |

### Phase 3 — Test du parcours démo complet (30 min)

Suis le parcours décrit dans `demo_deviwa/PRAESENTATION_DE.md`:
1. Cockpit → 2 min, basculer entre acteurs
2. Marktüberblick → 2 min
3. Kurzfristprognose → 3 min
4. Lastprofil-Pricing → 5 min, **upload un vrai profil**
5. Ihre Transaktionen → 5 min, drill-down par acteur
6. Programmqualität → 3 min, montrer les heatmaps
7. VaR & Stresstest → 3 min, jouer avec les paramètres
8. Marktdaten → 1 min (bonus)

Chronométre. Si > 25 min, raccourcir.

### Phase 4 — Polish (autant de temps que disponible)

Par ordre de priorité décroissante:

1. **Logo FMV**: si tu trouves un PNG du logo (probablement
   `static/fmv_logo.png` ou similaire), remplace l'en-tête texte dans
   `demo_deviwa/utils.py:render_header()` par une image+texte.

2. **Rafraîchir les données** avant la démo (à faire **lundi** soir):
   ```bash
   python -c "from pfc_shaping.data.ingest_epex import update_cache; update_cache()"
   python -c "from pfc_shaping.data.ingest_entso import update_cache; update_cache()"
   python -c "from pfc_shaping.data.ingest_forwards import update_cache; update_cache()"
   python run_pfc_production.py --short-horizon 10
   ```

3. **Proof-read du PRAESENTATION_DE.md** par un germanophone si dispo.
   L'allemand est volontairement simple mais peut être affiné.

4. **Cas limites** à corriger si tu les croises:
   - PFC ne couvre pas tous les mois Deviwa → Cockpit affichera "—" pour MtM.
     Idéalement étendre la PFC ou faire un fallback "Ø PFC × volume".
   - Deviwa n'a que 2026 → Programmqualität prendra 2025 (ProgrammeReal),
     vérifier que le filtre période fonctionne.
   - Acteur sans Budget → caption "Budget vs Real" disparaît, c'est OK.

5. **Add-on demandé par l'utilisateur** (si temps): "Do nothing
   counterfactual" sur le Cockpit. Calcul:
   ```python
   # P&L si tout avait été acheté au spot moyen au lieu d'avoir hedgé
   spot_avg = epex_ch["price"].mean()
   counterfactual_cost = total_intake_mwh * spot_avg
   actual_cost = sum(notional_intake)
   savings = counterfactual_cost - actual_cost
   ```
   Affiche en KPI: « **Ihre Hedging-Strategie hat bisher X EUR gespart** ».

### Phase 5 — Avant de quitter le projet

Commit + push tes changements sur `claude/review-repo-state-ftYuW`:
```bash
git add demo_deviwa/
git commit -m "Polish demo for Deviwa presentation"
git push -u origin claude/review-repo-state-ftYuW
```

## Choses à NE PAS faire

- ❌ Ne touche pas à `pfc_shaping/model/lear_forecaster.py` — un audit en
  cours, des fixes de leakage viennent d'être appliqués (commit e4fd93e).
- ❌ Ne touche pas au `dashboard/` interne, c'est l'outil des analystes.
- ❌ N'ajoute pas de nouvelle dépendance lourde sans demander (la démo
  doit rester légère).
- ❌ N'expose pas le serveur à l'extérieur (pas de `--server.address 0.0.0.0`,
  pas de tunnel ngrok). C'est localhost uniquement.
- ❌ Ne pousse pas sur `main`.
- ❌ N'écris pas dans `pfc_shaping/output/` (ces fichiers sont gérés par
  le pipeline FMV).

## Risques connus à monitorer

| Risque | Mitigation |
|---|---|
| Streamlit reload bug avec multipage et `st.cache_data` | Si une page reste figée, redémarrer le serveur |
| Plotly plotly>=6 + Streamlit<1.55: incompatibilités sur Heatmap | Si crash du Heatmap calendrier, downgrader plotly à 5.24 |
| Mauvaise détection tz dans Programme/Real | Si écarts énormes, vérifier la colonne Date (DD.MM.YYYY HH:MM) — peut être en CET vs UTC |
| Volume sign convention pour Withdrawal | Tester avec `Gesamter Pool` puis acteur seul: si totals incohérents, voir `deviwa_parser.py:summarize_actor` |
| Path imports cassés au lancement | Vérifier que toutes les pages ont `sys.path.insert(0, ...)` au début |

## Critère de succès

✅ La démo tourne du premier coup le mardi matin sur la machine du présentateur.

✅ Les 4 acteurs Deviwa voient leurs **vraies données** sur écran.

✅ Aucune erreur visible (rouge Streamlit) pendant la présentation.

✅ Le présentateur peut lire `PRAESENTATION_DE.md` sans surprise.

## Si tu as des questions

Toutes les décisions structurelles ont été prises. Si tu hésites:
- Garde la cohérence visuelle (couleurs FMV, formatage CH, allemand)
- Privilégie la stabilité au feature-creep (mardi est dans 4 jours)
- Pose la question à l'utilisateur avant de modifier plus de 50 lignes

Bonne démo ! 🇨🇭
