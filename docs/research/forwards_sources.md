# Phase 0 — Cadrage des sources EEX forwards

**Date :** 2026-05-05
**Owner :** E1 (Forward markets EEX) — reviewer E5 (Data architecture)
**Branche :** `claude/audit-pfc-forwards-q73iC`

---

## 1. Décision sources

Deux fichiers Excel maîtres maintenus par le desk FMV dans
`H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\` :

| Fichier | Couverture marché | Profondeur historique | Rôle |
|---|---|---|---|
| `Price_Report_EEX_Yearly.xlsx` | DE, CH, AT, IT, FR | snapshot courant + ~quelques mois | **source as-of multi-marché** — alimente la calibration PFC du jour |
| `Price_Report_EEX_CH_DE_Historique2019.xlsx` | CH, DE | depuis 2019 | **profondeur historique** — alimente fits saisonniers, peak ratios, structural break, backtest |

Le fichier legacy `Price_Report_EEX.xlsx` (codes `Y01_2027_BASE`) reste lisible en
mode rétro-compatible (déploiements existants), mais n'est plus la source canonique.

### Politique seasonal/peak ratios par marché

`ContractCascader.fit_seasonal_ratios()` et `fit_peak_ratios()` sont alimentés
par le **spot history** (EPEX Swiss / EEX DE) et non par les forwards. La
politique reste donc :

| Marché | Source spot pour fits | Politique de fallback |
|---|---|---|
| CH | EPEX Swiss (energy-charts) | — |
| DE | EPEX DE-LU (energy-charts/SMARD) | — |
| AT | EPEX AT (energy-charts si dispo) | fallback DE-LU (zone fortement couplée historiquement) |
| FR | EPEX FR (energy-charts) | fallback DE-LU (basis stable depuis CRE 2024) |
| IT | EPEX IT_NORD (GME via energy-charts si dispo) | fallback ratios maison à figer après Phase 2 |

Décision : **Phase 0 ne touche pas aux fits saisonniers**. La couverture spot
multi-marché est traitée en Phase 3 (activation production) avec des tests de
fallback explicites.

---

## 2. Contrat parser (entrée Phase 1)

### Format `Yearly` (nouveau, observé sur screenshot 2026-05-05)

Onglets : `DE`, `CH`, `AT`, `IT`, `FR` (un par marché).

Layout par feuille :

| Ligne | Colonne A | Colonnes B+ |
|---|---|---|
| 1 | _(vide)_ | ISIN EEX, ex. `DE000EEX0D8L1` |
| 2 | _(vide)_ | Libellé FR, ex. `Fév 27 BASE`, `Sep 27 BASE`, `Cal 28 PEAK` |
| 3 | `Date` | _(vide)_ |
| 4+ | `01.05.2024` (DD.MM.YYYY) | prix EUR/MWh, `0` si non quoté |

**Mapping libellé FR → code interne** (à implémenter en Phase 1) :

```
Mois FR → numéro :
{Jan:1, Fév:2, Mar:3, Avr:4, Mai:5, Juin:6,
 Juil:7, Aoû:8, Sep:9, Oct:10, Nov:11, Déc:12}

Trimestre FR → numéro :
{T1:1, T2:2, T3:3, T4:4}     (à confirmer Phase 1 avec snapshot réel)

Calendar :
"Cal 28 BASE" → "2028"
"Cal 28 PEAK" → "2028-Peak"

Trimestre :
"T2 27 BASE" → "2027-Q2"     (notation supposée — à vérifier)

Mois :
"Fév 27 BASE" → "2027-02"
"Fév 27 PEAK" → "2027-02-Peak"
```

⚠️ **Action Phase 1** : confirmer la notation des trimestres et la convention
d'année (`27` = `2027`, jamais `1927`) sur snapshot réel de la feuille DE.

### Format `Historique2019` (à confirmer Phase 1)

Hypothèse de travail : même layout que `Yearly` mais avec uniquement DE et CH
en onglets, et historique de marks remontant à 2019. À vérifier dès accès au
fichier.

### Format legacy (existant, à préserver)

`pfc_shaping/data/ingest_forwards.py:39-40` :
```
_EEX_PRODUCT_PATTERN = ^(Y01|Q\d{2}|M\d{2})_(\d{4})_(BASE|PEAK)$
```

Layout legacy :
- Ligne 1 = code produit `Y01_2027_BASE` directement (pas d'ISIN)
- Ligne 4+ = marks

### Sniff de format (logique Phase 1)

```
Lire ligne 1 colonnes B+ d'une feuille.
Si > 50% des cellules matchent regex ^[A-Z]{2}[0-9A-Z]{10}$  → ISIN → format `yearly`
Sinon, si > 50% matchent _EEX_PRODUCT_PATTERN              → format `legacy`
Sinon                                                        → ValueError explicite
```

En mode `yearly`, on lit la **ligne 2** comme source du libellé produit (pas
la ligne 1). L'ISIN est conservé comme clé secondaire pour traçabilité.

---

## 3. Inventaire à compléter sur poste FMV (action utilisateur)

Cette phase ne peut pas être validée à 100% depuis l'environnement Linux du
repo : il faut un sniff réel des deux fichiers depuis un poste avec accès
au lecteur `H:\` ou au chemin UNC `\\fmvfs1\data\Energy\…`.

**Script de sniff** (à lancer côté FMV avec `ppa_env` Python) :

```powershell
$env:PATH='C:\Users\jbattaglia\.conda\ppa_env;C:\Users\jbattaglia\.conda\ppa_env\Library\bin;C:\Users\jbattaglia\.conda\ppa_env\Scripts;' + $env:PATH
C:\Users\jbattaglia\.conda\ppa_env\python.exe scripts/phase0_sniff_forwards.py
```

Le script :
1. Ouvre les deux fichiers Excel (Yearly + Historique2019).
2. Pour chaque feuille, dump les 3 premières lignes × 10 premières colonnes en
   JSON Lines vers `docs/research/phase0_forwards_snapshot.jsonl`.
3. Détecte automatiquement le format (sniff ISIN vs legacy).
4. Vérifie présence des onglets attendus (`DE/CH/AT/IT/FR` pour Yearly,
   `DE/CH` pour Historique2019).
5. Logge les couvertures de dates par feuille.

Ce snapshot **fixe le contrat de tests unitaires** de la Phase 1.

---

## 4. Risques identifiés

| Risque | Impact | Mitigation |
|---|---|---|
| Notation trimestre non standard (`T1` vs `Q1` vs `Mars 27 BASE Q1`) | parser Phase 1 partiel | snapshot Phase 0 obligatoire avant codage |
| Onglets renommés par le desk (`DE-LU` au lieu de `DE`) | crash production | sniff insensible à la casse + alias `{"DE-LU":"DE", "ITN":"IT"}` |
| Cellules `0` = non-quoté vs vrai `0` €/MWh (prix négatifs prompt) | filtrage agressif perd info | Phase 1 ne filtre que `NaN` et `< -500`, pas le strict `> 0` actuel |
| Year-2-digit ambigu (`27` = 1927 ou 2027) | mapping faux Y+10 | clamp `year < 2000 → year + 2000` + assertion `2020 ≤ year ≤ 2040` |
| ISIN non présent en historique 2019 (codes EEX changés) | parser yearly inopérant sur Historique | accepter le mode `legacy` aussi sur ce fichier |

---

## 5. Définition of Done — Phase 0

- [x] Document de cadrage `docs/research/forwards_sources.md` (ce fichier)
- [x] Script de sniff `scripts/phase0_sniff_forwards.py` (livré côté repo)
- [ ] **Snapshot JSONL exécuté côté FMV** (`docs/research/phase0_forwards_snapshot.jsonl`)
- [ ] Validation du contrat parser (mois/trimestres/Cal en libellé FR confirmés)
- [ ] Décision actée par E5 sur la double-source (Yearly + Historique) en parquet historique unifié

Les deux dernières cases nécessitent l'exécution du script sur le poste FMV.

---

## 6. Sortie de phase → entrée Phase 1

Phase 1 ne démarre que si :
1. Snapshot JSONL est commit dans `docs/research/`.
2. Mapping mois/trimestre FR confirmé sur snapshot réel.
3. Couverture historique de `Historique2019` confirmée (date min, date max, nombre de produits).

Sinon Phase 1 risque de coder un parser sur des hypothèses fausses.
