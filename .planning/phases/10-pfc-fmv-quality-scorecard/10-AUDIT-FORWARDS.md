# Phase 10 — Audit du shaping des forwards (Pilier 1 Hildmann)

> **⚠ SUPERSEDED — 2026-05-30.** Le verdict initial ci-dessous (« aucun
> changement de modèle justifié ») a été **falsifié** par le perfect-foresight
> diagnostic construit ensuite : sous foresight parfait du niveau Cal, la
> corrélation mensuelle ne monte qu'à ~0.745 (et non ~0.95), ce qui prouve que
> le résidu vient du **profilage** et non du forecast des forwards.
> Voir `10-PERFECT-FORESIGHT-SHAPING.md` pour la méthodologie, la mesure et le
> stack SOTA qui en a résulté (regime-aware seasonal_ratios + hydro-aware
> peak_spreads + intra-day half-life=90d, ship: α=2.0, médian Cal 2025 0.918,
> 12/12 vintages clear the 0.85 gate). Le document ci-dessous est conservé pour
> la traçabilité historique.

**Date** : 2026-05-29
**Périmètre** : reproductibilité et validité du scorecard structurel SC#1 (Pilier 1),
focus sur le test `seasonal_profile` (échec 0.78 < 0.85) et le test `continuity`.
**Verdict (superseded)** : aucun changement de modèle justifié. L'échec du gate seasonal est un
artefact de fenêtre de validation, pas un défaut de forme des forwards.

---

## 1. Résultats vérifiés (autoritaires)

Le scorecard committé (`run_scorecard_full`, `forwards_source=real_eex_xlsx`,
24 vintages, env épinglé `pandas 2.3.3 / numpy 2.0.2`) est **correct et reproductible** :

| Test | Observé | Seuil | Pass |
|------|--------:|-------|:----:|
| arb_free | 1.275 | < 0.01 | ✗ |
| holiday_weekend | 0.7749 | [0.65, 0.95] | ✓ |
| seasonal_profile | 0.7815 | > 0.85 | ✗ |
| continuity | 0.046 | < 2.0 | ✓ |

## 2. Pièges de mesure (à ne PAS reproduire)

Trois divergences apparentes ont été tracées à des **erreurs de mesure**, pas à des
défauts du pipeline. Documenté ici pour éviter de futures fausses pistes :

- **Cache per-vintage périmé** → arb_free=36, seasonal=0.60. Toujours lancer en
  cache froid (`--fresh`) après modification d'un input.
- **Mauvais orchestrateur** : `run_scorecard_pillar_1` mesure la continuity sur
  `price_shape` (pas diurne 23h→00h ⇒ saut ~33 €/MWh) et la seasonal sur un seul
  agrégat. Le verdict réel vient de **`run_scorecard_full`** (continuity sur le
  backbone `B`, seasonal **par vintage**). N'auditer que via `run_scorecard_full`.
- **Environnement** : pandas 3.0.2 vs l'épinglé 2.3.3 donne des résultats
  **identiques à ~1e-11** — l'écart observé n'était pas dû à l'environnement.

## 3. Constat seasonal_profile (le cœur de l'audit)

Méthodo : chaque vintage comparé à sa fenêtre **trailing 2 ans** de réalisé EPEX ;
agrégat = **min** du Pearson sur 24 vintages.

- **22/24 vintages passent**, la plupart à **0.92–0.99**.
- Les 2 seuls échecs sont les **plus anciens** : `2024-01` (0.836) et `2024-02`
  (0.805). Leur fenêtre trailing est dominée par les **prix de crise 2022-2023**,
  non représentatifs ⇒ désaccord de forme Jan/Fév/Mar.
- **L'agrégat min est donc fixé par un artefact de fenêtre de référence**, pas par
  un défaut de shaping.

### Biais de forme Déc(+) / Sep(−) : régime-dépendant, pas un défaut de ratio

Résidu z signé (PFC vs réalisé trailing), moyenne sur 24 vintages :
**Déc = +0.28** (sur-pondéré, 79 % des vintages), **Sep = −0.21** (sous-pondéré, 75 %).

MAIS le signe **bascule selon le régime** (ex. z_Déc négatif sur les vintages
mi-2024, fortement positif en 2025 ; z_Sep très négatif mi-2024, ≈0 début 2025).
La PFC est **ancrée sur les forwards** (vue marché) et comparée au **réalisé** :
un z_Déc=+0.4 en 2025 signifie surtout que le **forward a price un premium d'hiver
décembre que le réalisé n'a pas tenu** — effet forwards-vs-réalisé, en partie
**irréductible**. Tuner `fit_seasonal_ratios` serait donc **incorrect** : on ne peut
pas écraser le niveau impliqué par les forwards.

## 4. Limitation du test continuity

`run_scorecard_full` mesure la continuity sur le backbone **`B` lissé par
construction** (cf. commentaire `scorecard.py` ~L1780). Un résultat quasi-nul
(0.046) **confirme que le lisseur fonctionne, il ne certifie PAS la continuité de la
HPFC livrée** aux frontières de mois (laquelle porte des pas de niveau saisonniers
légitimes de ~10-30 €/MWh). Le PASS est donc partiellement vacant. Choix de design
délibéré et documenté — signalé pour arbitrage.

## 5. Recommandations (décisions métier, pas de patch automatique)

1. **Ne pas modifier le modèle de shaping** : la forme va bien (22/24 ≥ 0.85, médiane ~0.94).
2. **Gate seasonal** : traiter l'échec comme une **limite connue** (fenêtre trailing
   chevauchant la crise sur les 2 premiers vintages), ou exclure explicitement les
   vintages dont la fenêtre de référence chevauche une rupture structurelle. Décision
   du propriétaire — ne pas ajuster le seuil 0.85 en douce pour faire passer le gate.
3. **Continuity** : si l'on veut certifier la HPFC livrée, ajouter une métrique de
   saut de niveau inter-mois **net de la saisonnalité** (diagnostic additif,
   non-gating, pour préserver le contrat de reproductibilité `atol=1e-12`).
