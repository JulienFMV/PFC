# Audit · Couverture de données par acteur et impact sur le pool VaR

> Ce document fait suite à l'observation : la couverture temporelle des
> deals et programmes diffère significativement entre acteurs. Voici ce
> que ça veut dire pour les pages **Cockpit**, **Programmqualität**,
> **Pool & Risiko**, et comment le présenter le mardi sans mentir.

---

## 1. Cartographie de la couverture (telle qu'observée)

| Acteur | Deals | Programme | Couverture commune |
|---|---|---|---|
| **RELL** | 2026 | 2026 | **2026 uniquement** |
| **EW Binn** | 2026, 2027 (2 deals) | 2026, 2027 | 2026, 2027 |
| **EVTL** | 2026 → 2029 | 2026 → 2028 | 2026 → 2028 (gap 2029) |
| **EDSH** | 2026 → 2029 | 2026 → 2029 | **2026 → 2029** |

**Conséquence** pour le pool VaR (Cal-N = nombre d'acteurs participant) :

| Lieferjahr | Acteurs avec exposition non-nulle | Pool diversifiable ? |
|---|---|---|
| Cal-26 | 4 (tous) | ✅ Oui — bénéfice maximal |
| Cal-27 | 3 (RELL absent) | ✅ Oui — bénéfice réduit |
| Cal-28 | 2-3 (RELL + EW Binn absents si pas de carry-forward) | ⚠️ Bénéfice marginal |
| Cal-29 | 1-2 (EDSH seul + EVTL avec deals mais sans programme) | ❌ Pas de diversification réelle |

---

## 2. Comment le code gère ça aujourd'hui (déjà bon)

Le module `portfolio_yearly.py` (lignes 475-530) applique du **carry-forward
par acteur** avec une règle commerciale explicite :

> *« For years with no deals AND no programme, we deliberately do NOT
> invent a programme : extrapolating RELL's 62 240 MWh 2026 forecast
> to 2027-2029 would inject ~190 GWh of phantom long exposure into
> the pool risk and inflate VaR by ~15× without any underlying
> commercial truth. »*

Concrètement :
- **Carry-forward activé** uniquement pour les années où l'acteur **a au moins un deal**
- Pas d'invention de demande pour les années sans contrat
- EVTL aura son programme 2028 reporté vers 2029 (parce qu'il a des deals 2029)
- RELL n'aura pas son programme 2026 reporté ailleurs (pas de deals 2027+)

C'est **mathématiquement honnête**. L'effet : Cal-29 montrera quasiment
zero exposition pool ⇒ pas de bénéfice de diversification.

Dans `pool_diversification.py` (ligne 405-415), un commentaire explicite
gère le cas extrême : si **aucun acteur** n'est valorisable pour une année,
on retourne `NaN` plutôt que 0 — la page distingue alors « pas de couverture »
vs « risque nul » (deux états sémantiquement différents).

---

## 3. Stratégie démo : comment le présenter sans mentir

### Année phare : **Cal-27** (déjà le défaut)

C'est l'année où le narratif fonctionne le mieux :
- 3 acteurs participants (RELL est en livraison sur 2026, EW Binn + EVTL + EDSH ouverts pour 2027)
- Chiffres réalistes du bénéfice de diversification
- Forwards EEX Cal-27 actifs et liquides

**Phrase clé en allemand :**
> *« Für **Cal-27** sind drei von vier Akteuren noch offen — das ist
> das **typische Bild** eines aktiven Pools. »*

### Pour Cal-28 et Cal-29 : honnêteté + argumentaire commercial

Plutôt que cacher la couverture qui se réduit, **inverser le récit** :

> *« Sehen Sie : Cal-29 hat heute nur EDSH. **Wenn alle vier Akteure
> bis 2029 dabei sind**, sparen Sie noch mehr Risiko. Genau **deshalb
> macht der Pool Sinn**. Je länger Sie zusammen sind, desto mehr Vorteil. »*

→ Le manque de couverture devient un **argument de vente** pour étendre
l'engagement, pas un défaut à cacher.

### Pour Cal-26 (en livraison)

L'exposition ouverte est faible (la plupart des deals sont déjà partis
en livraison). Le bénéfice de diversification mesurable est mineur.
À mentionner au passage :

> *« Cal-26 ist bereits in Lieferung — hier sehen Sie das **Restrisiko**
> der noch nicht gelieferten Stunden. »*

---

## 4. Recommandations techniques (à faire avant mardi si possible)

### A. Bandeau de couverture sur la page Pool & Risiko

Ajouter un sous-titre dynamique sous le KPI principal :

> *« Cal-27 : 3 von 4 Akteuren beteiligt · 1 Akteur (RELL) bereits in Lieferung »*

ou en cas dégradé :

> *« Cal-29 : nur 1 Akteur — kein Pool-Diversifikations-Vorteil ⇒ **Aufruf zur Ausweitung** »*

Ça évite que l'auditoire se demande pourquoi le bénéfice diminue.

### B. Garde anti-mensonge

Si `valid` < 2 dans `compute_pool_diversification`, ne pas afficher
le KPI **« Diversifikations-Vorteil »** — afficher à la place un encart
**« Erweiterung empfohlen »**.

### C. Choix de l'année par défaut sur le Cockpit

Vérifier que la carte **Cal-27** est en évidence (la plus grande, en haut
ou en avant-plan) puisque c'est l'année où la pitch fonctionne. Cal-29
peut être grisée avec mention « Erweiterung empfohlen ».

---

## 5. Audit du Lastprofil-Pricing (page 3)

### Problème observé

L'**exemple synthétique** était ancré sur `pfc.index.min()`. Avec une PFC
qui démarre en 2025 (cas du local), le profil tournait sur 2025 — pas
de sens commercial puisqu'on **price pour le futur**.

### Correction appliquée

- Nouveau fichier : `data/samples/sample_load_2027_industrial.csv`
  - 8760 lignes horaires, 2027-01-01 → 2027-12-31, fuseau Europe/Zurich
  - 22.4 GWh/an, peak 5.36 MW, baseload 2.5 MW
  - Profil industriel réaliste : day-shift Mo-Fr, weekend −35 %, hiver +20 %,
    arrêt été 2 semaines (août), réduction Noël, jours fériés CH
  - Format : `Datum;Last_MWh` (séparateur `;`, décimale `,`) — convention CH
- Page 3 modifiée : si le CSV existe (priorité), il est chargé ; sinon
  fallback sur le générateur synthétique.

### Pour H:\ (à faire côté machine locale)

Le sandbox n'a pas accès à `H:\`. Sur votre machine, copier :

```
data\samples\sample_load_2027_industrial.csv
```

vers un emplacement H:\ partagé, par exemple :

```
H:\Energy\GeCom\MARCHE & NEGOCE\Demo_Deviwa\sample_load_2027_industrial.csv
```

Ou laisser dans `data/samples/` au sein du repo (déjà commité, automatiquement
détecté par la page 3).

### Profils additionnels possibles (post-démo)

Le script de génération est intégré au commit. Pour produire d'autres
profils (distributeur résidentiel ~5 GWh, gros industriel ~80 GWh), modifier
les constantes `BASELOAD_MW`, `DAY_SHIFT_BUMP_MW` et regénérer.

---

## 6. Tableau récapitulatif des actions

| Action | Statut | Où |
|---|---|---|
| Audit code carry-forward | ✅ Vérifié, déjà correct | `portfolio_yearly.py:475-530` |
| Audit pool VaR fallback NaN | ✅ Vérifié, déjà correct | `pool_diversification.py:402-415` |
| Profil 2027 industriel | ✅ Créé | `data/samples/sample_load_2027_industrial.csv` |
| Page 3 utilise le profil 2027 | ✅ Patché | `pages/3_Lastprofil_Pricing.py:165-220` |
| Bandeau de couverture sur Pool & Risiko | 📋 Recommandé | Page 6 |
| Garde anti-mensonge si <2 acteurs | 📋 Recommandé | `pool_diversification.py` |
| Phrase clé Cal-27 dans présentation | 📋 À ajouter au Sprechtext | `PRAESENTATION_15MIN_DE.md` |

---

## 7. Trois phrases pour la démo

1. *« Für Cal-27 sind drei von vier Akteuren noch offen — das ist das
   typische Bild eines aktiven Pools. »*
2. *« Wenn alle vier Akteure bis 2029 dabei sind, sparen Sie noch mehr
   Risiko. Genau deshalb macht der Pool Sinn. »*
3. *« Cal-26 ist bereits in Lieferung — hier sehen Sie das Restrisiko
   der noch nicht gelieferten Stunden. »*
