# Audit · Défauts de calcul cachés sur la démo Deviwa

> Suite logique de l'audit Sensitivität. Même méthodologie : on cherche
> les **agrégations qui cachent une structure**, les **constantes magiques**,
> et les **formules qui donnent des chiffres défendables par accident**.

---

## Vue d'ensemble — gravité par défaut

| # | Page / Module | Défaut | Sévérité |
|---|---|---|---|
| 1 | Page 5 Programmqualität | Magic number `ASSUMED_IMBALANCE_PRICE = 150` | 🔴 CRITIQUE |
| 2 | Page 5 | MAPE explose sur petits dénominateurs | 🔴 CRITIQUE |
| 3 | Page 5 | « Genauigkeit = 100 − MAPE » incohérent | 🔴 CRITIQUE |
| 4 | Cockpit (`app.py`) | Do-nothing-Vergleich vs Ø spot historique | 🟠 ÉLEVÉE |
| 5 | Page 4 Transaktionen | Notional total = double-comptage Intake+Withdrawal | 🟠 ÉLEVÉE |
| 6 | Page 6 VaR | « Diversifikations-Vorteil » dépend de l'offset signe | 🟡 MOYENNE |
| 7 | Page 6 VaR | Vol = 20 % flat (pas de courbe par tenor) | 🟡 MOYENNE |
| 8 | Page 7 Marktdaten | Solar-share = somme/somme sans pondération | 🟡 MOYENNE |

---

## 🔴 1. ASSUMED_IMBALANCE_PRICE = 150 EUR/MWh (Page 5)

**Le problème.** Une constante en haut du fichier décide TOUS les chiffres EUR de la page :

```python
# pages/5_Programmqualitaet.py:51
ASSUMED_IMBALANCE_PRICE = 150.0  # EUR/MWh (Mittelwert CH 2024-2025)
```

Utilisée à 5 endroits : KPI principal, pool comparison, kumulierte Tageskosten, top-10 worst days, monthly accuracy. Si la constante est fausse, **toute la page raconte une histoire fausse**.

**Pourquoi 150 est défendable mais fragile :**
- L'ausgleichsenergie CH se situe entre 50 et 500 EUR/MWh selon l'heure
- Le **prix moyen pondéré** sur 2024-2025 est plutôt **~120 EUR/MWh** côté positif (over-consumed) et **~80 EUR/MWh** côté négatif (under-consumed)
- Le 150 est un mid-range pessimiste, mais sans citation

**Pire encore — la direction n'est pas modélisée.** Le code fait `Σ |Δ| × 150`, donc considère que **toute déviation coûte de l'argent**. En réalité :
- Si vous **sur-consommez** quand le marché est en surplus → vous **payez peu** voire **rien**
- Si vous **sous-consommez** quand le marché est en surplus → vous **revendez** à prix bas

Le code surestime structurellement le coût.

**Fix proposé :**
- Exposer `ASSUMED_IMBALANCE_PRICE` comme **slider sidebar** (50-300 EUR/MWh, défaut 120)
- Ajouter une note sur écran : « *basé sur Mittelwert Swissgrid 2024-25 — réelle Kosten variieren* »
- À long terme : utiliser le vrai prix Swissgrid heure par heure si dispo

---

## 🔴 2. MAPE explose sur petits dénominateurs (Page 5)

**Le problème :**

```python
# pages/5_Programmqualitaet.py:193
ape = abs_err / df["programme_mw"].abs().clip(lower=1e-3) * 100
accuracy = float(max(0.0, 100.0 - ape.mean()))
```

Pour EW Binn (peak ~0.4 MW), si à 03h00 le programme prévoit 0.05 MW et le réalisé fait 0.10 MW :
- Erreur absolue = 0.05 MW (anodine, < 100 W de marge)
- APE = 0.05 / 0.05 × 100 = **100 %**

Un seul outlier à 100 % APE peut faire chuter `accuracy` de 90 % à 50 %. **L'utilisateur voit 50 % et croit que sa qualité est mauvaise**, alors que l'erreur en MWh est minuscule.

C'est **pourquoi MAPE est universellement déconseillé** par les comités de mesure (M4, ESPN). On préfère **WAPE** (Weighted Absolute Percentage Error) :

```python
WAPE = Σ |err| / Σ |target|
```

WAPE pondère par le volume — un outlier sur 50 kW pèse 50 fois moins qu'un outlier sur 2.5 MW. C'est ce que tout fournisseur d'énergie utilise (Volue, Axpo, BKW).

**Fix proposé :**
```python
wape = float(abs_err.sum() / df["programme_mw"].abs().sum() * 100)
accuracy = max(0.0, 100.0 - wape)
```

Plus stable, plus défendable, **et donne typiquement un chiffre plus flatteur** pour le client (ce qu'on veut en démo).

---

## 🔴 3. « Genauigkeit = 100 % − MAPE » conceptuellement faux

Lié au #2. Avec MAPE qui peut dépasser 100 %, le `max(0, ...)` clamp à 0. Ça veut dire qu'une journée vraiment mauvaise vous laisse à 0 % définitivement, **peu importe** combien d'autres sont à 99 %.

**Fix lié au #2 :** WAPE est mathématiquement bornée à [0, 100 %] tant que les targets sont positifs et les erreurs absolues ne dépassent pas la somme des targets. La formule devient propre.

---

## 🟠 4. Do-nothing-Vergleich vs Ø spot historique (Cockpit)

**Le code (vue dans system-reminders) :**

```python
spot_avg_price = float(epex_ch["price"].mean())  # ← moyenne sur TOUT l'historique
counterfactual_cost = float(intake_volume.sum() * spot_avg_price)
actual_cost = float(intake_notional.sum())
hedging_savings = counterfactual_cost - actual_cost
```

`spot_avg_price` agrège depuis 2023 jusqu'à aujourd'hui. Pour un client qui hedge **2026-2029**, comparer son prix de couverture à la moyenne 2023-2026 du spot est **temporellement mal aligné** :

- En 2023, spot CH ≈ 95-110 EUR/MWh (post-crise)
- En 2024, spot ≈ 65-85 EUR/MWh
- En 2025-26 partiel : volatile

Le client peut voir « vous avez économisé 200 kEUR » alors que la vraie comparaison (vs forwards 2026 ou spot 2026 réalisé) donnerait un chiffre très différent.

**Fix proposé :**
- Pour les deals déjà livrés : comparer à la **moyenne spot CH sur leur période de livraison réelle** (et pas la moyenne globale).
- Pour les deals encore ouverts : pas de counterfactual réalisé, dire « in Berechnung — Lieferung läuft ».
- Ou bien : citer explicitement la fenêtre — « *Ihr Hedging-Preis vs Ø Spot über die Lieferperiode = X EUR gespart* » (transparent).

---

## 🟠 5. Notional total double-compte Intake et Withdrawal (Page 4)

```python
# pages/4_Ihre_Transaktionen.py:103-112
vol_total = float(...volume_sum...abs().sum())     # somme des |volumes|
notional_total = float(...notional_sum...abs().sum())  # somme des |notionals|
avg_price = notional_total / vol_total
```

Pour un client qui a 1000 MWh Intake @ 80 € et 500 MWh Withdrawal @ 90 € :
- `vol_total` = 1500 MWh (bien)
- `notional_total` = 80'000 + 45'000 = **125'000 EUR**
- `avg_price` = 125'000 / 1500 = 83.3 EUR/MWh

Mais le client n'a **pas** dépensé 125 kEUR — il a dépensé 80 kEUR pour acheter et reçu 45 kEUR pour vendre, soit un net cash de **35 kEUR sortants**. Le « avg_price 83.3 » n'a pas de sens commercial : c'est la moyenne pondérée des deux directions.

**Fix proposé :**
- Séparer **« Ø Einkaufspreis »** (Intake only) et **« Ø Verkaufspreis »** (Withdrawal only)
- Si un seul des deux existe (cas typique GRD pure consommateur), n'afficher que celui-là
- Le « Notional gesamt » devient « **Netto-Cashflow** » signé

---

## 🟡 6. « Diversifikations-Vorteil » dépend du sign offset (Page 6)

Le module documente honnêtement que le bénéfice vient de **sub-additivité** :
```
Σ |e_i| ≥ |Σ e_i|     (égalité ssi tous les signes sont identiques)
```

Pour les 4 distributeurs Deviwa, **tous sont net long** (under-hedged → exposition positive au prix). Donc `Σ |e_i| ≈ |Σ e_i|` et le bénéfice tombe à zéro / négligeable.

Le code détecte ce cas (ligne 153) et affiche un message gris « *Für Cal-X haben alle Akteure das gleiche Vorzeichen — kein Diversifikations-Vorteil aus dem natürlichen Long/Short-Offset* ».

**Le risque commercial :** si le client demande « pourquoi 28 % sur Cal-27 et 0 % sur Cal-28 ? », il faut savoir répondre. La **vraie diversification** entre distributeurs viendrait de la **corrélation des charges** (modèle 2-factor price+volume), pas du modèle actuel.

**Fix proposé :** ajouter une phrase explicative dans le titre de la KPI :
> *« Diversifikations-Vorteil aus signiertem Netto — nicht aus Last-Korrelation »*

et conserver le message gris quand le bénéfice est nul.

---

## 🟡 7. Volatilité 20 % flat (Page 6)

```python
# pool_diversification.py
vol_annual_pct = 20.0  # constant
sigma_eur = abs(exposure_eur) * (vol_annual_pct / 100.0) * math.sqrt(horizon)
```

Vol forward annualisée 20 % constante pour Cal-26..Cal-29. En vrai :
- Cal court terme (proche livraison) : σ ≈ 25-30 %
- Cal moyen : σ ≈ 18-22 %
- Cal long : σ ≈ 12-18 %

L'effet sur la VaR est ±25 % selon le tenor. Pour la démo c'est acceptable (« hypothèse standard 20 % » défendable), mais à mentionner si question.

---

## 🟡 8. Solar-share Page 7

```python
# pages/7_Marktdaten.py:71
solar_share = 100 * entso_sl["solar_mw"].sum() / max(entso_sl["load_mw"].sum(), 1e-9)
```

Si `solar_mw` a des `NaN` (heures non publiées), la somme les ignore mais la `load_sum` les compte. Pas critique sur ENTSO récent qui est complet.

---

## Priorités de correction

| # | Action | Effort | ROI démo |
|---|---|---|---|
| 2+3 | **WAPE au lieu de MAPE** sur Page 5 | 15 min | 🔥 stabilité KPI |
| 1 | **Slider imbalance price** + caption | 20 min | 💡 honnêteté |
| 5 | **Split Ø Einkauf / Verkauf** Page 4 | 20 min | 📊 lisibilité |
| 4 | **Counterfactual sur fenêtre alignée** | 30 min | 🎯 défendabilité |
| 6 | Note explicative sur Diversifikations-Vorteil | 5 min | 🎤 anticiper Q&A |

Je propose de fixer **2+3 et 1** maintenant (45 min combinés) puisque c'est la page Programmqualität qui est la plus exposée aux questions hostiles.

Le reste documenté ici est pour post-démo ou réponse de Q&A.
