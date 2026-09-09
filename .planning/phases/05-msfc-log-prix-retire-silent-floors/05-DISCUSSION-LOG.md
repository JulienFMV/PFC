# Phase 5: MSFC retire silent floors + PFC peut être négative - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in 05-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-19
**Phase:** 5 - MSFC log-prix + retire silent floors
**Areas discussed:** MSFC methodology, Floor strategy, F_WV multiplicatif vs additif, Peak synthesis + validation fixtures

---

## Area 1 — MSFC log-prix : methodology

| Option | Description | Selected |
|--------|-------------|----------|
| (a) Linéaire + retire floors | MSFC reste linéaire (PCHIP). NEG-05 garanti par construction (exact repricing). Titre "log-prix" droppé. Smoothness proportionnelle (TODO P1-01) hors scope. | ✓ |
| (c) Asinh-transform k=30 | Move MSFC à asinh(B/k) smooth log-like des 2 côtés. Hyperparam k à calibrer. +1 plan complexité. | |
| (d) Linéaire MSFC + log m_factor | MSFC inchangé, move arbitrage_free à log m_factor (cible TODO P1-01). Hors scope strict "retire floors". | |
| Combinaison ou autre | | |

**User's choice:** (a) Linéaire + retire floors (Recommended)
**Notes:** Le titre ROADMAP "MSFC log-prix" est pré-audit, requirements (NEG-01..05) sont autoritative. NEG-05 (exact repricing en signed) garanti naturellement par PCHIP linéaire. log(prix négatif) non défini → asinh nécessiterait k à calibrer + transformation inverse ajoute risque erreur repricing. Smoothness proportionnelle TODO P1-01 reste hors Phase 5.

---

## Area 2 — Floor strategy : 1 flag vs N flags

| Option | Description | Selected |
|--------|-------------|----------|
| (4-i) ctor args defaults OFF + flag audit | enforce_*=False sur 4 classes (negative-ready par défaut, NEG-01 littéral). Master flag PFC_LT_ALLOW_NEGATIVE_PRICES = audit-trail INFO log only, pas propagation. Rollback opérateur = enforce_*=True explicite aux 4 callsites. | ✓ |
| (4-ii) ctor args defaults ON + flag forces OFF | Pattern 5bis-A : defaults legacy (floors ON), flag PFC_LT_ALLOW_NEGATIVE_PRICES=1 désactive. PFC reste positive par défaut, math change gated. | |
| (3) Master flag + 4 sub-flags overrides | PFC_LT_ALLOW_NEGATIVE_PRICES master + 4 env-vars granulaires. Symétrique 5bis-B sigma_off/_on pattern. | |
| Autre lecture | | |

**User's choice:** (4-i) ctor args defaults OFF (negative-ready)
**Notes:** Lecture littérale NEG-01 "default off for LT" = option default False = floor désactivé par défaut. Pattern délibéré pour éviter silent-revert : si opérateur veut legacy, il touche 4 callsites explicitement (audit-able git). Master flag info-only. Baseline 5bis-A reste verte sur forwards positifs synth (floors ne mordent pas).

---

## Area 3 — F_WV multiplicatif vs additif en régime négatif

| Option | Description | Selected |
|--------|-------------|----------|
| (a) Delta additif (f_wv-1)×\|B\| | Réécrit en delta additif scalé. Sémantique économique correcte 2 côtés. Pas de re-fit β_WV. F_WV_FLOOR retiré par construction. Équivalence baseline_5bisA NON exacte (à mesurer en research). | ✓ |
| (c) Désactiver f_wv sur LT mid-market | Réponse littérale NEG-03. WaterValueCorrection retiré pipeline export. Simple mais perd correction hydro CH calibrée. | |
| (b) Skip f_wv par timestamp si sign(B)<0 | Garde multiplicatif sur B>0, skip B<0. Hybride asymétrique. Couverture inégale. | |
| Autre | | |

**User's choice:** (a) Delta additif
**Notes:** Math sign-invariant naturelle. Coefficient β_WV fitté reste valide. `delta_wv = (f_wv-1) × |B|` : scarcity rapproche de 0 (correct), abundance enfonce (correct) en régime négatif. Research dry-run mesure l'écart avec baseline_5bisA. Si >1e-12 → 2 baselines (legacy preservé via enforce_*=True + nouvelle phase05 canonique).

---

## Area 4 — Peak synthesis (4a) + validation fixtures (4b)

### 4a — Peak synthesis

| Option | Description | Selected |
|--------|-------------|----------|
| (a) Spread additif | fit_peak_spreads (€/MWh) remplace fit_peak_ratios. result = base + spread. Sign-invariant. | ✓ |
| (c) Garder ratio + gate flag | allow_negative_peak=True default. Legacy ratio*price. Sémantique fausse en négatif (Peak<Base). | |
| (d) Skip synth si Base<0 | Conservatif. Perd résolution Peak/Offpeak en négatif. | |
| Autre | | |

**User's choice:** (a) Spread additif
**Notes:** Extension naturelle ratio→spread, sign-invariant. Sur historique positif, spread typiquement +3 à +8 €/MWh. Migration 1 module (cascading.py). Backward-compat dual-attribute si peak_base_ratios_ utilisé en aval.

### 4b — Validation fixture pour NEG-05 (Cal'27=-10)

| Option | Description | Selected |
|--------|-------------|----------|
| (a) Synthetic mirror seed=42 | Duplicate baseline_pfc_seed42, inject Cal'27=-10. | (initially proposed, rejected by user) |
| (c) Synth + 2020-Q2 manuel | CI sur synth + validation manuelle 2020 réel. | |
| (d) Générateur paramétré | _generate_negative_fixture.py paramétré. | |
| **Reformulation après catch user** | Pas de fixture Cal=-10 (non-réaliste). 4 unit tests math sign-invariance + 1 system acceptance SC #2 ROADMAP (forwards réalistes Cal=30, July=20, bowl 5bis-B → h13 Sunday < -20). | ✓ |

**User's choice:** Reformulation (4 unit tests + 1 system acceptance gated par 5bis-B)
**Notes:** User a flag que NEG-05 dans REQUIREMENTS.md ("Cal'27=-10") est non-réaliste : Cal annuels ne sont jamais négatifs, ce sont des heures qui le sont. Erreur d'agent (literal reading sans sanity-check métier). Reformulation : 4 unit tests math + 1 system acceptance avec fixture réaliste forwards_phase05_seed42 (Cal positif, July dépressif). NEG-05 wording REQUIREMENTS.md à reformuler en deferred. SC #2 ROADMAP gated par "5bis-B bowl calibrated".

---

## Claude's Discretion

- Format exact du log telemetry (INFO vs DEBUG, format string)
- Sidecar persistence pour PFC_LT_ALLOW_NEGATIVE_PRICES : étendre shape_hourly.meta.parquet OU créer assembler.meta.parquet (trancher au planning research)
- Pattern Python pour backward-compat shim fit_peak_ratios deprecated → spreads (raise / shim / alias WARN)
- Variable de margin pour le clamp signed-aware msfc_spline.py:120 (0.5*np.ptp(y_knots) proposé)

## Deferred Ideas

- Reformulation NEG-05 dans REQUIREMENTS.md (immediate, à fixup pré-planning ou dans Plan 05-01)
- Phase 5ter distribution probabiliste : réutilise infra Phase 5
- Phase 10 backtest réel HFC OMPEX 2024-2025 : real-data validation gate
- TODO P1-01 smoothness proportionnelle log-space : phase ultérieure
- Hot-rollback opérateur `PFC_LT_FORCE_LEGACY_FLOORS=1` : si demande post-merge (~10 lignes)
- 2020-Q2 historical real-data validation : pas CI strict, manuel VERIFICATION.md
- Cleanup ROADMAP title "MSFC log-prix" → "MSFC retire silent floors + PFC peut être négative"
