---
phase: 10
reviewers: [codex]
reviewed_at: 2026-05-20T20:21:07Z
plans_reviewed: [10-01-PLAN.md, 10-02-PLAN.md, 10-03-PLAN.md, 10-04-PLAN.md]
skipped_reviewers:
  - reviewer: claude
    reason: self-CLI rule (CLAUDE_CODE_ENTRYPOINT=claude-vscode — review.md requires independence)
  - reviewer: gemini
    reason: explicit user request "oublie Gemini"
codex_model: gpt-5.3-codex
codex_tokens: 92457
---

# Cross-AI Plan Review — Phase 10

Phase 10 = **PFC FMV Quality Scorecard (5-pillar SOTA replication)** — 4 plans séquentiels (waves 1→2→3→4), SC#1 Hildmann 4/4 PASS sur Config 4 = UNIQUE GATE, OMPEX comparison déférée à Phase 10B.

---

## Codex Review

> Codex CLI 0.125.0, model `gpt-5.3-codex`, reasoning effort medium, sandbox workspace-write, 92457 tokens. Invoked 2026-05-20T20:21:07Z.

## 1) Summary
Le découpage en 4 waves est globalement solide, très traçable, et bien aligné avec le pivot Phase 10 (scorecard qualité absolue, OMPEX déféré 10B). Les plans couvrent correctement les risques clés (Pitfall 1, IC95 defer, Config 3 stabilité, reproductibilité, budget compute), avec de bons checkpoints humains. Le principal risque n'est pas la conception technique, mais la **lourdeur d'exécution** et quelques ambiguïtés d'implémentation qui peuvent retarder la livraison (notamment agrégation multi-vintages, logique forwards fallback, et coût/fragilité des tests de reproductibilité).

## 2) Strengths
- Très bon alignement au gate unique SC#1 (Hildmann 4/4 sur Config 4).
- Excellente traçabilité décisions ↔ requirements ↔ tests ↔ livrables.
- Gestion explicite des risques majeurs (IC95 non supporté, Pitfall 1, DM edge cases, Config 3).
- TDD bien pensé avec tests unitaires + intégration progressive.
- Bonne séparation LT/CT respectée.
- Approche pragmatique pour 10B deferred et scope probabiliste limité à IC80.
- Reproducibility contract explicitement formalisé (atol/rtol).

## 3) Concerns
- **HIGH**: Over-engineering documentaire et procédural (beaucoup de contraintes/artefacts/checks) peut ralentir fortement l'exécution réelle.
- **HIGH**: Risque de non-atteinte SC#1 via Pitfall 1 (holiday/weekend ratio) malgré mitigation; décision threshold peut devenir politiquement sensible si recalibrée.
- **MEDIUM**: `derive_forwards_from_epex_hist` comme fallback peut biaiser Pillar 1/4 (proxy ≠ vrais forwards), avec risque de conclusions fragiles.
- **MEDIUM**: `run_scorecard_pillar_1` en mode mock PASS n'apporte pas de valeur gate; risque de faux sentiment de sécurité.
- **MEDIUM**: Reproducibility à `atol=1e-12` potentiellement trop strict sur chaînes pandas/statsmodels selon machine/version.
- **MEDIUM**: 96-build + caches volumineux + figures + markdown assembly: pipeline lourd, risque de timeout/instabilité opérationnelle.
- **LOW**: Some API assumptions (statsmodels f_test string constraint names) restent fragiles malgré test de signature.

## 4) Suggestions
- Réduire la complexité opérationnelle: fusionner certaines vérifications/doc updates pour éviter la surcharge process.
- Ajouter un **"go/no-go preflight"** unique avant 10-04 full run (data freshness, forwards path, disk space, runtime benchmark).
- Encadrer explicitement l'usage du fallback forwards proxy: marquer les résultats concernés comme "diagnostic only".
- Assouplir reproductibilité à double niveau: `1e-12` cible + fallback `1e-10` accepté avec justification.
- Rendre SC#1 uniquement basé sur run réel parquet (et non sur mock), et l'indiquer plus tôt dans Plan 10-02.
- Pour Pitfall 1: figer à l'avance la règle de recalibration (méthode unique) pour éviter décisions ad hoc en cours d'exécution.
- Prévoir un mode "fast rerun" standardisé (subset configs/vintages) pour debug rapide après incident.

## 5) Risk Assessment
**Overall risk: MEDIUM-HIGH.**
Le design est techniquement robuste et complet, mais la charge d'exécution est élevée et plusieurs points critiques (Pitfall 1 threshold, disponibilité/qualité forwards, reproductibilité ultra-stricte, coût compute 96 builds) peuvent bloquer ou retarder le gate SC#1. Si ces points sont verrouillés via un preflight strict et une politique claire de fallback/tolérance, le risque redescend à **MEDIUM**.

---

## Consensus Summary

**Single-reviewer pass** (codex only). Pas de cross-check avec d'autres modèles donc pas de "consensus" stricto sensu — toutes les findings ci-dessous sont des observations codex single-pass, à arbitrer au plan-phase --reviews iter.

### Strengths (highlighted by codex)
- 5 piliers SOTA alignment + traçabilité décisions/req/tests
- Risques majeurs explicitement gérés (Pitfall 1, IC95 defer, Config 3, reproductibilité)
- TDD progression + checkpoints humains bien placés
- Reproducibility contract formalisé (atol=1e-12 rtol=0)

### Concerns to address (codex single-reviewer pass — to triage at replan)

| # | Severity | Concern | Suggested Action (codex) | Triage |
|---|----------|---------|---------------------------|--------|
| 1 | HIGH | Over-engineering documentaire/procédural ralentit exécution | Fusionner certaines vérifications/doc updates | À évaluer : risk de coupes blanches vs gains tangibles |
| 2 | HIGH | SC#1 Pitfall 1 holiday/weekend ratio politiquement sensible si recalibré | Figer méthode recalibration ex-ante (1 règle unique) | À adresser Plan 10-01 — méthode de recalibration documentée AVANT mesure empirique |
| 3 | MEDIUM | `derive_forwards_from_epex_hist` fallback biaise Pillar 1/4 (proxy ≠ vrais forwards) | Marquer résultats fallback "diagnostic only" dans scorecard | À adresser Plan 10-01 et 10-04 (annotation dans `10-VERIFICATION.md`) |
| 4 | MEDIUM | `run_scorecard_pillar_1` mode mock PASS = faux sentiment sécurité gate | SC#1 uniquement sur run réel parquet, mentionné early dans Plan 10-02 | À adresser Plan 10-02 — clarifier que mock test = unit test, gate = real-run |
| 5 | MEDIUM | atol=1e-12 trop strict cross-machine pandas/statsmodels | Double-niveau cible 1e-12 + fallback 1e-10 justifié | À adresser Plan 10-04 Task 5 — relax fallback documenté |
| 6 | MEDIUM | Pipeline lourd 96-build + caches + figures + markdown | Preflight go/no-go unique avant 10-04 full run | À adresser Plan 10-04 — ajout task preflight (data freshness, forwards path, disk space, runtime bench) |
| 7 | LOW | statsmodels f_test string constraint name fragile | Test de signature déjà prévu Plan 10-02 — confirmer son suffisance | Acceptable as-is |

### Divergent Views
N/A (single reviewer).

### Suggested next step
Run `/gsd:plan-phase 10 --reviews` pour incorporer chirurgicalement les 7 concerns codex (5 actionnables HIGH/MEDIUM + 1 LOW acceptable as-is + 1 HIGH "over-engineering" à évaluer subjectivement). Le replanner décidera lesquelles intégrer comme edits surgicaux et lesquelles defer/accept-as-is.

---

*Phase 10 cross-AI review — codex single-pass (gemini skipped per user, claude skipped self-CLI). MEDIUM-HIGH risk verdict réduit à MEDIUM si Plan 10-01 preflight + recalibration ex-ante + fallback tolerance double-niveau sont wired.*
