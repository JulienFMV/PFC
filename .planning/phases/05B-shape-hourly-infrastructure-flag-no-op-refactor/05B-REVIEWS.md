---
phase: 5bis-A
reviewers: [codex, gemini]
reviewers_skipped:
  - claude: "self-skip — running inside Claude Code (CLAUDE_CODE_ENTRYPOINT=claude-vscode)"
reviewed_at: 2026-05-18T19:18Z
plans_reviewed:
  - 05B-01-PLAN.md
  - 05B-02-PLAN.md
  - 05B-03-PLAN.md
  - 05B-04-PLAN.md
  - 05B-05-PLAN.md
---

# Cross-AI Plan Review — Phase 5bis-A

## Codex Review

## Summary
Le découpage en 5 vagues est globalement solide: il sépare correctement le gel du baseline (Plan 01), l'infra de persistance (02–03), l'API/dispatch sans changement métier (04), puis la preuve de non-régression (05). L'intention "no-op first, math later" est claire et bien défendue, avec des critères d'acceptance précis. Le principal point faible est l'écart entre certains seuils (`1e-10` vs exigence no-op `1e-12`) et quelques choix d'implémentation qui peuvent introduire des dérives non voulues si non verrouillés (sérialisation/meta schema, legacy fixtures, signature dispatch).

## Strengths
- Séquencement logique et défendable (baseline avant code, puis infra, puis tests).
- Critères de succès détaillés et testables plan par plan.
- Très bonne prise en compte de la compat legacy (warning + fallback sans crash).
- Mécanique de flag bien pensée (precedence ctor/env, freeze-init, parquet-wins-on-load).
- Focus explicite sur LT-only et absence de contamination CT.
- Tests ciblés sur invariants clés (vue 3D, roundtrip complet, régression baseline paramétrée OFF/ON).

## Concerns
- **HIGH**: Tolérance de preuve no-op incohérente avec l'exigence. Plusieurs endroits utilisent `assert_frame_equal(..., atol=1e-10)` alors que vous imposez no-op `1e-12` (et parfois "bit-for-bit").
- **HIGH**: "Bit-for-bit" est affirmé mais non garanti par Parquet entre environnements/versions (metadata/engine/compression). Le contrat devrait être "numériquement identique" + schéma/index identiques, pas byte-equal.
- **HIGH**: Plan 01 inclut des assertions Git/historique trop fragiles (`git log ... wc -l == 1`, SHA exact) qui peuvent échouer pour de mauvaises raisons et bloquer inutilement.
- **MEDIUM**: Plan 02 sidecar en "single table polymorphe" (`attr` + colonnes variables) augmente le risque d'erreurs silencieuses de parsing/type cast.
- **MEDIUM**: `global_factors_` apparaît dans l'inventaire d'attributs mais pas clairement persisté/restauré; ambiguïté à lever pour "all trained attrs".
- **MEDIUM**: Test legacy fixture généré "à la main" peut diverger du vrai format historique (dtypes/index/order), réduisant la valeur de compat.
- **MEDIUM**: Capability check `inspect.signature(type(self.sh).apply)` peut rater certains wrappers/décorateurs; bon choix mais à tester avec implémentations réelles.
- **LOW**: Exigences de commit/ordre dans les plans (commit message exact, "single commit only") sont utiles mais lourdes pour l'exécution agentique.

## Suggestions
- Unifier le contrat no-op: utiliser partout `atol=1e-12` (ou expliciter pourquoi `1e-10` est acceptable) et ajouter `rtol=0`.
- Remplacer "bit-for-bit parquet" par:
  - `assert_frame_equal(..., check_exact=False, atol=1e-12, rtol=0)`
  - contrôle strict colonnes, dtypes, index freq/tz, tri.
- Pour `_meta.parquet`, préférer un schéma explicite par section (ou plusieurs sidecars) plutôt qu'un tableau polymorphe unique.
- Clarifier explicitement le statut de `global_factors_` (persisté/reconstruit) dans critères et tests.
- Générer le fixture legacy à partir d'un artifact réellement produit par code pre-5bis-A (ou commit fixture figé provenant de ce code), pas seulement "format-compatible".
- Ajouter un test "no hidden behavior branch" qui vérifie que `_use_seasonal_hourly` n'est lu dans aucun chemin math de 5bis-A.
- Rendre les garde-fous git moins brittles: vérifier "commit baseline parent des plans 02–05" plutôt que SHA/compte exact d'entrées log.

## Risk Assessment
**MEDIUM**.
Architecture de plan robuste et très bien spécifiée, mais le risque principal est contractuel: incohérence de tolérance no-op et sur-promesse "bit-for-bit" via Parquet. Si ces points sont harmonisés et les ambiguïtés de persistance levées, le risque descendrait à LOW.

---

## Gemini Review

## 1. Summary
The plan sequence is exceptionally well-structured, adhering to high-standard "quant" engineering principles. It correctly prioritizes **reproducibility** (frozen baseline) and **safety** (feature flags, roundtrip fidelity) over immediate behavioral changes. The split into 5 waves ensures that the infrastructure is verified as numerically inert before any "duck curve" deepening occurs in Phase 5bis-B.

## 2. Strengths
- **Numerical Hygiene:** The requirement to commit `baseline_pfc_seed42.parquet` *ahead* of any logic changes (Plan 01) is a critical safeguard. It ensures the "no-op" proof is anchored to a known immutable state.
- **Robust Persistence:** Fixing the pre-existing attribute loss bug (Plan 02) via the `_meta.parquet` sidecar is a high-value "cleanup" that prevents train/serve skew, which would have plagued the upcoming Phase 5 math changes.
- **Defensive Flag Mechanics:** The "freeze-at-init" and "parquet-wins-over-env" logic (Plan 03) is best-in-class for production models. It prevents subtle bugs where environment changes mid-run could lead to inconsistent outputs.
- **Surgical Code Quality:** Replacing the `try/except TypeError` in the assembler with explicit `inspect.signature` introspection (Plan 04) significantly improves the maintainability and debuggability of the curve-routing logic.
- **Rigorous Verification:** Plan 05 provides a comprehensive "proof suite." Parametrizing the baseline regression over both flag states (ON/OFF) provides a mathematical guarantee that the refactor is indeed a no-op.

## 3. Concerns
- **Floating Point Tolerance (MEDIUM):**
  - *Observation:* Plan 05 uses `atol=1e-10` for the baseline regression test, while Plan 01 uses `atol=1e-12` for idempotency.
  - *Risk:* For a "bit-pour-bit" no-op refactor, `1e-12` (near machine epsilon for float64) is preferred. `1e-10` might allow subtle drifts to pass.
- **Parquet Index Persistence (LOW):**
  - *Observation:* Plan 01 uses `index=True` for the baseline, but Plan 02 uses `index=False` for the `_meta.parquet`.
  - *Risk:* Inconsistency in index handling can sometimes lead to alignment issues during `load()`, though the plan seems to mitigate this with explicit sorting by `heure`.
- **Sidecar Naming Collision (LOW):** Plan 02 uses `_meta.parquet`. If future modules (e.g., `ShapeIntraday`) also need sidecars in the same directory, there might be collisions if they are saved to the same path.
  - *Mitigation:* `ShapeHourly.save` uses `Path(path).with_name("_meta.parquet")`. Since `path` is usually `shape_hourly.parquet`, the sidecar lives in the same folder — but a model-specific name would be safer.

## 4. Suggestions
- **Tighten Tolerance:** In `test_baseline_regression`, attempt `atol=1e-12` first; only relax to `1e-10` if cross-version pandas/pyarrow drift makes the tight tolerance unreliable in CI.
- **Sidecar Uniqueness:** Consider naming the sidecar `shape_hourly.meta.parquet` instead of `_meta.parquet` to avoid collision with other model components in the same storage directory.
- **Convergence Logging:** In Plan 04 (Assembler check), it would be beneficial to log *which* implementation was detected (e.g., `"Detected ShapeHourly (table-based) — outages_forecast ignored"`) to assist in production audits.

## 5. Risk Assessment
**Risk Level: LOW**

The sequence is extremely safe due to:
1. **Isolation:** Plan 01 anchors the project to a pre-refactor state.
2. **No Math Changes:** Every task explicitly forbids changing numerical output.
3. **Formal Verification:** Plan 05 provides the automated regression tests required to merge with confidence.

The implementation of `factors_3d_` as a lazy Mapping facade (Plan 04) is particularly low-risk as it doesn't touch the training or normalization logic.

**Verdict: Approved for execution.** The sequence perfectly implements the "no-op refactor first, math change second" strategy requested by the expert panel.

---

## Claude Review

*Skipped intentionally — this review session was launched from inside Claude Code (`CLAUDE_CODE_ENTRYPOINT=claude-vscode`). Per GSD's reviewer-isolation rule, the self-CLI is excluded for independence. The plan-checker pass that ran during `/gsd:plan-phase 5bis-A` was Claude-internal; that already reflects Claude's view.*

---

## Consensus Summary

Two external reviewers returned: Codex (MEDIUM risk) and Gemini (LOW risk).
Codex is more contractual/pessimistic; Gemini is more architectural/optimistic. Both endorse the structure.

### Agreed Strengths (Codex + Gemini)
- Sequencing baseline → infra → flag → API → tests is the gold standard for refactoring safety; "no-op first, math later" is correctly enforced.
- Plan 01's separate-commit-ahead baseline anchors the no-op proof to an immutable state.
- Save/load roundtrip on all trained attrs + legacy compat warning is a real bug fix that prevents train/serve skew.
- Feature-flag precedence (constructor → env, freeze-at-init, parquet-wins-on-load) is best-in-class for production models.
- Replacing `try/except TypeError` with `inspect.signature` capability check improves maintainability.
- Parametrized `test_baseline_regression[False|True]` is the right formal no-op proof shape.

### Agreed Concerns (highest priority)

| # | Concern | Severity | Source | Suggested Fix |
|---|---------|----------|--------|---------------|
| 1 | **Tolerance inconsistency** — must_haves cite `numpy.allclose(atol=1e-12)`, baseline regression uses `assert_frame_equal(atol=1e-10)`. | Codex HIGH + Gemini MEDIUM | Both reviewers | Try `atol=1e-12, rtol=0` first; document explicitly if relaxing to `1e-10` is needed for cross-version pandas/pyarrow drift. |
| 2 | **"Bit-for-bit" parquet wording is over-promised** — parquet byte equivalence isn't guaranteed across pandas/pyarrow/python versions. | Codex HIGH | Codex | Rewrite "bit-for-bit" / "byte-equivalent" language to "numerically identical (`assert_frame_equal(check_exact=False, atol=1e-12, rtol=0)` + identical columns/dtypes/index)". |
| 3 | **Plan 01 git assertions are fragile** — `git log ... \| wc -l == 1` and SHA-pinning will trip in worktrees / squash-merge flows. | Codex HIGH | Codex | Replace with parent-of-commit assertions: "the commit that introduces `baseline_pfc_seed42.parquet` is the parent of any commit modifying `pfc_shaping/lt/model/shape_hourly.py` or `pfc_shaping/lt/model/assembler.py`". |
| 4 | **Plan 02 polymorphic sidecar schema risks silent type cast** | Codex MEDIUM | Codex | Either (a) split `_meta.parquet` into per-attribute files, or (b) document the unified schema explicitly with per-row dtype assertions in `save()`. |
| 5 | **`global_factors_` persistence ambiguous** — listed in Plan 02 inventory but reconstructed via `_compute_global_fallback()` on load. | Codex MEDIUM + plan-checker | Both | Add explicit must_have in Plan 02: "`global_factors_` is intentionally NOT persisted; `load()` reconstructs it deterministically via `_compute_global_fallback()` and the test asserts equivalence." |
| 6 | **Legacy fixture authenticity** — Plan 05's hand-crafted `shape_hourly_legacy.parquet` may diverge from real pre-5bis-A artifacts. | Codex MEDIUM | Codex | Generate the legacy fixture by running `ShapeHourly.save()` from `main@28dfd65` (the actual pre-5bis-A code), commit the binary directly. Alternative: keep hand-crafted but explicitly call this out as a "format-compatible mock". |
| 7 | **Capability check fragility** — `inspect.signature(type(self.sh).apply)` misses decorators/wrappers. | Codex MEDIUM | Codex | Add a regression test that runs the capability check against the real `ShapeHourlyMLP` from the codebase, not just a synthetic mock. |
| 8 | **Sidecar naming collision** — `_meta.parquet` is generic; collides if a sibling model (e.g. `ShapeIntraday`) saves to the same dir. | Gemini LOW | Gemini | Rename to `shape_hourly.meta.parquet` (or follow `${stem}.meta.parquet` pattern derived from the main artifact filename). |
| 9 | **Parquet index handling inconsistency** — Plan 01 uses `index=True`, Plan 02 uses `index=False`. | Gemini LOW | Gemini | Document the rationale: baseline is a DataFrame with a meaningful DatetimeIndex (keep `index=True`); `_meta.parquet` is a flat dispatch table (keep `index=False`). Add an inline comment in `save()`. |
| 10 | **No detection logging in capability check** — operators won't see which `apply()` flavor is in use at runtime. | Gemini suggestion | Gemini | One-line `logger.info("Detected sh=%s — outages_forecast %s", type(self.sh).__name__, "passed" if accepts else "skipped")` at first dispatch. |

### Divergent Views
- **Risk level**: Codex MEDIUM vs Gemini LOW. Reflects the same data through different lenses — Codex weights contractual ambiguities (tolerance/wording/git), Gemini weights structural soundness (sequencing, formal verification). Both endorse the structure; the disagreement is about how much polish to add before execution.
- **`global_factors_`**: only Codex + the plan-checker flagged it. Gemini silent (likely because Plan 02's audit text resolved it in the explanation — but the *plan files themselves* don't yet say it).
- **TypeError propagation test** (plan-checker warning): neither external reviewer surfaced this. The risk is real but lower priority — keep on the optional fix list.
- **In-task refactor of `_generate_baseline.py`** (plan-checker warning): neither external reviewer surfaced. Lower priority cleanliness.

### Recommendation

Cheap, high-value fixes worth applying before `/gsd:execute-phase`:
1. **Harmonize tolerance** to `atol=1e-12, rtol=0` across must_haves and acceptance criteria (one find/replace per plan).
2. **Replace "bit-for-bit" / "byte-equivalent" wording** with "numerically identical (atol=1e-12, rtol=0, same columns/dtypes/index)" in all 5 plans + CONTEXT.md.
3. **De-brittle Plan 01 git assertions** — use parent-of-commit relation instead of `wc -l == 1` / SHA pin.
4. **Add `global_factors_` non-persistence must_have** to Plan 02 explicitly.
5. **Rename sidecar** to `shape_hourly.meta.parquet` (per Gemini) — also follow same pattern for future model components.

Optional (lower priority):
6. Capture legacy fixture from real `main@28dfd65` artifact, not hand-crafted.
7. Add operator log in Plan 04's dispatch.
8. Add TypeError-propagation test + `test_no_hidden_behavior_branch` per Codex's "no hidden behavior branch" suggestion.

Apply via either:
- `/gsd:plan-phase 5bis-A --reviews` — re-spawn the planner with REVIEWS.md as input (cleanest).
- Hand-edit the plans before execution (faster if you want to keep tight control).
