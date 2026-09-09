# Plan 10-04 — Reproducibility Exceptions [C5 REVIEWS]

**Purpose** : Documenter UNIQUEMENT les tuples (pillar_key, machine, libversion) où la
tolerance primaire `atol=1e-12 rtol=0` du contrat D-A6-3 ne tient pas, justifier le
fallback `atol=1e-10`, et tracer le signoff.

**Convention** : aucune exception silencieuse. Chaque ligne représente une décision
explicite signée par l'opérateur. Le test
`tests/test_phase10_reproducibility.py::test_reproducibility_subset_4_builds_config_4`
parse ce fichier via `_load_documented_exceptions` ; un pillar_key absent ici provoque
un FAIL si la tolerance 1e-12 échoue.

**Format strict** (parseable par `_load_documented_exceptions`) :

| pillar_key | machine | libversion | observed_delta | suspected_root_cause | signoff |
|---|---|---|---|---|---|

**État Plan 10-04** : *vide (aucune exception nécessaire — le contrat 1e-12 tient
sur l'environnement Mac Mini standard testé : Python 3.12.12 + pandas 2.x + statsmodels
0.14.x + numpy 2.x).*

**Si une ligne apparaît dans le futur** : c'est un signal qu'une investigation
determinism est due (Phase ultérieure). La forme attendue d'une entrée :

```
| pillar2_df | Mac Mini M2 Pro macOS 14.6 | pandas 2.2.3 + statsmodels 0.14.6 + numpy 1.26 | 1.3e-11 sur colonne mz_p_value | statsmodels OLS chain non-déterministe en chunked computation | julien 2026-MM-DD |
```

Les 4 pillar_key acceptés sont :
- `pillar1.arb_free`, `pillar1.holiday_weekend`, `pillar1.seasonal_profile`, `pillar1.continuity`
- `pillar2_df`, `pillar3_df`, `pillar4_df`
