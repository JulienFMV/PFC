# Session handoff - AFRY calendar-free shape diagnostics v3

Date: 2026-08-03  
Branch: `fix/lt-audit-remediation`  
HEAD observed throughout: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Workspace: `C:\Users\jbattaglia\PFC_LT`

## Outcome

Batch 3 is closed as local descriptive evidence only. Two independent
diagnostic builds are byte-identical with content ID:

`8ef0290502359b6e1e16093bcf34e6400d8e74dad23f9b1baa091e0a81a372f6`

Canonical paths:

- `build/afry-shape-diagnostics-final-v3-a/8ef0290502359b6e1e16093bcf34e6400d8e74dad23f9b1baa091e0a81a372f6`
- `build/afry-shape-diagnostics-final-v3-b/8ef0290502359b6e1e16093bcf34e6400d8e74dad23f9b1baa091e0a81a372f6`

Both report:

- `PASS_LOCAL_DIAGNOSTIC_ONLY`;
- `NO_GO_MODEL_AND_PRODUCTION`;
- `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.

The source catalog ID is
`a7ca9238ad715e67269c73a862607c3a3301358821818bc5d2fa4f1581c54454`.
The diagnostics verify against the canonical v6-A/v6-B source paths even
though the final diagnostic materialization commands used byte-identical
v5-A/v5-B copies. Source paths are intentionally outside the deterministic
diagnostic content ID; the caller must supply the independent source anchor.

No PFC model, monthly solver, candidate, T057 outcome or production state was
changed. EEX and ENTSO-E Databricks access was not available and was not
substituted with legacy or synthetic data.

## Restricted qualitative result

The bundle contains 440 scenario/delivery/weather diagnostics, 88 unweighted
weather summaries and 420 adjacent-available-year changes. Across all four
vendor scenarios:

- deepening followed by recompression is present for every weather pattern;
- negative-price share is non-monotone for every weather pattern;
- duck-spread and negative-share weather dispersion are non-zero for every
  scenario/delivery-year summary.

Interpretation is exactly
`DESCRIPTIVE_VENDOR_OUTPUT_NOT_EMPIRICAL_OR_CAUSAL_VALIDATION`. No restricted
numeric table was copied into Git, documentation or logs.

## Files changed for this batch

- `AGENTS.md` - mandatory future-agent routing and immutable AFRY blocker rules
- `pfc_shaping/validation/afry_shape_diagnostics.py`
- `scripts/build_afry_shape_diagnostics.py`
- `tests/test_afry_shape_diagnostics.py`
- `tests/test_afry_scenarios.py` - one fixture correction from `len(table)` to
  `table.num_rows`; production catalog code was not changed by this fix
- `.planning/phases/14-lt-audit-remediation/AFRY-CH-2026-Q2-SHAPE-DIAGNOSTIC-CONTRACT-V1.json`
- `.planning/phases/14-lt-audit-remediation/AFRY-CH-2026-Q2-AGENT-DATA-CONTEXT.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

Observed final SHA-256 values:

- diagnostic module:
  `45b4f6e8db718b342975a7c32706a5ac5b5eb38fe4b4db6106c38bd991136d7f`;
- diagnostic entrypoint:
  `8264c97ea41852c1e4fca08e81d854627ae2d9a4d1e24397a865e3451db69715`;
- source catalog module:
  `366603118ca25382e7f688d484154ab5e86f041a1cb9c528129c2bb4f477d9ed`;
- diagnostic contract:
  `5730e6f96c5b758850e782f488feb9b219807d5a2bbe3727025fc0d7b7827ea3`;
- diagnostic manifest:
  `2f04306c23e59de78f7666863cb2ef2e5ef4598666a56bd198a24ef73adca826`
  over 3,163 bytes.

The bound runtime dependency tree is
`1695e210c93e93e5806f7d5e198576289016d1e384fd3403479fdc9e65f35caa`
over 1,429 files and 45,398,809 bytes.

## Contract and security closure

The final verifier:

- requires exact canonical workspace, observed cwd and Git top-level;
- requires a caller-supplied current source-catalog anchor;
- rejects source/output overlap, symlinks and Windows reparse ancestry;
- binds the current contract, diagnostic module, CLI and source catalog
  module;
- binds the repo-local Python executable and full dependency-tree receipt;
- captures stable bytes and replays all three Parquet frames plus the audit in
  memory from the two exact source artifacts;
- checks exact member, hash, schema, row, authority, finiteness, share-bound,
  key and term-arithmetic closure;
- adopts an identical concurrent publication only after verification;
- quarantines an invalid content-addressed final slot instead of deleting it;
- keeps verification side-effect-free.

Security and IT/Operations first roasts demonstrated P1 findings in these
areas. After correction, both independent final re-roasts report no residual
P0/P1 for the claimed local `NO_GO` boundary. External signed CAS/WORM,
portable CI, retention, observability and rollback remain production gates,
not claims of this local diagnostic.

## Commands and results

Every shell command first checked exact cwd and Git root. Mutable paths stayed
below `build/`.

Source catalog unit matrix after the annual-contract migration:

```powershell
build\pytest-runtime-v2-final\python.exe -I -B -m pytest tests\test_afry_scenarios.py tests\test_afry_scenario_extraction.py -q -p no:cacheprovider --basetemp build\pytest-afry-source-current-v7
```

Result: `72 passed in 72.59s`.

Source materialization was executed twice with distinct repo-local TEMP/TMP
and output roots:

```powershell
build\pytest-runtime-v2-final\python.exe -I -B -m scripts.materialize_afry_scenario_catalog --registration .planning\phases\14-lt-audit-remediation\AFRY-CH-2026-Q2-SOURCE-REGISTRATION.json --output-root build\afry-catalog-final-v5-a
build\pytest-runtime-v2-final\python.exe -I -B -m scripts.materialize_afry_scenario_catalog --registration .planning\phases\14-lt-audit-remediation\AFRY-CH-2026-Q2-SOURCE-REGISTRATION.json --output-root build\afry-catalog-final-v5-b
```

Results: identical source ID `a7ca9238...c54454`, integrity PASS and model /
production `NO_GO`. Canonical v6-A/v6-B copies have the same 11 byte-identical
members and catalog SHA-256
`f7aa9bd4d3cc0f13851225809b0271a138bb733c51fd563d9594508b32f65c9a`.

Diagnostic materialization:

```powershell
build\pytest-runtime-v2-final\python.exe -I -B -m scripts.build_afry_shape_diagnostics --source-catalog-bundle build\afry-catalog-final-v5-a\a7ca9238ad715e67269c73a862607c3a3301358821818bc5d2fa4f1581c54454 --output-root build\afry-shape-diagnostics-final-v3-a
build\pytest-runtime-v2-final\python.exe -I -B -m scripts.build_afry_shape_diagnostics --source-catalog-bundle build\afry-catalog-final-v5-b\a7ca9238ad715e67269c73a862607c3a3301358821818bc5d2fa4f1581c54454 --output-root build\afry-shape-diagnostics-final-v3-b
```

Results: identical diagnostic ID `8ef02905...372f6`; six members compared,
zero differences. Both also verify with the canonical v6 source anchors.

Dedicated diagnostic tests evolved from 12 to 24 cases. Final result:
`24 passed`; targeted Ruff: `All checks passed!`; all three AFRY JSON contracts
parse with `json.tool`.

Integrated matrix:

```powershell
build\pytest-runtime-v2-final\python.exe -I -B -m pytest tests\test_afry_scenarios.py tests\test_afry_scenario_extraction.py tests\test_afry_shape_diagnostics.py tests\test_electrification_scenarios_data.py tests\test_electrification_shape.py tests\test_lt_ct_imports.py tests\test_lt_package_contract.py -q -p no:cacheprovider -m "not slow" --basetemp build\pytest-afry-shape-batch3-reclosure-final
```

Result after adding the future-agent governance test:
`173 passed, 4 skipped in 72.35s`.

## Fail-closed evidence and corrections

- Older catalog v3 was rejected after the verifier schema moved forward.
- Two catalog rebuilds failed after about four minutes because
  `afry_scenarios.py` changed during materialization; no final bundle was
  published.
- A later source catalog was rejected because the newly required annual
  contract was absent. The source matrix was allowed to return green before
  rebuilding.
- One migrated test incorrectly used `len(pyarrow.Table)` when constructing a
  row-length authority-flip vector. It was corrected to `table.num_rows` and
  the full source matrix passed.
- Initial Security/IT roasts found self-authenticated output forgery,
  incomplete runtime closure, reparse/TOCTOU, missing canonical library guard,
  path overlap and publication-race weaknesses. All demonstrated P1 findings
  were corrected and re-roasted.
- Older diagnostic IDs, including `ab8e0fdf...f2bdc`, lack the current three-
  producer/runtime/replay closure and are superseded.

## Residual risks and next action

Batch 4 must not begin until governed EEX and ENTSO-E Databricks access is
available. Once available:

1. register and hash the point-in-time EEX and ENTSO-E inputs without exposing
   tokens or values;
2. establish an authoritative Swiss delivery-calendar mapping, including the
   current hourly regime and the separately verified future 15-minute
   transition;
3. compare AFRY-derived hypotheses against simple CH baselines in
   rolling-origin evaluation;
4. use a new independently frozen future holdout; T057 stays sealed;
5. only then consider a candidate shaping feature or scenario crosswalk.

No commit or staging was performed. `data/eex_forwards_history.parquet`, CT and
Power BI were not touched by this batch. Production remains strict `NO_GO`.
