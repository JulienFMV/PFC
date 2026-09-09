# Session handoff - CH LT estimand and economic contract

Date: 2026-07-24

## Canonical state

- Repository: `C:\Users\jbattaglia\PFC_LT` only.
- Branch: `fix/lt-audit-remediation`.
- HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree remains intentionally very dirty. Do not reset, clean, restore or
  mass-stage it.
- No commit, staging, T057/future-holdout consumption, candidate promotion or
  production promotion was performed.
- `data/eex_forwards_history.parquet`, CT and Power BI were not edited by this
  slice. The protected parquet already appears modified in the inherited
  worktree and must not be touched or staged.
- Monthly solver remains deterministic level authority. OMPEX remains
  benchmark-only. Production remains strict `NO_GO`.

## What this slice closes

The next CH LT evaluation now has a full, explicit, hash-bound structural
estimand/economic-design contract before any new candidate or holdout use:

- complete local delivery months M01-M36 in buckets M01-M06, M07-M12,
  M13-M24 and M25-M36;
- separate non-compensable layers for CH EEX/solver consistency, direct CH
  hourly shape, one native CH quarter-hour product frozen before development,
  and joint probabilistic scenarios;
- deterministic monthly level from the solver, with full-price scenario
  ensemble expectation equal to the solver forward while preserving governed
  scenario-specific monthly-level risk;
- exact origin cadence/overlap/dependence, simultaneous-inference, power/MDE,
  calendar/strata and candidate/baseline obligations;
- FIL/ACC full-price economics with direction, Bloc 13 as a frozen payoff, and
  hydro decisions under a common optimizer/constraints/information class with
  model-specific non-anticipative actions and feasible clairvoyant regret;
- explicit FX convention for CHF metrics;
- OMPEX, neighbor/proxy truth, hourly-to-quarter-hour fabrication and
  `DEFAULT_CAPTURE_PREMIUM` forbidden as truth or scientific/economic evidence.

Schema v1 is deliberately incapable of scientific admission, execution,
production authorization or promotion. Supplying syntactically valid local
hashes does not change this because the external admission blocker is
unconditional.

## Canonical contract identities

- JSON:
  `.planning/phases/14-lt-audit-remediation/CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-DRAFT-20260724.json`
- explanatory note:
  `.planning/phases/14-lt-audit-remediation/CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-20260724.md`
- schema: `ch_lt_estimand_and_economic_design.v1`
- source document SHA-256:
  `4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931`
- canonical semantic SHA-256:
  `41ce07d1cf04e77a6936dc2d6f6fece387cbb415aa4588fa40406072e741b384`
- contract ID:
  `da4090073a4566f662e47fa59e206e1485d683305a1134f2089ebb13a4daa344`
- validator policy SHA-256:
  `52c90167c51779724509d8a69ecc368c77a547f2eb2f3f55dbac10b98185a276`

The validator closes the exact top-level, lifecycle, policy and evidence key
inventories. Rehashed unknown approval fields, Python bool/int equivalence,
mapping/byte rebinding, duplicate/non-finite JSON, linked input and weakened
policies fail closed.

## Fifteen current blockers

1. origin-available CH EEX vintage manifest;
2. monthly solver configuration manifest;
3. frozen candidate/baseline identity manifest;
4. direct CH hourly truth manifest;
5. direct native CH quarter-hour truth manifest;
6. exact statistical decision design manifest;
7. versioned calendar/strata manifest;
8. probabilistic scenario design manifest;
9. exact FMV profile/payoff population manifest;
10. frozen hydro dispatch policy manifest;
11. valuation and FX convention manifest;
12. FMV Risk economic MDE approval receipt;
13. exact origin/target/mask inventory;
14. post-evaluation market-consistency audit manifest;
15. independently signed external estimand-admission envelope.

## Packaged CLI and exact exit semantics

The governed wheel now includes:

- `pfc_shaping/validation/ch_lt_estimand_contract.py`;
- `pfc_shaping/cli/audit_ch_lt_estimand_contract.py`;
- entry point `pfc-lt-audit-estimand`;
- checkout wrapper `python -m scripts.audit_ch_lt_estimand_contract`.

Supported packaged/module command:

```powershell
python -B -m pfc_shaping.cli.audit_ch_lt_estimand_contract `
  --contract .planning\phases\14-lt-audit-remediation\CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-DRAFT-20260724.json `
  --expected-contract-sha256 4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931 `
  --mode validate-draft
```

Observed exit semantics on final source:

- `validate-draft`: exit `0`, structural validation only;
- default `admit-evaluation`: exit `3`, fifteen blockers, every authority false;
- invalid hash/bytes/schema: exit `2`.

Durable output additionally requires an absolute pre-provisioned `--audit-root`
and an absolute `.json` `--output` directly under the allowed Phase 14 audit
namespace. Existing output is never overwritten. The final wheel-runtime audit
is:

- path:
  `output/phase14/ch_lt_estimand_contract_audits/validate-draft-4209931e28a7c1cf.json`;
- SHA-256:
  `95898331006a995b25e94bad70f7cbd5aaadaedcbc69459de889a2a1675bb688`;
- operation ID:
  `d622e8ca9e0e5375ab0e637793cb3f0226bf1157331ec65ec1a818d30c73d2c4`;
- embedded runtime source revision:
  `04f2fa0f223ce3ccf49224650963ff221f01a63b688027f93270fa603ec1e11f`;
- status: `STRUCTURAL_ESTIMAND_DRAFT_BLOCKED_NOT_EXECUTABLE`.

The earlier local checkout audit
`validate-draft-5798bcbb5ffc6c6d.json` binds a superseded draft and reports
`source_revision=null`. Retain its append-only bytes as diagnostic history but
never select it as current evidence.

## Final wheel evidence

PEP 517 `pip wheel` first failed before build on the known sandbox-created
temporary-directory ACL issue, both under `%TEMP%` and a workspace-local
`TEMP/TMP`. Those attempts are non-conclusive and not counted.

Two fresh direct setuptools backend builds were then produced without reusing
build bases:

```powershell
python setup.py --quiet build `
  --build-base build\estimand-build-20260724-c `
  bdist_wheel `
  --bdist-dir build\estimand-bdist-20260724-c `
  --dist-dir build\estimand-wheel-20260724-c
```

The exact command was repeated with suffix `d`. This path emits the expected
setuptools deprecation warning; it is local artifact evidence, not the final CI
build method.

Final wheel C/D results:

- `fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- 437,672 bytes;
- 81 members;
- byte-identical SHA-256:
  `1ba55cafe85514d6030aa867cb1026d09c163fd651190363573f37abe96a89d5`;
- embedded source revision:
  `04f2fa0f223ce3ccf49224650963ff221f01a63b688027f93270fa603ec1e11f`;
- both `python -m scripts.check_lt_wheel_contract <wheel>`:
  `PASS`, `promotion_eligible=false`.

An isolated `python -I` smoke inserted only
`build/estimand-extracted-final-20260724`, imported the validator/CLI/build
identity from that root, observed admission exit 3, found the exact entry point,
loaded no `scripts` or CT module, and passed.

The local pip wheel-install operation generated all three Windows launchers,
including `pfc-lt-audit-estimand.exe`. Native execution of that `.exe` was
denied by the sandbox before process start. Therefore launcher E2E on a normal
Windows service host and PEP 517 two-clean-tree reproducibility remain external
IT qualification evidence, not a local pass. No escalation or production
installation was performed.

## Final verification commands and results

Focused contract/package suite:

```powershell
python -m pytest tests\test_ch_lt_estimand_contract.py `
  tests\test_lt_package_contract.py -q -p no:cacheprovider
```

Result: `45 passed in 2.50s`.

Runtime/packaging/publication-CAS matrix:

```powershell
python -m pytest `
  tests\test_snapshot_publisher_runtime_closure.py `
  tests\test_snapshot_publisher_artifact.py `
  tests\test_lt_package_contract.py `
  tests\test_snapshot_publisher_container_contract.py `
  tests\test_snapshot_publication_external_contract.py `
  tests\test_snapshot_anchor_client.py `
  tests\test_snapshot_anchor_reference.py `
  tests\test_snapshot_bootstrap_signer.py `
  -q -p no:cacheprovider
```

Result: `169 passed, 13 skipped in 72.99s`.

Final acquisition/monthly/cascade/probabilistic/shaping matrix:

```powershell
python -m pytest `
  tests\test_ch_lt_estimand_contract.py `
  tests\test_lt_package_contract.py `
  tests\test_dependence_power_supersession.py `
  tests\test_dependence_power_preflight.py `
  tests\test_ch_lt_pit_preregistration.py `
  tests\test_governed_lt_acquisition.py `
  tests\test_lt_input_sources.py `
  tests\test_governed_lt_input_snapshot_v2.py `
  tests\test_monthly_forward_curve_constraints.py `
  tests\test_monthly_forward_curve_solver.py `
  tests\test_monthly_forward_curve_priors.py `
  tests\test_monthly_forward_curve_integration.py `
  tests\test_monthly_curve_sensitivity.py `
  tests\test_cascading.py `
  tests\test_arbitrage_free.py `
  tests\test_probabilistic_output_governance.py `
  tests\test_assembler_profile_type.py `
  tests\test_intraday_amplitude.py `
  -q -p no:cacheprovider
```

Result: `430 passed, 3 skipped in 115.28s`.

Targeted Ruff, in-memory Python compilation and scoped `git diff --check` pass.
The direct pytest form follows a separate exact cwd/root guard because wrapping
pytest in the sandbox PowerShell process can create inaccessible basetemp ACLs.

## Independent final roasts

- Security/Governance: no residual P0/P1 in the corrected source/structural
  scope; filesystem ACL anchoring, external CAS/signature/SBOM and Windows
  power-loss durability remain external boundaries.
- IT/Operations: no residual P0/P1 source/package defect; local packaging GO,
  normal-host Windows launcher qualification still required; production NO_GO.
- Quant/Data: no residual P0/P1 after changing full-price scenario consistency
  from pathwise fixed levels to solver-consistent ensemble expectation and
  tightening origin/calendar/hydro freeze semantics; science remains NO_GO
  because all fifteen external proofs are absent.

## Files changed by this slice

- `pfc_shaping/validation/ch_lt_estimand_contract.py`
- `pfc_shaping/cli/audit_ch_lt_estimand_contract.py`
- `scripts/audit_ch_lt_estimand_contract.py`
- `tests/test_ch_lt_estimand_contract.py`
- `pfc_shaping/package_contract.py`
- `scripts/check_lt_wheel_contract.py`
- `tests/test_lt_package_contract.py`
- `pyproject.toml`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/phases/14-lt-audit-remediation/CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-DRAFT-20260724.json`
- `.planning/phases/14-lt-audit-remediation/CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-20260724.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff.

## Immediate next actions

1. Obtain exact FMV Risk-owned definitions/manifests for FIL, ACC, Bloc 13,
   hydro dispatch, valuation/FX and economic MDE. Do not infer them from current
   code defaults or OMPEX.
2. Acquire fresh direct CH hourly and native quarter-hour truth plus exact CH
   EEX vintage and solver/candidate/baseline manifests through the governed PIT
   path. Keep all post-origin spot outcomes scoring-only.
3. Freeze the exact origin schedule/masks, calendar/strata, statistical power
   and simultaneous-inference design, probabilistic scenario design and CPU/GPU
   reproducibility tolerances before model selection.
4. Implement the receipt-free plan core and independently signed external
   estimand-admission envelope. Schema v1 must remain non-executable.
5. On a normal Windows/CI host, run two clean-tree PEP 517 builds, offline
   install and native `pfc-lt-audit-estimand.exe` 0/3/2 tests, SBOM/scans and
   external signature/CAS qualification.
6. Only after those gates, build a new auditable CH candidate and prospective
   rolling-origin evidence. T057 remains locked and cannot repair missing
   historical robustness.

Do not promote production.
