# Session handoff - monthly sensitivity and packaging reclosure - 2026-07-24

## Authority and workspace

- Canonical workspace: `C:\Users\jbattaglia\PFC_LT`; never use the old `H:` repo.
- Branch: `fix/lt-audit-remediation`.
- HEAD observed throughout: `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree is intentionally very dirty. No reset, clean, restore, commit,
  staging or promotion was performed.
- `data/eex_forwards_history.parquet` was not edited or staged by this work;
  observed SHA-256:
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- Production remains strict `NO_GO`. Monthly solver remains sole level
  authority. OMPEX remains benchmark-only. CT and Power BI were untouched.

Every PowerShell command was prefixed by the exact canonical-workspace guard:

```powershell
$expected='C:\Users\jbattaglia\PFC_LT'; $cwd=(Get-Location).Path; $root=(git rev-parse --show-toplevel).Trim().Replace('/','\'); if ($cwd -cne $expected -or $root -cne $expected) { throw "Workspace mismatch: cwd=$cwd root=$root" };
```

## Outcome

The local CH model runs reproducibly with exact monthly solver authority, and
its active quote-to-month mapping now has an explicit conditional stability
gate. This is useful local quality evidence, not a production candidate.

The fresh wheel investigation did not reveal a missing solver dependency:
`monthly_forward_curve.py` and `monthly_curve_authority.py` are present. Only
the new local non-promotional sensitivity diagnostic is absent, deliberately,
from the positive minimal runtime inventory. No production CLI was added for a
local-only audit.

## Current local PFC authority replays

Final replays after enforcing exact `constraint_tolerance=1e-9`:

- run14:
  `output/phase14/local_pfc_prospective_intraday_20260724_run14`
- run15:
  `output/phase14/local_pfc_prospective_intraday_20260724_run15`
- completion manifests:
  `16ff798bea7f4ff15d4e36ede81856222090bb013a4407cedaafabd43a804130`
  and
  `7ff70c65f49d53f0f1ceba29dacf71826df45547bf3c334fd327b3d03573540d`
- eight of ten outputs are byte-identical; only path-bound completion and
  summary documents differ by design.
- monthly manifest SHA-256 in both runs:
  `86bc2a3611c767d6dfa10fd1bebd13106865af2119cd88b634be91d538088714`
- monthly solution hash:
  `af6b5a5fb991f87e25cc882d94ac4c97d7d403bc52d7d3916a3f4ff7b1c5582a`
- active constraint hash:
  `c8adbcf686b3e0d943dea50f73c7984f623bab655ad1967581a07296f9719b8f`
- active config hash:
  `f21e2f49d76e790cfc94cf17cb66e8f37379f6598c2ffecbfd1996723e4337ed`
- KKT constraint residual: `2.842170943040401e-14`;
  stationarity: `1.5719360463372172e-13`; condition: `58.79108736225422`;
  rank `16`, nullspace `38`, no ridge and no least-squares fallback.
- source remains `TEST_FIXTURE`; hard-quote and promotion eligibility are
  false.

The runner now rejects any local monthly constraint tolerance other than exact
`1e-9`. Its code-source fields are explicitly observations made after import,
with no execution attestation.

## Quote-to-curve and cascade evidence

New source:

- `pfc_shaping/calibration/monthly_curve_sensitivity.py`
- `scripts/audit_monthly_quote_curve_sensitivity.py`
- `tests/test_monthly_curve_sensitivity.py`
- `tests/test_audit_monthly_quote_curve_sensitivity_script.py`

The audit differentiates the 16 active independent quotes with symmetric
`0.01 EUR/MWh` perturbations. It checks exact quote repricing, active
constraint derivatives, affine response, rank, order/repeat invariance,
redundant-parent removal, direct KKT use and declared objective labels.

After the independent Quant finding, schema v2 retains the full-grid weighted
gain only as a diagnostic and gates four horizon-invariant views: quote-support
weighted `<=2`, delivery-year `<=2`, cascade-bucket `<=sqrt(5)` and maximum
monthly row-L1 response to bounded quote shocks `<=3`. Under one disjoint
residual step with pre-covered hours `H_K <= H_F` residual hours, the first,
third and fourth caps follow from the residual algebra; the annual `2` cap is
separate conservative local policy. None is calibrated production risk
appetite.

The response domain is explicit:

- active independent basis only;
- hierarchy selection, priors and objective frozen at baseline;
- no full-feed end-to-end claim;
- `2026-Q4` and `2027` are redundant hierarchy boundaries;
- `2031` and `2032` lie outside the delivery domain;
- those four quotes are not ordinary zero-risk Jacobian columns.

Authoritative current artifacts:

- run11:
  `output/phase14/monthly_quote_curve_sensitivity_20260724_run11`
- run12:
  `output/phase14/monthly_quote_curve_sensitivity_20260724_run12`
- CSV SHA-256, byte-identical:
  `29de930ca7455b9eb42c3a9a3080032f38c12e36b744d190f320a746b762274f`
- summary SHA-256, byte-identical:
  `c1dfc8edd479213a7e210809546e3068b0df1649fbc509fa561710b45fc7ffac`
- module source observed at report time:
  `77bcb70ce6d500886b4c5fc0eea82adc1d5e9d43e02c4d45b2747c4e6363073d`
- script source observed at report time:
  `45edbe4c40163c6ecd3171d0b779afdff26678f3990f6d8b16b574cbd318dd15`
- status: `PASS_LOCAL_DETERMINISTIC_SENSITIVITY_NOT_PRODUCTION`;
- schemas: `monthly_quote_curve_sensitivity.v2` and artifact v2;
- 18/18 gates pass;
- max quote repricing Jacobian error: `3.6186609264632352e-12`;
- max active constraint derivative error: `3.106848112111038e-12`;
- max central linearity error: `8.526512829121202e-12`;
- response rank: `16/16`;
- max element: `1.988680099616147`;
- raw spectral/Frobenius norms: `5.198816412327312` /
  `8.824430767289481`;
- full-grid weighted gain, diagnostic-only: `0.7098418999802371`;
- support gain: `1.988680099614965 / 2.0` (margin `0.566%`);
- delivery-year gain: `1.5049065745334473 / 2.0`;
- cascade-bucket gain: `2.107983096212873 / sqrt(5)`;
- monthly row-L1: `2.9773601992346244 / 3.0` (margin `0.755%`).

The registered five-year narrow-residual counterexample has old full-grid gain
`1.5957989 < 2` while maximum monthly derivative is `11.80645`; all four v2
gates fail, preventing horizon dilution.

CSV and summary are written into a random 128-bit same-parent staging directory
whose Windows ACL inherits from `output/phase14`, then the complete directory
is renamed into the final path. Crash injection before summary leaves no final
directory and retry to the same path succeeds. Separate PowerShell processes
read run11/run12 and recompute their hashes. A crash may leave a hidden staging
residue requiring bounded inventory/quarantine/GC; it is not a final artifact
and does not block retry.

Run5/run6 are superseded schema-v1 evidence. Run7 through run10 were generated
during diagnosis of `tempfile.mkdtemp` ACL inheritance and are not valid
evidence; their final directories are not independently readable. Do not use
or promote any of run5 through run10.

Canonical invocation used twice:

```powershell
python -m scripts.audit_monthly_quote_curve_sensitivity --monthly-manifest output\phase14\local_pfc_prospective_intraday_20260724_run14\structural_fan_chart.monthly_curve_manifest.json --expected-monthly-manifest-sha256 86bc2a3611c767d6dfa10fd1bebd13106865af2119cd88b634be91d538088714 --forwards data\eex_forwards_history.parquet --expected-forwards-sha256 21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013 --output-dir output\phase14\monthly_quote_curve_sensitivity_20260724_run11
```

Repeat with `run12` as output. Direct `python scripts\...` is not the canonical
checkout invocation; use `python -m scripts...` from the guarded repo root.

## Governed wheel closure

Fresh build directories:

- `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\pfc-lt-wheel-quote-sensitivity-20260724-v1`
- `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\pfc-lt-wheel-quote-sensitivity-20260724-v2`

Build command used for each unique directory:

```powershell
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir <fresh-dir>
```

Results:

- wheel: `fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- size: `427931` bytes;
- SHA-256, byte-identical:
  `42e3ea813e11f4461747929f2c93c54fd7e0bb91784494d9e77ebe9b651790c0`;
- 79 members;
- embedded source revision:
  `59d90d36a5289441badbeedced9a9b4ecdeae223cc444a4e80de09b0d6a8e54a`;
- both `python -m scripts.check_lt_wheel_contract <wheel>` audits: `PASS`;
- `promotion_eligible=false`.

Fresh isolated target:

`C:\Users\jbattaglia\AppData\Local\pfc-lt-build\pfc-lt-installed-quote-sensitivity-20260724-v2`

After `pip install --no-deps --target`, a `python -I` smoke inserted only this
target and imported:

- `pfc_shaping.calibration.monthly_forward_curve`;
- `pfc_shaping.pipeline.monthly_curve_authority`;
- `pfc_shaping.cli.governed_release`;
- `pfc_shaping.cli.governed_acquisition_builder`.

Every imported file resolved under the isolated target, the checkout root was
absent from `sys.path`, no CT module loaded, and the local sensitivity module
was absent by design.

## Verification matrices

- Targeted Ruff: `All checks passed!`.
- Focused sensitivity/script suite after final source: `9 passed in 2.93s`.
- Monthly/solver/sensitivity matrix: `245 passed, 2 warnings in 156.51s`.
  The warnings are pre-existing all-NaN/empty-slice warnings in the explicit
  insufficient-history failure test.
- Packaging/runtime/acquisition matrix:
  `295 passed, 14 skipped in 168.17s`.
- Publication/CAS final recertification was split because the complete lot is
  longer than the shell timeout: candidate bundle/evidence/assembler
  `10 + 13 + 42 passed`, atomic promotion `116 passed, 2 skipped`, governed
  release `37 passed`, quality gate `27 passed`, external CAS `18 passed`.
- LT core/import/profile/intraday matrix:
  `71 passed, 1 skipped in 65.56s`.

The initial combined publication recertifications timed out because
`test_candidate_evidence_assembler.py` needs `498.41s` and
`test_atomic_promotion.py` needs `764.24s` on this machine. Those combined
timeouts are non-conclusive and not counted; every constituent file then
completed separately as reported above. One over-broad command that included
every `test_tier2_monthly_eex*.py` also timed out without a pytest summary and
is not counted. No Tier-2 source was changed by this slice.

## Independent read-only roasts

Final Security, IT/Operations and Quant/Data re-roasts on source v2 and
run11/run12 report no P0 and no local P1. All three return controlled local GO
and production `NO_GO`.

- Security confirms captured-byte input use, honest post-import hash claims,
  transaction/crash retry and inherited ACL readability. Residual P2 local / P1
  production: orphan staging GC, Windows power-loss, directory-handle/writer
  lease, POSIX no-replace/default ACL and execution/supply-chain attestation.
- IT/Operations confirms the former partial-final/non-retryable sensitivity
  finding is closed. Residual P2 local: staging GC, module-only invocation,
  unstructured tracebacks, and the main local PFC runner remains
  completion-last rather than group-transactional. Cross-principal ACL,
  container, alert delivery, rollback and DR remain unproved.
- Quant/Data confirms the narrow-residual counterexample is closed and the
  v2 claims are correctly conditional. Support and row-L1 margins are only
  `0.566%` and `0.755%`. Production still needs volatility/covariance/liquidity
  calibration and independent governance of objective operators and weights.

## Files changed by this focused continuation

- `pfc_shaping/calibration/monthly_curve_sensitivity.py` (new)
- `scripts/audit_monthly_quote_curve_sensitivity.py` (new)
- `tests/test_monthly_curve_sensitivity.py` (new)
- `tests/test_audit_monthly_quote_curve_sensitivity_script.py` (new)
- `scripts/build_local_test_ch_pfc.py`
- `tests/test_build_local_test_ch_pfc_script.py`
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- this handoff.

No CT, Power BI or heavy desk-data file belongs to this focused delta.

## Residual blockers and next direction

Production remains `NO_GO` because this local evidence still uses a
`TEST_FIXTURE` forward source, mixed-authority DE-LU intraday proxy, no approved
fresh CH point-in-time inputs, no calibrated probabilistic outputs, and no
external service-identity/CAS/ACL/freeze/container/registry/observability/DR
proof.

After final diff checks and roasts, the next scientific direction is:

1. acquire a fresh governed prospective CH/EEX snapshot with independent
   timestamp and raw-byte provenance;
2. re-evaluate T057 and preserve its locked one-shot semantics;
3. build a new CH candidate without OMPEX input;
4. validate shaping on rolling origins plus an unused future holdout, including
   DE-to-CH transfer and sparse seasonal coverage;
5. calibrate probabilistic/scenario outputs and exact EEX product repricing;
6. only then assemble independent production/local-export/lambda manifests.

Do not promote, publish, commit or stage the protected forward-history file.
