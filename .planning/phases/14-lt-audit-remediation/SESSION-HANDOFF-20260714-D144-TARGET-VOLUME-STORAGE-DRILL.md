# Session Handoff - 2026-07-14 - D144 Target-Volume Storage Drill

## Scope

D144 implements a retained, synthetic and non-promotional drill for the exact
Windows/SMB volume intended to host governed LT releases. No real candidate,
market data, signing key, promotion, rollback or target-volume run was used.

## Changed files

- `pfc_shaping/path_safety.py`
- `pfc_shaping/data/shared_data_root.py`
- `pfc_shaping/pipeline/atomic_promotion.py`
- `pfc_shaping/pipeline/candidate_evidence.py`
- `pfc_shaping/pipeline/candidate_evidence_assembler.py`
- `pfc_shaping/pipeline/governed_release.py`
- `pfc_shaping/pipeline/governed_release_cli_contract.py`
- `pfc_shaping/pipeline/storage_drill.py`
- `scripts/run_lt_release_storage_drill.py`
- `tests/test_atomic_promotion.py`
- `tests/test_lt_release_storage_drill.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

## Implemented contract

- Absolute existing drill root; exclusive run ID; no recursive cleanup.
- Refusal under any governed release/candidate ancestor.
- Shared Python 3.11 Windows reparse detection and hardlink controls.
- Synthetic crash exactly after the production hardlink publication point,
  exact inode recovery and decoy preservation.
- Four ready-synchronized exclusive writers with a typed collision exception.
- Live lock exclusion, post-release reacquisition and abandoned-lock fail-
  closed proof.
- Two byte-exact readers observing both generations across 1,000 atomic JSON
  replacements with monotonic deadlines.
- Absent-to-complete observation of exact production directory replacement.
- Typed, distinct drive/short-UNC/FQDN-UNC aliases with `samefile`, sentinel,
  cross-alias lock and atomic-write checks.
- Per-probe subprocess supervision and whole-CLI supervision. Windows workers
  self-own a `KILL_ON_JOB_CLOSE` Job Object before spawning descendants; POSIX
  retains a process group. A grandchild heartbeat test proves tree shutdown.
- Final inventory closure with only the final report declared as excluded.
- Exclusive-create/fsync report path independent of hardlink support and
  versioned emergency JSON on stdout.

## Commands and results

```text
python -m ruff check <D144 runtime, shared safety and test files>
  PASS

python -m pytest -q tests/test_lt_release_storage_drill.py
  37 passed, 2 skipped in 78.22s

python -m pytest -q tests/test_atomic_promotion.py \
  tests/test_governed_release.py \
  tests/test_run_governed_lt_release_script.py \
  tests/test_candidate_evidence.py \
  tests/test_candidate_evidence_assembler.py \
  tests/test_shared_data_root.py \
  tests/test_lt_release_storage_drill.py
  247 passed, 4 skipped, 1 warning in 240.07s
```

The integrated result preceded only the final operator-cancellation priority
tests; the final dedicated suite covers those last supervisor changes. The
warning is the existing pandas empty/all-NA concat future warning. Skips are
platform/capability tests with deterministic reparse and supervision coverage.

## Operational execution still required

Use the command in `pfc_shaping/tools/OPERATIONS.md` on a dedicated directory
of the exact target volume, under the exact service identity, with drive,
short UNC and FQDN UNC spellings. Preserve the entire run directory and stdout
hash. A local `PASS` or exit `0` is not production authorization.

External evidence still required:

- multi-client SMB lease/concurrency drill;
- WORM or independent monotonic anti-rewind;
- service-account ACL and least privilege;
- HSM/KMS key custody and separated identities;
- appliance write-through and power-loss durability;
- backup, restore, retention and disaster recovery.

## Verdict

D144 software verdict is `GO`: final Systems, Security and IT/Operations
re-roasts found no P0/P1. Global production remains `NO_GO`. No real candidate
was promoted and no target-volume drill was executed.
