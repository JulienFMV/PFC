# Session handoff - OMPEX full archive inventory and C/H boundary

Date: 2026-08-05  
Decision: D-20260805-221  
Status: `PASS_CURRENT_BYTE_INVENTORY_NO_GO_ROLLING_ORIGIN`

## Outcome

The workstation contract now makes `C:\Users\jbattaglia\PFC_LT` the only
routine workspace. `H:` is permitted only for an explicitly requested,
occasional, read-only OMPEX benchmark refresh. There is no `H:` default or
runtime dependency, and no write or metadata mutation occurred on `H:`.

Two complete recursive byte scans of the OMPEX source matched exactly. The
selected local inventory content ID is
`336700af0b38324bbfc99c5332b5f360a01e00f2fd14baab090ebcb8e087a57a`.
It covers 353 XLSX workbooks: 351 dated curves plus two templates. The dated
curves share one schema, contain no formulas or active/external links, and use
two horizon regimes: 136 files with 43,824 rows and 215 with 52,584 rows.

This is not rolling-origin evidence. There are 49 missing calendar dates over
the filename interval, including a 47-day initial gap, and filenames do not
authenticate desk availability. Countable scientific origins remain 0/351.
OMPEX keeps benchmark-only authority and no price values were emitted by this
inventory.

## Changed files

- `pfc_shaping/validation/ompex_archive_inventory.py`
- `scripts/audit_ompex_archive_inventory.py`
- `tests/test_ompex_archive_inventory.py`
- `tests/test_ompex_external_benchmark_access.py`
- `.planning/phases/14-lt-audit-remediation/OMPEX-EXTERNAL-BENCHMARK-ACCESS-CONTRACT-V1.json`
- `docs/research/OMPEX-ARCHIVE-DATA-QUALITY-REPORT-20260805.md`
- `docs/research/OMPEX-INDEPENDENT-BENCHMARK.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

## Evidence and hashes

Selected artifacts on `C:`:

- `build/ompex-benchmark/2026-08-05/full-inventory-v3-a`
- `build/ompex-benchmark/2026-08-05/full-inventory-v3-b`

Both scans have the same content ID above and identical output hashes:

- `inventory.json`: `7dc64cd40d341072cbe264402a0fffb0bbf80183977e49de9dbecc16f89317aa`
- `summary.json`: `23ffd8e78848595a47f87c12db32a30925181d3a7e60fae3d41ef903c2408670`
- `manifest.json`: `4166c84e8378a685c39ee636fe9ac119a8fc5467bf19d0fd5f565245a6fd77f6`

Bound implementation hashes:

- inventory module: `2569cc089eb8e9ccc6213a95d8341477bc422d88b7cbdec776982107f4bdee52`
- runner: `69c73d116bebad9d3afabb572e76867c902ed2764fea3dc6c85aafbdba0d060d`
- inventory tests: `87381a35786de97da259dff88c087a8fa8ce02d2a3fc3841edd2fc5e5219e487`
- access contract: `9d74beea0319ec730c03c2d1ffc1041c28562aac83430b220801910a5e1f3d1e`
- access tests: `2adb5163deef261d4e57849f032b7e0e1af863f7305114a6966e7b8cab8e040a`
- technical report: `e0658e4b51f15910a939e6418dcf27009657740a39794e9e18935d03736b1cc5`

## Commands and validation

Every command ran from and was guarded to
`C:\Users\jbattaglia\PFC_LT`. The source path was supplied explicitly only to
the inventory runner. Mutable outputs stayed below `build/`.

- two full runner executions completed with identical content and output
  hashes;
- focused and adjacent test matrix: `59 passed, 1 skipped`, with 13 existing
  Matplotlib/PyParsing warnings;
- access-specific focused matrix: `7 passed`;
- Ruff, JSON parsing, Python compilation and manifest-bound current-file hash
  checks passed;
- no Databricks request, SQL Warehouse start, network write or remote write;
- no `H:` write, move, delete or metadata mutation.

One early inventory attempt failed closed when the source root contained the
nested `Template` directory. The implementation was corrected to inventory
regular XLSX files recursively while rejecting links/reparse points and then
passed six focused tests. One earlier shell invocation timed out before
producing an artifact; the governed rerun completed normally.

## Risks and next permitted work

- Do not claim OMPEX rolling-origin performance until provider/desk
  availability semantics are authenticated independently.
- Do not backfill the 49 missing dates or score historical candidates against
  the latest available vintage.
- Routine PFC and scoring must use selected local, hash-bound evidence on `C:`.
  A future `H:` refresh must be deliberate, read-only and exceptional.
- OMPEX remains post-candidate benchmark-only. Monthly solver level authority,
  LT/CT separation, the sealed T057 boundary and the current production
  `NO_GO` remain unchanged.
