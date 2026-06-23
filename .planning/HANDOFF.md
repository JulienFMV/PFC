# Current Handoff

Latest active handoff:

`.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-SOLVER-EXPORT-GOVERNANCE-HARDENING.md`

Read order for new agents:

1. `AGENTS.md`
2. `CLAUDE.md` if running Claude Code
3. `.planning/HANDOFF.md`
4. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
5. Latest session handoff linked above

Do not treat older Phase 14 generated reports as accepted production evidence
unless the latest handoff or decision log names them explicitly.

Current verdict: Phase 14 solver/export governance hardening has been pushed
through `2b1614e5e`. Delivered-product normalization audit is now fail-closed,
lambda selected-config hashes match the widened production/export active
config payload, and Power BI strict export gates block missing PEAK evidence
and structural invariant failures. This is still not production promotion:
next phase must regenerate a fresh candidate and audit real delivered
artifacts without `--allow-failed-gates`.

