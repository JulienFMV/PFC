# Current Handoff

Latest active handoff:

`.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-QUOTE-CONFLICT-AUDIT-GOVERNANCE.md`

Read order for new agents:

1. `AGENTS.md`
2. `CLAUDE.md` if running Claude Code
3. `.planning/HANDOFF.md`
4. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
5. Latest session handoff linked above

Do not treat older Phase 14 generated reports as accepted production evidence
unless the latest handoff or decision log names them explicitly.

Current verdict: Phase 14 product audit now distinguishes delivered curve drift
from redundant source quote conflicts. The fresh local-test CH candidate has
quote-aware BASE/PEAK buckets passing and `critical_count=0` in the delivered
product audit, but still blocks with `quote_conflict_count=9` and
`unsupported_count=9`. Strict Power BI export still blocks on
`shape_score_10=6.75 < 8.50` and `monthly_split_critical_flags=1`. Production
remains NO-GO; next phase should decide cleaned-snapshot vs hierarchy-policy
handling for `QUOTE_CONFLICT`, then fix the remaining model-quality gates
through priors/objective/shape calibration, not month patches.

