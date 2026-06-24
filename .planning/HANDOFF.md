# Current Handoff

Latest active handoff:

`.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260624-PROMOTION-STATUS-ENUM-EXACT.md`

Read order for new agents:

1. `AGENTS.md`
2. `CLAUDE.md` if running Claude Code
3. `.planning/HANDOFF.md`
4. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
5. Latest session handoff linked above

Do not treat older Phase 14 generated reports as accepted production evidence
unless the latest handoff or decision log names them explicitly.

Current verdict: Phase 14 remains NO-GO for production. Roaster P0/P1
governance fail-open paths have been fixed in code/tests: selected config
approval uses case-sensitive exact `selection_status == "PRODUCTION_APPROVED"`,
selected/prod/export parity is required, and source hierarchy policies are
strictly typed, snapshot-bound and quote-conflict-count-bound with integer
counts only. The current local CH candidate still blocks with
`QUOTE_CONFLICT=9`, `UNSUPPORTED=9`, and strict Power BI
`cross_year_near_clone_warnings=1`. The current source hierarchy policy and
selected config artifacts remain diagnostic/draft with
`production_approved=false`. Remaining P2: a future production-approved source
hierarchy policy should bind exact conflict identities or input CSV hash.

