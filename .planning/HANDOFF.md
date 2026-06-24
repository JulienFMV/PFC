# Current Handoff

Latest active handoff:

`.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260624-SOURCE-HIERARCHY-IDENTITY-BINDING.md`

Read order for new agents:

1. `AGENTS.md`
2. `CLAUDE.md` if running Claude Code
3. `.planning/HANDOFF.md`
4. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
5. Latest session handoff linked above

Do not treat older Phase 14 generated reports as accepted production evidence
unless the latest handoff or decision log names them explicitly.

Current verdict: MIT roasters gave GO on closing P0/P1 governance exact-value
fail-open paths, and the remaining source hierarchy identity-binding P2 has
now been implemented in code/tests. Phase 14 remains NO-GO for production.
Selected config approval uses case-sensitive exact
`selection_status == "PRODUCTION_APPROVED"`, selected/prod/export parity is
required, and source hierarchy policies are strictly typed, snapshot-bound,
quote-conflict-count-bound, and bound to `input_csv_sha256`,
`quote_conflict_identity_hash`, or exact `expected_quote_conflicts` before prod
approval can accept conflicts. The current local CH candidate still blocks
with `QUOTE_CONFLICT=9`, `UNSUPPORTED=9`, and strict Power BI
`cross_year_near_clone_warnings=1`. The current source hierarchy policy and
selected config artifacts remain diagnostic/draft with
`production_approved=false`.

