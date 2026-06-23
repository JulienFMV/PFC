# Current Handoff

Latest active handoff:

`.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-CANDIDATE-AUDIT-BLOCKERS.md`

Read order for new agents:

1. `AGENTS.md`
2. `CLAUDE.md` if running Claude Code
3. `.planning/HANDOFF.md`
4. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
5. Latest session handoff linked above

Do not treat older Phase 14 generated reports as accepted production evidence
unless the latest handoff or decision log names them explicitly.

Current verdict: Phase 14 has a fresh local-test CH candidate generated with
monthly solver ON and final PEAK calibration ON, but it is still production
NO-GO. Delivered quote-aware BASE/PEAK buckets pass, while direct parent quote
gates fail because the 2026-06-17 EEX snapshot contains internally inconsistent
finer-vs-parent quotes. Strict Power BI export also blocks on
`shape_score_10=6.75 < 8.50` and `monthly_split_critical_flags=1`.
Next phase must resolve audit semantics for internally inconsistent redundant
quotes and calibrate the remaining monthly split / structural width blockers
without individual month patches.

