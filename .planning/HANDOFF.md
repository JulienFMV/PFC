# Current Handoff

Latest active handoff:

`.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260624-YOY50-CANDIDATE-PASSES-STRICT.md`

Read order for new agents:

1. `AGENTS.md`
2. `CLAUDE.md` if running Claude Code
3. `.planning/HANDOFF.md`
4. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
5. Latest session handoff linked above

Do not treat older Phase 14 generated reports as accepted production evidence
unless the latest handoff or decision log names them explicitly.

Current verdict: local Phase 14 CH candidate
`output/phase14/20260624_parent_local_prior_lshape25_yoy50_structural_s126/`
is auditable. Delivered-product audit passes with the exact artifact-bound
source hierarchy policy
`.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_lshape25_yoy50_structural_s126.json`
(`accepted_quote_conflict_count=9`, `UNSUPPORTED=0`, `OUT_OF_SCOPE=9`), and
strict Power BI export passes without `--allow-failed-gates`
(`powerbi_quality_gate_status=PASS`, cross-year warnings `0`). Selected config
artifact
`.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_lshape25_yoy50_structural_s126.json`
is scoped to local candidate selection only:
`production_promotion_approved=false`. Real production/export/selected capstone
check remains BLOCKED with 4 CRITICAL governance gates:
`lambda_calibration_artifact_present`, `production_export_path_parity`,
`selected_config_production_approval`, and
`selected_config_manifest_parity`. Phase 14 remains NO-GO for actual
production promotion until the real production manifest, local export manifest,
and selected config artifact are regenerated/checked as one manifest triad;
simulated parity on the local candidate is not production evidence.

