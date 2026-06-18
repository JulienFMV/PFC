# External Audit Prompt - LT Quant Shaping Redesign Plan - 2026-06-18

You are an external quantitative engineering auditor reviewing a redesign plan for the CH long-term electricity PFC/HFC shaping model.

Your mandate is to audit the plan critically before implementation. Do not assume the plan is correct because it was internally reviewed. Treat it as a pre-implementation model-risk document.

## Scope

Repository: `JulienFMV/PFC`  
Branch: `fix/lt-audit-remediation`  
Plan under review:

```text
.planning/phases/15-lt-quant-shaping-redesign/QUANT-SHAPING-REDESIGN-PLAN-20260618.md
```

Domain:

- Swiss long-term electricity forward curve generation;
- CH EEX BASE/PEAK calibration;
- cross-border priors from DE/FR/AT/IT_NORD or documented IT proxy;
- monthly/hourly shaping, negative prices, probabilistic tails, Power BI QA.

Current production status in the plan: `NO GO`.

## Context

The redesign was triggered by a visually and quantitatively suspect LT PFC candidate. Earlier local fixes preserved EEX calibration but still relied on diagnostic overlays and gates. The user rejected further patching and requested a scientifically defensible quant redesign aligned with HPFC/PFC literature and European market structure.

The plan should therefore be judged on whether it is a rigorous implementation and validation blueprint, not on whether it already implements code.

## Required Audit Work

Please audit whether the plan is sufficient for an external quant/model-risk reviewer.

Focus especially on:

1. Mathematical formulation:
   - convex QP v1;
   - hard EEX constraints;
   - null-space formulation;
   - rank/faisability checks;
   - BASE/PEAK/OFFPEAK convention;
   - partial horizon policy;
   - KKT diagnostics.

2. Cross-border market model:
   - DE/FR/AT/IT_NORD influence on CH;
   - zero-mean neighbor deviation projection;
   - no absolute level leakage;
   - NTC/interconnector/spread regime dataset;
   - Swiss hydro regime;
   - data coverage gates.

3. Validation:
   - rolling backtest protocol;
   - point-in-time anti-leakage contract;
   - benchmarks;
   - block and spread metrics;
   - statistical non-degradation rules;
   - production acceptance gates.

4. Probabilistic layer:
   - deterministic forward curve `p_t` separated from physical scenarios/quantiles;
   - scenario mean EEX consistency;
   - non-crossing quantiles;
   - proper scoring rules;
   - negative-price events, runs, co-occurrence, and stress cases.

5. Governance:
   - run manifest schema;
   - required output artifacts;
   - Power BI binding checks;
   - RACI;
   - waiver process;
   - definition of done.

## Expected Output

Return:

1. `Verdict`: `BLOCK`, `CONDITIONAL PASS`, or `PASS`.
2. `Score`: 0-10, with justification.
3. `Top Findings`: only material findings, ordered by severity:
   - `P0`: blocks external-audit readiness;
   - `P1`: must be fixed before implementation starts;
   - `P2`: can be fixed during implementation planning.
4. `Missing Evidence`: any data, literature, equations, thresholds, or governance artifacts required.
5. `Recommended Patch`: precise edits to the plan before implementation.

Do not audit generated CSV artifacts or Power BI outputs. This is a plan audit only.
