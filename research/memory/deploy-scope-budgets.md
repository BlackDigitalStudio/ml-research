---
name: deploy-scope-budgets
description: "Deploy-scope validation runs = AxB t5/t10 ONLY (BUDGETS=5,10, the script default); do not inherit the exploratory-era BUDGETS=5,10,20,40 from old run scripts; noA output is hardwired in the frozen optuna script — ignore it, don't report it"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2fba63ea-658b-4410-928d-a8c01aff03c4
---

User feedback 2026-07-09 (year-scale anchored run): running budgets t20/t40 and
reporting noA in a DEPLOY-scope validation is out of scope — a mistake.

**Why:** the deployed policy is AxB at t5 (t10 as the near-alternative). Extra
budgets/policies in a confirmatory run blur the deliverable and waste attention;
the exploratory 4-budget sweep era is over for this cell.

**How to apply:** for confirmatory/validation runs of subs60_xgb_optuna_ic.py use
`BUDGETS=5,10` (the script default — the original h150_reruns.sh override
`5,10,20,40` is NOT to be copied). noA is computed unconditionally inside the
frozen script: leave the script frozen, but do not surface noA numbers in reports.
Related: [[h150-sim-live-parity]], [[scope-bound-claims]].
