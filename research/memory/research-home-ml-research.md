---
name: research-home-ml-research
description: "ml-research (ex scalper-bot, renamed) = canonical research repo (user decision 2026-07-11); trading_algorithm = deploy showcase pinned by ledger commits; agent memory canonical store is research/memory/ in-repo"
metadata: 
  node_type: memory
  type: project
  originSessionId: e405d5fd-5d12-437d-a431-d396bf143b9a
---

User decision (2026-07-11): the research base — ledger, RESEARCH_LOG, run machinery
(`research/runtime/`), and agent memory — lives in the CURRENT repo, renamed
`BlackDigitalStudio/ml-research` (ex `scalper-bot`; remote redirects). No physical
move: git history and branch `claude/husdc-rev1` are load-bearing protocol evidence
(ledger commit refs, bit-parity provenance).

- `BlackDigitalStudio/trading_algorithm` = clean showcase of the deployed algorithm;
  ledger deploy records pin its commits. Token local: `C:\Разработки\ml_research_token.txt`.
- **Agent memory canonical store = `research/memory/` in-repo**; local auto-memory is
  a cache. Mirror new/changed memories there in the same commit as the related work.
- `REPO_MAP.md` (root) orients agents: active vs frozen vs historical code.
Related: [[research-runtime-infra]], [[xsym-cross-symbol-run]].
