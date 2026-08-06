---
name: execute-explicit-commands-directly
description: "On an explicit simple operational command, execute directly — no self-added precautionary checks the user didn't ask for"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2f125ab3-dff7-4906-8f09-cd7955db840b
---

When the user gives an explicit, simple, reversible operational command (e.g. "just stop that 8-vCPU VM"), **execute it directly**. Do not wrap it in my own precautionary steps — process snapshots, "let me first check if a job is running", confirmation prompts — that they didn't request.

**Why:** twice in the billing-crisis session the user rejected my added checks ("просто отключи", "без проверок... останавливай"). They found the extra ceremony slowing and unwanted on a clear, cheap-to-reverse action.

**How to apply:** for an explicit action command, just do it and report the result + impact. This does NOT lower the bar for *factual claims* — keep the "prove, don't assume" rigor for analysis/assertions (see [[capture-all-information]]). The nuance: verification for claims = yes; precautionary gating on an explicit reversible action = no. If an action is truly destructive/irreversible (delete, overwrite, push) the normal confirm-first rule still applies.
