# research/memory — shared agent memory (canonical store)

This directory is the CANONICAL, repo-versioned agent memory for the research
program. Any agent's machine-local auto-memory (`~/.claude/projects/*/memory/`) is a
cache of this; on divergence, this directory + git history win.

Discipline:
- At session start: skim `MEMORY.md` (index); read bodies before ACTING on a claim
  (CLAUDE.md "Interpreting records" rules apply — one-liners strip conditions).
- When you save a durable memory locally, mirror it here in the same commit as the
  work that produced it. Delete/correct stale entries here (git keeps the history —
  unlike the append-only ledgers, memory is a living state, not an event log).
- Every quantitative claim is a measured cell (symbol × period × execution ×
  features × protocol) — see `scope-bound-claims.md` FIRST.
- What belongs here: cross-session operational knowledge (VM/deploy mechanics,
  protocol constants, debugged pitfalls, user directives on how to work).
  What does NOT: anything derivable from the ledger/RESEARCH_LOG/code, or
  session-scoped state.
