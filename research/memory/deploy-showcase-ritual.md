---
name: deploy-showcase-ritual
description: trading_algorithm repo must be updated in the same change-window as ANY deploy add/remove/resize/re-weight — DEPLOYMENTS.md manifest + units + validation snapshots (user directive 2026-07-11)
metadata:
  type: feedback
---

User directive (2026-07-11): «Репозиторий надо обновлять каждый раз когда что-то
добавляем/убираем» — the deploy showcase `BlackDigitalStudio/trading_algorithm`
must never diverge from production.

**Why:** two live symbols already have different VMs, weights, sizing and results;
a stale showcase repo misleads anyone (human or agent) reading it as truth.

**How to apply:** any deploy change (add/remove symbol, resize, re-weight, unit/env
edit) ships a same-window commit to trading_algorithm: `DEPLOYMENTS.md` (per-instance
manifest: VM/IP, units, venue, sizing, weights GCS paths, tau seed, validation cells,
ledger ids, open items) + updated `live/` unit files + new validation tools. Clone with
token `C:\Разработки\ml_research_token.txt`. Rule mirrored in both repos' CLAUDE.md.
Related: [[live-trading-deploy]], [[research-home-ml-research]].
