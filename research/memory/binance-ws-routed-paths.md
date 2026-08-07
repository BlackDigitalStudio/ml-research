---
name: binance-ws-routed-paths
description: "Binance futures WS migrated to routed /public,/market,/private paths (2026-04-23); legacy /ws,/stream now deliver ONLY /public"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2f125ab3-dff7-4906-8f09-cd7955db840b
---

Binance USDⓈ-M futures WebSocket migrated to **routed paths** effective **2026-04-23**:
- `wss://fstream.binance.com/public` — book depth, raw `@trade`, book ticker (high-freq)
- `wss://fstream.binance.com/market` — `@aggTrade`, `@markPrice`, `@forceOrder` (liquidations), kline, ticker
- `wss://fstream.binance.com/private` — user data (listenKey)

Legacy unrouted `…/ws/<stream>` and `…/stream?streams=` still connect but now deliver **only /public** streams and **silently drop /market** ones (subscription acks, LIST_SUBSCRIPTIONS shows active, but 0 frames). Format is otherwise unchanged: routed combined URL is `…/market/stream?streams=a/b/c` and still returns `{"stream","data"}` envelopes.

**Why it mattered:** the Chronos recorder connected on legacy `/ws` so `@forceOrder` (liquidations) + `@markPrice` produced 0 frames for ~2 months — misdiagnosed as Tokyo/IP jurisdiction gating. It was NOT geo: proven by legacy `/ws` markPrice=0 vs routed `/market` markPrice≈12/12s **from the same Tokyo egress**, which also caught real BTCUSDT liquidations.

**How to apply:** any Binance futures WS work must route streams by category (`/public` vs `/market`); don't trust legacy `/ws`. Spot API has an analogous migration — check before assuming a stream is "blocked". Fixed in crypto-market-recorder commit 479bcb0 (`gateway.py` `_binance_route_path`). See [[recorder-deploy-mechanics]], [[recorder-vm-live]].
