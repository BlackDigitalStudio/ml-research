---
name: recorder-deploy-mechanics
description: "How the Chronos recorder is actually deployed/restarted on the Tokyo VM (paths differ from the repo's systemd template)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2f125ab3-dff7-4906-8f09-cd7955db840b
---

Deploying a code change to the live Chronos recorder (VM `scalper-recorder`, zone `asia-northeast1-b`):

- **Live code lives at** `/home/scalper/crypto-market-recorder/` (NOT `/opt/chronos` — the repo's `systemd/chronos.service` template is stale/wrong vs reality). Package: `…/chronos/`, entrypoint `chronos_run.py`, venv `…/venv/bin/python`. Runs as user `scalper`.
- **Deploy dir is a plain copy, NOT a git checkout** → deploy by file copy, not `git pull`. SSH as `delmi` has **passwordless sudo**; files are `scalper:scalper`.
- **Logs:** `/home/scalper/crypto-market-recorder/logs/chronos.log` (NOT journald, NOT `/var/log/chronos`). **Data:** `/home/scalper/crypto-market-recorder/data/<exchange>/<symbol>/<stream>/{raw,.parts}/`.
- **Steps:** `gcloud compute scp <file> scalper-recorder:/tmp/… --zone=asia-northeast1-b` → `sudo cp` into place + `sudo chown scalper:scalper` → verify by importing as scalper (`sudo -u scalper venv/bin/python -c "import chronos.gateway"`; running py_compile as `delmi` fails only on writing `__pycache__`, not syntax) → `sudo systemctl restart chronos` (graceful SIGINT, 30s flush; ~secs gap, books re-seed on reconnect) → confirm `systemctl is-active` + grep log for `WS binance-(public|market)-N connected`.

**Why:** rediscovering the real paths each deploy is slow; the repo template misleads. **How to apply:** follow this for any recorder hotfix. [[audit-before-long-runs]] still applies (import-check before restart). See [[recorder-vm-live]], [[binance-ws-routed-paths]].
