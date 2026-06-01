# Recorder VM (Tokyo, 24/7)

Runs `scripts/record_data.py` on a cheap Linux VM near Binance — the recorder
was built for Linux + systemd, so the VM path needs zero code changes (vs. the
several Windows incompatibilities: `loop.add_signal_handler`, `/tmp` health
file, POSIX paths).

## Layout

- **VM**: `scalper-recorder`, `e2-small` (2 vCPU / 2 GB), `asia-northeast1-b`,
  Debian 12, 30 GB pd-standard. Region chosen because Binance Futures is
  reachable from Tokyo (the original prod VPS was there) and `us-*` is
  geo-blocked by Binance.
- **Data**: written locally to `/home/scalper/scalper-bot/data` (7-day
  retention, ~1.5 GB/day depth), then mirrored hourly to
  `gs://recorder-data-asia-0998ac51/recorder/<host>/` (additive — survives the
  local 7-day rotation). Bucket is in Tokyo, co-located with the VM, so the
  sync egress is ~free.
- **Auth**: recorder uses only PUBLIC Binance streams + public REST, so
  `config.env` carries placeholder API keys; the user-data stream self-disables
  on `PermissionError`. GCS sync auths via the VM's attached service account
  (metadata) — no keys on disk.

## Services (systemd)

| unit | role |
|------|------|
| `scalper-recorder.service` | the recorder; `Restart=always` |
| `scalper-gcs-sync.timer` | hourly `gcloud storage rsync` data → GCS |
| `scalper-watchdog.timer` | every 2 min: restart recorder if health file stale >120 s |

## Provision

```bash
bash deploy/recorder-vm/deploy.sh
```

## Monitor

From Windows: `powershell -File deploy\recorder-vm\status.ps1`

Or directly:
```bash
gcloud compute ssh scalper-recorder --zone asia-northeast1-b \
  --command 'tail -n 40 /home/scalper/scalper-bot/logs/recorder.log'
```

## Redeploy code

```bash
git archive --format=tar.gz -o /tmp/scalper-bot.tar.gz HEAD
gcloud storage cp /tmp/scalper-bot.tar.gz gs://recorder-data-asia-0998ac51/_bootstrap/scalper-bot.tar.gz
gcloud compute ssh scalper-recorder --zone asia-northeast1-b \
  --command 'sudo google_metadata_script_runner startup && sudo systemctl restart scalper-recorder'
```

## Stop / destroy (stop billing)

```bash
gcloud compute instances stop  scalper-recorder --zone asia-northeast1-b   # pause (~disk-only cost)
gcloud compute instances delete scalper-recorder --zone asia-northeast1-b  # full teardown
```
