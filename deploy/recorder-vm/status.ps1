# Check the recorder VM from Windows. Usage: powershell -File deploy\recorder-vm\status.ps1
$VM   = "scalper-recorder"
$ZONE = "asia-northeast1-b"
gcloud compute ssh $VM --zone $ZONE --command @'
echo "===== systemd ====="
systemctl --no-pager --lines=0 status scalper-recorder.service | head -6
systemctl list-timers --no-pager | grep -E "scalper|NEXT" || true
echo "===== last stats ====="
grep -a "Stats:" /home/scalper/scalper-bot/logs/recorder.log 2>/dev/null | tail -3
echo "===== data on disk ====="
du -sh /home/scalper/scalper-bot/data 2>/dev/null
echo "===== health file age (s) ====="
if [ -f /tmp/scalper_recorder_health ]; then echo $(( $(date +%s) - $(stat -c %Y /tmp/scalper_recorder_health) )); else echo "MISSING"; fi
'@
