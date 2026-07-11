#!/bin/bash
# Save maker paths for all 8 symbols, full history, 4-way parallel (per-day npz streamed to GCS).
cd /tmp
SYMS="BNB-USDT-PERP BTC-USDT-PERP DOGE-USDT-PERP ETH-USDT-PERP LINK-USDT-PERP LTC-USDT-PERP SOL-USDT-PERP XRP-USDT-PERP"
echo "MASTER START $(date -u)"
printf '%s\n' $SYMS | xargs -P 4 -I{} bash -c \
  'echo "START {} $(date -u)"; python3 subs60_save_maker_paths.py --symbols {} > /tmp/sp_{}.log 2>&1; echo "DONE {} rc=$? $(date -u)"'
echo "MASTER ALL DONE $(date -u)"
