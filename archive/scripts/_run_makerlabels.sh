#!/bin/bash
# Full maker-label build: all 8 symbols, full history, 4-way parallel.
cd /tmp
SYMS="BNB-USDT-PERP BTC-USDT-PERP DOGE-USDT-PERP ETH-USDT-PERP LINK-USDT-PERP LTC-USDT-PERP SOL-USDT-PERP XRP-USDT-PERP"
echo "MASTER START $(date -u)"
printf '%s\n' $SYMS | xargs -P 4 -I{} bash -c \
  'echo "START {} $(date -u)"; python3 subs60_makerlabel_build.py --symbols {} > /tmp/ml_{}.log 2>&1; echo "DONE {} rc=$? $(date -u)"'
echo "MASTER ALL DONE $(date -u)"
