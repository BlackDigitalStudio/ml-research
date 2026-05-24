#!/bin/bash
set -x
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip
pip3 install --break-system-packages -q numpy pandas pyarrow xgboost scikit-learn google-cloud-storage
curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/freecode" > /root/hdata_rawbars.py
cd /root
N_SYMBOLS=8 OUT_TAG=_smoke python3 hdata_rawbars.py && N_SYMBOLS=120 python3 hdata_rawbars.py
echo "RAWBARS_STARTUP_DONE rc=$?"
sleep 15
poweroff
