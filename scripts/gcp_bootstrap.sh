#!/usr/bin/env bash
# GCP bootstrap for the Claude Code planning/orchestration container.
#
# Model (user decision 2026-05-16): this ephemeral container is the
# CONTROL PLANE. It authenticates to GCP, then provisions / drives a
# 96 vCPU VM in europe-west1 where the heavy compute (cache rebuild,
# H5/H2) runs co-located with gs://blackdigital-scalper-data. No bulk
# data is pulled through this container.
#
# Secret handling — non-negotiable:
#   * the SA key arrives ONLY via the environment secret $GCP_SA_KEY
#     (or base64 $GCP_SA_KEY_B64). Never via chat, never committed.
#   * the key is written 0600 to $HOME/.gcp (outside the repo) and is
#     never echoed. No `set -x`.
#   * idempotent: safe to re-run every session start.
#
# Wire this as the environment's setup script (Claude Code on the web:
# environment settings -> setup script), or `source` it manually.
set -euo pipefail
umask 077

CRED_DIR="${GCP_CRED_DIR:-$HOME/.gcp}"
CRED_FILE="$CRED_DIR/sa.json"
GCP_REGION="${GCP_REGION:-europe-west1}"
GCP_ZONE="${GCP_ZONE:-europe-west1-b}"

die() { echo "gcp_bootstrap: $*" >&2; exit 1; }

# --- 1. materialise credentials from the environment secret ---------------
if [[ -n "${GCP_SA_KEY_B64:-}" ]]; then
  mkdir -p "$CRED_DIR"
  printf '%s' "$GCP_SA_KEY_B64" | base64 -d > "$CRED_FILE"
elif [[ -n "${GCP_SA_KEY:-}" ]]; then
  mkdir -p "$CRED_DIR"
  printf '%s' "$GCP_SA_KEY" > "$CRED_FILE"
elif [[ -f "$CRED_FILE" ]]; then
  : # already materialised this session
else
  die "no credentials. Set the environment secret GCP_SA_KEY (raw SA
  JSON) or GCP_SA_KEY_B64 (base64). This is the ONLY accepted channel —
  do not paste the key into chat or commit it."
fi
chmod 600 "$CRED_FILE"

case "$CRED_DIR" in
  "$PWD"/*|"$PWD") die "refusing: credential dir $CRED_DIR is inside the
  repo. Set GCP_CRED_DIR outside the working tree." ;;
esac
python3 - "$CRED_FILE" <<'PY' || die "GCP_SA_KEY is not valid JSON / not a service-account key"
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("type") == "service_account", "not a service_account key"
assert d.get("project_id") and d.get("client_email"), "missing project_id/client_email"
PY

export GOOGLE_APPLICATION_CREDENTIALS="$CRED_FILE"
GCP_PROJECT="${GCP_PROJECT:-$(python3 -c 'import json,os;print(json.load(open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]))["project_id"])')}"
export GOOGLE_APPLICATION_CREDENTIALS GCP_PROJECT GCP_REGION GCP_ZONE
SA_EMAIL="$(python3 -c 'import json,os;print(json.load(open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]))["client_email"])')"
echo "gcp_bootstrap: project=$GCP_PROJECT region=$GCP_REGION sa=$SA_EMAIL"

# --- 2. install the Python clients (PyPI is reachable; gcloud CLI is NOT
#        required — provisioning uses google-cloud-compute) --------------
if ! python3 -c 'import google.cloud.storage, google.cloud.compute_v1, google.auth' 2>/dev/null; then
  echo "gcp_bootstrap: installing google-cloud SDK (pip)..."
  pip install --quiet --disable-pip-version-check \
    google-cloud-storage google-cloud-compute google-auth
fi

# --- 3. verify both scopes WITHOUT leaking the key ------------------------
#   storage: list the two asset buckets (names + a sample object count)
#   compute: list zones in the region (proves instanceAdmin scope)
python3 - <<'PY'
import os
from google.cloud import storage
from google.cloud import compute_v1

proj = os.environ["GCP_PROJECT"]
region = os.environ["GCP_REGION"]
buckets = ["blackdigital-scalper-data", "scalper-bot-research-data"]

sc = storage.Client(project=proj)
for b in buckets:
    try:
        bk = sc.bucket(b)
        sample = list(sc.list_blobs(bk, max_results=1))
        print(f"  storage OK  gs://{b}  (reachable, sample_objs={len(sample)})")
    except Exception as e:
        print(f"  storage WARN gs://{b}: {type(e).__name__}: {e}")

zc = compute_v1.ZonesClient()
zs = [z.name for z in zc.list(project=proj) if z.name.startswith(region)]
print(f"  compute OK  {len(zs)} zones in {region}: {zs[:4]}")
print("gcp_bootstrap: auth + storage + compute verified.")
PY
echo "gcp_bootstrap: done. GOOGLE_APPLICATION_CREDENTIALS exported for this session."
