#!/usr/bin/env bash
# GCP bootstrap for the Claude Code planning/orchestration container.
#
# Model (user decision 2026-05-16): this ephemeral container is the
# CONTROL PLANE. It authenticates to GCP, then provisions / drives a
# 96 vCPU VM in europe-west1 where the heavy compute (cache rebuild,
# H5/H2) runs co-located with gs://blackdigital-scalper-data. No bulk
# data is pulled through this container.
#
# Credential — non-negotiable:
#   * accepts EITHER a service-account key OR a user-ADC
#     ("authorized_user", from `gcloud auth application-default login`).
#     The ADC path exists because org policy
#     iam.disableServiceAccountKeyCreation forbids SA keys for this org
#     (user decision 2026-05-17); ADC is NOT an SA key and is allowed.
#   * arrives ONLY via the environment secret $GCP_SA_KEY (or base64
#     $GCP_SA_KEY_B64). Never via chat, never committed.
#   * written 0600 to $HOME/.gcp (outside the repo), never echoed,
#     no `set -x`. Idempotent: safe to re-run every session start.
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

# Default project when the credential carries none (ADC / bare token):
# the canonical Cryptolake-asset project (recon 2026-05-16, RESEARCH_LOG
# / CRYPTOLAKE_SCHEMA.md). Overridable via $GCP_PROJECT.
DEFAULT_GCP_PROJECT="project-26a24ad0-1059-4f73-93b"

# --- 1. materialise the credential -----------------------------------------
# Credential is resolved by CONTENT, not by which env var holds it — the
# GCP_SA_KEY slot is the one empirically confirmed to propagate into a
# live session here, so it accepts EITHER form:
#   * value starts with `ya29.`  -> a raw OAuth2 bearer token (~1 h).
#       Phone path: `gcloud auth print-access-token` in Cloud Shell.
#       The container only holds creds for SHORT bursts (VM launch
#       ~1-2 min, status/ingest ~sec); the multi-hour screen runs on
#       the VM's OWN attached SA (metadata) — ~1 h is ample, expiry
#       mid-run loses nothing (results durable in GCS, VM self-deletes).
#   * value starts with `{`      -> JSON: an SA key OR a user-ADC
#       ("authorized_user", from `gcloud auth application-default
#       login`; org policy forbids SA keys here — ADC is allowed).
#   * value starts with `4/0`    -> an OAuth AUTHORIZATION CODE: refuse
#       with guidance (single-use code, not a credential).
# Priority: GCP_ACCESS_TOKEN, then GCP_SA_KEY_B64, then GCP_SA_KEY,
# then a pre-materialised $CRED_FILE. Whitespace in a token is stripped.
_RAW=""
if [[ -n "${GCP_ACCESS_TOKEN:-}" ]]; then
  _RAW="$GCP_ACCESS_TOKEN"
elif [[ -n "${GCP_SA_KEY_B64:-}" ]]; then
  _RAW="$(printf '%s' "$GCP_SA_KEY_B64" | base64 -d 2>/dev/null || true)"
elif [[ -n "${GCP_SA_KEY:-}" ]]; then
  _RAW="$GCP_SA_KEY"
fi
_RAW_TRIM="$(printf '%s' "$_RAW" | tr -d '[:space:]')"
if [[ -z "$_RAW" && -f "$CRED_FILE" ]]; then
  CRED_MODE="json"                       # already materialised this session
elif [[ "$_RAW_TRIM" == ya29.* || "$_RAW_TRIM" == ya29_* ]]; then
  CRED_MODE="token"; GCP_ACCESS_TOKEN="$_RAW_TRIM"
elif [[ "$_RAW_TRIM" == 4/0* ]]; then
  die "credential is an OAuth AUTHORIZATION CODE (4/0...), not a usable \
credential. In Cloud Shell run 'gcloud auth print-access-token' and put \
THAT (starts with ya29.) into the GCP_SA_KEY secret."
elif [[ "${_RAW#"${_RAW%%[![:space:]]*}"}" == \{* ]]; then
  mkdir -p "$CRED_DIR"; printf '%s' "$_RAW" > "$CRED_FILE"; CRED_MODE="json"
elif [[ -n "$_RAW" ]]; then
  die "credential is set but unrecognised (not ya29.* token, not {...} \
JSON, not 4/0 code). Put either an access token or an SA/ADC JSON into \
the GCP_SA_KEY env secret."
else
  die "no credentials. Set the GCP_SA_KEY env secret to EITHER a one-line
  access token (gcloud auth print-access-token, starts ya29.) OR an
  SA/ADC JSON. Env-secret channel ONLY — never chat, never committed."
fi

case "$CRED_DIR" in
  "$PWD"/*|"$PWD") die "refusing: credential dir $CRED_DIR is inside the
  repo. Set GCP_CRED_DIR outside the working tree." ;;
esac

if [[ "$CRED_MODE" == "token" ]]; then
  # bare token: nothing on disk; google libs get explicit creds (see
  # scripts/phase_b_vm.py::_creds). project must come from env/default.
  # Heal a wrapped/space-polluted paste (strip ALL whitespace — no
  # whitespace is valid inside a bearer token).
  GCP_ACCESS_TOKEN="$(printf '%s' "$GCP_ACCESS_TOKEN" | tr -d '[:space:]')"
  [[ -n "$GCP_ACCESS_TOKEN" ]] || die "GCP_ACCESS_TOKEN is empty after \
trimming whitespace."
  case "$GCP_ACCESS_TOKEN" in
    4/0*) die "GCP_ACCESS_TOKEN looks like an OAuth AUTHORIZATION CODE \
(4/0...), not an access token. In Cloud Shell run \
'gcloud auth print-access-token' and paste THAT (starts with ya29.)." ;;
    \{*)  die "GCP_ACCESS_TOKEN looks like JSON — that belongs in \
GCP_SA_KEY, not GCP_ACCESS_TOKEN. For the token path paste the one-line \
output of 'gcloud auth print-access-token' (starts with ya29.)." ;;
  esac
  unset GOOGLE_APPLICATION_CREDENTIALS || true
  GCP_PROJECT="${GCP_PROJECT:-$DEFAULT_GCP_PROJECT}"
  export GCP_ACCESS_TOKEN GCP_PROJECT GCP_REGION GCP_ZONE
  echo "gcp_bootstrap: project=$GCP_PROJECT region=$GCP_REGION identity=oauth-access-token(~1h)"
else
  chmod 600 "$CRED_FILE"
  python3 - "$CRED_FILE" <<'PY' || die "credential is not valid JSON / not a service_account key nor an authorized_user ADC"
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("type")
if t == "service_account":
    assert d.get("project_id") and d.get("client_email"), "SA key missing project_id/client_email"
elif t == "authorized_user":
    assert d.get("client_id") and d.get("client_secret") and d.get("refresh_token"), \
        "authorized_user ADC missing client_id/client_secret/refresh_token"
else:
    raise SystemExit(f"unsupported credential type {t!r} (need service_account or authorized_user)")
PY
  export GOOGLE_APPLICATION_CREDENTIALS="$CRED_FILE"
  GCP_PROJECT="${GCP_PROJECT:-$(python3 -c 'import json,os
d=json.load(open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]))
print(d.get("project_id") or d.get("quota_project_id") or "")')}"
  GCP_PROJECT="${GCP_PROJECT:-$DEFAULT_GCP_PROJECT}"
  export GOOGLE_APPLICATION_CREDENTIALS GCP_PROJECT GCP_REGION GCP_ZONE
  CRED_ID="$(python3 -c 'import json,os
d=json.load(open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]))
print(d.get("client_email") or ("authorized_user:"+d.get("client_id","")[:18]+"... (ADC)"))')"
  echo "gcp_bootstrap: project=$GCP_PROJECT region=$GCP_REGION identity=$CRED_ID"
fi

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

# token mode: explicit bearer creds (no ADC on disk). JSON mode: None
# -> google libs discover GOOGLE_APPLICATION_CREDENTIALS as before.
_tok = os.environ.get("GCP_ACCESS_TOKEN")
creds = None
if _tok:
    import google.oauth2.credentials
    class _StaticToken(google.oauth2.credentials.Credentials):
        def refresh(self, request):   # bare token: never refresh
            return
    creds = _StaticToken(token="".join(_tok.split()))

sc = storage.Client(project=proj, credentials=creds)
for b in buckets:
    try:
        bk = sc.bucket(b)
        sample = list(sc.list_blobs(bk, max_results=1))
        print(f"  storage OK  gs://{b}  (reachable, sample_objs={len(sample)})")
    except Exception as e:
        print(f"  storage WARN gs://{b}: {type(e).__name__}: {e}")

zc = compute_v1.ZonesClient(credentials=creds)
zs = [z.name for z in zc.list(project=proj) if z.name.startswith(region)]
print(f"  compute OK  {len(zs)} zones in {region}: {zs[:4]}")
print("gcp_bootstrap: auth + storage + compute verified.")
PY
echo "gcp_bootstrap: done. GOOGLE_APPLICATION_CREDENTIALS exported for this session."
