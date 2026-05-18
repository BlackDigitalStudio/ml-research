#!/usr/bin/env bash
# HD1-seq Modal bootstrap — makes an ephemeral Claude-web container able
# to drive Modal. Idempotent. Needed because the container is reclaimed
# between sessions (pip install, the certifi CA patch and the token file
# do not survive). The Anthropic egress proxy is a TLS-inspection MITM,
# so the local Modal gRPC client must trust its CA (remote Modal
# containers are NOT behind that proxy and need no patch).
#
# Env expected (set as environment secrets, not pasted in chat):
#   MODAL_TOKEN_ID / MODAL_TOKEN_SECRET   (or run `modal token set` once)
#   GCP_SA_KEY                            (service-account JSON, read GCS)
#
# Usage:  bash tools/modal_bootstrap.sh
set -euo pipefail

echo "[bootstrap] pip install modal (pinned)"
python3 -m pip install -q --disable-pip-version-check 'modal==1.4.2' \
  'numpy==2.2.4' scikit-learn >/dev/null

echo "[bootstrap] trust Anthropic egress TLS-inspection CA in certifi"
CERTIFI="$(python3 -c 'import certifi;print(certifi.where())')"
if ! grep -q "sandbox-egress" "$CERTIFI" 2>/dev/null; then
  for c in /usr/local/share/ca-certificates/*.crt; do
    [ -f "$c" ] && { printf '\n'; cat "$c"; } >> "$CERTIFI"
  done
  echo "[bootstrap] appended egress CA(s) -> $CERTIFI"
else
  echo "[bootstrap] certifi already patched"
fi

if [ -n "${MODAL_TOKEN_ID:-}" ] && [ -n "${MODAL_TOKEN_SECRET:-}" ]; then
  echo "[bootstrap] modal token set from env"
  python3 -m modal token set --token-id "$MODAL_TOKEN_ID" \
    --token-secret "$MODAL_TOKEN_SECRET" --no-verify --activate
fi

echo "[bootstrap] modal profile:"
python3 -m modal profile list 2>&1 | tail -4 || true

if [ -n "${GCP_SA_KEY:-}" ]; then
  echo "[bootstrap] (re)create Modal secret hd1-gcp from GCP_SA_KEY"
  python3 -m modal secret create hd1-gcp "GCP_SA_KEY=${GCP_SA_KEY}" \
    --force >/dev/null && echo "[bootstrap] secret hd1-gcp ready"
fi

echo "[bootstrap] done. Run:  modal run scripts/hd1_seq_modal.py --dry 1"
