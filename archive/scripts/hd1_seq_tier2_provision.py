#!/usr/bin/env python3
"""HD1 Tier-2 (rev45) GCP build-VM provisioner.

Phase 1 (no spend): bundle the Rust crate + the frozen-faithful build
scripts, upload to a GCS staging prefix (ADC, intra-project).
Phase 2 (SPEND starts): create ONE on-demand n2-standard-96 VM in
europe-west1-b (co-located with the europe-west1 bucket so the GCS read
is free intra-region), default compute SA (already has objectAdmin on
the bucket), with a startup-script that builds the Rust binary, runs the
frozen MAX_L=1536 build for {BTC,ETH,LTC}, stages the packed cache back
to GCS, and then SHUTS THE VM DOWN.

Self-stop is deliberate: per the rev45 measured-spend discipline the VM
must NOT auto-continue to the Modal transfer/sweep — it halts so the
user sees the actual build cost before the next stage is authorized.

Region/machine/symbols are determined (not configurable): the bucket is
europe-west1 regional, only N2 has >=96 vCPU quota, and the rev45 cells
use only BTC/ETH/LTC. SOL is still listed for the window plan only.
"""
from __future__ import annotations

import io
import os
import sys
import tarfile
import time

PROJECT = "project-26a24ad0-1059-4f73-93b"
BUCKET = "blackdigital-scalper-data"
ZONE = "europe-west1-b"
REGION = "europe-west1"
MACHINE = "n2-standard-96"
SA_EMAIL = "908838972123-compute@developer.gserviceaccount.com"
STAGE_PREFIX = "hd1seq_tier2_pack"
BUNDLE_KEY = f"{STAGE_PREFIX}/bundle/tier2_bundle.tar.gz"
INSTANCE = "hd1-tier2-build"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STARTUP = r"""#!/bin/bash
# NO `set -u` (unset $HOME under the GCE metadata script runner would
# kill the script before any GCS write) and NO `set -e` (build failure
# must still reach the trap so the log + FAILED marker are uploaded).
set -x
set -o pipefail
exec > >(tee /var/log/tier2_startup.log) 2>&1
export HOME=/root CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup
export PATH=/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin:/sbin
GS="gs://%(bucket)s/%(stage)s"
# gsutil ships on GCE Google images and auto-auths via the instance SA;
# use it (not pip lib) for the CRITICAL fetch + the always-on
# post-mortem so a pip/network failure can never blind us again.
upload_log(){ gsutil -q cp /var/log/tier2_startup.log \
    "$GS/l1536/tier2_startup.log" 2>/dev/null || true; }
trap 'rc=$?; echo "[startup] EXIT rc=$rc"; upload_log; \
      shutdown -h now' EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update -y || true
apt-get install -y python3-pip build-essential pkg-config libssl-dev \
    curl tar ca-certificates google-cloud-cli || true
pip3 install --break-system-packages -q google-cloud-storage numpy \
    || pip3 install -q google-cloud-storage numpy || true
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    --default-toolchain stable

mkdir -p /opt/tier2 && cd /opt/tier2
gsutil -q cp "$GS/bundle/tier2_bundle.tar.gz" /opt/tier2/b.tgz
tar xzf b.tgz

cd /opt/tier2/rust_ingest
/root/.cargo/bin/cargo build --release -p depth_parser \
    --bin hd1_seq_build
RB=/opt/tier2/rust_ingest/target/release/hd1_seq_build
cd /opt/tier2

python3 scripts/hd1_seq_tier2_gcpbuild.py --max-l 1536 \
    --work /var/tier2work --rust-bin "$RB" \
    --build-syms BTC-USDT-PERP,ETH-USDT-PERP,LTC-USDT-PERP \
    --stage "$GS/l1536"
RC=$?
echo "[startup] build rc=$RC"
if [ "$RC" -ne 0 ]; then
  echo "rc=$RC see tier2_startup.log" | \
    gsutil -q cp - "$GS/l1536/TIER2_BUILD_FAILED.txt" || true
fi
echo "[startup] complete rc=$RC"
# trap (EXIT) uploads the full log and powers the VM off.
""" % {"bucket": BUCKET, "stage": STAGE_PREFIX}


def _bundle_bytes() -> bytes:
    """tar.gz of rust_ingest/ (minus target/) + the two scripts."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        def add(rel):
            tf.add(os.path.join(REPO, rel), arcname=rel)
        for rel in ("scripts/hd1_seq_tier2_gcpbuild.py",
                    "scripts/hd1_seq_core.py"):
            add(rel)
        ri = os.path.join(REPO, "rust_ingest")
        for root, dirs, files in os.walk(ri):
            if "target" in dirs:
                dirs.remove("target")
            for fn in files:
                ap = os.path.join(root, fn)
                tf.add(ap, arcname=os.path.relpath(ap, REPO))
    return buf.getvalue()


def phase1_bundle():
    from google.cloud import storage
    data = _bundle_bytes()
    cl = storage.Client(project=PROJECT)
    cl.bucket(BUCKET).blob(BUNDLE_KEY).upload_from_string(
        data, content_type="application/gzip")
    print(f"[bundle] uploaded gs://{BUCKET}/{BUNDLE_KEY} "
          f"({len(data)} bytes)")


def phase2_provision():
    from google.cloud import compute_v1
    ic = compute_v1.InstancesClient()
    try:
        cur = ic.get(project=PROJECT, zone=ZONE, instance=INSTANCE)
        print(f"[vm] '{INSTANCE}' already exists status={cur.status} "
              f"-- not recreating")
        return
    except Exception:
        pass

    # pd-standard (HDD): counts against DISKS_TOTAL_GB (2458), NOT the
    # europe-west1 SSD_TOTAL_GB quota (300, which pd-balanced/pd-ssd
    # consume). Single 700GB root holds OS + /var/tier2work; the build
    # is CPU/network-bound with sequential npz IO, fine on pd-standard.
    boot = compute_v1.AttachedDisk(
        boot=True, auto_delete=True,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            source_image="projects/debian-cloud/global/images/family/"
                          "debian-12",
            disk_size_gb=700, disk_type=(
                f"zones/{ZONE}/diskTypes/pd-standard")))

    inst = compute_v1.Instance(
        name=INSTANCE,
        machine_type=f"zones/{ZONE}/machineTypes/{MACHINE}",
        disks=[boot],
        network_interfaces=[compute_v1.NetworkInterface(
            access_configs=[compute_v1.AccessConfig(
                name="External NAT", type_="ONE_TO_ONE_NAT")])],
        service_accounts=[compute_v1.ServiceAccount(
            email=SA_EMAIL,
            scopes=["https://www.googleapis.com/auth/cloud-platform"])],
        scheduling=compute_v1.Scheduling(
            provisioning_model="STANDARD",      # on-demand (user choice)
            automatic_restart=True,
            on_host_maintenance="MIGRATE"),
        metadata=compute_v1.Metadata(items=[
            compute_v1.Items(key="startup-script", value=STARTUP)]),
        labels={"job": "hd1-tier2-build", "rev": "rev45"},
        tags=compute_v1.Tags(items=["hd1-tier2"]))

    print(f"[vm] creating {INSTANCE} {MACHINE} @ {ZONE} "
          f"(on-demand; SPEND STARTS NOW)")
    t0 = time.time()
    op = ic.insert(project=PROJECT, zone=ZONE, instance_resource=inst)
    try:
        op.result(timeout=240)                  # ExtendedOperation
    except Exception as e:
        print(f"[vm] insert op wait note: {type(e).__name__} {e}")
    got = ic.get(project=PROJECT, zone=ZONE, instance=INSTANCE)
    ext = ""
    try:
        ext = got.network_interfaces[0].access_configs[0].nat_i_p
    except Exception:
        pass
    print(f"[vm] {INSTANCE} status={got.status} ext_ip={ext} "
          f"create_op={op.status} ({time.time()-t0:.0f}s)")
    print(f"[vm] cost clock running ~$4.66/hr on-demand {MACHINE}. "
          f"It self-stops after build; DONE marker -> "
          f"gs://{BUCKET}/{STAGE_PREFIX}/l1536/TIER2_BUILD_DONE.json")


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if "--provision-only" not in args:
        phase1_bundle()
    if "--bundle-only" not in args:
        phase2_provision()
