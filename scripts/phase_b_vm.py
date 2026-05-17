#!/usr/bin/env python3
"""Phase B orchestrator — runs in the planning container, drives a GCP VM.

`launch`  : tar the repo -> GCS, create an n2-highcpu-96 in europe-west1
            with a HARD cost cap (scheduling.max_run_duration +
            instance_termination_action=DELETE — Compute-enforced
            self-destruct even if this ephemeral container dies or the
            startup script wedges), record the VM identity durably.
`status`  : poll the serial console + GCS results.json.
`ingest`  : pull results.json from GCS, append ledger rows via ledger.py.

Credentials: ADC (the user OAuth from gcp_bootstrap). The VM authenticates
via its attached default compute SA (metadata server) — no key shipped.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/opt/gcp-libs")
sys.path.insert(0, str(REPO))

PROJECT = "project-26a24ad0-1059-4f73-93b"
ZONE = "europe-west1-b"
BUCKET = "blackdigital-scalper-data"
MACHINE = "n2-highcpu-96"
DEFAULT_SA = "908838972123-compute@developer.gserviceaccount.com"
MAX_RUN_SEC = 14400          # 4 h hard cap -> ~$14 worst case, then DELETE
VM_RUNS = REPO / "research" / "vm_runs.jsonl"

STARTUP = r"""#!/bin/bash
set -uo pipefail
export HOME=/root        # startup-script runs under systemd with no HOME
md(){ curl -s -H "Metadata-Flavor: Google" "http://metadata/computeMetadata/v1/$1"; }
RUN_ID=$(md instance/attributes/run_id)
PROJ=$(md project/project-id)
GIT=$(md instance/attributes/git_commit)
NAME=$(md instance/name)
ZONE=$(md instance/zone | awk -F/ '{print $NF}')
exec > >(tee /var/log/phaseb.log) 2>&1
echo "PHASE_B_BOOT run=$RUN_ID git=$GIT $(date -u)"
upload(){ python3 - "$1" "$2" <<'PY' 2>/dev/null || true
import sys
from google.cloud import storage
storage.Client().bucket("blackdigital-scalper-data").blob(sys.argv[2]).upload_from_filename(sys.argv[1])
PY
}
cleanup(){
  # Stop compute charges immediately on completion/failure. The Compute-
  # enforced hard cap (max_run_duration + instance_termination_action=
  # DELETE) is the durable teardown guarantee; the orchestrator also
  # deletes promptly on the terminal monitor signal. No compute.delete
  # permission needed on the VM SA.
  upload /var/log/phaseb.log "research_runs/$RUN_ID/phaseb.log"
  poweroff -f || true
}
trap cleanup EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y build-essential curl pkg-config libssl-dev python3-pip git
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. /root/.cargo/env || export PATH=/root/.cargo/bin:$PATH
pip3 install --quiet --break-system-packages numpy pyarrow google-cloud-storage xgboost scikit-learn
python3 - <<PY
from google.cloud import storage
storage.Client().bucket("blackdigital-scalper-data").blob(f"research_runs/$RUN_ID/src.tar.gz").download_to_filename("/tmp/src.tgz")
PY
mkdir -p /opt/app && tar xzf /tmp/src.tgz -C /opt/app
NEED_RUST=$(md instance/attributes/need_rust)
if [ "$NEED_RUST" = "true" ]; then
  cd /opt/app/rust_ingest && cargo build --release --bins
fi
cd /opt/app
export SCALPER_USE_RUST=1 RUN_ID GIT
RUNNER=$(md instance/attributes/runner_cmd)
eval "$RUNNER"          # eval (not bash -c): $RUN_ID/$GIT resolve here
echo "PHASE_B_RUNNER_EXIT=$? $(date -u)"
"""


def _sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def _gcs():
    from google.cloud import storage
    return storage.Client(project=PROJECT).bucket(BUCKET)


def cmd_launch(a) -> int:
    from google.cloud import compute_v1

    git = _sh(["git", "rev-parse", "--short", "HEAD"])
    # GCE instance names must match [a-z]([-a-z0-9]{0,61}[a-z0-9])? —
    # lowercase, hyphens only. run_id doubles as the instance name.
    run_id = time.strftime("phaseb-%Y%m%d-%H%M%S", time.gmtime())
    mode = getattr(a, "mode", "alpha")
    if mode in ("alpha", "ha5", "h3"):
        script = {"alpha": "alpha_screen", "ha5": "ha5_screen",
                  "h3": "h3_screen"}[mode]
        runner = (f'python3 scripts/{script}.py --run-id "$RUN_ID" '
                  f'--git-commit "$GIT" --symbols LINK-USDT-PERP '
                  f'SOL-USDT-PERP --days 90')
        need_rust = "false"   # screens have no sim -> skip cargo build
    else:
        runner = (f'python3 scripts/phase_b_run.py --run-id "$RUN_ID" '
                  f'--git-commit "$GIT" --symbols LINK-USDT-PERP '
                  f'SOL-USDT-PERP --days 90 --timeout-sec 120')
        need_rust = "true"
    print(f"run_id={run_id} git={git} mode={mode} need_rust={need_rust}")

    # 1. source tarball (tracked files only) -> GCS
    tar = Path(f"/tmp/{run_id}_src.tar.gz")
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(tar),
                    "HEAD"], cwd=REPO, check=True)
    bk = _gcs()
    bk.blob(f"research_runs/{run_id}/src.tar.gz").upload_from_filename(str(tar))
    print(f"uploaded src.tar.gz ({tar.stat().st_size/1e6:.1f} MB)")

    # 2. create the VM with a Compute-enforced hard cost cap
    img = compute_v1.ImagesClient().get_from_family(
        project="debian-cloud", family="debian-12").self_link
    inst = compute_v1.Instance(
        name=run_id,
        machine_type=f"zones/{ZONE}/machineTypes/{MACHINE}",
        disks=[compute_v1.AttachedDisk(
            boot=True, auto_delete=True,
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                source_image=img, disk_size_gb=200,
                disk_type=f"zones/{ZONE}/diskTypes/pd-standard"))],
        network_interfaces=[compute_v1.NetworkInterface(
            access_configs=[compute_v1.AccessConfig(
                name="External NAT", type_="ONE_TO_ONE_NAT")])],
        service_accounts=[compute_v1.ServiceAccount(
            email=DEFAULT_SA,
            scopes=["https://www.googleapis.com/auth/cloud-platform"])],
        scheduling=compute_v1.Scheduling(
            provisioning_model="STANDARD",
            instance_termination_action="DELETE",
            max_run_duration=compute_v1.Duration(seconds=MAX_RUN_SEC)),
        metadata=compute_v1.Metadata(items=[
            compute_v1.Items(key="startup-script", value=STARTUP),
            compute_v1.Items(key="run_id", value=run_id),
            compute_v1.Items(key="git_commit", value=git),
            compute_v1.Items(key="runner_cmd", value=runner),
            compute_v1.Items(key="need_rust", value=need_rust)]),
        labels={"purpose": "scalper-phase-b", "run": run_id.lower()[:60]})
    op = compute_v1.InstancesClient().insert(
        project=PROJECT, zone=ZONE, instance_resource=inst)
    op.result(timeout=120)
    print(f"VM created: {inst.name} ({MACHINE}, {ZONE}) "
          f"hard-cap {MAX_RUN_SEC}s -> auto-DELETE")

    # 3. record durably (survives this container's death) + commit/push
    rec = {"run_id": run_id, "instance": inst.name, "zone": ZONE,
           "project": PROJECT, "machine": MACHINE,
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "max_run_sec": MAX_RUN_SEC, "git_commit": git,
           "gcs_prefix": f"gs://{BUCKET}/research_runs/{run_id}/",
           "status": "launched"}
    with VM_RUNS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    try:
        subprocess.run(["git", "add", str(VM_RUNS)], cwd=REPO, check=True)
        subprocess.run(["git", "commit", "-m",
                        f"chore(phase-b): record VM {inst.name} "
                        f"(hard-cap {MAX_RUN_SEC}s, auto-delete)\n\n"
                        f"https://claude.ai/code/session_014vNLDoetV7qyjB9LyLzDxT"],
                       cwd=REPO, check=True)
        subprocess.run(["git", "push", "-u", "origin",
                        "claude/explore-trading-algorithm-6oLeW"],
                       cwd=REPO, check=True)
        print("VM identity recorded + pushed (findable/killable if I die)")
    except subprocess.CalledProcessError as e:
        print(f"WARN: durable-record push failed: {e}")
    print(f"\nMonitor: python scripts/phase_b_vm.py status --run-id {run_id}")
    return 0


def cmd_status(a) -> int:
    from google.cloud import compute_v1
    name = a.run_id.replace("_", "-")
    ic = compute_v1.InstancesClient()
    try:
        vm = ic.get(project=PROJECT, zone=ZONE, instance=name)
        print(f"VM {name}: {vm.status}")
    except Exception as e:
        print(f"VM {name}: gone/unknown ({type(e).__name__}) "
              f"— likely completed + auto-deleted")
    bk = _gcs()
    rb = bk.blob(f"research_runs/{a.run_id}/results.json")
    if rb.exists():
        print("results.json:\n", rb.download_as_text()[:4000])
    else:
        try:
            out = ic.get_serial_port_output(
                project=PROJECT, zone=ZONE, instance=name)
            print("serial tail:\n", "\n".join(
                out.contents.splitlines()[-25:]))
        except Exception:
            print("no results yet, no serial (VM may be gone)")
    return 0


def cmd_ingest(a) -> int:
    bk = _gcs()
    res = json.loads(bk.blob(
        f"research_runs/{a.run_id}/results.json").download_as_text())
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ledger", REPO / "research" / "ledger.py")
    L = importlib.util.module_from_spec(spec)
    sys.modules["ledger"] = L
    spec.loader.exec_module(L)
    # alpha runner emits a flat res["records"]; phase_b emits
    # res["symbols"][sym]["ledger_record"]. Handle both.
    recs = list(res.get("records", []))
    for sym, r in res.get("symbols", {}).items():
        if r.get("ledger_record"):
            recs.append(r["ledger_record"])
        elif r.get("error"):
            print(f"{sym}: no record ({r.get('error')})")
    n = 0
    for rec in recs:
        if "error" in rec or "experiment_id" not in rec:
            print(f"skip: {rec.get('symbol')} {rec.get('error','')[:80]}")
            continue
        L.validate_experiment(rec)
        with (REPO / "research" / "experiments.jsonl").open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        n += 1
        print(f"appended {rec['experiment_id']} status={rec['status']} "
              f"ric={rec.get('rank_ic_oos')} "
              f"eS={rec.get('economic_pass_strict')}")
    print(f"ingested {n} rows; run `python3 research/ledger.py check`")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="phase_b_vm")
    s = p.add_subparsers(dest="cmd", required=True)
    lp = s.add_parser("launch")
    lp.add_argument("--mode", choices=("alpha", "ha5", "h3", "phaseb"),
                    default="alpha")
    for c in ("status", "ingest"):
        sp = s.add_parser(c)
        sp.add_argument("--run-id", required=True)
    a = p.parse_args(argv)
    return {"launch": cmd_launch, "status": cmd_status,
            "ingest": cmd_ingest}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
