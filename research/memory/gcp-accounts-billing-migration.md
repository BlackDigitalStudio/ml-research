---
name: gcp-accounts-billing-migration
description: GCP billing crisis + Variant-B relink to a new $300 account; data-bucket topology across old/new accounts
metadata: 
  node_type: memory
  type: project
  originSessionId: 2f125ab3-dff7-4906-8f09-cd7955db840b
---

**~2026-07-04 billing crisis + migration (Variant B).**

- **Old account** `virgin.ship03@gmail.com` / project `project-0998ac51-36ba-445c-bc7`: its billing account `018133-BBEBCE-A6335A` went **delinquent** (card Visa ···1090 declined; balance **$150.41** owed; $300 free-trial fully consumed). Delinquency blocked GCS (403) on both its buckets. **The $150.41 was NOT paid** — it still sits on 018133.
- **Fix = Variant B:** created a **new Google account** with billing `01CB22-236825-287C72` (fresh **$300 free trial, ~90 days** from Jul 4) and project `project-d39e90d0-62e9-416a-aaf`, then **relinked `project-0998ac51` to billing 01CB22** (needed a cross-account IAM grant: new account gave `virgin.ship03` the *Billing Account User* role on 01CB22). Result: old project's GCS unblocked, recorder sync resumed, **no $150 paid**.
- **Data buckets:**
  - Live/original (in old project, now on new billing): `gs://recorder-data-asia-0998ac51` (asia-northeast1, recorder, ~226 GiB) + `gs://market-data-0998ac51` (europe-west1, Cryptolake ~585 GB).
  - **Full backup copies on the NEW account** (project-d39e90d0): `gs://recorder-data-asia-d39e90d0` (asia) + `gs://market-data-eu-d39e90d0` (europe). Copied same-region (no egress) via `gcloud storage rsync`; recorder copy verified **byte-exact** (242,616,235,839 B).
  - The **live recorder still writes to the OLD bucket** `recorder-data-asia-0998ac51`; the `-d39e90d0` copies are point-in-time snapshots (re-rsync to refresh).
- **Runway:** the recorder VM + storage + copies now all bill to the $300 trial; burn ≈ **$100–130/mo** → **~2–2.5 months**. Decide long-term home before it lapses. Old blackdigital account (`gs://blackdigital-scalper-data`) billing is **closed** (also blocked). See [[recorder-vm-live]], [[cryptolake-data-locations]] if present.
