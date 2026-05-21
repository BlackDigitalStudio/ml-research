#!/usr/bin/env python3
"""Build ONE unified SQLite ledger from all append-only JSONL event logs
plus the per-rev probe result JSONs. Idempotent: rebuilds from scratch.

Sources (all under research/):
  experiments.jsonl       -> experiments      (confirmatory HM1 store)
  hypotheses.jsonl        -> hypotheses        (hypothesis/rev registry)
  hardware_ledger.jsonl   -> hardware_ledger   (per-app hw/cost)
  vm_runs.jsonl           -> vm_runs           (GCP VM run log)
  rev{50,52,54,56}_*.json -> session_cells     (per-cell probe summaries)
                          -> session_epochs    (per-epoch learning curves)

Each row keeps the most-queried columns AS columns plus the full record in
raw_json (so nothing is lost as schemas evolve). Run:
    python3 research/build_full_ledger_db.py
Output: research/full_ledger.db
"""
import glob
import json
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "full_ledger.db")


def _load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass
    return rows


def main():
    if os.path.exists(OUT):
        os.remove(OUT)
    db = sqlite3.connect(OUT)
    c = db.cursor()
    c.executescript("""
    CREATE TABLE experiments (
      experiment_id TEXT, ts TEXT, hypothesis_id TEXT, kind TEXT,
      status TEXT, model_family TEXT, symbols TEXT, horizon_sec TEXT,
      rank_ic_oos REAL, delta_ic REAL, auc_oos REAL, n_samples INTEGER,
      n_eff REAL, baseline_ref TEXT, economic_pass_strict TEXT,
      setup TEXT, note TEXT, git_commit TEXT, raw_json TEXT);
    CREATE TABLE hypotheses (
      hypothesis_id TEXT, rev INTEGER, ts TEXT, status TEXT,
      priority_rank INTEGER, result_experiment_id TEXT,
      statement TEXT, note TEXT, raw_json TEXT,
      PRIMARY KEY (hypothesis_id, rev));
    CREATE TABLE hardware_ledger (
      rev INTEGER, ts TEXT, app_id TEXT, workspace TEXT, stage TEXT,
      hw_class TEXT, gpu TEXT, vcpu TEXT, memory_gib TEXT,
      wall_s REAL, cost_usd REAL, note TEXT, raw_json TEXT);
    CREATE TABLE vm_runs (raw_json TEXT);
    CREATE TABLE session_cells (
      rev INTEGER, source_file TEXT, cell_tag TEXT, arch TEXT,
      L INTEGER, D INTEGER, dropout REAL, wd REAL, seed INTEGER,
      n_params INTEGER, n_fit INTEGER, n_val INTEGER, n_te INTEGER,
      block INTEGER, best_val_ric REAL, best_val_ep INTEGER,
      test_at_best_val REAL, best_test_ric REAL, best_test_ep INTEGER,
      last_train_loss REAL, placebo_ric REAL, boot_se REAL, gpu_s REAL);
    CREATE TABLE session_epochs (
      rev INTEGER, cell_tag TEXT, ep INTEGER, lr REAL, train_loss REAL,
      val_logloss REAL, val_ric REAL, test_logloss REAL, test_ric REAL);
    """)

    # --- experiments.jsonl ---
    for o in _load_jsonl(os.path.join(HERE, "experiments.jsonl")):
        c.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            o.get("experiment_id"), o.get("ts"), o.get("hypothesis_id"),
            o.get("kind"), o.get("status"), o.get("model_family"),
            json.dumps(o.get("symbols")), str(o.get("horizon_sec", o.get("horizon_s"))),
            o.get("rank_ic_oos"), o.get("delta_ic"), o.get("auc_oos"),
            o.get("n_samples"), o.get("n_eff"), o.get("baseline_ref"),
            str(o.get("economic_pass_strict")), o.get("setup"), o.get("note"),
            o.get("git_commit"), json.dumps(o)))

    # --- hypotheses.jsonl ---
    for o in _load_jsonl(os.path.join(HERE, "hypotheses.jsonl")):
        if "hypothesis_id" not in o or "rev" not in o:
            continue
        c.execute("INSERT OR REPLACE INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?)", (
            o.get("hypothesis_id"), o.get("rev"), o.get("ts"),
            o.get("status"), o.get("priority_rank"),
            o.get("result_experiment_id"), o.get("statement"),
            o.get("note"), json.dumps(o)))

    # --- hardware_ledger.jsonl ---
    for o in _load_jsonl(os.path.join(HERE, "hardware_ledger.jsonl")):
        hw = o.get("hw", {})
        tim = o.get("timing", {})
        bil = o.get("billing", {})
        c.execute("INSERT INTO hardware_ledger VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            o.get("refs", {}).get("rev"), o.get("ts"), o.get("modal_app_id"),
            o.get("workspace"), o.get("stage"), hw.get("class"),
            hw.get("gpu"), str(hw.get("vcpu")), str(hw.get("memory_gib")),
            tim.get("wall_s"), bil.get("actual_cost_usd"),
            o.get("note"), json.dumps(o)))

    # --- vm_runs.jsonl ---
    for o in _load_jsonl(os.path.join(HERE, "vm_runs.jsonl")):
        c.execute("INSERT INTO vm_runs VALUES (?)", (json.dumps(o),))

    # --- per-rev probe result JSONs (session cells + epochs) ---
    def ins_cell(rev, src, r):
        s = r["summary"]
        bv, bt, le = s["by_best_val_ric"], s["by_best_test_ric"], s["last_ep"]
        fd = r.get("final_ep_diagnostics", {})
        cfg = r.get("cfg", {})
        c.execute("INSERT INTO session_cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            rev, src, r.get("cell_tag", cfg.get("sym", "")), r.get("arch"),
            r.get("L", cfg.get("L")), r.get("D", cfg.get("D")),
            r.get("dropout", cfg.get("DROPOUT")), r.get("wd", cfg.get("WD")),
            r.get("seed"), r.get("n_params", cfg.get("n_params")),
            r.get("n_fit"), r.get("n_val"), r.get("n_te"), r.get("block"),
            bv["val_ric"], bv["ep"], bv["test_ric"],
            bt["test_ric"], bt["ep"], le["train_loss"],
            fd.get("placebo_ric"), fd.get("boot_se"), r.get("gpu_s")))
        tag = r.get("cell_tag", cfg.get("sym", "lcurve"))
        for h in r["history"]:
            c.execute("INSERT INTO session_epochs VALUES (?,?,?,?,?,?,?,?,?)", (
                rev, tag, h["ep"], h["lr"], h["train_loss"],
                h["val_logloss"], h["val_ric"], h.get("test_logloss"),
                h["test_ric"]))

    rev_files = {50: "rev50_lcurve_seed0.json", 52: "rev52_regsweep.json",
                 54: "rev54_c1_engineered_dense.json", 56: "rev56_lsweep.json"}
    for rev, fn in rev_files.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        if "full_histories" in r:
            for cell in r["full_histories"]:
                ins_cell(rev, fn, cell)
        else:
            ins_cell(rev, fn, r)

    db.commit()
    # report
    for t in ("experiments", "hypotheses", "hardware_ledger", "vm_runs",
              "session_cells", "session_epochs"):
        n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {n} rows")
    hw_tot = c.execute(
        "SELECT SUM(cost_usd) FROM hardware_ledger").fetchone()[0]
    print(f"  hardware_ledger total cost: ${hw_tot:.2f}")
    db.close()
    print(f"\nDB written: {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
