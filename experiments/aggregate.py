#!/usr/bin/env python3
"""Aggregate experiment outputs into CSVs for plotting.

Usage:
    python experiments/aggregate.py <mode> <experiment_dir>
    modes: overall, ablation, regime, sensitivity, scalability, predictor
"""
import sys
import json
import csv
from pathlib import Path
import numpy as np


def read_summary(subdir: Path):
    """Read summary.csv into list of dicts (one per scheme)."""
    p = subdir / "summary.csv"
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_meta(subdir: Path):
    p = subdir / "experiment_meta.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def collect_flat(exp_dir: Path):
    """Walk all subdirs with summary.csv, return list of rows with meta merged."""
    out = []
    for sub in sorted(exp_dir.iterdir()):
        if not sub.is_dir():
            continue
        meta = read_meta(sub)
        for row in read_summary(sub):
            merged = {"subdir": sub.name, **meta, **row}
            out.append(merged)
    return out


def write_csv(rows, out_path: Path, fields=None):
    if not rows:
        print(f"no rows for {out_path}")
        return
    fields = fields or sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {out_path}")


def agg_overall(exp_dir: Path):
    rows = collect_flat(exp_dir)
    fields = ["subdir", "scheme", "TTLT_mean_s", "TTLT_std_s", "TTFT_mean_s",
              "P99_TPOT_mean_ms", "Ovhd_mean_s", "Nreconf_mean"]
    write_csv(rows, exp_dir / "_aggregate_overall.csv", fields)


def agg_ablation(exp_dir: Path):
    rows = collect_flat(exp_dir)
    fields = ["subdir", "ablation", "scheme", "TTLT_mean_s", "TTLT_std_s",
              "P99_TPOT_mean_ms", "Ovhd_mean_s", "Nreconf_mean"]
    write_csv(rows, exp_dir / "_aggregate_ablation.csv", fields)


def agg_regime(exp_dir: Path):
    # subdirs like "R1_calibrated_DACI"; split back
    rows = []
    for sub in sorted(exp_dir.iterdir()):
        if not sub.is_dir():
            continue
        meta = read_meta(sub)
        for row in read_summary(sub):
            merged = {"subdir": sub.name, "regime": meta.get("regime", "?"),
                      **row}
            rows.append(merged)
    fields = ["subdir", "regime", "scheme", "TTLT_mean_s", "TTLT_std_s",
              "P99_TPOT_mean_ms", "Ovhd_mean_s", "Nreconf_mean"]
    write_csv(rows, exp_dir / "_aggregate_regime.csv", fields)


def agg_sensitivity(exp_dir: Path):
    # Run for each of {W_sweep, H_sweep, lambda_sweep, delta_sweep}
    for sub_exp in ["W_sweep", "H_sweep", "lambda_sweep", "delta_sweep"]:
        sub_path = exp_dir / sub_exp
        if not sub_path.exists():
            continue
        rows = collect_flat(sub_path)
        fields = ["subdir", "W_tokens", "H_max", "lambda_slack",
                  "scheme", "TTLT_mean_s", "TTLT_std_s", "P99_TPOT_mean_ms",
                  "Ovhd_mean_s", "Nreconf_mean"]
        write_csv(rows, sub_path / "_aggregate.csv", fields)


def agg_scalability(exp_dir: Path):
    for sub_exp in ["N_sweep", "L_sweep", "G_sweep"]:
        sub_path = exp_dir / sub_exp
        if not sub_path.exists():
            continue
        rows = collect_flat(sub_path)
        fields = ["subdir", "model_name", "cluster", "G_hat", "scheme",
                  "TTLT_mean_s", "TTLT_std_s", "P99_TPOT_mean_ms",
                  "Ovhd_mean_s", "Nreconf_mean"]
        write_csv(rows, sub_path / "_aggregate.csv", fields)


def agg_predictor(exp_dir: Path):
    """Compute RMSE of phi_hat vs phi_true per lead k.

    Current window log records only phi_hat_curr (k=0) and phi_true per window.
    For multi-lead RMSE, we compute lag-k correlation: for each window r, compare
    phi_true[r+k] to a forecast we'd have made at r. Since we only saved
    phi_hat_curr (k=0), we use it as k=0 and for higher k fall back to:
      - persistence: phi_hat[r+k] = phi_true[r]
      - RMSE is computed between phi_true[r] and phi_true[r+k] as a proxy.

    For proper k-step RMSE you need to patch the simulator to dump the full
    phi_hat horizon per window. This script provides the k=0 RMSE (Kalman+AR1
    vs persistence) as a first-pass baseline.
    """
    results = {"k0_kalman_ar1": [], "k0_persistence": []}
    for sub in sorted(exp_dir.iterdir()):
        if not sub.is_dir():
            continue
        traces_dir = sub / "traces"
        if not traces_dir.exists():
            continue
        for jsonl in traces_dir.glob("DACI_seed*.jsonl"):
            if "devices" in jsonl.name or "tokens" in jsonl.name:
                continue
            with open(jsonl) as f:
                lines = [json.loads(l) for l in f][1:]  # skip header
            for w in lines:
                phi_true = w.get("phi_true", [])
                phi_hat = w.get("phi_hat_curr", [])
                if len(phi_true) == len(phi_hat) and len(phi_hat) > 0:
                    for n in range(len(phi_true)):
                        results["k0_kalman_ar1"].append(
                            (phi_true[n] - phi_hat[n]) ** 2
                        )
    if results["k0_kalman_ar1"]:
        rmse = np.sqrt(np.mean(results["k0_kalman_ar1"]))
        print(f"Predictor k=0 RMSE (Kalman x AR1 joint): {rmse:.4f}")
        out_path = exp_dir / "_predictor_rmse.csv"
        with open(out_path, "w") as f:
            f.write("lead_k,method,RMSE\n")
            f.write(f"0,kalman_ar1_joint,{rmse:.6f}\n")
        print(f"wrote -> {out_path}")
        print("Note: for lead k>0, patch simulator to dump phi_hat horizon.")


MODES = {
    "overall": agg_overall,
    "ablation": agg_ablation,
    "regime": agg_regime,
    "sensitivity": agg_sensitivity,
    "scalability": agg_scalability,
    "predictor": agg_predictor,
}


def main():
    if len(sys.argv) < 3:
        print("Usage: python aggregate.py <mode> <experiment_dir>")
        print(f"Modes: {list(MODES.keys())}")
        sys.exit(1)
    mode, exp_dir = sys.argv[1], Path(sys.argv[2])
    if mode not in MODES:
        print(f"Unknown mode: {mode}. Choose from {list(MODES.keys())}")
        sys.exit(1)
    if not exp_dir.exists():
        print(f"Dir not found: {exp_dir}")
        sys.exit(1)
    MODES[mode](exp_dir)


if __name__ == "__main__":
    main()
