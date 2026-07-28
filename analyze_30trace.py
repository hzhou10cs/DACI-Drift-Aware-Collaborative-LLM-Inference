"""Regenerate Table 3 and every derived figure, from the traces, in one pass.

§6a.4 / §0g: the current draft's §5.2 prose and Table 3 were generated from
different runs and disagree — the text claims DACI leads FM by 16% where the
table's own numbers give 10.7%. That happened because derived percentages were
hand-transcribed. Everything this script prints is computed from the same
traces in the same pass, so prose and table cannot drift apart again.

Reads:
  <before_root>/outputs/BEFORE30*  — pre-fix runs (pristine simulator)
  <after_root>/outputs/AFTER30*    — post-fix runs (four correctness fixes)

Emits: markdown to stdout, plus --csv and --tex artifacts.

Mechanism counters (§5b.4): the per-window traces carry `b`, `a`, `accepted`,
so `a`-changed rate, acceptance rate and #reconfigs are all recoverable. Pool
size and permutations-enumerated are NOT — `control/mechanism.py` post-dates
these runs — and are reported as unavailable rather than silently omitted.
The `a`-changed rate is the one that answers "have RT and FM gone inert
again", which is what §5b.4 exists to detect.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from typing import Dict, List, Optional

SCHEMES = ["SDA", "RT", "FM", "DACI"]

# Paper Table 3 as printed, Qwen3-14B row (recorded in plan §0g). The other two
# model rows are not quoted in the plan, so they are absent rather than guessed.
PAPER_TABLE3 = {
    "qwen3-14b": {
        "SDA":  {"TTLT": 425.16},
        "RT":   {"TTLT": 488.17, "Ovhd": 90.29},
        "FM":   {"TTLT": 416.50},
        "DACI": {"TTLT": 371.89, "Ovhd": 7.69},
    },
}
# The §5.2 prose figures, as written, for comparison against regenerated ones.
PAPER_PROSE = {
    "qwen3-14b": {"DACI_TTLT": 367.8, "FM_TTLT": 437.5, "RT_TTLT": 488.2,
                  "lead_vs_FM_pct": 16.0, "lead_vs_RT_pct": 25.0,
                  "RT_Ovhd": 85.3, "DACI_Ovhd": 8.2},
}

MODELS = ["qwen3-14b", "gemma3-4b", "llama-3.2-8b"]


def _summary(root: str, run_id: str) -> Optional[Dict[str, Dict[str, float]]]:
    p = os.path.join(root, "outputs", run_id, "summary.csv")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["scheme"]] = {k: float(v) for k, v in row.items() if k != "scheme"}
    return out


def _mechanism(root: str, run_id: str) -> Dict[str, Dict[str, float]]:
    """a-changed and acceptance rates, from the per-window traces."""
    out: Dict[str, Dict[str, float]] = {}
    tdir = os.path.join(root, "outputs", run_id, "traces")
    for scheme in SCHEMES:
        a_changed = accepted = windows = 0
        traces = sorted(glob.glob(os.path.join(tdir, f"{scheme}_seed*.jsonl")))
        n_traces_with_a_change = 0
        for f in traces:
            prev_a = None
            this_trace_changed = False
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if "header" in rec:
                        continue
                    windows += 1
                    if rec.get("accepted"):
                        accepted += 1
                    a = rec.get("a")
                    if a is not None:
                        if prev_a is not None and list(a) != list(prev_a):
                            a_changed += 1
                            this_trace_changed = True
                        prev_a = a
            if this_trace_changed:
                n_traces_with_a_change += 1
        if windows:
            out[scheme] = {
                "windows": windows,
                "accept_rate_pct": 100.0 * accepted / windows,
                "a_changed_rate_pct": 100.0 * a_changed / windows,
                "a_changes_total": a_changed,
                "traces_with_a_change": n_traces_with_a_change,
                "n_traces": len(traces),
            }
    return out


def _lead(d: Dict[str, Dict[str, float]], other: str) -> float:
    """DACI's TTLT advantage over `other`, in percent."""
    return 100.0 * (d[other]["TTLT_mean_s"] - d["DACI"]["TTLT_mean_s"]) / d[other]["TTLT_mean_s"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-root", default="../daci-sim-orig")
    ap.add_argument("--after-root", default=".")
    ap.add_argument("--csv", default="results/m5a_fixes/table3_regenerated.csv")
    ap.add_argument("--tex", default="results/m5a_fixes/table3_regenerated.tex")
    args = ap.parse_args()

    rows = []
    print("# Table 3, regenerated — paper vs BEFORE vs AFTER\n")
    print("All figures computed from the traces in one pass (§6a.4). "
          "30 seeds (42–71) per cell.\n")

    for model in MODELS:
        suffix = "" if model == "qwen3-14b" else f"_{model}"
        before = _summary(args.before_root, f"BEFORE30{suffix}")
        after = _summary(args.after_root, f"AFTER30{suffix}")
        if not before or not after:
            print(f"## {model}\n\n_(missing runs: "
                  f"before={'ok' if before else 'MISSING'} "
                  f"after={'ok' if after else 'MISSING'})_\n")
            continue
        mech_b = _mechanism(args.before_root, f"BEFORE30{suffix}")
        mech_a = _mechanism(args.after_root, f"AFTER30{suffix}")
        paper = PAPER_TABLE3.get(model, {})

        print(f"## {model}\n")
        print("| scheme | paper TTLT | BEFORE TTLT | AFTER TTLT | "
              "P99 TPOT b→a | Ovhd b→a | #Rec b→a | accept% b→a | "
              "**a-changed% b→a** |")
        print("|---|---|---|---|---|---|---|---|---|")
        for s in SCHEMES:
            b, a = before[s], after[s]
            mb, ma = mech_b.get(s, {}), mech_a.get(s, {})
            pv = paper.get(s, {}).get("TTLT")
            print(f"| {s} | {pv if pv is not None else '—'} "
                  f"| {b['TTLT_mean_s']:.1f}±{b['TTLT_std_s']:.1f} "
                  f"| {a['TTLT_mean_s']:.1f}±{a['TTLT_std_s']:.1f} "
                  f"| {b['P99_TPOT_mean_ms']:.1f}→{a['P99_TPOT_mean_ms']:.1f} "
                  f"| {b['Ovhd_mean_s']:.2f}→{a['Ovhd_mean_s']:.2f} "
                  f"| {b['Nreconf_mean']:.2f}→{a['Nreconf_mean']:.2f} "
                  f"| {mb.get('accept_rate_pct', float('nan')):.2f}→"
                  f"{ma.get('accept_rate_pct', float('nan')):.2f} "
                  f"| {mb.get('a_changed_rate_pct', float('nan')):.3f}→"
                  f"**{ma.get('a_changed_rate_pct', float('nan')):.3f}** |")
            rows.append({
                "model": model, "scheme": s, "paper_TTLT": pv,
                "before_TTLT": b["TTLT_mean_s"], "after_TTLT": a["TTLT_mean_s"],
                "before_TTLT_std": b["TTLT_std_s"], "after_TTLT_std": a["TTLT_std_s"],
                "before_P99TPOT": b["P99_TPOT_mean_ms"], "after_P99TPOT": a["P99_TPOT_mean_ms"],
                "before_Ovhd": b["Ovhd_mean_s"], "after_Ovhd": a["Ovhd_mean_s"],
                "before_Nrec": b["Nreconf_mean"], "after_Nrec": a["Nreconf_mean"],
                "before_accept_pct": mb.get("accept_rate_pct"),
                "after_accept_pct": ma.get("accept_rate_pct"),
                "before_a_changed_pct": mb.get("a_changed_rate_pct"),
                "after_a_changed_pct": ma.get("a_changed_rate_pct"),
                "after_a_changes_total": ma.get("a_changes_total"),
                "after_traces_with_a_change": ma.get("traces_with_a_change"),
                "n_traces": ma.get("n_traces"),
            })

        print()
        print("**DACI's TTLT lead, regenerated:**\n")
        print("| vs | BEFORE | AFTER |")
        print("|---|---|---|")
        for other in ("SDA", "RT", "FM"):
            print(f"| {other} | {_lead(before, other):+.2f}% | {_lead(after, other):+.2f}% |")
        print()

        prose = PAPER_PROSE.get(model)
        if prose:
            print("**§5.2 prose vs regenerated (§0g):**\n")
            print("| quantity | prose says | Table 3 as printed | AFTER (regenerated) |")
            print("|---|---|---|---|")
            print(f"| DACI TTLT | {prose['DACI_TTLT']} | {paper['DACI']['TTLT']} "
                  f"| {after['DACI']['TTLT_mean_s']:.2f} |")
            print(f"| FM TTLT | {prose['FM_TTLT']} | {paper['FM']['TTLT']} "
                  f"| {after['FM']['TTLT_mean_s']:.2f} |")
            print(f"| RT TTLT | {prose['RT_TTLT']} | {paper['RT']['TTLT']} "
                  f"| {after['RT']['TTLT_mean_s']:.2f} |")
            print(f"| DACI lead vs FM | {prose['lead_vs_FM_pct']:.1f}% | "
                  f"{100*(paper['FM']['TTLT']-paper['DACI']['TTLT'])/paper['FM']['TTLT']:.1f}% "
                  f"| **{_lead(after, 'FM'):.2f}%** |")
            print(f"| DACI lead vs RT | {prose['lead_vs_RT_pct']:.1f}% | "
                  f"{100*(paper['RT']['TTLT']-paper['DACI']['TTLT'])/paper['RT']['TTLT']:.1f}% "
                  f"| **{_lead(after, 'RT'):.2f}%** |")
            print(f"| RT overhead | {prose['RT_Ovhd']} s | {paper['RT']['Ovhd']} s "
                  f"| {after['RT']['Ovhd_mean_s']:.2f} s |")
            print(f"| DACI overhead | {prose['DACI_Ovhd']} s | {paper['DACI']['Ovhd']} s "
                  f"| {after['DACI']['Ovhd_mean_s']:.2f} s |")
            print()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(args.tex, "w", encoding="utf-8") as fh:
        fh.write("% Generated by analyze_30trace.py -- do not edit by hand (§0g).\n")
        fh.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        fh.write("Model & Scheme & TTLT (s) & P99 TPOT (ms) & Ovhd (s) & "
                 "\\#Rec & $a$-chg (\\%) \\\\\n\\midrule\n")
        for r in rows:
            fh.write(f"{r['model']} & {r['scheme']} & "
                     f"{r['after_TTLT']:.2f} $\\pm$ {r['after_TTLT_std']:.2f} & "
                     f"{r['after_P99TPOT']:.1f} & {r['after_Ovhd']:.2f} & "
                     f"{r['after_Nrec']:.2f} & "
                     f"{(r['after_a_changed_pct'] or 0):.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nwrote {args.csv}\nwrote {args.tex}")

    print("\n## Counter availability (§5b.4)\n")
    print("| counter | available | why |")
    print("|---|---|---|")
    print("| #reconfigs | yes | summary.csv |")
    print("| acceptance rate | yes | per-window `accepted` |")
    print("| **a-changed rate** | **yes** | per-window `a`; this is the "
          "\"has the baseline gone inert\" test |")
    print("| pool size | **no** | `control/mechanism.py` post-dates these runs |")
    print("| placements enumerated | **no** | same |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
