"""
make_tables.py — builds the tables (Markdown + CSV) and figures of Article 3 from the JSON results.
Run after: run_art3.py bench | prng | nsize, run_missp.py, identifiability.py
Output: tables/*.md, tables/*.csv, figures/*.png
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import art3_core as C

RES = Path(__file__).resolve().parent / "results"
T = RES / "tables"; T.mkdir(parents=True, exist_ok=True)
FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)
B = json.load(open(RES / "results_bench.json"))
P = json.load(open(RES / "results_prng.json"))
N = json.load(open(RES / "results_nsize.json"))
MS = json.load(open(RES / "results_missp.json"))
SCH = ("complete", "C3", "C5")
LAB = {"F": "N, M fixed", "I": "Model I", "II": "Model II", "III": "Model III"}
GEN = list(C.FILES)


def write(name, header, rows, title):
    with open(T / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(T / f"{name}.md", "w") as f:
        f.write(f"### {title}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")


f2 = lambda x: f"{x:.3f}"
f1 = lambda x: f"{x:.1f}"

# ---- Table 1: benchmark, scenario a (lambda, mu known)
rows = []
for net in C.NETS:
    for m in C.MODELS:
        r = [LAB[m], net]
        for s in SCH:
            v = B[f"{m}|{net}|a|{s}"]
            r += [f1(v["exact"]), f2(v["K_mse"]), f2(v["K_mean"] - 9)]
        rows.append(r)
write("table1_benchmark_known", ["Model", "Net"] + [f"{s} {k}" for s in SCH for k in ("%ex", "MSE", "bias")], rows,
      "Exact benchmark, n = 33, lambda and mu known (R = 200 000)")

# ---- Table 2: scenario b vs a (cost of estimating the ZTP parameters)
rows = []
for net in C.NETS:
    for m in ("I", "II", "III"):
        for s in SCH:
            a, b = B[f"{m}|{net}|a|{s}"], B[f"{m}|{net}|b|{s}"]
            r = [LAB[m], net, s, f1(a["exact"]), f1(b["exact"]), f2(a["K_mse"]), f2(b["K_mse"]),
                 f"{b['K_mse'] / a['K_mse']:.2f}" if a["K_mse"] > 0 else "-"]
            for p in ("lam", "mu"):
                r += [f"{b[p + '_median']:.2f} / {b[p + '_mae']:.2f}" if f"{p}_median" in b else "-"]
            rows.append(r)
write("table2_known_vs_estimated",
      ["Model", "Net", "Data", "%ex (a)", "%ex (b)", "MSE (a)", "MSE (b)", "MSE ratio b/a",
       "lambda: median / MAD", "mu: median / MAD"], rows,
      "Cost of estimating lambda, mu together with K (benchmark, n = 33; a: R = 200 000, b: R = 20 000)")

# ---- Table 3: PRNG sources, scenario a, %ex and MSE, flag outside the 95% band
rows = []
outside = 0; total = 0
for m in C.MODELS:
    for net in C.NETS:
        for s in SCH:
            lo, hi = B[f"{m}|{net}|a|{s}"]["band_exact_R1000"]
            r = [LAB[m], net, s, f1(B[f"{m}|{net}|a|{s}"]["exact"]), f"{lo:.1f}-{hi:.1f}"]
            for meth in ("inverse", "structural"):
                for g in GEN:
                    v = P[f"{g}|{m}|{meth}|{net}|a|{s}"]["exact"]
                    ok = lo <= v <= hi
                    outside += not ok; total += 1
                    r.append(f1(v) + ("" if ok else "†"))
            rows.append(r)
write("table3_prng_known", ["Model", "Net", "Data", "Benchmark %ex", "95% band"] +
      [f"{meth[:3]} {g.split()[0]}" for meth in ("inverse", "structural") for g in GEN], rows,
      f"P(K̂ = 9), %, for the three sources (R = 1000, n = 33, scenario a); † outside the band: {outside} of {total}")

# ---- Table 4: misspecification
rows = []
for nn in (33, 100, 300):
    for t in ("I", "II", "III"):
        for net in C.NETS:
            ff = MS[f"{nn}|{t}|F|{net}|complete"]; tt = MS[f"{nn}|{t}|{t}|{net}|complete"]
            rows.append([nn, LAB[t], net, f"{ff['K_mean']:.2f}", f1(ff["exact"]), f2(ff["K_mse"]),
                         f"{tt['K_mean']:.2f}", f1(tt["exact"]), f2(tt["K_mse"])])
write("table4_misspecification", ["n", "True model", "Net", "fixed-model fit: mean K̂", "%ex", "MSE",
                                  "correct fit: mean K̂", "%ex", "MSE"], rows,
      "Fitting the fixed model (N = M = 3) to data from models with random structure (complete data, R = 20 000)")

# ---- Table 5: accuracy explained by p_K = P(X = K)
rows = []
for net in C.NETS:
    for m in C.MODELS:
        pK = C.pmf(9, m, net)[9]
        rows.append([LAB[m], net, f"{pK:.4f}", f1(100 * (1 - (1 - pK) ** 33)), f1(B[f"{m}|{net}|a|complete"]["exact"])])
write("table5_pK", ["Model", "Net", "p_K = P(X = 9)", "P(max = K) = 1-(1-p_K)^33, %", "benchmark %ex"], rows,
      "P(max = K) = 1 - (1 - p_K)^n versus the benchmark P(K̂ = K) (complete data, n = 33)")

# ---- Figure: %ex vs n
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
ns = (10, 33, 100, 300)
col = {"F": "#444444", "I": "#1f77b4", "II": "#d62728", "III": "#2ca02c"}
for k, net in enumerate(C.NETS):
    for m in C.MODELS:
        ax[k].plot(ns, [N[f"{n}|{m}|{net}|a|complete"]["exact"] for n in ns], "-o", color=col[m], label=f"{LAB[m]} (a)")
        if m != "F":
            ax[k].plot(ns, [N[f"{n}|{m}|{net}|b|complete"]["exact"] for n in ns], "--s", color=col[m], mfc="white",
                       label=f"{LAB[m]} (b)")
    ax[k].set_xscale("log"); ax[k].set_xticks(ns); ax[k].set_xticklabels(ns)
    ax[k].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[k].set_title(f"{net} network"); ax[k].set_xlabel("sample size n"); ax[k].grid(alpha=.3)
ax[0].set_ylabel("P(K̂ = 9), %")
ax[1].legend(fontsize=7, ncol=2, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig_exact_vs_n.png", dpi=300)

# ---- Figure: misspecification
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for k, net in enumerate(C.NETS):
    for j, t in enumerate(("I", "II", "III")):
        ax[k].plot(ns[1:], [MS[f"{n}|{t}|F|{net}|complete"]["K_mean"] for n in ns[1:]], "--o", color=col[t],
                   label=f"{LAB[t]} data, fixed-model fit")
        ax[k].plot(ns[1:], [MS[f"{n}|{t}|{t}|{net}|complete"]["K_mean"] for n in ns[1:]], "-s", color=col[t],
                   label=f"{LAB[t]} data, correct model")
    ax[k].axhline(9, color="k", lw=.8); ax[k].set_xscale("log"); ax[k].set_xticks(ns[1:]); ax[k].set_xticklabels(ns[1:])
    ax[k].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[k].set_title(f"{net} network"); ax[k].set_xlabel("sample size n"); ax[k].grid(alpha=.3)
ax[0].set_ylabel("mean of K̂")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig_misspecification.png", dpi=300)
print("tables and figures written")
