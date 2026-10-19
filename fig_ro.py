"""fig_ro.py — Romanian versions of the two article figures."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
from pathlib import Path
RES=Path(__file__).resolve().parent/"results"; FIG=RES/"figures"; FIG.mkdir(parents=True,exist_ok=True)
N=json.load(open(RES/"results_nsize.json")); MS=json.load(open(RES/"results_missp.json"))
LAB={"F":"N, M fixe","I":"Modelul I","II":"Modelul II","III":"Modelul III"}
col={"F":"#444444","I":"#1f77b4","II":"#d62728","III":"#2ca02c"}
ns=(10,33,100,300)
fig,ax=plt.subplots(1,2,figsize=(10,3.9),sharey=True)
for k,net in enumerate(("SP","PS")):
    for m in ("F","I","II","III"):
        ax[k].plot(ns,[N[f"{n}|{m}|{net}|a|complete"]["exact"] for n in ns],"-o",color=col[m],label=f"{LAB[m]}, λ, μ cunoscuți")
        if m!="F":
            ax[k].plot(ns,[N[f"{n}|{m}|{net}|b|complete"]["exact"] for n in ns],"--s",color=col[m],mfc="white",label=f"{LAB[m]}, λ, μ estimați")
    ax[k].set_xscale("log"); ax[k].set_xticks(ns); ax[k].set_xticklabels(ns); ax[k].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[k].set_title("Rețeaua "+("S-P" if net=="SP" else "P-S")); ax[k].set_xlabel("volumul eșantionului n"); ax[k].grid(alpha=.3)
ax[0].set_ylabel("P(K̂ = 9), %"); ax[1].legend(fontsize=7,ncol=1,loc="lower right")
fig.tight_layout(); fig.savefig(FIG/"Figura1_exact_vs_n_ro.png",dpi=300)
fig,ax=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
for k,net in enumerate(("SP","PS")):
    for t in ("I","II","III"):
        ax[k].plot(ns[1:],[MS[f"{n}|{t}|F|{net}|complete"]["K_mean"] for n in ns[1:]],"--o",color=col[t],label=f"date din {LAB[t]}, estimare cu modelul fix")
        ax[k].plot(ns[1:],[MS[f"{n}|{t}|{t}|{net}|complete"]["K_mean"] for n in ns[1:]],"-s",color=col[t],label=f"date din {LAB[t]}, modelul corect")
    ax[k].axhline(9,color="k",lw=.8); ax[k].set_xscale("log"); ax[k].set_xticks(ns[1:]); ax[k].set_xticklabels(ns[1:]); ax[k].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[k].set_title("Rețeaua "+("S-P" if net=="SP" else "P-S")); ax[k].set_xlabel("volumul eșantionului n"); ax[k].grid(alpha=.3)
ax[0].set_ylabel("media lui K̂"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG/"Figura2_model_gresit_ro.png",dpi=300)
