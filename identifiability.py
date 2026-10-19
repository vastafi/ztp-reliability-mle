"""
identifiability.py — identifiability of (K, lambda, mu) for models I, II, III.

1. For every K' != 9, the smallest Kullback-Leibler divergence KL(P_true || P_{K',theta})
   over the ZTP parameters theta. A strictly positive minimum means that a wrong K cannot
   be compensated by changing lambda or mu (K is identifiable).
2. The Fisher information matrix of (lambda, mu) per observation in model III at K = 9:
   correlation of the estimators and the sample size needed for a standard error of 0.1.
Output: results_identifiability.json
"""
import json

import numpy as np
from scipy.optimize import minimize

import art3_core as C
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)

K0 = C.K_TRUE
SUP = 40


def P(K, model, net, lam, mu):
    return C.pmf(K, model, net, lam, mu, support=SUP)


def kl(p, q):
    m = p > 1e-15
    if np.any(q[m] <= 0):
        return np.inf
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


out = {"kl": {}, "fisher_III": {}}
for model in ("I", "II", "III"):
    for net in C.NETS:
        p = P(K0, model, net, C.LAM, C.MU)
        for K in range(K0 + 1, K0 + 4):
            if model == "I":
                f = lambda t: kl(p, P(K, model, net, np.exp(t[0]), C.MU)); x0 = [np.log(C.LAM)]
            elif model == "II":
                f = lambda t: kl(p, P(K, model, net, C.LAM, np.exp(t[0]))); x0 = [np.log(C.MU)]
            else:
                f = lambda t: kl(p, P(K, model, net, np.exp(t[0]), np.exp(t[1]))); x0 = [np.log(C.LAM), np.log(C.MU)]
            best = min((minimize(f, np.array(x0) + d, method="Nelder-Mead",
                                 options={"xatol": 1e-9, "fatol": 1e-15, "maxiter": 5000}) for d in (0, 0.7, -0.7)),
                       key=lambda r: r.fun)
            out["kl"][f"{model}|{net}|{K}"] = dict(KL=best.fun, theta=np.exp(best.x).tolist())
            print(model, net, K, f"KL_min={best.fun:.4f}", np.round(np.exp(best.x), 3))

h = 1e-5
for net in C.NETS:
    p = P(K0, "III", net, C.LAM, C.MU)
    lg = lambda l, m: np.log(np.where(p > 0, P(K0, "III", net, l, m), 1.0))
    dl = (lg(C.LAM + h, C.MU) - lg(C.LAM - h, C.MU)) / (2 * h)
    dm = (lg(C.LAM, C.MU + h) - lg(C.LAM, C.MU - h)) / (2 * h)
    I = np.array([[np.sum(p * dl * dl), np.sum(p * dl * dm)], [np.sum(p * dm * dl), np.sum(p * dm * dm)]])
    Cov = np.linalg.inv(I)
    r = Cov[0, 1] / np.sqrt(Cov[0, 0] * Cov[1, 1])
    out["fisher_III"][net] = dict(I=I.tolist(), corr=r, n_for_se01=np.ceil(np.diag(Cov) / 0.01).tolist())
    print("III", net, f"corr={r:.3f}", "n for SE 0.1 (lam, mu):", np.ceil(np.diag(Cov) / 0.01))

json.dump(out, open(RES / "results_identifiability.json", "w"), indent=1)
