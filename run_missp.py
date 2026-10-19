"""
run_missp.py — misspecification study for Article 3.
Data come from model I, II or III (random numbers of units / sub-networks), but K is
estimated with the simpler fixed model F (N = M = 3), as an engineer would do when the
randomness of the structure is ignored. Exact benchmark, R = 20 000, n in {33, 100, 300}.
Usage: python run_missp.py
"""
import json
import time

import numpy as np

import art3_core as C
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)

SEED = 2026
out = {}
t0 = time.time()
for nn in (33, 100, 300):
    for true in ("I", "II", "III"):
        for net in C.NETS:
            rng = np.random.default_rng([SEED, 11, nn, C.MODELS.index(true), C.NETS.index(net)])
            X = C.sim_exact(rng, 20_000, nn, true, net)
            for fit in ("F", true):
                for s in C.SCHEMES:
                    r = C.summary(C.mle(X, fit, net, s, "a", chunk=2000), fit, "a")
                    out[f"{nn}|{true}|{fit}|{net}|{s}"] = r
                    print(f"{time.time()-t0:6.0f}s n={nn} true={true:3s} fit={fit:3s} {net} {s:8s} "
                          f"mean={r['K_mean']:.2f} %ex={r['exact']:.1f} MSE={r['K_mse']:.3f}", flush=True)
json.dump(out, open(RES / "results_missp.json", "w"), indent=1)
print("done missp")
