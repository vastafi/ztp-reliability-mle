"""
run_art3.py  —  experiments for Article 3 (MLE across models F, I, II, III).

stage 'bench' : exact benchmark (NumPy PCG64), n = 33
                scenario a: R = 200 000; scenario b: R = 20 000
stage 'prng'  : the three digit sources, R = 1000, n = 33, methods inverse + structural
stage 'nsize' : benchmark vs sample size n in {10, 33, 100, 300}, R = 5000,
                complete data and C5, both scenarios
Usage: python run_art3.py bench|prng|nsize
"""
import json
import sys
import time

import numpy as np

import art3_core as C
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)

SEED = 2026
R, n = 1000, 33


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def scenarios(model):
    return ("a",) if model == "F" else ("a", "b")


def band(p, R_):
    se = np.sqrt(p * (1 - p) / R_)
    return [100 * (p - 1.96 * se), 100 * (p + 1.96 * se)]


stage = sys.argv[1]
out = {}
t0 = time.time()

if stage == "bench":
    for model in C.MODELS:
        for net in C.NETS:
            for sc in scenarios(model):
                Rb = 200_000 if sc == "a" else 20_000
                rng = np.random.default_rng([SEED, C.MODELS.index(model), C.NETS.index(net)])
                X = C.sim_exact(rng, Rb, n, model, net)
                for s in C.SCHEMES:
                    e = C.mle(X, model, net, s, sc, chunk=2000 if sc == "a" else 500)
                    r = C.summary(e, model, sc)
                    r["band_exact_R1000"] = band(r["exact"] / 100, R)
                    r["R"] = Rb
                    out[f"{model}|{net}|{sc}|{s}"] = r
                    print(f"{time.time()-t0:7.0f}s {model:3s} {net} {sc} {s:8s} %ex={r['exact']:.2f} MSE={r['K_mse']:.4f}", flush=True)

elif stage == "prng":
    for g, files in C.FILES.items():
        d = C.load_digits(files)
        for model in C.MODELS:
            data = {}
            for net in C.NETS:
                data[("inverse", net)], _ = C.sim_inverse(d, R, n, model, net)
            U, V, used = C.sim_structural(d, R, n, model)
            data[("structural", "SP")], data[("structural", "PS")] = U, V
            out[f"{g}|{model}|digits_structural"] = used
            for (meth, net), X in data.items():
                for sc in scenarios(model):
                    for s in C.SCHEMES:
                        e = C.mle(X, model, net, s, sc)
                        r = C.summary(e, model, sc)
                        r["sq_err_K"] = ((e["K"] - C.K_TRUE) ** 2).tolist()
                        out[f"{g}|{model}|{meth}|{net}|{sc}|{s}"] = r
                        print(f"{time.time()-t0:7.0f}s {g[:6]} {model:3s} {meth[:5]} {net} {sc} {s:8s} %ex={r['exact']:.1f} MSE={r['K_mse']:.3f}", flush=True)

elif stage == "nsize":
    for nn in (10, 33, 100, 300):
        for model in C.MODELS:
            for net in C.NETS:
                rng = np.random.default_rng([SEED, 7, nn, C.MODELS.index(model), C.NETS.index(net)])
                X = C.sim_exact(rng, 5000, nn, model, net)
                for sc in scenarios(model):
                    for s in ("complete", "C5"):
                        r = C.summary(C.mle(X, model, net, s, sc), model, sc)
                        out[f"{nn}|{model}|{net}|{sc}|{s}"] = r
                        print(f"{time.time()-t0:7.0f}s n={nn} {model:3s} {net} {sc} {s:8s} %ex={r['exact']:.1f} MSE={r['K_mse']:.3f}", flush=True)

json.dump(to_py(out), open(RES / f"results_{stage}.json", "w"), indent=1)
print("done", stage, f"{time.time()-t0:.0f}s")
