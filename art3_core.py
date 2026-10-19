"""
art3_core.py
Maximum-likelihood estimation of K (and of the ZTP parameters lambda, mu) for S-P / P-S
networks whose unit lifetimes are i.i.d. discrete uniform on {0, 1, ..., K}.

Models (distributions taken from Article 2, Eqs. 14a-14h; A(t) = e^t - 1):
    F   : N = 3, M = 3 fixed                 (reference case, as in the JES article)
    I   : M = 3 fixed,  N_k ~ ZTP(lambda)
    II  : N = 3 fixed,  M ~ ZTP(mu)
    III : M ~ ZTP(mu),  N_k ~ ZTP(lambda)
S-P: U = min over sub-networks of max over units;  P-S: V = max of min.

Estimation scenarios:
    'a' : K only, lambda and mu known
    'b' : K together with the ZTP parameters of the model (lambda for I, mu for II,
          lambda and mu for III); profile likelihood over K, grid over the nuisance
          parameters.
Data types: complete, interval-censored C3 = [0,4),[4,7),[7,inf) and
C5 = [0,5),[5,7),[7,8),[8,9),[9,inf) (same schemes as in the JES article).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

K_TRUE, LAM, MU = 9, 2.5, 2.5
N_FIX, M_FIX = 3, 3
K_MAX = 30
MODELS = ("F", "I", "II", "III")
NETS = ("SP", "PS")
C3 = [0, 4, 7]
C5 = [0, 5, 7, 8, 9]
SCHEMES = {"complete": None, "C3": C3, "C5": C5}

# grids for the nuisance parameters (log-spaced)
GRID_1D = np.exp(np.linspace(math.log(0.02), math.log(50.0), 600))
GRID_2D = np.exp(np.linspace(math.log(0.02), math.log(50.0), 140))


def _A(t):
    return np.expm1(t)


def _G(lam, s):
    """P(min over N ~ ZTP(lam) of ... ) building block: E[s^N] for N ~ ZTP(lam), s in [0,1]."""
    return _A(lam * s) / _A(lam)


def cdf(x, K, model, net, lam=LAM, mu=MU, N=N_FIX, M=M_FIX):
    """CDF of the network lifetime at integer x; lam, mu may be arrays (broadcast)."""
    x = np.asarray(x, dtype=float)
    F = np.clip((x + 1) / (K + 1), 0.0, 1.0)
    Q = 1.0 - F
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if model == "F":
        out = 1 - (1 - F ** N) ** M if net == "SP" else (1 - Q ** N) ** M
    elif model == "I":
        out = 1 - (1 - _G(lam, F)) ** M if net == "SP" else (1 - _G(lam, Q)) ** M
    elif model == "II":
        out = 1 - _G(mu, 1 - F ** N) if net == "SP" else _G(mu, 1 - Q ** N)
    elif model == "III":
        out = 1 - _G(mu, 1 - _G(lam, F)) if net == "SP" else _G(mu, 1 - _G(lam, Q))
    else:
        raise ValueError(model)
    out = np.where(x < 0, 0.0, np.where(x >= K, 1.0, out))
    return np.clip(out, 0.0, 1.0)


def pmf(K, model, net, lam=LAM, mu=MU, support=K_TRUE + 1):
    x = np.arange(support)
    return cdf(x, K, model, net, lam, mu) - cdf(x - 1, K, model, net, lam, mu)


def theory(K=K_TRUE, model="F", net="SP", lam=LAM, mu=MU):
    p = pmf(K, model, net, lam, mu, support=K + 1)
    x = np.arange(K + 1)
    m = float((x * p).sum())
    return dict(p=p, mean=m, var=float(((x - m) ** 2 * p).sum()))


# ------------------------------------------------------------------ cell probabilities
def cells(K, model, net, scheme, lam=LAM, mu=MU):
    """Cell probabilities for the given data type; trailing axis = cells.
    Complete data: cells x = 0..9 (the true K = 9 makes larger values impossible,
    so only these cells can carry observations)."""
    lam = np.asarray(lam, float)[..., None]
    mu = np.asarray(mu, float)[..., None]
    if scheme is None:
        x = np.arange(K_TRUE + 1)
        P = cdf(x, K, model, net, lam, mu) - cdf(x - 1, K, model, net, lam, mu)
    else:
        a = np.array(scheme, float)
        b = np.r_[a[1:], np.inf]
        Fa = cdf(a - 1, K, model, net, lam, mu)
        Fb = np.where(np.isinf(b), 1.0, cdf(np.where(np.isinf(b), 0, b) - 1, K, model, net, lam, mu))
        P = Fb - Fa
    shape = np.broadcast_shapes(lam.shape, mu.shape)[:-1] + (P.shape[-1],)
    return np.broadcast_to(P, shape)


def counts_of(X, scheme):
    X = np.asarray(X)
    if scheme is None:
        return np.stack([(X == v).sum(axis=1) for v in range(K_TRUE + 1)], axis=1)
    edges = list(scheme) + [np.inf]
    return np.stack([((X >= a) & (X < b)).sum(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)


def _loglik(cnt, P):
    """cnt (R, c), P (G, c) -> LL (R, G); cells with n_j = 0 contribute p^0 = 1."""
    with np.errstate(divide="ignore"):
        logP = np.where(P > 0, np.log(np.where(P > 0, P, 1.0)), 0.0)
    LL = cnt @ logP.T
    bad = (cnt[:, None, :] > 0) & (P[None, :, :] <= 0)
    LL[bad.any(axis=2)] = -np.inf
    return LL


def nuisance_grid(model):
    """Returns (lam_values, mu_values) of the grid used in scenario 'b'."""
    if model == "I":
        return GRID_1D, np.full_like(GRID_1D, MU)
    if model == "II":
        return np.full_like(GRID_1D, LAM), GRID_1D
    if model == "III":
        L, Mu = np.meshgrid(GRID_2D, GRID_2D, indexing="ij")
        return L.ravel(), Mu.ravel()
    return np.array([LAM]), np.array([MU])


_CACHE = {}


def _cell_table(model, net, scheme_name, scenario):
    key = (model, net, scheme_name, scenario)
    if key not in _CACHE:
        scheme = SCHEMES[scheme_name]
        if scenario == "a" or model == "F":
            lam, mu = np.array([LAM]), np.array([MU])
        else:
            lam, mu = nuisance_grid(model)
        T = np.stack([cells(K, model, net, scheme, lam, mu) for K in range(1, K_MAX + 1)])  # (nK, G, c)
        _CACHE[key] = (T, lam, mu)
    return _CACHE[key]


def mle(X, model, net, scheme_name, scenario="a", chunk=500):
    """Profile MLE. Returns dict with arrays K_hat, lam_hat, mu_hat (length R).
    Ties: smallest K, then first grid point."""
    T, lam, mu = _cell_table(model, net, scheme_name, scenario)
    cnt = counts_of(X, SCHEMES[scheme_name]).astype(float)
    R = cnt.shape[0]
    Kh = np.empty(R, int)
    gh = np.empty(R, int)
    for s in range(0, R, chunk):
        c = cnt[s:s + chunk]
        best = np.full(c.shape[0], -np.inf)
        bk = np.zeros(c.shape[0], int)
        bg = np.zeros(c.shape[0], int)
        for i in range(T.shape[0]):
            LL = _loglik(c, T[i])
            j = LL.argmax(axis=1)
            v = LL[np.arange(len(j)), j]
            upd = v > best + 1e-12
            best[upd], bk[upd], bg[upd] = v[upd], i + 1, j[upd]
        Kh[s:s + chunk], gh[s:s + chunk] = bk, bg
    return dict(K=Kh, lam=lam[gh], mu=mu[gh])


def summary(est, model, scenario):
    K = est["K"].astype(float)
    out = dict(K_mean=K.mean(), K_var=K.var(), K_mse=((K - K_TRUE) ** 2).mean(),
               exact=100 * (K == K_TRUE).mean(), under=100 * (K < K_TRUE).mean(),
               over=100 * (K > K_TRUE).mean())
    if scenario == "b":
        for p, true, est_on in (("lam", LAM, model in ("I", "III")), ("mu", MU, model in ("II", "III"))):
            if est_on:
                v = est[p]
                out[f"{p}_median"] = float(np.median(v))
                out[f"{p}_mean"] = float(v.mean())
                out[f"{p}_mse"] = float(((v - true) ** 2).mean())
                out[f"{p}_mae"] = float(np.median(np.abs(v - true)))
                out[f"{p}_bound"] = float(100 * ((v <= GRID_1D[0] * 1.001) | (v >= GRID_1D[-1] * 0.999)).mean())
    return {k: float(v) for k, v in out.items()}


# ------------------------------------------------------------------ simulation
DATA = Path(__file__).resolve().parent / "data"   # same digit files as the JES and Article 2 repositories
FILES = {
    "Java ThreadLocalRandom": [DATA / "java_threadlocal" / f"threadlocal_data{s}.csv" for s in ["1", "2", "3", "4", "5", ""]],
    "Python secrets": [DATA / "python_secrets" / f"secrets_data{s}.csv" for s in ["1", "2", "3", "4", "5", ""]],
    "Digits of π": [DATA / "pi" / f"pi_digits_part{s}.csv" for s in ["", "1", "2", "3", "4", "5"]],
}


def load_digits(files):
    parts = []
    for f in files:
        raw = np.frombuffer(Path(f).read_bytes(), dtype=np.uint8)
        parts.append(raw[(raw >= 48) & (raw <= 57)] - 48)
    return np.concatenate(parts).astype(np.int64)


def sim_exact(rng, R, n, model, net):
    return rng.choice(K_TRUE + 1, size=(R, n), p=pmf(K_TRUE, model, net))


def sim_inverse(d, R, n, model, net, offset=0):
    """9 consecutive digits -> u in [0,1); lifetime = min{k : F(k) >= u} (JES, method 1)."""
    need = R * n * 9
    blk = d[offset:offset + need].reshape(R * n, 9)
    u = blk @ (10 ** np.arange(8, -1, -1, dtype=np.int64)) / 1e9
    F = cdf(np.arange(K_TRUE + 1), K_TRUE, model, net)
    return np.searchsorted(F, u, side="left").reshape(R, n), need


class Stream:
    def __init__(s, d):
        s.d, s.pos = d, 0

    def take(s, k):
        if s.pos + k > len(s.d):
            raise StopIteration
        out = s.d[s.pos:s.pos + k]
        s.pos += k
        return out

    def unif(s):
        b = s.take(5)
        return (b[0] * 10000 + b[1] * 1000 + b[2] * 100 + b[3] * 10 + b[4]) / 100000


def ztp(s, lam):
    """Knuth's Poisson algorithm with 5-digit uniforms, zero rejected (Article 2)."""
    L = math.exp(-lam)
    while True:
        k, p = 0, 1.0
        while True:
            k += 1
            p *= s.unif()
            if p <= L:
                break
        if k - 1 > 0:
            return k - 1


def sim_structural(d, R, n, model):
    """Builds each network from the digit stream (unit lifetime = one digit, K = 9),
    exactly as in Article 2 (05_modele_I_II_III.py, interleaved order). Returns (U, V)."""
    s = Stream(d.tolist())
    U = np.empty(R * n, int)
    V = np.empty(R * n, int)
    for r in range(R * n):
        M = M_FIX if model in ("F", "I") else ztp(s, MU)
        tmax, tmin = [], []
        for _ in range(M):
            N = N_FIX if model in ("F", "II") else ztp(s, LAM)
            x = s.take(N)
            tmax.append(max(x))
            tmin.append(min(x))
        U[r], V[r] = min(tmax), max(tmin)
    return U.reshape(R, n), V.reshape(R, n), s.pos
