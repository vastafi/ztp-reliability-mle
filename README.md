# MLE of K in S-P / P-S networks with a random structure

Code and results for the third article of the series by V. Astafi and A. Leahu (Technical University of Moldova):
maximum-likelihood estimation of the parameter *K* of the discrete uniform unit lifetime on {0, 1, …, *K*}
in serial-parallel (S-P) and parallel-serial (P-S) networks whose numbers of units and/or sub-networks are
zero-truncated Poisson (ZTP) random variables.


## Models

| Code | Number of sub-networks *M* | Units per sub-network *N* |
|---|---|---|
| `F` | 3 (fixed) | 3 (fixed) — reference case |
| `I` | 3 (fixed) | *N_k* ~ ZTP(λ) |
| `II` | *M* ~ ZTP(μ) | 3 (fixed) |
| `III` | *M* ~ ZTP(μ) | *N_k* ~ ZTP(λ) |

S-P lifetime: U = min over sub-networks of max over units. P-S lifetime: V = max of min.
True values: *K* = 9, λ = μ = 2.5.

## Experiments

| Script | What it does | Output (in `results/`) |
|---|---|---|
| `art3_core.py` | models, cell probabilities, profile MLE, simulation (inverse CDF and structural) | — |
| `identifiability.py` | minimum KL divergence for every wrong *K*; Fisher information of (λ, μ) in model III (Table 2) | `results_identifiability.json` |
| `run_art3.py bench` | exact benchmark with NumPy PCG64, *n* = 33 (scenario a: R = 200 000; b: R = 20 000) (Tables 4, 5) | `results_bench.json` |
| `run_art3.py prng` | Java ThreadLocalRandom, Python secrets, digits of π; R = 1000, *n* = 33; inverse and structural simulation (Section 6.5) | `results_prng.json` |
| `run_art3.py nsize` | benchmark for *n* = 10, 33, 100, 300 (R = 5000) (Figure 1) | `results_nsize.json` |
| `run_missp.py` | misspecification: data from models I–III, *K* estimated with the fixed model F (Table 6, Figure 2) | `results_missp.json` |
| `decomposition.py` | P(K̂ = K) split by the event max = K (Table 3) | `results_decomp.json` |
| `check_kmax.py` | sensitivity of scenario b with censored data to the bound K_max (Section 6.2) | `results_kmax.json` |
| `make_tables.py` | all tables and the English figures | `tables/*.md`, `tables/*.csv`, `figures/*.png` |
| `fig_ro.py` | the two figures of the article, Romanian labels | `figures/Figura*_ro.png` |

## Running

```bash
pip install -r requirements.txt
python identifiability.py
python run_art3.py bench      # ~1 h (scenario b of model III is the slow part)
python run_art3.py prng       # ~30 min
python run_art3.py nsize      # ~40 min
python run_missp.py           # ~1 min
python decomposition.py       # ~1 min
python check_kmax.py          # ~10 min, needs results_bench.json
python make_tables.py
python fig_ro.py
```

All outputs are written to `results/`. The folder is committed, because the article cites these numbers;
`.gitignore` excludes only caches, IDE and OS files and logs.

The `data/` folder holds the same digit files as the two previous repositories
(`java_threadlocal/`, `python_secrets/`, `pi/`, one digit per line).