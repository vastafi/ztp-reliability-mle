import numpy as np, art3_core as C, json
from pathlib import Path
RES = Path(__file__).resolve().parent / 'results'; RES.mkdir(exist_ok=True)
rng=np.random.default_rng(5)
out={}
print("--- decomposition: P(Khat=K | max=K), P(Khat=K, max<K)")
for net in C.NETS:
  for m in C.MODELS:
    X=C.sim_exact(rng,40000,33,m,net); e=C.mle(X,m,net,'complete','a',chunk=4000)['K']
    mx=X.max(1)==9; pK=C.pmf(9,m,net)[9]
    r=dict(pK=pK, bound=1-(1-pK)**33, ex=(e==9).mean(), ex_given_max=(e[mx]==9).mean(), ex_and_less=((e==9)&~mx).mean(), under=(e<9).mean(), over=(e>9).mean())
    out[f'{m}|{net}']=r
    print(m,net,{k:round(float(v),4) for k,v in r.items()})
json.dump({k:{a:float(b) for a,b in v.items()} for k,v in out.items()},open(RES / 'results_decomp.json','w'),indent=1)
