import numpy as np, json, art3_core as C
from pathlib import Path
RES = Path(__file__).resolve().parent / 'results'
B=json.load(open(RES / 'results_bench.json')); out={}
for kmax in (20,):
  C.K_MAX=kmax; C._CACHE.clear()
  for m in ('I','II','III'):
    for net in C.NETS:
      rng=np.random.default_rng([2026,C.MODELS.index(m),C.NETS.index(net)])
      Rb=20000 if m!='III' else 4000
      X=C.sim_exact(rng,Rb,33,m,net)
      for s in ('C3','C5'):
        r=C.summary(C.mle(X,m,net,s,'b'),m,'b'); b=B[f'{m}|{net}|b|{s}']
        K=None
        out[f'{m}|{net}|{s}']=dict(ex20=r['exact'],mse20=r['K_mse'],ex30=b['exact'],mse30=b['K_mse'],over20=r['over'])
        print(m,net,s,f"%ex K<=30: {b['exact']:.1f}  K<=20: {r['exact']:.1f} | MSE 30: {b['K_mse']:.2f} 20: {r['K_mse']:.2f}",flush=True)
json.dump(out,open(RES / 'results_kmax.json','w'),indent=1)
