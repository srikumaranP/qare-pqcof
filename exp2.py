import numpy as np, json, math, time
from qlib import *
rng0=np.random.default_rng(1)
cfg=[(2,3),(3,3),(4,3)]   # n=6,9,12
NINST=15; res={}
for (G,K) in cfg:
    n=G*K; rows=[]
    for s in range(NINST):
        Q,_,_=make_instance(G,K,seed=1000*n+s)
        E,X=qubo_energies(Q); Emin,Emax=E.min(),E.max()
        feas=np.array([all(X[b,g*K:(g+1)*K].sum()==1 for g in range(G)) for b in range(2**n)])
        opt_feas=E[feas].min(); assert abs(opt_feas-Emin)<1e-9   # penalty makes optimum feasible
        row={"n":n,"seed":s,"exact_evals":2**n}
        # classical baselines: random sampling & simulated annealing & greedy, matched to 1000 evaluations
        S=1000
        r=np.random.default_rng(s); idx=r.integers(0,2**n,S); row["rand_best_gap"]=float((E[idx].min()-Emin)/(Emax-Emin))
        # (a) generic single-flip SA, geometric cooling, S evals
        r=np.random.default_rng(100+s); x=r.integers(0,2,n); e=float(x@Q@x); best=e
        for t in range(S):
            T=1.0*(0.02/1.0)**(t/S); j=r.integers(n); y=x.copy(); y[j]^=1; ey=float(y@Q@y)
            if ey<e or r.random()<math.exp(-(ey-e)/T): x,e=y,ey
            best=min(best,e)
        row["sa_best_gap"]=(best-Emin)/(Emax-Emin); row["sa_hit"]=bool(abs(best-Emin)<1e-9)
        # (b) structure-aware SA: feasible-subspace move (re-select one option in one group), S evals
        r=np.random.default_rng(200+s); x=np.zeros(n)
        for g in range(G): x[g*K+r.integers(K)]=1
        e=float(x@Q@x); best=e
        for t in range(S):
            T=0.5*(0.01/0.5)**(t/S); g=r.integers(G); y=x.copy(); y[g*K:(g+1)*K]=0; y[g*K+r.integers(K)]=1; ey=float(y@Q@y)
            if ey<e or r.random()<math.exp(-(ey-e)/T): x,e=y,ey
            best=min(best,e)
        row["sas_best_gap"]=(best-Emin)/(Emax-Emin); row["sas_hit"]=bool(abs(best-Emin)<1e-9)
        row["rand_hit"]=bool(abs(E[idx].min()-Emin)<1e-9)
        # QAOA p=1,2,3 (exact sim)
        xprev=None
        for p in (1,2,3):
            t=time.time()
            x,fun,nfev,Es=optimise_qaoa(E,n,p,restarts=5,seed=s,maxiter=150,x_prev=xprev); xprev=x
            psi=qaoa_state(Es,None,None,n,x[:p],x[p:]); prob=np.abs(psi)**2
            row[f"p{p}_ar"]=float(1-fun)               # (Emax-<E>)/(Emax-Emin) on rescaled energies
            row[f"p{p}_popt"]=float(prob[np.isclose(E,Emin)].sum())
            row[f"p{p}_pfeas"]=float(prob[feas].sum())
            # 1000-shot best sample
            r=np.random.default_rng(500+s); samp=r.choice(2**n,size=S,p=prob/prob.sum())
            row[f"p{p}_best_gap"]=float((E[samp].min()-Emin)/(Emax-Emin)); row[f"p{p}_hit"]=bool(abs(E[samp].min()-Emin)<1e-9)
            row[f"p{p}_nfev"]=int(nfev)
        row["rand_ar"]=float((Emax-E.mean())/(Emax-Emin))
        rows.append(row)
    res[n]=rows
    print("n=",n,"done",flush=True)
json.dump(res,open("exp2.json","w"))
for n,rows in res.items():
    f=lambda k:(np.mean([r[k] for r in rows]),np.std([r[k] for r in rows],ddof=1))
    print(n,{k:tuple(round(v,3) for v in f(k)) for k in ["rand_ar","p1_ar","p2_ar","p3_ar","p1_popt","p2_popt","p3_popt","p3_pfeas"]},
      "hit rates rand/SA/SAswap/p1/p2/p3:",[np.mean([r[k] for r in rows]) for k in ["rand_hit","sa_hit","sas_hit","p1_hit","p2_hit","p3_hit"]])
