import numpy as np, json, math, time
from scipy import stats
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qlib import *

def edge_colour_layers(edges):
    """Greedy edge colouring of commuting ZZ terms -> layers of disjoint pairs (depth reduction)."""
    layers=[]
    for e in sorted(edges,key=lambda e:(e[0],e[1])):
        for L in layers:
            if all(e[0] not in f and e[1] not in f for f in L): L.append(e); break
        else: layers.append([e])
    return [e for L in layers for e in L]

def build(n,h,J,gam,bet,order="naive"):
    qc=QuantumCircuit(n); qc.h(range(n))
    edges=list(J.keys())
    if order=="colour": edges=edge_colour_layers(edges)
    for g,b in zip(gam,bet):
        for i in range(n): qc.rz(2*g*h[i],i)
        for (i,j) in edges:
            qc.cx(i,j); qc.rz(2*g*J[(i,j)],j); qc.cx(i,j)
        for i in range(n): qc.rx(2*b,i)
    return qc

def ising_scaled(Q,E):
    Es_scale=1.0/(E.max()-E.min())
    Qs=Q*Es_scale
    c,h,J=to_ising(Qs); return h,J

res={"circuits":{}, "prune":{}, "noise":{}}
# ---------- 3A: circuit resource reduction (p=3) ----------
for (G,K) in [(2,3),(3,3),(4,3)]:
    n=G*K; cm=CouplingMap.from_grid(*{6:(2,3),9:(3,3),12:(3,4)}[n]); rows=[]
    for s in range(8):
        Q,_,_=make_instance(G,K,seed=1000*n+s); E,X=qubo_energies(Q); h,J=ising_scaled(Q,E)
        gam=[0.3,0.5,0.7]; bet=[0.6,0.4,0.2]; row={"m":len(J)}
        for order in ("naive","colour"):
            qc=build(n,h,J,gam,bet,order)
            lg=transpile(qc,basis_gates=["cx","rz","sx","x"],optimization_level=0)
            row[f"{order}_logical_cx"]=lg.count_ops().get("cx",0); row[f"{order}_logical_depth"]=lg.depth()
            for lvl in (0,3):
                t=transpile(qc,coupling_map=cm,basis_gates=["cx","rz","sx","x"],optimization_level=lvl,seed_transpiler=7)
                row[f"{order}_L{lvl}_cx"]=t.count_ops().get("cx",0); row[f"{order}_L{lvl}_depth"]=t.depth()
        rows.append(row)
    res["circuits"][n]=rows; print("circ",n,flush=True)

# ---------- 3B: term pruning ablation (n=9, p=2) ----------
G,K=3,3; n=9
for tau in (0.0,0.02,0.05,0.10):
    rows=[]
    for s in range(10):
        Q,_,_=make_instance(G,K,seed=1000*n+s); E,X=qubo_energies(Q); Emin,Emax=E.min(),E.max()
        Qp=Q.copy()
        for i in range(n):
            for j in range(i+1,n):
                if i//K!=j//K and abs(Q[i,j])/4 < tau*1.0: Qp[i,j]=0.0  # penalty J = 2*lam/4 = 1.0 (reference scale)
        Ep,_=qubo_energies(Qp)
        c,h,J=to_ising(Qp)
        x,fun,nfev,Eps=optimise_qaoa(Ep,n,2,restarts=4,seed=s,maxiter=120)
        psi=qaoa_state(Eps,None,None,n,x[:2],x[2:]); prob=np.abs(psi)**2
        ar_true=float((Emax-(prob*E).sum())/(Emax-Emin)); popt=float(prob[np.isclose(E,Emin)].sum())
        rows.append({"zz":len(to_ising(Qp)[2]),"ar":ar_true,"popt":popt})
    res["prune"][str(tau)]=rows; print("prune",tau,np.mean([r["zz"] for r in rows]),np.mean([r["ar"] for r in rows]),flush=True)

# ---------- 3C: noise, measurement mitigation, ZNE (n=6, p=2) ----------
def noise_model(p2,p1=0.001,ro=0.02):
    nm=NoiseModel(); nm.add_all_qubit_quantum_error(depolarizing_error(p1,1),["sx","x","rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2,2),["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1-ro,ro],[ro,1-ro]])); return nm

def fold(tc,k):
    """global unitary folding U (U^dag U)^k on a measurement-free circuit"""
    out=tc.copy()
    for _ in range(k):
        out.compose(tc.inverse(),inplace=True); out.compose(tc,inplace=True)
    return out

def energy_from_counts(counts,E,shots):
    return sum(c*E[int(k,2)] for k,c in counts.items())/shots

G,K=2,3; n=6; SH=8192
for p2 in (0.002,0.005,0.01):
    nm=noise_model(p2); sim=AerSimulator(noise_model=nm,method="density_matrix"); ideal=AerSimulator(method="statevector")
    # readout calibration (tensored, per-qubit)
    cal=[]
    for prep in (0,1):
        c=QuantumCircuit(n)
        if prep: c.x(range(n))
        c.measure_all(); cal.append(sim.run(transpile(c,sim,optimization_level=0),shots=SH,seed_simulator=11).result().get_counts())
    A=[]
    for q in range(n):
        p0=sum(v for k,v in cal[0].items() if k[::-1][q]=="0")/SH; p1=sum(v for k,v in cal[1].items() if k[::-1][q]=="1")/SH
        A.append(np.array([[p0,1-p1],[1-p0,p1]]))
    Ainv=[np.linalg.inv(a) for a in A]
    def mitigate(counts):
        p=np.zeros(2**n)
        for k,v in counts.items(): p[int(k,2)]=v/SH
        T=p.reshape([2]*n)    # axis 0 = qubit n-1 ... axis n-1 = qubit 0
        for q in range(n):
            ax=n-1-q; T=np.moveaxis(np.tensordot(Ainv[q],np.moveaxis(T,ax,0),axes=(1,0)),0,ax)
        return T.reshape(-1)
    rows=[]
    for s in range(12):
        Q,_,_=make_instance(G,K,seed=1000*n+s); E,X=qubo_energies(Q); h,J=ising_scaled(Q,E)
        x,fun,nfev,Es=optimise_qaoa(E,n,2,restarts=4,seed=s,maxiter=120)
        qc=build(n,h,J,x[:2],x[2:],"colour")
        tc=transpile(qc,basis_gates=["cx","rz","sx","x"],optimization_level=1,seed_transpiler=3)
        # validation: noiseless circuit vs analytic simulator
        sv=tc.copy(); sv.save_statevector(); st=ideal.run(sv).result().get_statevector().probabilities()
        Eideal=float((st*Es).sum()); psi=qaoa_state(Es,None,None,n,x[:2],x[2:]); chk=float(np.real(np.vdot(psi,Es*psi)))
        assert abs(Eideal-chk)<1e-6,(Eideal,chk)
        ests={}; ests_mem={}
        for lam,k in ((1,0),(3,1),(5,2)):
            f=fold(tc,k); f.measure_all()
            counts=sim.run(transpile(f,sim,optimization_level=0),shots=SH,seed_simulator=100+s+lam).result().get_counts()
            ests[lam]=energy_from_counts(counts,Es,SH); ests_mem[lam]=float((mitigate(counts)*Es).sum())
        rich=lambda e:(15*e[1]-10*e[3]+3*e[5])/8      # Richardson (quadratic) extrapolation to lambda=0
        lin=lambda e:(3*e[1]-e[3])/2                   # linear extrapolation using lambda=1,3
        rows.append({"cx":tc.count_ops().get("cx",0),"depth":tc.depth(),"ideal":Eideal,
            "raw":ests[1],"mem":ests_mem[1],"zne":rich(ests),"zne_lin":lin(ests),"zne_mem":rich(ests_mem),"zne_lin_mem":lin(ests_mem)})
    res["noise"][str(p2)]=rows
    err=lambda k:np.mean([abs(r[k]-r["ideal"]) for r in rows])
    print("noise p2=",p2,{k:round(err(k),4) for k in ["raw","mem","zne","zne_lin","zne_mem","zne_lin_mem"]},"cx",rows[0]["cx"],flush=True)
json.dump(res,open("exp3.json","w"))
