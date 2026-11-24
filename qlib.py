import numpy as np, itertools, math
from scipy.optimize import minimize

def make_instance(G,K,seed,lam=None):
    """Synthetic implementation-configuration QUBO: G operation groups, K options each (one-hot).
    Per-option cost = weighted normalised latency/memory/energy; pairwise coupling = cross-group contention."""
    rng=np.random.default_rng(seed); n=G*K
    lat=rng.uniform(0.2,1.0,n); mem=rng.uniform(0.2,1.0,n); en=rng.uniform(0.2,1.0,n)
    w=np.array([0.4,0.3,0.3])
    lin=w[0]*lat+w[1]*mem+w[2]*en
    Q=np.zeros((n,n))
    for i in range(n): Q[i,i]=lin[i]
    for i in range(n):
        for j in range(i+1,n):
            if i//K!=j//K: Q[i,j]=rng.normal(0,0.15)   # cross-group interaction
    lam = lam if lam is not None else 2.0
    for g in range(G):                               # one-hot penalty lam*(sum x -1)^2
        idx=range(g*K,(g+1)*K)
        for i in idx:
            Q[i,i]+=-lam
            for j in idx:
                if j>i: Q[i,j]+=2*lam
    return Q,G,K

def qubo_energies(Q):
    n=Q.shape[0]
    X=((np.arange(2**n)[:,None]>>np.arange(n))&1).astype(float)   # bit i = qubit i
    E=np.einsum('bi,ij,bj->b',X,Q,X)
    return E,X

def to_ising(Q):
    """x=(1-z)/2 -> E = c + sum h_i z_i + sum J_ij z_i z_j (J on i<j)."""
    n=Q.shape[0]; h=np.zeros(n); J={}; c=0.0
    for i in range(n):
        c+=Q[i,i]/2; h[i]-=Q[i,i]/2
    for i in range(n):
        for j in range(i+1,n):
            q=Q[i,j]
            if q!=0:
                c+=q/4; h[i]-=q/4; h[j]-=q/4; J[(i,j)]=q/4
    return c,h,J

def qaoa_state(E,h,J,n,gammas,betas):
    """Statevector QAOA using diagonal cost E (full QUBO energy) -- exact simulator."""
    psi=np.ones(2**n,complex)/math.sqrt(2**n)
    for g,b in zip(gammas,betas):
        psi=psi*np.exp(-1j*g*E)
        psi=psi.reshape([2]*n)
        c,s=math.cos(b),-1j*math.sin(b)
        for q in range(n):
            ax=n-1-q
            a0=np.take(psi,0,axis=ax); a1=np.take(psi,1,axis=ax)
            psi=np.stack([c*a0+s*a1, s*a0+c*a1],axis=ax)
        psi=psi.reshape(-1)
    return psi

def interp(x,p):
    g,b=x[:p],x[p:]; q=p+1
    f=lambda a:np.interp(np.linspace(0,1,q),np.linspace(0,1,p),a) if p>1 else np.repeat(a,q)
    return np.concatenate([f(g),f(b)])

def optimise_qaoa(E,n,p,restarts=6,seed=0,maxiter=120,x_prev=None):
    Es=(E-E.min())/(E.max()-E.min())  # rescale for stable angles
    rng=np.random.default_rng(seed); best=None; nfev=0
    def f(x):
        psi=qaoa_state(Es,None,None,n,x[:p],x[p:])
        return float(np.real(np.vdot(psi,Es*psi)))
    for r in range(restarts):
        x0=np.concatenate([rng.uniform(0,2.0,p),rng.uniform(0,math.pi/2,p)])
        if r==0 and x_prev is not None: x0=interp(x_prev,p-1)
        o=minimize(f,x0,method='COBYLA',options={'maxiter':maxiter})
        nfev+=o.nfev
        if best is None or o.fun<best.fun: best=o
    return best.x,best.fun,nfev,Es
