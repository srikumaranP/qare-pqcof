import time, json, statistics as st, platform, sys, math
import numpy as np
from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
import importlib.metadata as md

def ci95(x):
    x=np.array(x,float); n=len(x)
    from scipy import stats
    h=stats.t.ppf(0.975,n-1)*x.std(ddof=1)/math.sqrt(n)
    return float(x.mean()),float(x.std(ddof=1)),float(h)

def timeit(f,reps):
    out=[]
    for _ in range(reps):
        t=time.perf_counter(); r=f(); out.append((time.perf_counter()-t)*1e3)
    return out,r

res={"env":{"python":sys.version.split()[0],"kyber_py":md.version("kyber-py"),"dilithium_py":md.version("dilithium-py"),
            "numpy":np.__version__,"platform":platform.platform()},"kem":{},"dsa":{}}
msg=b"QARE-PQCOF benchmark message"*4
for name,K in [("ML-KEM-512",ML_KEM_512),("ML-KEM-768",ML_KEM_768),("ML-KEM-1024",ML_KEM_1024)]:
    R=40
    kg,(pk,sk)=timeit(K.keygen,R)
    ec=[];dc=[]
    pk,sk=K.keygen()
    ec,(key,ct)=timeit(lambda:K.encaps(pk),R)
    dc,key2=timeit(lambda:K.decaps(sk,ct),R)
    assert key==key2
    res["kem"][name]={"pk":len(pk),"sk":len(sk),"ct":len(ct),"reps":R,
        "keygen":ci95(kg),"encaps":ci95(ec),"decaps":ci95(dc),
        "raw":{"keygen":kg,"encaps":ec,"decaps":dc}}
    print(name,len(pk),len(sk),len(ct),[round(ci95(v)[0],2) for v in (kg,ec,dc)],flush=True)
for name,D in [("ML-DSA-44",ML_DSA_44),("ML-DSA-65",ML_DSA_65),("ML-DSA-87",ML_DSA_87)]:
    R=20
    kg,(pk,sk)=timeit(D.keygen,R)
    pk,sk=D.keygen()
    sg,sig=timeit(lambda:D.sign(sk,msg),R)
    vf,ok=timeit(lambda:D.verify(pk,msg,sig),R)
    assert ok
    res["dsa"][name]={"pk":len(pk),"sk":len(sk),"sig":len(sig),"reps":R,
        "keygen":ci95(kg),"sign":ci95(sg),"verify":ci95(vf),
        "raw":{"keygen":kg,"sign":sg,"verify":vf}}
    print(name,len(pk),len(sk),len(sig),[round(ci95(v)[0],2) for v in (kg,sg,vf)],flush=True)
json.dump(res,open("exp1.json","w"))
