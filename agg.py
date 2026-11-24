import json, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
e1=json.load(open("exp1.json")); e2=json.load(open("exp2.json")); e3=json.load(open("exp3.json"))
out={}
ms=lambda a:(float(np.mean(a)),float(np.std(a,ddof=1)))
# ---- Exp2 summary
t2={}
for n,rows in e2.items():
    d={}
    for k in ["rand_ar","p1_ar","p2_ar","p3_ar","p1_popt","p2_popt","p3_popt","p1_pfeas","p2_pfeas","p3_pfeas","p1_nfev","p2_nfev","p3_nfev"]:
        d[k]=ms([r[k] for r in rows])
    for k in ["rand_hit","sa_hit","sas_hit","p1_hit","p2_hit","p3_hit"]:
        d[k]=(int(sum(r[k] for r in rows)),len(rows))
    d["wil_p1_p3"]=float(stats.wilcoxon([r["p1_ar"] for r in rows],[r["p3_ar"] for r in rows]).pvalue)
    t2[n]=d
out["exp2"]=t2
# ---- Exp3A circuits
t3={}
for n,rows in e3["circuits"].items():
    d={"m":rows[0]["m"]}
    for k in ["naive_logical_cx","naive_logical_depth","colour_logical_depth","naive_L0_cx","naive_L0_depth","naive_L3_cx","naive_L3_depth","colour_L0_cx","colour_L0_depth","colour_L3_cx","colour_L3_depth"]:
        d[k]=ms([r[k] for r in rows])
    t3[n]=d
out["circ"]=t3
# ---- pruning
tp={}
for tau,rows in e3["prune"].items():
    tp[tau]={"zz":ms([r["zz"] for r in rows]),"ar":ms([r["ar"] for r in rows]),"popt":ms([r["popt"] for r in rows])}
base=[r["ar"] for r in e3["prune"]["0.0"]]
for tau in ("0.02","0.05","0.1"):
    tp[tau]["wil_vs_0"]=float(stats.wilcoxon(base,[r["ar"] for r in e3["prune"][tau]]).pvalue)
out["prune"]=tp
# ---- noise
tn={}
for p2,rows in e3["noise"].items():
    d={"cx":rows[0]["cx"],"depth":ms([r["depth"] for r in rows])}
    for k in ["raw","mem","zne","zne_lin","zne_mem","zne_lin_mem"]:
        d[k]=ms([abs(r[k]-r["ideal"]) for r in rows])
    raw=[abs(r["raw"]-r["ideal"]) for r in rows]
    d["wil_mem"]=float(stats.wilcoxon(raw,[abs(r["mem"]-r["ideal"]) for r in rows]).pvalue)
    d["wil_zne"]=float(stats.wilcoxon(raw,[abs(r["zne"]-r["ideal"]) for r in rows]).pvalue)
    d["wil_znemem"]=float(stats.wilcoxon(raw,[abs(r["zne_mem"]-r["ideal"]) for r in rows]).pvalue)
    d["ideal_mean"]=ms([r["ideal"] for r in rows])
    tn[p2]=d
out["noise"]=tn
json.dump(out,open("agg.json","w"),indent=1)
print(json.dumps(out,indent=1)[:6000])

# ================= Figures =================
# Fig1 architecture
fig,ax=plt.subplots(figsize=(6.6,3.0)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,46)
def box(x,y,w,h,txt,fc,ec="#333"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",fc=fc,ec=ec,lw=0.9))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=7.4)
def arr(x1,y1,x2,y2,st="->"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=st,mutation_scale=9,lw=0.9,color="#333"))
C1,C2,C3="#e8eef7","#fdf0d9","#e6f3e6"
box(2,31,22,11,"L1  Classical PQC\nexecution layer\nML-KEM / ML-DSA\n(unmodified, FIPS 203/204)",C1)
box(2,16,22,11,"L2  Profiling layer\nlatency, memory,\nenergy, bytes on wire",C1)
box(30,23,24,16,"L3  Optimisation and\ndecision engine\nconfiguration space x,\nobjective F(x), constraints",C2)
box(60,31,22,11,"L4  Quantum layer\nQUBO to QAOA,\ncircuit reduction",C3)
box(60,16,22,11,"L5  Error mitigation\nMEM, ZNE\n(PEC modelled)",C3)
box(30,3,24,10,"Classical baselines (mandatory)\nexhaustive, SA, structure-aware SA",C2)
box(60,3,22,10,"L6  Evaluation and\nsecurity validation\nKAT, tests, paired stats",C1)
box(87,16,12,26,"Selected\nconfiguration\nx*\n\nback to L1",C2)
arr(24,36,30,34); arr(24,21,30,27); arr(54,34,60,36); arr(54,28,60,22); arr(71,31,71,27)
arr(42,23,42,13); arr(54,8,60,8); arr(82,36,87,36); arr(82,21,87,24); arr(71,16,71,13)
ax.text(2,44,"Fig. 1.  QARE-PQCOF: the quantum layers are only consulted for the configuration search.",fontsize=8,style="italic")
plt.savefig("fig1.png",dpi=220,bbox_inches="tight"); plt.close()

# Fig2: optimisation quality vs n
ns=[6,9,12]
fig,axs=plt.subplots(1,2,figsize=(6.6,2.5))
w=0.16; labs=["Random","QAOA p=1","QAOA p=2","QAOA p=3"]; keys=["rand_ar","p1_ar","p2_ar","p3_ar"]; cols=["#bbb","#9ecae1","#4292c6","#08519c"]
for i,(k,l,c) in enumerate(zip(keys,labs,cols)):
    axs[0].bar(np.arange(3)+(i-1.5)*w,[t2[str(n)][k][0] for n in ns],w,yerr=[t2[str(n)][k][1] for n in ns],color=c,label=l,capsize=1.5,lw=0.5)
axs[0].set_xticks(range(3)); axs[0].set_xticklabels([f"n={n}" for n in ns]); axs[0].set_ylim(0.6,1.0); axs[0].set_ylabel("Approximation ratio"); axs[0].legend(fontsize=6.5,ncol=2,frameon=False,loc="upper right"); axs[0].set_title("(a) Expected-energy quality",fontsize=8.5)
hk=[("rand_hit","Random 1000"),("sa_hit","SA 1000"),("sas_hit","Structure-aware SA 1000"),("p3_hit","QAOA p=3, 1000 shots")]
for i,(k,l) in enumerate(hk):
    axs[1].bar(np.arange(3)+(i-1.5)*w,[t2[str(n)][k][0]/t2[str(n)][k][1] for n in ns],w,color=["#bbb","#fdae6b","#e6550d","#08519c"][i],label=l,lw=0.5)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels([f"n={n}" for n in ns]); axs[1].set_ylim(0,1.25); axs[1].set_ylabel("Fraction of instances, optimum found"); axs[1].legend(fontsize=6,frameon=False,loc="upper right"); axs[1].set_title("(b) Optimum found within budget",fontsize=8.5)
plt.tight_layout(); plt.savefig("fig2.png",dpi=220); plt.close()

# Fig3: circuit cost & pruning
fig,axs=plt.subplots(1,2,figsize=(6.6,2.5))
labs=["Naive order, L0","Naive order, L3","Colour order, L0","Colour order, L3"]; ks=["naive_L0_cx","naive_L3_cx","colour_L0_cx","colour_L3_cx"]; cols=["#d9d9d9","#9ecae1","#fdae6b","#31a354"]
for i,(k,l,c) in enumerate(zip(ks,labs,cols)):
    axs[0].bar(np.arange(3)+(i-1.5)*0.2,[t3[str(n)][k][0] for n in ns],0.2,yerr=[t3[str(n)][k][1] for n in ns],color=c,label=l,capsize=1.2,lw=0.5)
axs[0].set_xticks(range(3)); axs[0].set_xticklabels([f"n={n}" for n in ns]); axs[0].set_ylabel("CNOT count after routing (p=3)"); axs[0].legend(fontsize=6,frameon=False,loc="upper left"); axs[0].set_title("(a) Transpilation on a grid device",fontsize=8.5)
taus=["0.0","0.02","0.05","0.1"]; zz=[tp[t]["zz"][0] for t in taus]; ar=[tp[t]["ar"][0] for t in taus]; arsd=[tp[t]["ar"][1] for t in taus]
axs[1].errorbar(zz,ar,yerr=arsd,fmt="o-",color="#08519c",capsize=2,ms=4)
for z,a,t in zip(zz,ar,taus): axs[1].annotate(f"τ={t}",(z,a),textcoords="offset points",xytext=(3,6),fontsize=6.5)
axs[1].set_xlabel("ZZ terms per QAOA layer (n=9)"); axs[1].set_ylabel("Approximation ratio (true energy)"); axs[1].set_title("(b) Term pruning, p=2",fontsize=8.5)
plt.tight_layout(); plt.savefig("fig3.png",dpi=220); plt.close()

# Fig4: error mitigation
fig,ax=plt.subplots(figsize=(6.6,2.5))
p2s=["0.002","0.005","0.01"]; labs=["Raw noisy","MEM only","ZNE only","ZNE + MEM"]; ks=["raw","mem","zne","zne_mem"]; cols=["#d9d9d9","#9ecae1","#fdae6b","#31a354"]
for i,(k,l,c) in enumerate(zip(ks,labs,cols)):
    ax.bar(np.arange(3)+(i-1.5)*0.2,[tn[p][k][0] for p in p2s],0.2,yerr=[tn[p][k][1] for p in p2s],color=c,label=l,capsize=1.2,lw=0.5)
ax.set_xticks(range(3)); ax.set_xticklabels([f"2-qubit error {float(p)*100:.1f}%" for p in p2s]); ax.set_ylabel("|error| in normalised energy"); ax.legend(fontsize=7,frameon=False); ax.set_title("Absolute error of the QAOA energy estimate, n=6, p=2, 12 instances",fontsize=8.5)
plt.tight_layout(); plt.savefig("fig4.png",dpi=220); plt.close()
