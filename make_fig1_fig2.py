"""Regenerates Fig. 1 (architecture) and the final Fig. 2 (legends below axes). Run after agg.py, in the same folder."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Serif"})
# ---- Fig. 1
fig,ax=plt.subplots(figsize=(6.8,3.0)); ax.axis("off"); ax.set_xlim(0,104); ax.set_ylim(0,44)
def box(x,y,w,h,txt,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",fc=fc,ec="#333",lw=0.9))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=6.3)
def arr(a,b,c,d): ax.add_patch(FancyArrowPatch((a,b),(c,d),arrowstyle="->",mutation_scale=9,lw=0.9,color="#333"))
C1,C2,C3="#e8eef7","#fdf0d9","#e6f3e6"
box(1,30,25,12,"L1 Classical PQC layer\nML-KEM / ML-DSA\nunmodified FIPS 203/204",C1)
box(1,14,25,12,"L2 Profiling layer\nlatency, memory,\nenergy, bytes on wire",C1)
box(33,22,25,18,"L3 Decision engine\nconfiguration space x\nobjective F(x)\nsecurity constraints",C2)
box(65,30,24,12,"L4 Quantum layer\nQUBO to QAOA\ncircuit reduction",C3)
box(65,14,24,12,"L5 Error mitigation\nMEM, ZNE\n(PEC modelled only)",C3)
box(33,2,25,11,"Classical baselines\nexhaustive, SA,\nstructure-aware SA",C2)
box(65,2,24,9,"L6 Evaluation and\nsecurity validation",C1)
box(94,14,9,28,"Selected\nconfig.\nx*\n\nback\nto L1",C2)
arr(26,36,33,34); arr(26,20,33,26); arr(58,34,65,36); arr(58,27,65,21); arr(77,30,77,26)
arr(45,22,45,13); arr(58,7,65,7); arr(89,36,94,36); arr(89,20,94,22); arr(77,14,77,11)
plt.savefig("fig1.png",dpi=220,bbox_inches="tight"); plt.close()
# ---- Fig. 2
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
t2=json.load(open("agg.json"))["exp2"]; ns=[6,9,12]
fig,axs=plt.subplots(1,2,figsize=(6.6,2.9)); w=0.19
labs=["Random","QAOA p=1","QAOA p=2","QAOA p=3"]; keys=["rand_ar","p1_ar","p2_ar","p3_ar"]; cols=["#bbb","#9ecae1","#4292c6","#08519c"]
for i,(k,l,c) in enumerate(zip(keys,labs,cols)):
    axs[0].bar(np.arange(3)+(i-1.5)*w,[t2[str(n)][k][0] for n in ns],w,yerr=[t2[str(n)][k][1] for n in ns],color=c,label=l,capsize=1.5,lw=0.5)
axs[0].set_xticks(range(3)); axs[0].set_xticklabels([f"n={n}" for n in ns]); axs[0].set_ylim(0.6,1.0); axs[0].set_ylabel("Approximation ratio"); axs[0].set_title("(a) Expected-energy quality",fontsize=8.5)
axs[0].legend(fontsize=6.5,ncol=2,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.14))
hk=[("rand_hit","Random, 1000 samples"),("sa_hit","Generic SA, 1000"),("sas_hit","Structure-aware SA, 1000"),("p3_hit","QAOA p=3, 1000 shots")]
for i,(k,l) in enumerate(hk):
    axs[1].bar(np.arange(3)+(i-1.5)*w,[t2[str(n)][k][0]/t2[str(n)][k][1] for n in ns],w,color=["#bbb","#fdae6b","#e6550d","#08519c"][i],label=l,lw=0.5)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels([f"n={n}" for n in ns]); axs[1].set_ylim(0,1.05); axs[1].set_ylabel("Fraction of instances, optimum found"); axs[1].set_title("(b) Optimum found within budget",fontsize=8.5)
axs[1].legend(fontsize=6.5,ncol=2,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.14))
plt.tight_layout(); plt.savefig("fig2.png",dpi=220); plt.close()
