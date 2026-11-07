# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/phase2b/fig_ranking_saison.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Abb. Ranking (n=129, Bootstrap-CI) und Saison-Boxplot aus dem eingefrorenen Stand."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
HERE=Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from paths import RESULTS as FR, FIGURES as FIG  # noqa: E402
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size":9.5,"axes.grid":True,"grid.color":"#e3e6ea","grid.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":600})
CM=1/2.54; W=16*CM
NAME={"AO":"Ensemble ½·LAW + ½·Oraku","RO":"Ensemble ½·R + ½·Oraku","A":"TiRex-2 (Lufttemp., Abfluss, Wind) – LAW","T":"TiRex-2 (Lufttemperatur) – L","R":"TiRex-2 (zweistufig) – R","Oraku":"Aare-Oraku (Referenz)","B_persist":"TiRex-2 (zweistufig, Thun-Persistenz)","persistence":"Persistenz","climatology":"Klimatologie"}
COL={"AO":"#CC3311","RO":"#CC3311","A":"#009E73","T":"#0072B2","R":"#0072B2","Oraku":"#D55E00","B_persist":"#0072B2","persistence":"#6b7280","climatology":"#6b7280"}
r=pd.read_csv(FR/"stats_dst"/"prev_runs_ranking.csv").sort_values("MAE",ascending=False)
fig,ax=plt.subplots(figsize=(W,7.5*CM))
y=np.arange(len(r))
ax.barh(y,r.MAE,color=[COL[m] for m in r.model],alpha=0.85,height=0.6)
ax.errorbar(r.MAE,y,xerr=[r.MAE-r.CI_lo_95,r.CI_hi_95-r.MAE],fmt="none",ecolor="black",elinewidth=0.9,capsize=2.5)
for yy,(mae,hi) in enumerate(zip(r.MAE,r.CI_hi_95)): ax.text(hi+0.02,yy,f"{mae:.3f}",va="center",fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in r.model],fontsize=8.5); ax.set_xlabel("MAE [°C]")
ax.set_xlim(0,1.75); fig.tight_layout(); fig.savefig(FIG/"abb_ranking_n129.png",bbox_inches="tight"); fig.savefig(FIG/"abb_ranking_n129.pdf",bbox_inches="tight"); plt.close(fig)
s=pd.read_csv(FR/"all_seasons_prev_runs_dst_summary.csv",parse_dates=["t0"])
SE={12:"Winter",1:"Winter",2:"Winter",3:"Frühling",4:"Frühling",5:"Frühling",6:"Sommer",7:"Sommer",8:"Sommer",9:"Herbst",10:"Herbst",11:"Herbst"}
s["season"]=s.t0.dt.month.map(SE)
fig,ax=plt.subplots(figsize=(W,7*CM))
order=["Winter","Frühling","Sommer","Herbst"]; models=[("Oraku_MAE","Aare-Oraku","#D55E00"),("T_MAE","TiRex-2 (Lufttemperatur)","#0072B2"),("AO_MAE","Ensemble ½·LAW + ½·Oraku","#CC3311")]
pos=np.arange(len(order)); w=0.25
for i,(c,lab,col) in enumerate(models):
    data=[s[s.season==se][c].to_numpy() for se in order]
    bp=ax.boxplot(data,positions=pos+(i-1)*w,widths=w*0.85,patch_artist=True,showfliers=True,flierprops=dict(marker=".",ms=3,mec=col,alpha=0.6),medianprops=dict(color="black",lw=1),whiskerprops=dict(color=col),capprops=dict(color=col),boxprops=dict(facecolor=col,alpha=0.35,edgecolor=col))
    ax.plot([],[],color=col,lw=6,alpha=0.5,label=lab)
ax.set_xticks(pos); ax.set_xticklabels([f"{se}\n(n = {int((s.season==se).sum())})" for se in order]); ax.set_ylabel("MAE je Prognosefall [°C]"); ax.set_yscale("log"); ax.set_yticks([0.05,0.1,0.2,0.5,1,2]); ax.set_yticklabels(["0,05","0,1","0,2","0,5","1","2"])
ax.legend(loc="upper left",fontsize=8); fig.tight_layout(); fig.savefig(FIG/"abb_saison_boxplot_n129.png",bbox_inches="tight"); fig.savefig(FIG/"abb_saison_boxplot_n129.pdf",bbox_inches="tight")
print("-> abb_ranking_n129, abb_saison_boxplot_n129")
