"""Erzeugt die zwei neuen Ganzjahres-Figuren fuer paper_v2_ganzjahr.typ.

Output:
  figures/abb_allseasons_ranking.png  — Modell-Ranking n=128
  figures/abb_allseasons_seasons.png  — Boxplot per Saison + Modell
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "figures"
SRC = BASE / "src"

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.linewidth": 0.6})


def fig_ranking():
    dfs = pd.read_csv(SRC / "all_seasons_ranking.csv")
    # short labels
    label_map = {
        "AO": "Ensemble(A, Oraku)  *",
        "RO": "Ensemble(R, Oraku)  *",
        "T": "TiRex-2 (nur tt)",
        "R": "R: Rekursive 2-Stage",
        "A": "TiRex-2 (tt+flow+wind)",
        "Oraku": "Aare Oraku (Baseline)",
        "B_persist": "TiRex-2 + Thun (Persist.)",
        "persistence": "Persistenz (Baseline)",
        "climatology": "Klimatologie (Baseline)",
    }
    dfs["short"] = dfs["model"].map(label_map)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors, edges, lws, alphas = [], [], [], []
    for m in dfs["model"]:
        if m in ("AO", "RO"):
            colors.append("#c0392b"); edges.append("#4a0000"); lws.append(1.8); alphas.append(1.0)
        elif m in ("climatology", "persistence"):
            colors.append("#7f7f7f"); edges.append("black"); lws.append(0.6); alphas.append(0.65)
        elif m == "Oraku":
            colors.append("#e67e22"); edges.append("black"); lws.append(0.6); alphas.append(0.85)
        else:
            colors.append("#2874a6"); edges.append("black"); lws.append(0.6); alphas.append(0.85)

    yy = np.arange(len(dfs))
    err_lo = dfs["MAE"] - dfs["CI_lo_95"]
    err_hi = dfs["CI_hi_95"] - dfs["MAE"]
    for i, (mae_v, elo, ehi, c, e, lw, a) in enumerate(
            zip(dfs["MAE"], err_lo, err_hi, colors, edges, lws, alphas)):
        ax.barh(i, mae_v, xerr=[[elo], [ehi]],
                color=c, ec=e, lw=lw, alpha=a,
                error_kw=dict(ecolor="black", elinewidth=1.0, capsize=3))
    for i, (m, v) in enumerate(zip(dfs["MAE"], dfs["CI_hi_95"])):
        is_ens = dfs["model"].iloc[i] in ("AO", "RO")
        ax.text(v + 0.05, i, f"{m:.3f}", va="center",
                fontsize=8.5 if is_ens else 7.5,
                fontweight="bold" if is_ens else "normal",
                color="#4a0000" if is_ens else "black")
    ax.set_yticks(yy)
    ax.set_yticklabels(dfs["short"])
    for tick, m in zip(ax.get_yticklabels(), dfs["model"]):
        if m in ("AO", "RO"):
            tick.set_fontweight("bold")
            tick.set_color("#4a0000")
    ax.invert_yaxis()
    ax.set_xlabel("MAE (deg C) mit 95% Bootstrap-CI, n = 128 Fenster @ 48h "
                  "(2025-11-04 bis 2026-07-30)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(dfs["CI_hi_95"]) + 0.3)
    fig.suptitle("Modell-Ranking Aare Bern, Ganzjahres-Test (n = 128 Fenster, 4 Jahreszeiten)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "abb_allseasons_ranking.png")
    plt.close(fig)
    print("  Ganzjahres-Ranking gespeichert")


def fig_seasons():
    df = pd.read_csv(SRC / "all_seasons_summary.csv", parse_dates=["t0"])
    df["month"] = df["t0"].dt.month
    df["season"] = df["month"].map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Frühling",4:"Frühling",5:"Frühling",
        6:"Sommer",7:"Sommer",8:"Sommer",
        9:"Herbst",10:"Herbst",11:"Herbst"})

    seasons = ["Winter","Frühling","Sommer","Herbst"]
    models = [("T_MAE","T"), ("A_MAE","A"), ("R_MAE","R"),
              ("Oraku_MAE","Oraku"), ("AO_MAE","AO"), ("RO_MAE","RO")]
    colors = ["#2874a6","#2874a6","#2874a6","#e67e22","#c0392b","#c0392b"]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.2), sharey=True)
    for ax, s in zip(axes, seasons):
        sub = df[df["season"] == s]
        if len(sub) == 0: continue
        data = [sub[m].dropna().values for m, _ in models]
        labels = [lab for _, lab in models]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="red", markersize=4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.set_title(f"{s}\n(n={len(sub)}, T̄={sub['truth_mean'].mean():.1f} °C)",
                     fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7.5)
    axes[0].set_ylabel("Fenster-MAE (deg C)")
    fig.suptitle("MAE pro Fenster nach Jahreszeit (n = 128 gesamt, Median + Mittelwert)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "abb_allseasons_seasons.png")
    plt.close(fig)
    print("  Saison-Boxplot gespeichert")


if __name__ == "__main__":
    print("Generiere Ganzjahres-Figuren ...")
    fig_ranking()
    fig_seasons()
    print("Fertig.")
