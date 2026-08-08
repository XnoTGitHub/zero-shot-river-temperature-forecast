"""Zusaetzliche Grafiken fuer das ueberarbeitete Paper."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# =============================================================================
# Abb. Extended: Fenster-MAE Boxplots pro Jahr fuer T, A, R, Persistenz
# =============================================================================
def fig_extended_years():
    df = pd.read_csv(BASE / "src/extended_evaluation.csv")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    years = sorted(df["year"].dropna().unique().astype(int))
    models = ["T_MAE", "A_MAE", "R_MAE", "persistence_MAE"]
    labels = {"T_MAE": "T", "A_MAE": "A", "R_MAE": "R",
              "persistence_MAE": "Persistenz"}
    colors = {"T_MAE": "#7f7f7f", "A_MAE": "#2874a6",
              "R_MAE": "#922b21", "persistence_MAE": "#e2b12a"}

    positions = []
    all_data = []
    all_colors = []
    xticks = []
    xticklabels = []
    slot_w = 0.7
    for y_idx, year in enumerate(years):
        d = df[df["year"] == year]
        for m_idx, m in enumerate(models):
            pos = y_idx * (len(models) + 1) + m_idx
            positions.append(pos)
            vals = d[m].dropna().to_numpy()
            all_data.append(vals)
            all_colors.append(colors[m])
        xticks.append(y_idx * (len(models) + 1) + (len(models) - 1) / 2)
        xticklabels.append(str(year))

    bp = ax.boxplot(all_data, positions=positions, widths=slot_w,
                    patch_artist=True, showmeans=True, meanline=True,
                    meanprops=dict(color="red", lw=1.2))
    for patch, c in zip(bp["boxes"], all_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("MAE (°C)")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=colors[m], alpha=0.75, label=labels[m])
                      for m in models]
    ax.legend(handles=legend_handles, loc="upper right", ncol=2, fontsize=7.5)
    fig.suptitle("Kreuzjahres-Validierung: Fenster-MAE pro Sommer",
                 fontsize=10, fontweight="bold")
    fig.savefig(FIG / "abb_extended_years.png")
    plt.close(fig)
    print("  abb_extended_years.png")


# =============================================================================
# Abb. Fehler-Korrelation A vs Oraku Scatter
# =============================================================================
def fig_error_correlation():
    df = pd.read_csv(BASE / "aare_recursive_thun.csv", parse_dates=["t0"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))

    # A vs Oraku
    ax = axes[0]
    r, p = stats.pearsonr(df["A_MAE"], df["Oraku_MAE"])
    ax.scatter(df["A_MAE"], df["Oraku_MAE"], s=40, c="#2874a6",
               alpha=0.75, edgecolor="black", lw=0.6)
    # Regression line
    slope, intercept = np.polyfit(df["A_MAE"], df["Oraku_MAE"], 1)
    xs = np.array([df["A_MAE"].min(), df["A_MAE"].max()])
    ax.plot(xs, slope * xs + intercept, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("MAE A (TiRex-2) [°C]")
    ax.set_ylabel("MAE Aare Oraku [°C]")
    ax.set_title(f"A vs. Oraku\nPearson r = {r:+.2f} (p = {p:.2f})",
                 fontsize=9)
    ax.grid(alpha=0.3)

    # A vs R
    ax = axes[1]
    r, p = stats.pearsonr(df["A_MAE"], df["R_recursive_MAE"])
    ax.scatter(df["A_MAE"], df["R_recursive_MAE"], s=40, c="#922b21",
               alpha=0.75, edgecolor="black", lw=0.6)
    slope, intercept = np.polyfit(df["A_MAE"], df["R_recursive_MAE"], 1)
    xs = np.array([df["A_MAE"].min(), df["A_MAE"].max()])
    ax.plot(xs, slope * xs + intercept, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("MAE A (TiRex-2) [°C]")
    ax.set_ylabel("MAE R (recursive) [°C]")
    ax.set_title(f"A vs. R\nPearson r = {r:+.2f} (p = {p:.3f})",
                 fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Fehler-Korrelation der Modell-Paare (n = 15 Fenster, Juli 2026)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "abb_error_correlation.png")
    plt.close(fig)
    print("  abb_error_correlation.png")


if __name__ == "__main__":
    fig_extended_years()
    fig_error_correlation()
    print("Fertig.")
