"""Scatter Vorhersage vs. Wahrheit + Kontext-Ablation Grafik."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig_scatter_pred_truth():
    df = pd.read_csv(BASE / "src/predictions_wide.csv",
                     parse_dates=["t0", "datetime"])
    df["h"] = pd.to_numeric(df["h"], errors="coerce")
    print(f"  {len(df)} predictions loaded")

    # Fuege Oraku dazu aus dem Fenster-CSV (nur Fenster-MAE) — wir nutzen truth vs T und A
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    for ax, model, color, title in zip(
        axes,
        ["T_pred", "A_pred"],
        ["#7f7f7f", "#2874a6"],
        ["TiRex-2 T (nur Lufttemperatur)", "TiRex-2 A (tt+flow+wind)"],
    ):
        truth = df["truth"].to_numpy()
        pred = df[model].to_numpy()
        mask = ~np.isnan(truth) & ~np.isnan(pred)
        truth, pred = truth[mask], pred[mask]

        # Color-code by horizon
        h_norm = (df.loc[mask, "h"].to_numpy() - 1) / 47
        sc = ax.scatter(truth, pred, s=6, c=h_norm, cmap="viridis",
                        alpha=0.6, edgecolor="none")

        vmin = min(truth.min(), pred.min())
        vmax = max(truth.max(), pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k-", lw=1, alpha=0.5)
        ax.set_xlim(vmin - 0.5, vmax + 0.5)
        ax.set_ylim(vmin - 0.5, vmax + 0.5)
        ax.set_xlabel("Gemessene Wassertemperatur (°C)")
        ax.set_ylabel("Prognose (°C)")
        mae = np.mean(np.abs(pred - truth))
        r = np.corrcoef(truth, pred)[0, 1]
        ax.set_title(f"{title}\nMAE = {mae:.3f} °C, r = {r:.3f}",
                     fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Horizont (0 = t₀+1 h, 1 = t₀+48 h)", fontsize=8)

    fig.suptitle("Scatter: Prognose vs. Wahrheit (Juli 2026, 720 Datenpunkte pro Modell)",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.savefig(FIG / "abb_scatter_pred_truth.png")
    plt.close(fig)
    print("  abb_scatter_pred_truth.png")


def fig_context_ablation():
    df = pd.read_csv(BASE / "src/context_ablation.csv")
    agg = df.groupby("context")[["T_MAE", "A_MAE"]].agg(["mean", "std"])
    print(agg.round(4))

    contexts = sorted(df["context"].unique())
    x = np.arange(len(contexts))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    T_means = [df[df["context"] == c]["T_MAE"].mean() for c in contexts]
    T_std = [df[df["context"] == c]["T_MAE"].std() for c in contexts]
    A_means = [df[df["context"] == c]["A_MAE"].mean() for c in contexts]
    A_std = [df[df["context"] == c]["A_MAE"].std() for c in contexts]

    ax.bar(x - w/2, T_means, w, yerr=T_std, capsize=3,
           label="T (nur tt)", color="#7f7f7f", ec="black", lw=0.5, alpha=0.85)
    ax.bar(x + w/2, A_means, w, yerr=A_std, capsize=3,
           label="A (tt+flow+wind)", color="#2874a6", ec="black", lw=0.5, alpha=0.85)

    for i, (v, s) in enumerate(zip(T_means, T_std)):
        ax.text(i - w/2, v + s + 0.02, f"{v:.3f}", ha="center", fontsize=7)
    for i, (v, s) in enumerate(zip(A_means, A_std)):
        ax.text(i + w/2, v + s + 0.02, f"{v:.3f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"T = {c}" for c in contexts])
    ax.set_ylabel("MAE (°C)")
    ax.set_ylim(0, max(T_means + A_means) * 1.35)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.suptitle("Kontext-Längen-Ablation TiRex-2 (Juli 2026, 15 Fenster, H = 48 h)",
                 fontsize=10, fontweight="bold")
    fig.savefig(FIG / "abb_context_ablation.png")
    plt.close(fig)
    print("  abb_context_ablation.png")


if __name__ == "__main__":
    import sys
    if "--scatter" in sys.argv:
        fig_scatter_pred_truth()
    if "--context" in sys.argv:
        fig_context_ablation()
    if len(sys.argv) == 1:
        fig_scatter_pred_truth()
        fig_context_ablation()
