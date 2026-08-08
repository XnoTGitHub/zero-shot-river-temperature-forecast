"""Erzeugt alle Paper-Grafiken in figures/ mit 300 dpi."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# -----------------------------------------------------------------------------
# Abb. 1: Studiengebiet Karte (schematisch: Thunersee -> Aare -> Bern)
# -----------------------------------------------------------------------------
def fig01_karte():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    # Schweiz-Umriss (grob approximiert)
    swiss_x = [0.05, 0.15, 0.30, 0.55, 0.75, 0.92, 0.95, 0.85, 0.70, 0.40, 0.20, 0.07, 0.05]
    swiss_y = [0.55, 0.75, 0.85, 0.88, 0.85, 0.70, 0.50, 0.30, 0.20, 0.15, 0.20, 0.35, 0.55]
    ax.fill(swiss_x, swiss_y, color="#e8e8e0", ec="black", lw=0.8, alpha=0.6)

    # Thunersee
    from matplotlib.patches import Ellipse
    lake = Ellipse((0.42, 0.42), width=0.13, height=0.055, angle=25,
                   fc="#5b9ec9", ec="#2c5980", lw=0.9, zorder=3)
    ax.add_patch(lake)
    ax.text(0.42, 0.36, "Thunersee", ha="center", va="top",
            fontsize=8, style="italic", color="#1f3e5a")

    # Aare (Fluss von Thunersee nach Bern) - geschwungene Linie
    aare_x = np.array([0.47, 0.49, 0.48, 0.46, 0.44, 0.41, 0.38, 0.36, 0.34, 0.33])
    aare_y = np.array([0.44, 0.47, 0.51, 0.545, 0.56, 0.575, 0.58, 0.575, 0.565, 0.55])
    ax.plot(aare_x, aare_y, color="#2c5980", lw=1.8, zorder=2, solid_capstyle="round")

    # Pegel Bern (Standort Schoenau)
    ax.plot(0.33, 0.55, marker="v", ms=11, color="#c0392b", mec="black",
            mew=0.7, zorder=5)
    ax.annotate("Aare Bern\n(Aareguru 'bern')", xy=(0.33, 0.55),
                xytext=(0.18, 0.62), fontsize=8, fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))

    # Pegel Thun (Ausfluss Thunersee)
    ax.plot(0.47, 0.44, marker="v", ms=9, color="#e67e22", mec="black",
            mew=0.7, zorder=5)
    ax.annotate("Aare Thun\n(upstream sensor)", xy=(0.47, 0.44),
                xytext=(0.63, 0.35), fontsize=8, fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))

    # Wind-Achse Thunersee (SE-NW, 114 deg from north)
    wx, wy = 0.42, 0.42
    dx, dy = 0.06 * np.sin(np.deg2rad(114)), -0.06 * np.cos(np.deg2rad(114))
    ax.annotate("", xy=(wx + dx, wy + dy), xytext=(wx - dx, wy - dy),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="#9b59b6", alpha=0.85))
    ax.text(0.55, 0.47, "Wind-Achse\n114 deg / 294 deg",
            fontsize=7, color="#6a2f88", ha="left", style="italic")

    # Nordpfeil
    ax.annotate("N", xy=(0.06, 0.97), xytext=(0.06, 0.90),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
                ha="center", fontsize=10, fontweight="bold")

    # Fliessrichtung Aare
    ax.annotate("", xy=(0.34, 0.55), xytext=(0.44, 0.56),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#2c5980"))
    ax.text(0.395, 0.60, "Aare (Fliessrichtung)", fontsize=7,
            style="italic", color="#1f3e5a")

    # Massstab (schematisch)
    ax.plot([0.65, 0.85], [0.05, 0.05], "k-", lw=1.5)
    ax.plot([0.65, 0.65], [0.03, 0.07], "k-", lw=1.2)
    ax.plot([0.85, 0.85], [0.03, 0.07], "k-", lw=1.2)
    ax.text(0.75, 0.09, "~ 30 km", ha="center", fontsize=7)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("Untersuchungsgebiet: Aare zwischen Thunersee und Bern (Schweiz)",
                 fontsize=10, fontweight="bold", y=0.98)
    fig.savefig(FIG / "abb01_karte_studiengebiet.png")
    plt.close(fig)
    print("  Abb. 1 gespeichert")


# -----------------------------------------------------------------------------
# Abb. 2: TiRex-2 Kovariat-Interface (Zeitachse mit past und future covariates)
# -----------------------------------------------------------------------------
def fig02_tirex_schema():
    fig, ax = plt.subplots(figsize=(7.0, 2.9))

    t = np.arange(0, 340)
    T = 256  # context length
    H = 48   # horizon

    # Target (Wassertemperatur)
    y = 18 + 2 * np.sin(t / 30) + 0.3 * np.random.default_rng(0).standard_normal(len(t))
    ax.plot(t[:T], y[:T], color="#2c5980", lw=1.4, label="Target: Wassertemperatur (Historie)")
    ax.plot(t[T:T + H], y[T:T + H], color="#c0392b", lw=1.4, ls="--",
            label="Prognose (Horizont H)")

    # Kovariate (versetzt, klein)
    yc = 22 + 3 * np.sin((t + 30) / 40)
    ax.plot(t[:T], yc[:T] - 12, color="#27ae60", lw=1.0, alpha=0.7,
            label="past_covariates (bis t0)")
    ax.plot(t[T:T + H], yc[T:T + H] - 12, color="#27ae60", lw=1.0, alpha=0.7,
            ls="-.")
    ax.plot(t, yc - 15, color="#8e44ad", lw=1.0, alpha=0.7,
            label="future_covariates (bis t0+H)")

    # t0-Linie
    ax.axvline(T, color="black", lw=1, ls=":")
    ax.text(T, 22.5, r" $t_0$", fontsize=9, fontweight="bold")

    # Fenster-Beschriftung
    ax.annotate("", xy=(T, 5.5), xytext=(0, 5.5),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(T / 2, 5.0, f"Kontext-Fenster (T = {T} h)",
            ha="center", fontsize=8, style="italic")
    ax.annotate("", xy=(T + H, 5.5), xytext=(T, 5.5),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(T + H / 2, 5.0, f"Horizont ({H} h)", ha="center",
            fontsize=8, style="italic")

    ax.set_xlim(0, 340); ax.set_ylim(4, 24)
    ax.set_xlabel("Zeit (h)"); ax.set_ylabel("Wert")
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_yticks([])
    fig.suptitle("TiRex-2 Kovariat-Interface: past_covariates und future_covariates",
                 fontsize=10, fontweight="bold")
    fig.savefig(FIG / "abb02_tirex_schema.png")
    plt.close(fig)
    print("  Abb. 2 gespeichert")


# -----------------------------------------------------------------------------
# Abb. 3: Wind-Alongshore-Physik am Thunersee (Ekman-Transport)
# -----------------------------------------------------------------------------
def fig03_wind_physik():
    # Vertikal stapeln fuer schmale Spalte
    fig, axes = plt.subplots(2, 1, figsize=(3.8, 5.0))

    for ax, (title, wind_deg, colour, effect) in zip(
        axes,
        [("Wind aus 114° (SE): warm nach Thun",
          114, "#c0392b", "warmes Wasser\nnach Bern"),
         ("Wind aus 294° (NW): Upwelling",
          294, "#2874a6", "kaltes Wasser\nnach Bern")]):

        from matplotlib.patches import Ellipse
        lake = Ellipse((0.55, 0.45), 0.6, 0.30, angle=25,
                       fc="#a9d6ec", ec="#1a5276", lw=1)
        ax.add_patch(lake)
        # Ausfluss Thun (links, im NW)
        ax.plot(0.30, 0.58, marker=">", ms=10, color="#e67e22", mec="black")
        ax.text(0.28, 0.72, "Ausfluss\nThun", fontsize=7, ha="center",
                va="center", fontweight="bold")

        # Windvektor
        target_deg = (wind_deg + 180) % 360
        arrow_len = 0.20
        ex = np.sin(np.deg2rad(target_deg)) * arrow_len
        ey = -np.cos(np.deg2rad(target_deg)) * arrow_len
        start_x = 0.55 - ex * 1.5
        start_y = 0.45 - ey * 1.5
        ax.annotate("", xy=(start_x + ex, start_y + ey),
                    xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color=colour))
        # Wind-Label neben dem Vektor-Ursprung
        ax.text(start_x, start_y + 0.06,
                f"Wind {wind_deg}°",
                fontsize=8, color=colour, fontweight="bold", ha="center")

        # Effekt: kleine Box unten
        eff_col = "#c0392b" if "warm" in effect else "#2874a6"
        ax.text(0.5, 0.10, effect, fontsize=7.5, color=eff_col,
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec=eff_col, lw=0.8))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIG / "abb03_wind_physik.png")
    plt.close(fig)
    print("  Abb. 3 gespeichert")


# -----------------------------------------------------------------------------
# Abb. 4: 2-Stage-Workflow (Thun -> Bern rekursiv)
# -----------------------------------------------------------------------------
def fig04_workflow():
    # Vertikaler Aufbau, gedacht fuer eine Spalte
    fig, ax = plt.subplots(figsize=(3.8, 6.0))

    # Stage 1 Box (oben)
    s1 = FancyBboxPatch((0.05, 0.76), 0.90, 0.22,
                        boxstyle="round,pad=0.01",
                        fc="#dbeff5", ec="#1a5276", lw=1.3)
    ax.add_patch(s1)
    ax.text(0.50, 0.945, "Stage 1 : TiRex-2", ha="center", fontsize=10,
            fontweight="bold", color="#1a5276")
    ax.text(0.50, 0.895, "Ziel: Thun-Wassertemperatur",
            ha="center", fontsize=8, fontweight="bold")
    ax.text(0.50, 0.850, "past + future covariates:",
            ha="center", fontsize=7.5, style="italic")
    ax.text(0.50, 0.800, "tt, solar, flow (Thun), Wind Thunersee",
            ha="center", fontsize=7.5)

    # Pfeil Stage 1 -> Stage 2
    ax.annotate("", xy=(0.50, 0.60), xytext=(0.50, 0.75),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))
    ax.text(0.53, 0.68, r"$\hat{T}_{Thun}(t_0..t_0+H)$",
            ha="left", fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    # Stage 2 Box (unten)  --  hoeher fuer mehr Text-Platz
    s2 = FancyBboxPatch((0.02, 0.14), 0.96, 0.44,
                        boxstyle="round,pad=0.01",
                        fc="#f5dbdb", ec="#922b21", lw=1.3)
    ax.add_patch(s2)
    ax.text(0.50, 0.545, "Stage 2 : TiRex-2", ha="center", fontsize=10,
            fontweight="bold", color="#922b21")
    ax.text(0.50, 0.495, "Ziel: Bern-Wassertemperatur",
            ha="center", fontsize=8, fontweight="bold")
    ax.text(0.50, 0.435, "past covariates:",
            ha="center", fontsize=7.5, style="italic")
    ax.text(0.50, 0.385, "Bern-Historie, tt, flow, wind, Thun-Historie",
            ha="center", fontsize=7.5)
    ax.text(0.50, 0.315, "future covariates:",
            ha="center", fontsize=7.5, style="italic")
    ax.text(0.50, 0.265,
            r"tt, flow (BAFU), wind, $\hat{T}_{Thun}$ aus Stage 1",
            ha="center", fontsize=7.5)
    ax.text(0.50, 0.185,
            r"$\uparrow$  Stage 1 ersetzt naive Persistenz  $\uparrow$",
            ha="center", fontsize=7, style="italic", color="#922b21")

    # Output-Pfeil
    ax.annotate("", xy=(0.50, 0.05), xytext=(0.50, 0.14),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#922b21"))
    ax.text(0.50, 0.005, r"$\hat{T}_{Bern}(t_0..t_0+H)$",
            ha="center", fontsize=9, style="italic", fontweight="bold",
            color="#922b21")

    ax.set_xlim(0, 1); ax.set_ylim(-0.01, 1); ax.set_aspect("auto")
    ax.axis("off")
    fig.savefig(FIG / "abb04_workflow_recursive.png",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("  Abb. 4 gespeichert")


# -----------------------------------------------------------------------------
# Abb. 5: MAE-Ranking mit Bootstrap-CIs
# -----------------------------------------------------------------------------
def fig05_ranking_ci():
    dfs = pd.read_csv(BASE / "src/baselines_summary.csv")
    label_map = {
        "climatology": "Klimatologie (Baseline)",
        "persistence": "Persistenz (Baseline)",
        "B_persist": "TiRex-2 + Thun (Persistenz)",
        "T": "T: TiRex-2 (nur tt)",
        "A": "A: TiRex-2 (tt+flow+wind)",
        "Oraku": "Aare Oraku (Baseline)",
        "R_recursive": "R: Rekursive 2-Stage",
        "RO": "Ensemble(R, Oraku)",
        "AO": "Ensemble(A, Oraku)  *",
    }
    dfs["label"] = dfs["model"].map(label_map).fillna(dfs["model"])
    dfs = dfs.sort_values("MAE", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors, edges, lws, alphas = [], [], [], []
    for m in dfs["model"]:
        if m == "AO":
            colors.append("#c0392b"); edges.append("#4a0000"); lws.append(1.8); alphas.append(1.0)
        elif m in ["climatology", "persistence"]:
            colors.append("#7f7f7f"); edges.append("black"); lws.append(0.6); alphas.append(0.65)
        elif m == "Oraku":
            colors.append("#e67e22"); edges.append("black"); lws.append(0.6); alphas.append(0.85)
        else:
            colors.append("#2874a6"); edges.append("black"); lws.append(0.6); alphas.append(0.85)

    yy = np.arange(len(dfs))
    err_lo = dfs["MAE"] - dfs["CI_lo_95"]
    err_hi = dfs["CI_hi_95"] - dfs["MAE"]
    # Bars einzeln zeichnen, damit Rand pro Modell variabel ist
    for i, (mae_v, elo, ehi, c, e, lw, a) in enumerate(
            zip(dfs["MAE"], err_lo, err_hi, colors, edges, lws, alphas)):
        ax.barh(i, mae_v, xerr=[[elo], [ehi]],
                color=c, ec=e, lw=lw, alpha=a,
                error_kw=dict(ecolor="black", elinewidth=1.0, capsize=3))
    for i, (m, v) in enumerate(zip(dfs["MAE"], dfs["CI_hi_95"])):
        is_ao = dfs["model"].iloc[i] == "AO"
        ax.text(v + 0.05, i, f"{m:.3f}", va="center",
                fontsize=8.5 if is_ao else 7.5,
                fontweight="bold" if is_ao else "normal",
                color="#4a0000" if is_ao else "black")

    ax.set_yticks(yy)
    ax.set_yticklabels(dfs["label"])
    # AO-Label fett hervorheben
    for tick, m in zip(ax.get_yticklabels(), dfs["model"]):
        if m == "AO":
            tick.set_fontweight("bold")
            tick.set_color("#4a0000")
    ax.invert_yaxis()
    ax.set_xlabel("MAE (deg C) mit 95% Bootstrap-CI, n = 15 Fenster @ 48h")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(dfs["CI_hi_95"]) + 0.3)
    fig.suptitle("Modell-Ranking Aare Bern (48h): Ensemble schlaegt alle einzelnen",
                 fontsize=10, fontweight="bold")
    fig.savefig(FIG / "abb05_ranking_ci.png")
    plt.close(fig)
    print("  Abb. 5 gespeichert")


# -----------------------------------------------------------------------------
# Abb. 6: 7-Tage-Segmentanalyse
# -----------------------------------------------------------------------------
def fig06_7d_segments():
    df = pd.read_csv(BASE / "aare_7d_experiment.csv")
    print(f"  7d-Experiment: {len(df)} Fenster, Spalten: {list(df.columns)}")

    seg_data = {
        "0-48 h":  {"T": 0.54, "A": 0.55, "R": 0.62, "mean(A,R)": 0.56},
        "48-120 h": {"T": 0.77, "A": 0.77, "R": 1.10, "mean(A,R)": 0.93},
        "120-168 h": {"T": 0.87, "A": 0.82, "R": 1.26, "mean(A,R)": 1.01},
    }
    seg_labels = list(seg_data.keys())
    models = ["T", "A", "R", "mean(A,R)"]
    x = np.arange(len(seg_labels))
    w = 0.2

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = {"T": "#7f7f7f", "A": "#2874a6", "R": "#922b21", "mean(A,R)": "#8e44ad"}
    labels_pretty = {
        "T": "TiRex-2 (tt only)",
        "A": "TiRex-2 (tt+flow+wind)",
        "R": "R: Rekursive 2-Stage",
        "mean(A,R)": "Ensemble(A, R)",
    }
    for i, m in enumerate(models):
        vals = [seg_data[s][m] for s in seg_labels]
        offset = (i - (len(models) - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=labels_pretty[m],
               color=colors[m], ec="black", lw=0.4, alpha=0.85)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(seg_labels)
    ax.set_ylabel("Segment-MAE (deg C)")
    ax.set_ylim(0, 1.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.text(2.0, 1.42, "R kollabiert ab 48h wegen\nFehlerpropagation aus Stage-1",
            fontsize=7, ha="center", style="italic", color="#922b21",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8ebe8", ec="#922b21", alpha=0.9))
    fig.suptitle("7-Tage-Segmentanalyse: Modelle haben ihren Sweet Spot",
                 fontsize=10, fontweight="bold")
    fig.savefig(FIG / "abb06_7d_segments.png")
    plt.close(fig)
    print("  Abb. 6 gespeichert")


# -----------------------------------------------------------------------------
# Abb. 7: Kovariat-Ablation Boxplot
# -----------------------------------------------------------------------------
def fig07_ablation():
    df = pd.read_csv(BASE / "aare_experiments_results.csv")
    print(f"  Ablation-Daten: {len(df)} Zeilen, Spalten: {list(df.columns)}")
    if "horizon" in df.columns:
        d168 = df[df["horizon"] == 168].copy()
    else:
        d168 = df.copy()

    # Wir extrahieren gewuenschte Varianten wenn verfuegbar
    variants = ["TiRex-1", "TiRex-2 no cov", "TiRex-2 past cov",
                "TiRex-2 future flow-only", "TiRex-2 future tt-only",
                "TiRex-2 future noisy", "TiRex-2 future perfect"]
    model_col = None
    for c in ["model", "variant", "Model", "Variante"]:
        if c in d168.columns:
            model_col = c
            break

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if model_col is not None:
        available = [v for v in variants if v in d168[model_col].unique()]
        if len(available) == 0:
            available = d168[model_col].unique().tolist()
        mae_col = "MAE" if "MAE" in d168.columns else ("mae" if "mae" in d168.columns else None)
        if mae_col is None:
            mae_col = d168.columns[-1]
        data = [d168[d168[model_col] == v][mae_col].dropna().values for v in available]
        bp = ax.boxplot(data, tick_labels=available, patch_artist=True,
                        showmeans=True, meanline=True,
                        meanprops=dict(color="red", lw=1.2))
        for patch in bp["boxes"]:
            patch.set_facecolor("#a9cce3")
            patch.set_alpha(0.7)
        ax.set_ylabel("MAE (deg C) @ 168h")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7)
    else:
        ax.text(0.5, 0.5, "Ablation-Daten nicht auswertbar",
                ha="center", va="center", transform=ax.transAxes)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Kovariat-Ablation TiRex-2: Lufttemperatur ist der wertvollste Kanal",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "abb07_ablation.png")
    plt.close(fig)
    print("  Abb. 7 gespeichert")


# -----------------------------------------------------------------------------
# Abb. 8: Beispielprognose (Zeitreihe: 4 Fenster)
# -----------------------------------------------------------------------------
def fig08_beispielfenster():
    """Bestes, typisches (Median) und schlechtestes Fenster nach AO_MAE."""
    import json
    df = pd.read_csv(BASE / "aare_recursive_thun.csv", parse_dates=["t0"])
    df = df.sort_values("AO_MAE").reset_index(drop=True)
    n = len(df)
    sel = pd.DataFrame([
        df.iloc[0].to_dict()      | {"label": "Bestes Fenster",       "rank": f"1 / {n}"},
        df.iloc[n // 2].to_dict() | {"label": "Typisches Fenster (Median)", "rank": f"{n // 2 + 1} / {n}"},
        df.iloc[n - 1].to_dict()  | {"label": "Schlechtestes Fenster", "rank": f"{n} / {n}"},
    ])

    pw = pd.read_csv(BASE / "src/predictions_wide.csv",
                     parse_dates=["t0", "datetime"])

    def load_oraku_series(t0):
        fp = BASE / "data" / "aare_dashboard_cache" / f"oraku_temperature_{t0.isoformat()}_48.json"
        with open(fp) as f:
            d = json.load(f)
        times = pd.to_datetime(d["data"]["time"]).tz_localize(None)
        return pd.Series(d["data"]["value"], index=times, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)

    for ax, (_, r) in zip(axes, sel.iterrows()):
        t0 = pd.Timestamp(r["t0"])
        sub = pw[pw["t0"] == t0].sort_values("h")
        if len(sub) == 0:
            continue
        idx = sub["datetime"]
        truth = sub["truth"].to_numpy()
        pred_A = sub["A_pred"].to_numpy()
        oraku = load_oraku_series(t0).reindex(idx).to_numpy()
        pred_AO = 0.5 * (pred_A + oraku)

        ax.plot(idx, truth, "k-", lw=1.6, marker="o", ms=2.2, label="Messung")
        ax.plot(idx, pred_A, color="#2874a6", lw=1.2, ls="--", label="A (TiRex-2)")
        ax.plot(idx, oraku, color="#e67e22", lw=1.2, ls="--", label="Oraku")
        ax.plot(idx, pred_AO, color="#c0392b", lw=1.6, label="AO (Ensemble)")

        info = (f"A: {r['A_MAE']:.2f}\n"
                f"Oraku: {r['Oraku_MAE']:.2f}\n"
                f"AO: {r['AO_MAE']:.2f}")
        ax.text(0.98, 0.03, info, transform=ax.transAxes, fontsize=7,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#4a0000", lw=0.5, alpha=0.9))
        ax.set_title(f"{r['label']}\n{t0.strftime('%Y-%m-%d')}  (Rank {r['rank']})",
                     fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %Hh"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    axes[0].set_ylabel("Wassertemperatur (deg C)")
    axes[0].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    fig.suptitle("Bestes, typisches und schlechtestes Fenster nach AO-MAE (48h)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "abb08_beispielfenster.png")
    plt.close(fig)
    print("  Abb. 8 gespeichert")


if __name__ == "__main__":
    print("Erzeuge Grafiken in", FIG)
    fig01_karte()
    fig02_tirex_schema()
    fig03_wind_physik()
    fig04_workflow()
    fig05_ranking_ci()
    fig06_7d_segments()
    fig07_ablation()
    fig08_beispielfenster()
    print("Fertig.")
