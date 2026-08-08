"""Reviewer-Robustheitsanalysen zur Ganzjahres-Auswertung (n=128).

Drei Fragen:
  1. Sensitivitaet: bleiben die Hauptbefunde bestehen, wenn die 7 Fenster
     entfernt werden, die zeitlich in die Juli-2026-Entwicklungsphase fallen?
  2. Serielle Abhaengigkeit: bleiben die Befunde bei Startabstand 4d (jedes
     zweite Fenster) und 8d (jedes vierte) bestehen? Zusatz: Block-Bootstrap.
  3. Bias-Mechanismus: sind die gegenlaeufigen Bias-Signaturen von A und
     Oraku ein Ganzjahres-Phaenomen oder saisonal begrenzt?

Output:
  reviewer_sensitivity.csv       - Wilcoxon-Tests auf 121 Fenstern
  reviewer_thinning.csv          - Wilcoxon-Tests bei 4d/8d Sampling
  reviewer_block_bootstrap.csv   - Block-Bootstrap-CIs (Blocklaenge 3, 5, 7)
  reviewer_bias_by_season.csv    - Mean/Median Residual pro Modell & Saison
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "src"

SUMMARY = SRC / "all_seasons_summary.csv"
HOURLY = SRC / "all_seasons_hourly.csv"

# Fenster, deren [t0, t0+48h] mit dem Juli-2026-Sommer-Fokus-Datensatz
# (Startzeitpunkte 13.-27.07.2026, Horizont 48h -> abgedeckt 13.-29.07.2026)
# ueberlappen. Aus dem 128er-Sampler alle 2 Tage betrifft das:
DEV_JULY_WINDOWS = [
    "2026-07-14", "2026-07-16", "2026-07-18", "2026-07-20",
    "2026-07-22", "2026-07-24", "2026-07-26",
]

PAIRS = [
    ("AO_MAE", "persistence_MAE"),
    ("AO_MAE", "Oraku_MAE"),
    ("RO_MAE", "Oraku_MAE"),
    ("R_MAE",  "Oraku_MAE"),
    ("A_MAE",  "Oraku_MAE"),
    ("T_MAE",  "Oraku_MAE"),
    ("R_MAE",  "B_persist_MAE"),
    ("AO_MAE", "A_MAE"),
    ("RO_MAE", "R_MAE"),
    ("R_MAE",  "A_MAE"),
    ("R_MAE",  "T_MAE"),
    ("T_MAE",  "A_MAE"),
    ("AO_MAE", "RO_MAE"),
]

MODELS = ["T_MAE","A_MAE","R_MAE","B_persist_MAE","Oraku_MAE",
          "AO_MAE","RO_MAE","persistence_MAE","climatology_MAE"]


def wilcoxon_row(df, a, b, tag):
    d = df[a] - df[b]
    d_nonzero = d[d != 0]
    w, p = stats.wilcoxon(df[a], df[b], alternative="less",
                          zero_method="wilcox", method="auto")
    _, p_two = stats.wilcoxon(df[a], df[b], alternative="two-sided",
                              zero_method="wilcox", method="auto")
    cd = d.mean() / d.std() if d.std() > 0 else np.nan
    return {
        "subset": tag,
        "A": a.replace("_MAE",""), "B": b.replace("_MAE",""),
        "n": len(df),
        "wins_A_over_B": int((d < 0).sum()),
        "mean_diff": d.mean(),
        "wilcoxon_p_less": p,
        "wilcoxon_p_two_sided": p_two,
        "cohens_d_z": cd,
    }


def bootstrap_mean_ci(x, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=n, replace=True).mean()
    return np.percentile(means, [2.5, 97.5])


def moving_block_bootstrap_mean_ci(x, block_len, n_boot=10000, seed=42):
    """Circular moving block bootstrap fuer den Mittelwert."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block_len))
    means = np.empty(n_boot)
    idx_all = np.arange(n)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([
            np.take(x, np.arange(s, s + block_len) % n) for s in starts
        ])[:n]
        means[i] = sample.mean()
    return np.percentile(means, [2.5, 97.5])


def moving_block_bootstrap_diff_pvalue(a, b, block_len, n_boot=10000, seed=42):
    """Block-Bootstrap p-Wert: Anteil boot-Samples mit mean(a-b) >= 0.

    Testet H1: mean(a) < mean(b) analog Wilcoxon einseitig.
    """
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(d)
    n_blocks = int(np.ceil(n / block_len))
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([
            np.take(d, np.arange(s, s + block_len) % n) for s in starts
        ])[:n]
        boot_means[i] = sample.mean()
    # p = P(boot_mean >= 0) unter der beobachteten Verteilung, zentriert.
    # Alternative: Anteil der Bootstrap-Mittelwerte, die >= 0 sind.
    # Wir nehmen dies als einfache einseitige Approximation.
    return float((boot_means >= 0).mean())


def main():
    df = pd.read_csv(SUMMARY, parse_dates=["t0"]).sort_values("t0").reset_index(drop=True)
    df["month"] = df["t0"].dt.month
    df["season"] = df["month"].map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Fruehling",4:"Fruehling",5:"Fruehling",
        6:"Sommer",7:"Sommer",8:"Sommer",
        9:"Herbst",10:"Herbst",11:"Herbst"})
    print(f"Basis: n={len(df)} Fenster, {df['t0'].min().date()} .. {df['t0'].max().date()}")

    # =========================================================================
    # 1) Sensitivitaet: 121 Fenster (ohne 7 Juli-Entwicklungsfenster)
    # =========================================================================
    dev_dates = pd.to_datetime(DEV_JULY_WINDOWS)
    mask_dev = df["t0"].isin(dev_dates)
    print(f"\n[1] Sensitivitaet: entferne {mask_dev.sum()} Juli-2026-Fenster "
          f"(zeitliche Ueberlappung mit dem 15er Entwicklungs-Datensatz).")
    df_sens = df[~mask_dev].reset_index(drop=True)
    print(f"    Neu: n={len(df_sens)}")

    rows_s = []
    for a, b in PAIRS:
        rows_s.append(wilcoxon_row(df_sens, a, b, f"sensitivity_n{len(df_sens)}"))
    df_s = pd.DataFrame(rows_s)
    df_s.to_csv(SRC / "reviewer_sensitivity.csv", index=False)
    print("\n    Wilcoxon-Tests ohne Juli-Entwicklungsfenster:")
    print(df_s[["A","B","n","wins_A_over_B","mean_diff","wilcoxon_p_less","cohens_d_z"]]
          .to_string(index=False))

    # MAE-Ranking auf Sens-Subset
    print("\n    MAE-Ranking:")
    for m in MODELS:
        print(f"      {m.replace('_MAE',''):10s}  {df_sens[m].mean():.4f}   "
              f"(vs. n=128: {df[m].mean():.4f})")

    # =========================================================================
    # 2) Serielle Abhaengigkeit: 4d (jedes zweite), 8d (jedes vierte)
    # =========================================================================
    print("\n[2] Serielle Abhaengigkeit: ausgeduenntes Sampling.")
    thin_rows = []
    for step, tag in [(2, "thin_4d"), (4, "thin_8d")]:
        for offset in range(step):
            df_t = df.iloc[offset::step].reset_index(drop=True)
            for a, b in PAIRS:
                r = wilcoxon_row(df_t, a, b, f"{tag}_offset{offset}")
                r["step"] = step
                r["offset"] = offset
                thin_rows.append(r)
    df_thin = pd.DataFrame(thin_rows)
    df_thin.to_csv(SRC / "reviewer_thinning.csv", index=False)

    # Zusammenfassung: fuer jedes Paar geben wir die p-Werte ueber offsets aus.
    print("\n    Zweiseitige p-Werte bei ausgeduenntem Sampling (min/median/max ueber Offsets):")
    for step in [2, 4]:
        print(f"\n    Step={step} (Startabstand {step*2}d):")
        subset = df_thin[df_thin["step"] == step]
        for (a, b), grp in subset.groupby(["A","B"], sort=False):
            ps = grp["wilcoxon_p_two_sided"].values
            print(f"      {a:8s} vs {b:14s}  n={int(grp['n'].mean()):3d}  "
                  f"p in [{ps.min():.4g}; {ps.max():.4g}] median={np.median(ps):.4g}  "
                  f"meandiff={grp['mean_diff'].mean():+.4f}")

    # =========================================================================
    # 2b) Moving-Block-Bootstrap-CIs fuer die Hauptpaare
    # =========================================================================
    print("\n[2b] Moving-Block-Bootstrap-CIs (n=128) fuer Hauptpaare, Blocklaenge 3/5/7:")
    key_pairs = [("AO_MAE","Oraku_MAE"), ("A_MAE","Oraku_MAE"),
                 ("T_MAE","Oraku_MAE"), ("R_MAE","Oraku_MAE"),
                 ("AO_MAE","A_MAE"), ("AO_MAE","T_MAE"),
                 ("R_MAE","T_MAE")]
    block_rows = []
    for a, b in key_pairs:
        d = (df[a] - df[b]).to_numpy()
        row = {"A": a.replace("_MAE",""), "B": b.replace("_MAE",""),
               "n": len(d), "mean_diff": d.mean()}
        for L in [3, 5, 7]:
            lo, hi = moving_block_bootstrap_mean_ci(d, L, n_boot=5000, seed=42)
            row[f"block{L}_ci_lo"] = lo
            row[f"block{L}_ci_hi"] = hi
        block_rows.append(row)
    df_block = pd.DataFrame(block_rows)
    df_block.to_csv(SRC / "reviewer_block_bootstrap.csv", index=False)
    print(df_block.to_string(index=False))

    # =========================================================================
    # 3) Bias-Analyse auf allen 128 Fenstern, nach Saison
    # =========================================================================
    print("\n[3] Bias-Analyse auf hourly-Residuen (n=128 Fenster).")
    h = pd.read_csv(HOURLY, parse_dates=["t0","datetime"])
    h["month"] = h["t0"].dt.month
    h["season"] = h["month"].map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Fruehling",4:"Fruehling",5:"Fruehling",
        6:"Sommer",7:"Sommer",8:"Sommer",
        9:"Herbst",10:"Herbst",11:"Herbst"})

    bias_rows = []
    for scope, sub in [("Alle 128 Fenster", h)] + \
                      [(f"Nur {s}", h[h["season"]==s]) for s in ["Winter","Fruehling","Sommer","Herbst"]]:
        n_win = sub["t0"].nunique()
        n_hr = len(sub)
        for model_pred, tag in [("T_pred","T"),("A_pred","A"),
                                 ("R_pred","R"),("Oraku_pred","Oraku"),
                                 ("AO_pred","AO"),("RO_pred","RO")]:
            resid = sub[model_pred] - sub["truth"]
            bias_rows.append({
                "scope": scope,
                "n_windows": n_win,
                "n_hours": n_hr,
                "model": tag,
                "mean_bias": resid.mean(),
                "median_bias": resid.median(),
                "resid_std": resid.std(),
                "mean_abs_error": resid.abs().mean(),
            })
    df_bias = pd.DataFrame(bias_rows)
    df_bias.to_csv(SRC / "reviewer_bias_by_season.csv", index=False)

    print("\n    Bias-Signaturen (mean of predicted - truth):")
    print(df_bias[["scope","n_windows","model","mean_bias","median_bias","resid_std"]]
          .to_string(index=False))

    # Kompakte Zusammenfassung fuer den Text
    print("\n    Kompakte Bias-Uebersicht A vs Oraku (Sanity: ist der Sommer-Effekt saisonal?):")
    for scope in ["Alle 128 Fenster","Nur Winter","Nur Fruehling","Nur Sommer","Nur Herbst"]:
        row_A = df_bias[(df_bias["scope"]==scope) & (df_bias["model"]=="A")].iloc[0]
        row_O = df_bias[(df_bias["scope"]==scope) & (df_bias["model"]=="Oraku")].iloc[0]
        print(f"      {scope:22s}  A: {row_A['mean_bias']:+.3f} °C   "
              f"Oraku: {row_O['mean_bias']:+.3f} °C   "
              f"(gegenlaeufig? {'JA' if np.sign(row_A['mean_bias'])!=np.sign(row_O['mean_bias']) else 'NEIN'})")


if __name__ == "__main__":
    main()
