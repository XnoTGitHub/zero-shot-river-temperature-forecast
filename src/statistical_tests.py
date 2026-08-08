"""Paarweise statistische Tests auf existierenden Fenster-MAEs.

Wilcoxon signed rank test (nonparametrisch, paired) und einfacher paired
t-test fuer alle kritischen Modellpaare aus aare_recursive_thun.csv
sowie baselines_per_window.csv.

Output: statistical_tests.csv mit p-Werten und Effect Sizes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent


def main():
    rt = pd.read_csv(BASE / "aare_recursive_thun.csv", parse_dates=["t0"])
    bl = pd.read_csv(BASE / "src/baselines_per_window.csv", parse_dates=["t0"])
    df = rt.merge(bl[["t0", "persistence_MAE", "climatology_MAE"]], on="t0", how="left")

    # Zu vergleichende Modell-Paare (Model_A besser bei negativer Differenz)
    pairs = [
        ("AO_MAE", "Oraku_MAE", "Ensemble(A,Oraku) vs. Oraku"),
        ("AO_MAE", "A_MAE", "Ensemble(A,Oraku) vs. A"),
        ("R_recursive_MAE", "Oraku_MAE", "R (recursive 2-stage) vs. Oraku"),
        ("R_recursive_MAE", "A_MAE", "R vs. A"),
        ("A_MAE", "Oraku_MAE", "A vs. Oraku"),
        ("RO_MAE", "Oraku_MAE", "Ensemble(R,Oraku) vs. Oraku"),
        ("AO_MAE", "RO_MAE", "Ensemble(A,Oraku) vs. Ensemble(R,Oraku)"),
        ("A_MAE", "persistence_MAE", "A vs. Persistenz"),
        ("Oraku_MAE", "persistence_MAE", "Oraku vs. Persistenz"),
        ("AO_MAE", "persistence_MAE", "Ensemble(A,Oraku) vs. Persistenz"),
    ]

    rows = []
    for a_col, b_col, label in pairs:
        a = df[a_col].dropna().to_numpy()
        b = df[b_col].dropna().to_numpy()
        # Restrict to paired non-null
        mask = df[a_col].notna() & df[b_col].notna()
        a = df.loc[mask, a_col].to_numpy()
        b = df.loc[mask, b_col].to_numpy()
        diff = a - b            # negative = a besser

        # Wilcoxon signed rank (H0: median(diff)=0)
        try:
            w_stat, w_p = stats.wilcoxon(a, b, alternative="less")
            _, w_p_two = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            w_stat, w_p, w_p_two = np.nan, np.nan, np.nan
        # Paired t-test (H0: mean(diff)=0)
        t_stat, t_p = stats.ttest_rel(a, b, alternative="less")
        # Effect size (Cohen's d fuer paired samples)
        d_paired = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
        # Win rate
        wins = int(np.sum(a < b))
        n = len(a)

        rows.append({
            "comparison": label,
            "n": n,
            "wins_A_over_B": wins,
            "mean_diff": float(np.mean(diff)),
            "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
            "wilcoxon_p_one_sided": float(w_p) if not np.isnan(w_p) else np.nan,
            "wilcoxon_p_two_sided": float(w_p_two) if not np.isnan(w_p_two) else np.nan,
            "ttest_stat": float(t_stat),
            "ttest_p_one_sided": float(t_p),
            "cohens_d_paired": float(d_paired),
        })

    dfs = pd.DataFrame(rows)
    out = BASE / "src/statistical_tests.csv"
    dfs.to_csv(out, index=False)

    print("=" * 90)
    print(f"Wilcoxon signed rank test (one-sided, H1: A < B) und paired t-test")
    print(f"n = 15 Fenster @ 48 h, Sommerperiode Juli 2026")
    print("=" * 90)
    for _, r in dfs.iterrows():
        sig_w = "***" if r["wilcoxon_p_one_sided"] < 0.001 else \
                "**" if r["wilcoxon_p_one_sided"] < 0.01 else \
                "*" if r["wilcoxon_p_one_sided"] < 0.05 else "n.s."
        print(f"  {r['comparison']:45s}"
              f"  wins={r['wins_A_over_B']:2d}/{r['n']}"
              f"  d={r['mean_diff']:+.4f}"
              f"  Wilcoxon p={r['wilcoxon_p_one_sided']:.4f} {sig_w}"
              f"  t-p={r['ttest_p_one_sided']:.4f}"
              f"  Cohen d={r['cohens_d_paired']:+.2f}")

    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
