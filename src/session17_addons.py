"""Session 17 add-ons: T ins Sommer-Ranking, TOST-Aequivalenz, nicht-zirkulaerer Block-Bootstrap.

Erzeugt (Ueberschreibt teilweise):
- baselines_summary.csv                    (jetzt inkl. T-Zeile)
- summer_extra_tests.csv                   (Wilcoxon fuer 15 Sommer-Fenster inkl. T-Vergleiche)
- tost_ao_vs_t.csv                         (TOST-Aequivalenztest AO vs T, 128 Fenster)
- reviewer_block_bootstrap_noncirc.csv     (nicht-zirkulaerer Bootstrap, 128 Fenster)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).parent


def mae(pred, truth):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(truth))))


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


# =============================================================================
# 1) T in Sommer-Ranking (15 Fenster)
# =============================================================================
def summer_ranking_with_T():
    pw = pd.read_csv(BASE / "predictions_wide.csv")
    per_win = (pw.groupby("t0")
                 .apply(lambda g: pd.Series({
                     "T_MAE": mae(g["T_pred"], g["truth"]),
                     "A_MAE": mae(g["A_pred"], g["truth"]),
                 }))
                 .reset_index())
    per_win.to_csv(BASE / "summer_T_per_window.csv", index=False)

    # Bootstrap-CI fuer T
    T_maes = per_win["T_MAE"].to_numpy()
    lo, hi = bootstrap_ci(T_maes)
    sem = float(np.std(T_maes, ddof=1) / np.sqrt(len(T_maes)))
    T_row = pd.DataFrame([{
        "model": "T", "n": len(T_maes), "MAE": float(T_maes.mean()),
        "CI_lo_95": lo, "CI_hi_95": hi, "SEM": sem,
    }])

    # Bestehende baselines_summary.csv erweitern
    bs = pd.read_csv(BASE / "baselines_summary.csv")
    bs = bs[bs["model"] != "T"]  # falls schon vorhanden
    bs = pd.concat([bs, T_row], ignore_index=True).sort_values("MAE").reset_index(drop=True)
    bs.to_csv(BASE / "baselines_summary.csv", index=False)
    print("[1] Sommer-Ranking mit T (Bootstrap-CI):")
    print(bs.to_string(index=False))
    return per_win


def summer_wilcoxon_with_T(per_win_T):
    """Wilcoxon Vergleiche mit T in den 15 Sommer-Fenstern."""
    ap = pd.read_csv(BASE / "all_models_per_window.csv")
    # Merge T aus predictions_wide-Aggregat
    df = ap.merge(per_win_T[["t0", "T_MAE"]], on="t0", how="inner")
    assert len(df) == 15, f"expected 15, got {len(df)}"

    def wilcox(name, a, b):
        diffs = a - b
        two = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        one = stats.wilcoxon(diffs, alternative="less", zero_method="wilcox")  # A < B
        # Cohens d_z
        dz = float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else np.nan
        return {
            "comparison": name,
            "n": len(diffs),
            "wins_A": int((diffs < 0).sum()),
            "delta_MAE": float(np.mean(diffs)),
            "p_two_sided": float(two.pvalue),
            "p_one_sided_less": float(one.pvalue),
            "cohens_dz": dz,
        }

    tests = pd.DataFrame([
        wilcox("T vs Oraku",  df["T_MAE"].to_numpy(), df["Oraku_MAE"].to_numpy()),
        wilcox("T vs A",      df["T_MAE"].to_numpy(), df["A_MAE"].to_numpy()),
        wilcox("R vs T",      df["R_recursive_MAE"].to_numpy(), df["T_MAE"].to_numpy()),
        wilcox("AO vs T",     df["AO_MAE"].to_numpy(), df["T_MAE"].to_numpy()),
        wilcox("RO vs T",     df["RO_MAE"].to_numpy(), df["T_MAE"].to_numpy()),
    ])
    tests.to_csv(BASE / "summer_extra_tests.csv", index=False)
    print("\n[1b] Sommer-Wilcoxon-Tests mit T (n=15):")
    print(tests.to_string(index=False))
    return tests


# =============================================================================
# 2) TOST-Aequivalenztest AO vs T auf 128 Fenster
# =============================================================================
def tost_ao_vs_t():
    """Two one-sided tests: teste ob |mean(AO_MAE - T_MAE)| < margin.

    Praxisgrenze: 0.05 degC (aus Reviewer-Vorschlag).
    """
    df = pd.read_csv(BASE / "all_seasons_summary.csv")
    df = df[df["status"] == "ok"] if "status" in df.columns else df

    # Fallback: nutze all_seasons_hourly falls summary keine T-Spalte hat
    if "T_MAE" not in df.columns:
        h = pd.read_csv(BASE / "all_seasons_hourly.csv")
        agg = h.groupby("t0").apply(lambda g: pd.Series({
            "T_MAE": mae(g["T_pred"], g["truth"]),
            "AO_MAE": mae(g["AO_pred"], g["truth"]),
        })).reset_index()
    else:
        agg = df[["t0", "T_MAE", "AO_MAE"]].copy()

    diffs = agg["AO_MAE"].to_numpy() - agg["T_MAE"].to_numpy()
    n = len(diffs)
    mean_diff = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    se = sd / np.sqrt(n)

    margins = [0.02, 0.03, 0.05, 0.10]
    rows = []
    for m in margins:
        # H0_lower: mean_diff <= -m; H0_upper: mean_diff >= +m
        # TOST rejects H0 iff both one-sided t-tests reject.
        t_lower = (mean_diff - (-m)) / se        # test mean > -m
        p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
        t_upper = (mean_diff - (+m)) / se        # test mean < +m
        p_upper = stats.t.cdf(t_upper, df=n - 1)
        p_tost = max(p_lower, p_upper)
        equiv = p_tost < 0.05

        # Non-parametric analog: Bootstrap-CI der mean_diff, check ob 90%-CI in (-m, +m)
        rng = np.random.default_rng(42)
        boots = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(10000)])
        boot_lo, boot_hi = float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))
        equiv_boot = (boot_lo > -m) and (boot_hi < +m)

        rows.append({
            "margin_degC": m,
            "n": n,
            "mean_diff_AO_minus_T": mean_diff,
            "sd": sd,
            "p_tost_parametric": p_tost,
            "equivalence_parametric_alpha05": equiv,
            "boot_90CI_lo": boot_lo, "boot_90CI_hi": boot_hi,
            "equivalence_boot_90CI": equiv_boot,
        })
    tost = pd.DataFrame(rows)
    tost.to_csv(BASE / "tost_ao_vs_t.csv", index=False)
    print(f"\n[2] TOST AO vs T (128 Fenster; mean_diff = {mean_diff:+.4f} degC):")
    print(tost.to_string(index=False))
    return tost


# =============================================================================
# 3) Nicht-zirkulaerer Block-Bootstrap
# =============================================================================
def noncirc_block_bootstrap(pairs=None, blocks=(3, 5, 7), n_boot=5000):
    """Compare circular vs non-circular moving-block bootstrap for key pairs.

    Sampled per pair over 128 Fenster-MAE-Differenzen.
    """
    df = pd.read_csv(BASE / "all_seasons_summary.csv")
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    # Fallback: aus hourly aggregieren, falls Spalten fehlen
    needed_cols = ["AO_MAE", "T_MAE", "A_MAE", "R_MAE", "Oraku_MAE"]
    if not all(c in df.columns for c in needed_cols):
        h = pd.read_csv(BASE / "all_seasons_hourly.csv")
        df = h.groupby("t0").apply(lambda g: pd.Series({
            "T_MAE":  mae(g["T_pred"], g["truth"]),
            "A_MAE":  mae(g["A_pred"], g["truth"]),
            "R_MAE":  mae(g["R_pred"], g["truth"]),
            "Oraku_MAE": mae(g["Oraku_pred"], g["truth"]),
            "AO_MAE": mae(g["AO_pred"], g["truth"]),
        })).reset_index()

    df = df.sort_values("t0").reset_index(drop=True)
    n = len(df)

    def get(col_a, col_b):
        return (df[col_a] - df[col_b]).to_numpy()

    if pairs is None:
        pairs = [
            ("AO vs Oraku", "AO_MAE", "Oraku_MAE"),
            ("T vs Oraku",  "T_MAE",  "Oraku_MAE"),
            ("A vs Oraku",  "A_MAE",  "Oraku_MAE"),
            ("R vs Oraku",  "R_MAE",  "Oraku_MAE"),
            ("AO vs A",     "AO_MAE", "A_MAE"),
            ("AO vs T",     "AO_MAE", "T_MAE"),
            ("R vs T",      "R_MAE",  "T_MAE"),
        ]

    def circular_block_boot(x, block, n_boot, seed):
        rng = np.random.default_rng(seed)
        n = len(x)
        n_blocks = int(np.ceil(n / block))
        means = np.empty(n_boot)
        for b in range(n_boot):
            starts = rng.integers(0, n, size=n_blocks)
            picks = []
            for s in starts:
                picks.extend([(s + i) % n for i in range(block)])
            means[b] = x[picks[:n]].mean()
        return means

    def noncirc_block_boot(x, block, n_boot, seed):
        rng = np.random.default_rng(seed)
        n = len(x)
        max_start = n - block + 1  # nur Bloecke die intern zu Ende passen
        n_blocks = int(np.ceil(n / block))
        means = np.empty(n_boot)
        for b in range(n_boot):
            starts = rng.integers(0, max_start, size=n_blocks)
            picks = []
            for s in starts:
                picks.extend(range(s, s + block))
            means[b] = x[picks[:n]].mean()
        return means

    rows = []
    for name, ca, cb in pairs:
        diffs = get(ca, cb)
        for block in blocks:
            m_circ = circular_block_boot(diffs, block, n_boot, seed=42)
            m_nonc = noncirc_block_boot(diffs, block, n_boot, seed=42)
            rows.append({
                "comparison": name,
                "block": block,
                "n": len(diffs),
                "mean_diff": float(diffs.mean()),
                "circular_CI_lo": float(np.quantile(m_circ, 0.025)),
                "circular_CI_hi": float(np.quantile(m_circ, 0.975)),
                "noncirc_CI_lo": float(np.quantile(m_nonc, 0.025)),
                "noncirc_CI_hi": float(np.quantile(m_nonc, 0.975)),
            })
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "reviewer_block_bootstrap_noncirc.csv", index=False)
    print(f"\n[3] Nicht-zirkulaerer Block-Bootstrap (Vergleich zu zirkulaer, n={n}):")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    per_win_T = summer_ranking_with_T()
    summer_wilcoxon_with_T(per_win_T)
    tost_ao_vs_t()
    noncirc_block_bootstrap()
    print("\nDone.")
