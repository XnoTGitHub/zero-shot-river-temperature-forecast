"""Statistische Auswertung des 128-Fenster-Ganzjahres-Tests.

Output:
  src/all_seasons_ranking.csv  — Modell x MAE x Bootstrap-CI-95
  src/all_seasons_tests.csv    — Paarweise Wilcoxon
  src/all_seasons_by_season.csv — Saison-Zeilen pro Modell
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "src"

MODELS = ["T_MAE","A_MAE","R_MAE","B_persist_MAE","Oraku_MAE",
          "AO_MAE","RO_MAE","persistence_MAE","climatology_MAE"]

LABEL_MAP = {
    "T_MAE": "T: TiRex-2 (nur tt)",
    "A_MAE": "A: TiRex-2 (tt+flow+wind)",
    "R_MAE": "R: rekursive 2-Stage",
    "B_persist_MAE": "B_persist: TiRex-2 + Thun (Persistenz)",
    "Oraku_MAE": "Aare Oraku",
    "AO_MAE": "AO: Ensemble (A, Oraku)",
    "RO_MAE": "RO: Ensemble (R, Oraku)",
    "persistence_MAE": "Persistenz",
    "climatology_MAE": "Klimatologie",
}


def bootstrap_ci(x, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(x, size=n, replace=True))
    return np.percentile(means, [2.5, 97.5])


def main():
    df = pd.read_csv(SRC / "all_seasons_summary.csv", parse_dates=["t0"])
    df["month"] = df["t0"].dt.month
    df["season"] = df["month"].map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Fruehling",4:"Fruehling",5:"Fruehling",
        6:"Sommer",7:"Sommer",8:"Sommer",
        9:"Herbst",10:"Herbst",11:"Herbst"})
    print(f"n Fenster: {len(df)}   Zeitraum: {df['t0'].min().date()} .. {df['t0'].max().date()}")

    # --- Ranking mit Bootstrap-CI ---
    rows = []
    for m in MODELS:
        vals = df[m].dropna().to_numpy()
        lo, hi = bootstrap_ci(vals)
        rows.append({"model": m.replace("_MAE",""),
                     "label": LABEL_MAP[m],
                     "MAE": vals.mean(),
                     "median": np.median(vals),
                     "CI_lo_95": lo, "CI_hi_95": hi,
                     "n": len(vals),
                     "std": vals.std()})
    df_rank = pd.DataFrame(rows).sort_values("MAE")
    df_rank.to_csv(SRC / "all_seasons_ranking.csv", index=False)
    print("\nRanking:")
    print(df_rank[["label","MAE","CI_lo_95","CI_hi_95"]].to_string(index=False))

    # --- Paarweise Wilcoxon ---
    pairs = [
        ("AO_MAE", "persistence_MAE"),
        ("AO_MAE", "Oraku_MAE"),
        ("RO_MAE", "Oraku_MAE"),
        ("R_MAE",  "Oraku_MAE"),
        ("A_MAE",  "Oraku_MAE"),
        ("T_MAE",  "Oraku_MAE"),
        ("R_MAE",  "B_persist_MAE"),
        ("AO_MAE", "A_MAE"),
        ("AO_MAE", "T_MAE"),
        ("RO_MAE", "R_MAE"),
        ("RO_MAE", "T_MAE"),
        ("R_MAE",  "A_MAE"),
        ("R_MAE",  "T_MAE"),
        ("T_MAE",  "A_MAE"),
        ("AO_MAE", "RO_MAE"),
    ]
    test_rows = []
    for a, b in pairs:
        d = df[a] - df[b]
        w, p = stats.wilcoxon(df[a], df[b], alternative="less")
        _, p_two = stats.wilcoxon(df[a], df[b], alternative="two-sided")
        cd = d.mean() / d.std()
        test_rows.append({
            "A": a.replace("_MAE",""), "B": b.replace("_MAE",""),
            "n": len(df),
            "wins_A_over_B": int((d < 0).sum()),
            "mean_diff": d.mean(),
            "wilcoxon_p_less": p,
            "wilcoxon_p_two_sided": p_two,
            "cohens_d": cd,
        })
    df_tests = pd.DataFrame(test_rows)
    df_tests.to_csv(SRC / "all_seasons_tests.csv", index=False)
    print("\nPaarweise Tests:")
    print(df_tests[["A","B","wins_A_over_B","mean_diff","wilcoxon_p_less","cohens_d"]].to_string(index=False))

    # --- Nach Saison ---
    season_rows = []
    for season in ["Winter","Fruehling","Sommer","Herbst"]:
        sub = df[df["season"] == season]
        if len(sub) < 3: continue
        for m in MODELS:
            vals = sub[m].dropna().to_numpy()
            season_rows.append({
                "season": season,
                "n": len(vals),
                "model": m.replace("_MAE",""),
                "MAE": vals.mean(),
                "median": np.median(vals),
                "std": vals.std(),
                "truth_mean_season": sub["truth_mean"].mean(),
            })
    df_season = pd.DataFrame(season_rows)
    df_season.to_csv(SRC / "all_seasons_by_season.csv", index=False)

    print("\nNach Saison — Bestes Ensemble je Saison:")
    for season in ["Winter","Fruehling","Sommer","Herbst"]:
        sub = df_season[df_season["season"] == season]
        if len(sub) == 0: continue
        best = sub.sort_values("MAE").iloc[0]
        print(f"  {season} (n={best['n']}, T={best['truth_mean_season']:.1f}°C):  "
              f"best = {best['model']} @ {best['MAE']:.3f}")

    print(f"\nOutput: all_seasons_ranking.csv, all_seasons_tests.csv, all_seasons_by_season.csv")


if __name__ == "__main__":
    main()
