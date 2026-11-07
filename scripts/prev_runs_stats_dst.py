# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/frozen_2026-10-14/scripts/prev_runs_stats_dst.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Vollstaendige statistische Auswertung des strikt-kausalen Primaertests
(Previous Runs API). Konsolidiert die Logik aus:
  all_seasons_stats.py, session18_holm.py, reviewer_robustness.py,
  session17_addons.py (TOST).

Input:
  paper_hywa/src/all_seasons_prev_runs_summary.csv
  paper_hywa/src/all_seasons_prev_runs_hourly.csv

Output:
  paper_hywa/src/prev_runs_ranking.csv        Modell-Ranking + Bootstrap-CI
  paper_hywa/src/prev_runs_tests.csv          Paarweise Wilcoxon
  paper_hywa/src/prev_runs_holm.csv           Holm-Korrektur ueber 5 Primaertests
  paper_hywa/src/prev_runs_block_bootstrap.csv Moving-Block-Bootstrap-CIs
  paper_hywa/src/prev_runs_tost.csv           TOST-Aequivalenz AO vs T
  paper_hywa/src/prev_runs_bias_by_season.csv Bias-Signaturen A vs Oraku
  paper_hywa/src/prev_runs_thinning.csv       Sensitivitaet 4d/8d Sampling
  paper_hywa/src/prev_runs_delay6h_note.txt   Kurznotiz zum Vergleich mit dem
                                              alten historical-forecast-Setup
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else "dst"
SUBSET = sys.argv[2] if len(sys.argv) > 2 else ""
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS as RES  # noqa: E402
SUMMARY = RES / f"all_seasons_prev_runs_{TAG}_summary.csv"
HOURLY = RES / f"all_seasons_prev_runs_{TAG}_hourly.csv"
OUT = RES / f"stats_{TAG}{SUBSET}"
OUT.mkdir(exist_ok=True)

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

PAIRS = [
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


def bootstrap_ci(x, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(x, size=n, replace=True))
    return np.percentile(means, [2.5, 97.5])


def wilcoxon_row(df, a, b, tag=""):
    d = df[a] - df[b]
    w, p = stats.wilcoxon(df[a], df[b], alternative="less",
                          zero_method="wilcox", method="auto")
    _, p_two = stats.wilcoxon(df[a], df[b], alternative="two-sided",
                              zero_method="wilcox", method="auto")
    cd = d.mean() / d.std() if d.std() > 0 else np.nan
    return {
        "subset": tag, "A": a.replace("_MAE",""), "B": b.replace("_MAE",""),
        "n": len(df), "wins_A_over_B": int((d < 0).sum()),
        "mean_diff": d.mean(),
        "wilcoxon_p_less": p, "wilcoxon_p_two_sided": p_two,
        "cohens_d_z": cd,
    }


def moving_block_bootstrap_mean_ci(x, block_len, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block_len))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([
            np.take(x, np.arange(s, s + block_len) % n) for s in starts
        ])[:n]
        means[i] = sample.mean()
    return np.percentile(means, [2.5, 97.5])


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, k in enumerate(order):
        val = p[k] * (m - i)
        running = max(running, val)
        adj[k] = min(running, 1.0)
    return adj


def main():
    df = pd.read_csv(SUMMARY, parse_dates=["t0"]).sort_values("t0").reset_index(drop=True)
    if SUBSET == "_n128":
        df = df[df["t0"] != pd.Timestamp("2026-03-28")].reset_index(drop=True)
    df["month"] = df["t0"].dt.month
    df["season"] = df["month"].map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Fruehling",4:"Fruehling",5:"Fruehling",
        6:"Sommer",7:"Sommer",8:"Sommer",
        9:"Herbst",10:"Herbst",11:"Herbst"})
    print(f"n Fenster: {len(df)}   Zeitraum: {df['t0'].min().date()} .. {df['t0'].max().date()}")
    print(f"Saison-Verteilung: {df.groupby('season').size().to_dict()}")

    # ------------------------------------------------------------------
    # 1) Ranking + Bootstrap-CI
    # ------------------------------------------------------------------
    rows = []
    for m in MODELS:
        vals = df[m].dropna().to_numpy()
        lo, hi = bootstrap_ci(vals)
        rows.append({"model": m.replace("_MAE",""),
                     "label": LABEL_MAP[m],
                     "MAE": vals.mean(),
                     "median": np.median(vals),
                     "CI_lo_95": lo, "CI_hi_95": hi,
                     "n": len(vals), "std": vals.std()})
    df_rank = pd.DataFrame(rows).sort_values("MAE")
    df_rank.to_csv(OUT / "prev_runs_ranking.csv", index=False)
    print("\n[1] Ranking (aufsteigend nach MAE):")
    print(df_rank[["label","MAE","median","CI_lo_95","CI_hi_95"]].to_string(index=False,
        float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------
    # 2) Paarweise Wilcoxon (alle 15 Vergleiche)
    # ------------------------------------------------------------------
    test_rows = [wilcoxon_row(df, a, b, f"primary_n{len(df)}") for a, b in PAIRS]
    df_tests = pd.DataFrame(test_rows)
    df_tests.to_csv(OUT / "prev_runs_tests.csv", index=False)
    print("\n[2] Paarweise Wilcoxon (Auszug):")
    print(df_tests[["A","B","wins_A_over_B","mean_diff",
                    "wilcoxon_p_two_sided","cohens_d_z"]].to_string(index=False,
                    float_format=lambda x: f"{x:.4g}"))

    # ------------------------------------------------------------------
    # 3) Holm-Korrektur ueber die 5 Primaertests
    # ------------------------------------------------------------------
    primary_keys = [
        ("Q1  T vs Oraku",       ("T",  "Oraku")),
        ("Q2a R vs B_persist",   ("R",  "B_persist")),
        ("Q2b R vs T",           ("R",  "T")),
        ("Q3a AO vs Oraku",      ("AO", "Oraku")),
        ("Q3b AO vs T",          ("AO", "T")),
    ]
    lookup = {(r["A"], r["B"]): r for _, r in df_tests.iterrows()}
    holm_rows = []
    for label, key in primary_keys:
        r = lookup[key]
        holm_rows.append({
            "question": label,
            "comparison": f"{key[0]} vs {key[1]}",
            "n": int(r["n"]),
            "delta_MAE": float(r["mean_diff"]),
            "p_two_sided_raw": float(r["wilcoxon_p_two_sided"]),
        })
    df_holm = pd.DataFrame(holm_rows)
    df_holm["p_holm_adj"] = holm_adjust(df_holm["p_two_sided_raw"].to_numpy())
    df_holm["sig_raw_alpha05"] = df_holm["p_two_sided_raw"] < 0.05
    df_holm["sig_holm_alpha05"] = df_holm["p_holm_adj"] < 0.05
    df_holm.to_csv(OUT / "prev_runs_holm.csv", index=False)
    print("\n[3] Holm-Korrektur (5 Primaertests):")
    with pd.option_context("display.float_format", "{:.3e}".format):
        print(df_holm.to_string(index=False))

    # ------------------------------------------------------------------
    # 4) Moving-Block-Bootstrap-CIs fuer Kernpaare
    # ------------------------------------------------------------------
    key_pairs = [("AO_MAE","Oraku_MAE"), ("A_MAE","Oraku_MAE"),
                 ("T_MAE","Oraku_MAE"), ("R_MAE","Oraku_MAE"),
                 ("AO_MAE","A_MAE"), ("AO_MAE","T_MAE"),
                 ("R_MAE","T_MAE"), ("RO_MAE","Oraku_MAE"),
                 ("RO_MAE","T_MAE")]
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
    df_block.to_csv(OUT / "prev_runs_block_bootstrap.csv", index=False)
    print("\n[4] Moving-Block-Bootstrap 95%-CIs:")
    print(df_block.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------
    # 5) TOST-Aequivalenztest AO vs T
    # ------------------------------------------------------------------
    diffs = (df["AO_MAE"] - df["T_MAE"]).to_numpy()
    n = len(diffs); mean_diff = float(diffs.mean()); sd = float(diffs.std(ddof=1))
    se = sd / np.sqrt(n)
    tost_rows = []
    for m in [0.02, 0.03, 0.05, 0.10]:
        t_lower = (mean_diff - (-m)) / se
        p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
        t_upper = (mean_diff - (+m)) / se
        p_upper = stats.t.cdf(t_upper, df=n - 1)
        p_tost = max(p_lower, p_upper)
        tost_rows.append({
            "margin_degC": m, "n": n,
            "mean_diff_AO_minus_T": mean_diff, "sd": sd,
            "p_tost_parametric": p_tost,
            "equivalence_alpha05": p_tost < 0.05,
        })
    df_tost = pd.DataFrame(tost_rows)
    df_tost.to_csv(OUT / "prev_runs_tost.csv", index=False)
    print(f"\n[5] TOST AO vs T  (mean_diff = {mean_diff:+.4f} degC):")
    print(df_tost.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ------------------------------------------------------------------
    # 6) Bias-Analyse (nach Saison, stuendliche Residuen)
    # ------------------------------------------------------------------
    h = pd.read_csv(HOURLY, parse_dates=["t0"])
    if SUBSET == "_n128":
        h = h[h["t0"] != pd.Timestamp("2026-03-28")]
    h["month"] = h["t0"].dt.month
    h["season"] = h["month"].map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Fruehling",4:"Fruehling",5:"Fruehling",
        6:"Sommer",7:"Sommer",8:"Sommer",
        9:"Herbst",10:"Herbst",11:"Herbst"})
    bias_rows = []
    for scope, sub in [(f"Alle {h.t0.nunique()} Fenster", h)] + \
                      [(f"Nur {s}", h[h["season"]==s]) for s in ["Winter","Fruehling","Sommer","Herbst"]]:
        n_win = sub["t0"].nunique()
        for model_pred, tag in [("T_pred","T"),("A_pred","A"),
                                 ("R_pred","R"),("Oraku_pred","Oraku"),
                                 ("AO_pred","AO"),("RO_pred","RO")]:
            resid = sub[model_pred] - sub["truth"]
            bias_rows.append({
                "scope": scope, "n_windows": n_win, "n_hours": len(sub),
                "model": tag,
                "mean_bias": resid.mean(),
                "median_bias": resid.median(),
                "resid_std": resid.std(),
                "mean_abs_error": resid.abs().mean(),
            })
    df_bias = pd.DataFrame(bias_rows)
    df_bias.to_csv(OUT / "prev_runs_bias_by_season.csv", index=False)
    print("\n[6] Bias-Signaturen (mean of predicted - truth):")
    print(df_bias[["scope","n_windows","model","mean_bias","resid_std"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # ------------------------------------------------------------------
    # 7) Serielle Sensitivitaet (4d, 8d Sampling)
    # ------------------------------------------------------------------
    thin_rows = []
    for step, tag in [(2, "thin_4d"), (4, "thin_8d")]:
        for offset in range(step):
            df_t = df.iloc[offset::step].reset_index(drop=True)
            for a, b in PAIRS:
                r = wilcoxon_row(df_t, a, b, f"{tag}_offset{offset}")
                r["step"] = step; r["offset"] = offset
                thin_rows.append(r)
    df_thin = pd.DataFrame(thin_rows)
    df_thin.to_csv(OUT / "prev_runs_thinning.csv", index=False)
    print("\n[7] Serielle Sensitivitaet (min/median/max ueber Offsets, zweiseitig):")
    for step in [2, 4]:
        subset = df_thin[df_thin["step"] == step]
        print(f"\n   Step={step} (Startabstand {step*2}d):")
        for (a, b), grp in subset.groupby(["A","B"], sort=False):
            ps = grp["wilcoxon_p_two_sided"].values
            print(f"     {a:8s} vs {b:14s}  n={int(grp['n'].mean()):3d}  "
                  f"p [{ps.min():.3g}; {ps.max():.3g}]  meandiff {grp['mean_diff'].mean():+.4f}")

    # ------------------------------------------------------------------
    # 8) Vergleich mit dem alten (historical-forecast) Setup, falls vorhanden
    # ------------------------------------------------------------------
    old = RES / f"all_seasons_hist_{TAG}_summary.csv"
    note_lines = []
    note_lines.append("PREVIOUS-RUNS-vs-HISTORICAL-FORECAST VERGLEICH  (dieselben 128 Fenster)")
    note_lines.append("=" * 72)
    if old.exists():
        df_old = pd.read_csv(old, parse_dates=["t0"])
        merged = df_old.merge(df, on="t0", suffixes=("_hist", "_prev"))
        note_lines.append(f"gemeinsame Fenster: {len(merged)}")
        for m in ["T_MAE","A_MAE","R_MAE","Oraku_MAE","AO_MAE","RO_MAE"]:
            h_m = merged[f"{m}_hist"].mean()
            p_m = merged[f"{m}_prev"].mean()
            note_lines.append(f"  {m:12s}  hist={h_m:.4f}   prev-runs={p_m:.4f}   "
                              f"delta={p_m - h_m:+.4f} degC")
        note_lines.append("\nInterpretation: identische Fenster, dieselbe TiRex-Pipeline,")
        note_lines.append("einzige Aenderung: Wetter-Kovariaten aus previous_day1/day2 statt")
        note_lines.append("historical-forecast. Kleine MAE-Aenderungen sind erwartbar, weil")
        note_lines.append("die Prognose fuer den Horizont nun aus einem 24-48h aelteren")
        note_lines.append("Modelllauf stammt (leicht groessere Kovariat-Ungenauigkeit).")
    else:
        note_lines.append("Hinweis: all_seasons_summary.csv (Original) nicht gefunden.")
    (OUT / "prev_runs_delay6h_note.txt").write_text("\n".join(note_lines))
    print("\n[8] " + note_lines[0])
    for line in note_lines[1:]:
        print("   " + line)


if __name__ == "__main__":
    main()
