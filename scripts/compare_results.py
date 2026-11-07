# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/frozen_2026-10-14/scripts/compare_results.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Vergleich Original (n=128) vs. DST-korrigiert (n=128, gleiche Fenster) vs. korrigiert inkl. 28.03. (n=129).

Liest: paper_hywa/src/*.csv (Original), results/*.csv (neu), results/stats_*/ (neue Statistik).
Schreibt: results/compare_tables.json und results/compare_summary.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from paths import RESULTS as RES, V100_REFERENCE as SRC  # noqa: E402
TZ = "Europe/Zurich"
MODELS = ["T_MAE", "A_MAE", "R_MAE", "B_persist_MAE", "Oraku_MAE", "AO_MAE", "RO_MAE", "persistence_MAE", "climatology_MAE"]
PRIMARY = [("Q1 T vs Oraku", "T_MAE", "Oraku_MAE"), ("Q2a R vs B_persist", "R_MAE", "B_persist_MAE"),
           ("Q2b R vs T", "R_MAE", "T_MAE"), ("Q3a AO vs Oraku", "AO_MAE", "Oraku_MAE"), ("Q3b AO vs T", "AO_MAE", "T_MAE")]
EXTRA = [("RO vs Oraku", "RO_MAE", "Oraku_MAE"), ("A vs Oraku", "A_MAE", "Oraku_MAE"), ("R vs Oraku", "R_MAE", "Oraku_MAE"),
         ("AO vs A", "AO_MAE", "A_MAE"), ("T vs A", "T_MAE", "A_MAE"), ("AO vs RO", "AO_MAE", "RO_MAE")]
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Fruehling", 4: "Fruehling", 5: "Fruehling",
          6: "Sommer", 7: "Sommer", 8: "Sommer", 9: "Herbst", 10: "Herbst", 11: "Herbst"}

out = {}
lines = []


def P(s=""):
    print(s); lines.append(s)


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        run = max(run, p[k] * (m - i)); adj[k] = min(run, 1.0)
    return adj


def mbb_ci(d, L, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed); n = len(d); nb = int(np.ceil(n / L)); means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        means[i] = np.concatenate([np.take(d, np.arange(s, s + L) % n) for s in starts])[:n].mean()
    return np.percentile(means, [2.5, 97.5])


def boot_ci(x, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed); n = len(x)
    return np.percentile([rng.choice(x, n).mean() for _ in range(n_boot)], [2.5, 97.5])


def tost(d, m):
    n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    return max(1 - stats.t.cdf((d.mean() + m) / se, n - 1), stats.t.cdf((d.mean() - m) / se, n - 1))


def test_table(df, pairs, tag):
    rows = []
    for name, a, b in pairs:
        d = (df[a] - df[b]).to_numpy()
        rows.append({"test": name, "dataset": tag, "n": len(d), "delta": d.mean(), "median": float(np.median(d)),
                     "wins": int((d < 0).sum()), "p_two": stats.wilcoxon(d).pvalue,
                     "p_sign": stats.binomtest(int((d < 0).sum()), len(d)).pvalue,
                     "d_z": d.mean() / d.std(ddof=1), "mbb5_lo": mbb_ci(d, 5)[0], "mbb5_hi": mbb_ci(d, 5)[1]})
    t = pd.DataFrame(rows)
    prim = t[t.test.str.startswith("Q")].copy()
    t.loc[prim.index, "p_holm"] = holm(prim.p_two.to_numpy())
    return t


def ranking(df, tag):
    r = pd.DataFrame({"model": [m.replace("_MAE", "") for m in MODELS],
                      "MAE": [df[m].mean() for m in MODELS],
                      "ci_lo": [boot_ci(df[m].dropna().to_numpy())[0] for m in MODELS],
                      "ci_hi": [boot_ci(df[m].dropna().to_numpy())[1] for m in MODELS]}).sort_values("MAE")
    r["rank"] = range(1, len(r) + 1); r["dataset"] = tag
    return r


# ---------------------------------------------------------------------------
orig = pd.read_csv(SRC / "all_seasons_prev_runs_summary.csv", parse_dates=["t0"]).sort_values("t0")
new = pd.read_csv(RES / "all_seasons_prev_runs_dst_summary.csv", parse_dates=["t0"]).sort_values("t0")
new128 = new[new.t0.isin(orig.t0)].reset_index(drop=True)
assert len(new128) == len(orig) == 128, (len(new128), len(orig))
datasets = {"orig_n128": orig.reset_index(drop=True), "dst_n128": new128, "dst_n129": new.reset_index(drop=True)}
P("=" * 78); P("A  Fensterbestand"); P("=" * 78)
P(f"Original n={len(orig)}; DST-korrigiert gleiche Fenster n={len(new128)}; korrigiert inkl. 28.03. n={len(new)}")
P(f"Neu aufgenommen: {sorted(set(new.t0) - set(orig.t0))}; verloren: {sorted(set(orig.t0) - set(new.t0))}")
if "flow_source" in new.columns:
    P(f"Abflussquelle im korrigierten Lauf: {new.flow_source.value_counts().to_dict()}  "
      f"(persist = BAFU-Antwort leer, 24-h-Persistenz wie im Original)")
    P(f"  Fenster mit Persistenz-Fallback: {[d.strftime('%Y-%m-%d') for d in new[new.flow_source=='persist'].t0][:40]}")

# Ueberlappung 28.03/30.03
if "fc_end" in new.columns:
    r28 = new[new.t0 == pd.Timestamp("2026-03-28")]; r30 = new[new.t0 == pd.Timestamp("2026-03-30")]
    if len(r28) and len(r30):
        P(f"Fenster 28.03.: {r28.fc_start.iloc[0]} .. {r28.fc_end.iloc[0]};  Fenster 30.03.: {r30.fc_start.iloc[0]} .. {r30.fc_end.iloc[0]}")
        P("  -> letzter Zeitpunkt von 28.03. (30.03. 00:00 CEST) = erster Zeitpunkt von 30.03.: EIN gemeinsamer Messzeitpunkt (1 h Ueberlappung).")

# ---------------------------------------------------------------------------
P(); P("=" * 78); P("B  Mittlere MAE je Modell"); P("=" * 78)
mae_tab = pd.DataFrame({k: [v[m].mean() for m in MODELS] for k, v in datasets.items()}, index=[m.replace("_MAE", "") for m in MODELS])
mae_tab["delta_dst128_vs_orig"] = mae_tab["dst_n128"] - mae_tab["orig_n128"]
P(mae_tab.to_string(float_format=lambda x: f"{x:+.4f}" if abs(x) < 0.05 else f"{x:.4f}"))
out["mae"] = mae_tab.reset_index().rename(columns={"index": "model"}).to_dict("records")

ranks = pd.concat([ranking(v, k) for k, v in datasets.items()])
P(); P("Rangfolge:")
rk = ranks.pivot(index="model", columns="dataset", values="rank").loc[[m.replace("_MAE", "") for m in MODELS]]
P(rk.to_string())
out["ranking"] = ranks.to_dict("records")

# ---------------------------------------------------------------------------
P(); P("=" * 78); P("C  Primaertests und weitere Vergleiche"); P("=" * 78)
tests = pd.concat([test_table(v, PRIMARY + EXTRA, k) for k, v in datasets.items()])
pd.set_option("display.width", 220)
for name in [p[0] for p in PRIMARY + EXTRA]:
    sub = tests[tests.test == name][["dataset", "n", "delta", "median", "wins", "p_two", "p_holm", "p_sign", "d_z", "mbb5_lo", "mbb5_hi"]]
    P(f"\n{name}"); P(sub.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
out["tests"] = tests.to_dict("records")

# TOST AO vs T
P(); P("TOST AO vs T (p bei Grenze 0.02/0.03/0.05):")
for k, v in datasets.items():
    d = (v.AO_MAE - v.T_MAE).to_numpy()
    P(f"  {k}: mean={d.mean():+.4f}  p(0.02)={tost(d,0.02):.3f}  p(0.03)={tost(d,0.03):.3f}  p(0.05)={tost(d,0.05):.2e}")
    out.setdefault("tost", []).append({"dataset": k, "mean": d.mean(), "p002": tost(d, .02), "p003": tost(d, .03), "p005": tost(d, .05)})

# ---------------------------------------------------------------------------
P(); P("=" * 78); P("D  Saisonale MAE (dst_n129 vs orig)"); P("=" * 78)
for k in ["orig_n128", "dst_n129"]:
    v = datasets[k].copy(); v["season"] = v.t0.dt.month.map(SEASON)
    g = v.groupby("season")[["T_MAE", "A_MAE", "R_MAE", "Oraku_MAE", "AO_MAE", "RO_MAE"]].mean().loc[["Winter", "Fruehling", "Sommer", "Herbst"]]
    g["n"] = v.groupby("season").size()
    P(f"\n{k}"); P(g.to_string(float_format=lambda x: f"{x:.3f}"))
    out[f"season_{k}"] = g.reset_index().to_dict("records")
# saisonale Tests AO vs Oraku, T vs Oraku
P(); P("Saisonale Tests (p zweiseitig Wilcoxon):")
for k in ["orig_n128", "dst_n129"]:
    v = datasets[k].copy(); v["season"] = v.t0.dt.month.map(SEASON)
    for s in ["Winter", "Fruehling", "Sommer", "Herbst"]:
        g = v[v.season == s]
        P(f"  {k} {s:9s} n={len(g):2d}  AO-Oraku Δ={(g.AO_MAE-g.Oraku_MAE).mean():+.3f} p={stats.wilcoxon(g.AO_MAE-g.Oraku_MAE).pvalue:.2g}  "
          f"T-Oraku Δ={(g.T_MAE-g.Oraku_MAE).mean():+.3f} p={stats.wilcoxon(g.T_MAE-g.Oraku_MAE).pvalue:.2g}")

# ---------------------------------------------------------------------------
P(); P("=" * 78); P("E  Aenderung je Fenster (dst vs orig, gleiche 128 Fenster)"); P("=" * 78)
m = orig.merge(new128, on="t0", suffixes=("_o", "_n"))
m["cet"] = m.t0 < "2026-03-29"
for col in ["T_MAE", "A_MAE", "R_MAE", "B_persist_MAE", "AO_MAE", "RO_MAE"]:
    dcol = m[f"{col}_n"] - m[f"{col}_o"]
    P(f"{col:14s} Δ mean={dcol.mean():+.4f}  CET={dcol[m.cet].mean():+.4f} (n={int(m.cet.sum())})  CEST={dcol[~m.cet].mean():+.4f}  "
      f"|Δ|>0 in {int((dcol.abs()>1e-6).sum())} Fenstern  max|Δ|={dcol.abs().max():.3f}  Wilcoxon p(Δ≠0, CET)={stats.wilcoxon(dcol[m.cet]).pvalue:.3g}")
    out.setdefault("per_window_change", []).append({"model": col, "mean": dcol.mean(), "cet": dcol[m.cet].mean(), "cest": dcol[~m.cet].mean(),
                                                    "n_changed": int((dcol.abs() > 1e-6).sum()), "max_abs": dcol.abs().max(),
                                                    "p_cet": stats.wilcoxon(dcol[m.cet]).pvalue})
P(f"Oraku/truth identisch in allen Fenstern: Oraku max|Δ|={(m.Oraku_MAE_n-m.Oraku_MAE_o).abs().max():.2e}; "
  f"CEST-Fenster T max|Δ|={(m.T_MAE_n-m.T_MAE_o)[~m.cet].abs().max():.2e}")
top = m.assign(dT=m.T_MAE_n - m.T_MAE_o).sort_values("dT")
P("Groesste Verbesserung T (dst-orig): " + ", ".join(f"{r.t0.date()} {r.dT:+.3f}" for r in top.head(3).itertuples()))
P("Groesste Verschlechterung T:       " + ", ".join(f"{r.t0.date()} {r.dT:+.3f}" for r in top.tail(3).itertuples()))

# ---------------------------------------------------------------------------
P(); P("=" * 78); P("F  6-h-Delay-Test"); P("=" * 78)
d_old = pd.read_csv(SRC / "delay6h_prev_runs_summary.csv", parse_dates=["t0"])
d_new_p = RES / "delay6h_prev_runs_dst_summary.csv"
if d_new_p.exists():
    d_new = pd.read_csv(d_new_p, parse_dates=["t0"])
    for tag, dd in [("orig_n128", d_old), ("dst_n128", d_new[d_new.t0.isin(d_old.t0)]), ("dst_n129", d_new)]:
        P(f"\n{tag} (n={len(dd)}): MAE T={dd.T_MAE.mean():.3f} A={dd.A_MAE.mean():.3f} R={dd.R_MAE.mean():.3f} AO={dd.AO_MAE.mean():.3f} RO={dd.RO_MAE.mean():.3f} Oraku={dd.Oraku_MAE.mean():.3f}")
        for a in ["T_MAE", "A_MAE", "R_MAE", "AO_MAE", "RO_MAE"]:
            d = (dd[a] - dd.Oraku_MAE).to_numpy(); lo, hi = boot_ci(d)
            P(f"   {a[:-4]:3s}-Oraku Δ={d.mean():+.4f} CI=[{lo:+.4f},{hi:+.4f}] p={stats.wilcoxon(d).pvalue:.3g} TOST(0.05)={tost(d,0.05):.3f} d_z={d.mean()/d.std(ddof=1):+.2f}")
            out.setdefault("delay", []).append({"dataset": tag, "model": a[:-4], "delta": d.mean(), "lo": lo, "hi": hi,
                                                "p": stats.wilcoxon(d).pvalue, "tost05": tost(d, .05), "dz": d.mean() / d.std(ddof=1)})
        d = (dd.AO_MAE - dd.T_MAE).to_numpy()
        P(f"   AO-T (beide delay) Δ={d.mean():+.4f} p={stats.wilcoxon(d).pvalue:.2g}")
else:
    P("delay6h_prev_runs_dst_summary.csv fehlt noch")

# ---------------------------------------------------------------------------
P(); P("=" * 78); P("G  Historical-Forecast-Vergleich"); P("=" * 78)
h_old = pd.read_csv(SRC / "all_seasons_summary.csv", parse_dates=["t0"])
h_new_p = RES / "all_seasons_hist_dst_summary.csv"
if h_new_p.exists():
    h_new = pd.read_csv(h_new_p, parse_dates=["t0"])
    for tag, hh, pp in [("orig", h_old, orig), ("dst", h_new, new)]:
        mm = hh.merge(pp, on="t0", suffixes=("_hist", "_prev"))
        P(f"\n{tag}: gemeinsame Fenster {len(mm)}")
        for col in ["T_MAE", "A_MAE", "R_MAE", "AO_MAE", "RO_MAE", "Oraku_MAE"]:
            P(f"   {col:9s} hist={mm[f'{col}_hist'].mean():.4f} prev-runs={mm[f'{col}_prev'].mean():.4f} Δ={mm[f'{col}_prev'].mean()-mm[f'{col}_hist'].mean():+.4f}")
            out.setdefault("hist", []).append({"dataset": tag, "model": col[:-4], "hist": mm[f"{col}_hist"].mean(), "prev": mm[f"{col}_prev"].mean()})
        for a in ["T_MAE", "AO_MAE"]:
            d = (mm[f"{a}_hist"] - mm.Oraku_MAE_hist).to_numpy()
            P(f"   hist: {a[:-4]}-Oraku Δ={d.mean():+.4f} p={stats.wilcoxon(d).pvalue:.2g}")
else:
    P("all_seasons_hist_dst_summary.csv fehlt noch")

(RES / "compare_summary.txt").write_text("\n".join(lines))
json.dump(out, open(RES / "compare_tables.json", "w"), default=float, indent=1)
print("\n-> results/compare_summary.txt, results/compare_tables.json")
