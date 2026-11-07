# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/phase2a/metrics_hydro.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Z1 – Hydrologische Kennzahlen aus dem eingefrorenen Datensatz (n = 129).

Liest ausschliesslich review_runde1/frozen_2026-10-14/results/.
Schreibt nach phase2a/tables/ (CSV) und phase2a/tables/metrics.json.
Keine Modellläufe. Keine Änderung an Originaldaten.

Statistik-Konvention:
  - "gesichert": gepaarte Fensterdifferenzen, Wilcoxon zweiseitig + Vorzeichentest +
    zirkulärer Moving-Block-Bootstrap (Blocklänge 5, 5000 Replikate) auf Fensterebene,
    damit Stunden/Tage desselben Fensters nicht als unabhängig gezählt werden.
  - alles ohne diese Absicherung ist "deskriptiv/explorativ" und so gekennzeichnet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from paths import RESULTS as FROZEN, TABLES as OUT  # noqa: E402
OUT.mkdir(parents=True, exist_ok=True)
TZ = "Europe/Zurich"
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Fruehling", 4: "Fruehling", 5: "Fruehling",
          6: "Sommer", 7: "Sommer", 8: "Sommer", 9: "Herbst", 10: "Herbst", 11: "Herbst"}
SEASONS = ["Winter", "Fruehling", "Sommer", "Herbst", "ALLE"]
MODELS = {"Oraku": "Oraku_pred", "T": "T_pred", "A": "A_pred", "R": "R_pred", "AO": "AO_pred", "RO": "RO_pred"}
MAIN = ["Oraku", "T", "AO"]
THRESHOLDS = [20.0, 22.0]  # illustrativ, ohne oekologische/regulatorische Interpretation
log = []


def P(s=""):
    print(s, flush=True); log.append(s)


def mbb_ci(d, L=5, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed); d = np.asarray(d, float); n = len(d); nb = int(np.ceil(n / L))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        means[i] = np.concatenate([np.take(d, np.arange(s, s + L) % n) for s in starts])[:n].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(d):
    """Gepaarte Fensterdifferenzen -> Kennzahlen inkl. Unsicherheit."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 6:
        return {"n": int(len(d)), "delta": float(np.mean(d)) if len(d) else np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_wilcoxon": np.nan, "p_sign": np.nan, "wins": int((d < 0).sum())}
    lo, hi = mbb_ci(d)
    return {"n": int(len(d)), "delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "p_sign": float(stats.binomtest(int((d < 0).sum()), int((d != 0).sum()) or 1).pvalue),
            "wins": int((d < 0).sum())}


# ---------------------------------------------------------------------------
h = pd.read_csv(FROZEN / "all_seasons_prev_runs_dst_hourly.csv", parse_dates=["t0"])
h["dt"] = pd.to_datetime(h["datetime"], utc=True).dt.tz_convert(TZ)
h["season"] = h.t0.dt.month.map(SEASON)
h["hod"] = h.dt.dt.hour
h["day"] = h.dt.dt.date
s = pd.read_csv(FROZEN / "all_seasons_prev_runs_dst_summary.csv", parse_dates=["t0"]).sort_values("t0").reset_index(drop=True)
s["season"] = s.t0.dt.month.map(SEASON)
P(f"Datenbasis: {len(s)} Fenster, {len(h)} Stundenwerte, {h.dt.min()} .. {h.dt.max()}")

# ---------------------------------------------------------------------------
P("\n[0] Konsistenzpruefung: MAE aus Stundendaten vs. eingefrorene Fenster-MAE")
chk = h.groupby("t0").apply(lambda g: pd.Series({m: (g[c] - g.truth).abs().mean() for m, c in MODELS.items()}), include_groups=False)
mx = 0.0
for m in MODELS:
    diff = (chk[m] - s.set_index("t0")[f"{m}_MAE"]).abs().max()
    mx = max(mx, diff); P(f"  {m:6s} max |Δ| = {diff:.2e}")
assert mx < 1e-6, "MAE-Werte stimmen nicht mit dem eingefrorenen Stand ueberein"
P(f"  -> identisch bis auf CSV-Rundung der float32-Prognosen (max {mx:.1e} degC) [OK]")

# ---------------------------------------------------------------------------
P("\n[1] Tab. I – Guetemasse je Modell und Saison (Stundenwerte) + gepaarte Fensterdifferenzen vs. Oraku")
rows, diffs = [], []
for se in SEASONS:
    g = h if se == "ALLE" else h[h.season == se]
    gs = s if se == "ALLE" else s[s.season == se]
    for m, c in MODELS.items():
        e = g[c] - g.truth
        rows.append({"season": se, "model": m, "n_windows": int(gs.shape[0]), "n_hours": int(len(g)),
                     "MAE": e.abs().mean(), "bias": e.mean(), "RMSE": float(np.sqrt((e ** 2).mean())),
                     "r": float(np.corrcoef(g[c], g.truth)[0, 1]), "share_gt05": float((e.abs() > 0.5).mean()),
                     "share_gt10": float((e.abs() > 1.0).mean()), "max_abs": float(e.abs().max())})
    # Fensterebene: MAE-Differenz und Differenz des Anteils grosser Fehler
    pw = g.groupby("t0").apply(lambda w: pd.Series({**{f"{m}_mae": (w[c] - w.truth).abs().mean() for m, c in MODELS.items()},
                                                    **{f"{m}_gt05": ((w[c] - w.truth).abs() > 0.5).mean() for m, c in MODELS.items()}}),
                               include_groups=False)
    for m in ["T", "A", "R", "AO", "RO"]:
        for kind in ["mae", "gt05"]:
            r = paired(pw[f"{m}_{kind}"] - pw[f"Oraku_{kind}"])
            diffs.append({"season": se, "comparison": f"{m} - Oraku", "metric": kind, **r})
tab1 = pd.DataFrame(rows); tab1.to_csv(OUT / "tab1_metrics_by_season.csv", index=False)
tab1d = pd.DataFrame(diffs); tab1d.to_csv(OUT / "tab1b_paired_diffs_vs_oraku.csv", index=False)
P(tab1[tab1.model.isin(MAIN)].pivot(index="model", columns="season", values="MAE").round(3).to_string())
P(tab1d[(tab1d.metric == "mae") & tab1d.comparison.isin(["T - Oraku", "AO - Oraku"])][["season", "comparison", "n", "delta", "ci_lo", "ci_hi", "p_wilcoxon"]].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
P("\n[2] Tab. II – Tagesmaxima/-minima (lokale Kalendertage, nur vollstaendige Tage >= 23 h)")
dcount = h.groupby(["t0", "day"]).size()
complete = dcount[dcount >= 23].index
partial = dcount[dcount < 23]
P(f"  Fenster-Tage gesamt {len(dcount)}, vollstaendig {len(complete)}, unvollstaendig (ausgeschlossen) {len(partial)}: "
  + ", ".join(f"{t0.date()}/{d} ({n} h)" for (t0, d), n in partial.items()))
dst_days = dcount[[d in (pd.Timestamp("2026-03-29").date(),) for (_, d) in dcount.index]]
P(f"  DST-Beginn 29.03.2026: {dst_days.to_dict()} -> 23-h-Tag korrekt als ein Kalendertag gefuehrt")
hc = h.set_index(["t0", "day"]).loc[complete].reset_index()
g = hc.groupby(["t0", "day"])
daily = pd.DataFrame({"truth_max": g.truth.max(), "truth_min": g.truth.min(),
                      "truth_tmax_h": g.apply(lambda w: w.loc[w.truth.idxmax(), "hod"], include_groups=False)})
for m, c in MODELS.items():
    daily[f"{m}_max"] = g[c].max(); daily[f"{m}_min"] = g[c].min()
    daily[f"{m}_tmax_h"] = g.apply(lambda w: w.loc[w[c].idxmax(), "hod"], include_groups=False)
daily = daily.reset_index(); daily["season"] = daily.t0.dt.month.map(SEASON)
for m in MODELS:
    daily[f"{m}_emax"] = daily[f"{m}_max"] - daily.truth_max
    daily[f"{m}_emin"] = daily[f"{m}_min"] - daily.truth_min
    off = daily[f"{m}_tmax_h"] - daily.truth_tmax_h
    daily[f"{m}_toff_ordinary"] = off
    daily[f"{m}_toff"] = ((off + 12) % 24) - 12   # zirkulaere Differenz in [-12, 12) h
daily.to_csv(OUT / "daily_extremes_per_day.csv", index=False)
rows, diffs = [], []
for se in SEASONS:
    d = daily if se == "ALLE" else daily[daily.season == se]
    for m in MODELS:
        rows.append({"season": se, "model": m, "n_days": len(d), "n_windows": d.t0.nunique(),
                     "Tmax_bias": d[f"{m}_emax"].mean(), "Tmax_MAE": d[f"{m}_emax"].abs().mean(),
                     "Tmin_bias": d[f"{m}_emin"].mean(), "Tmin_MAE": d[f"{m}_emin"].abs().mean(),
                     "toff_mean_h": d[f"{m}_toff"].mean(), "toff_abs_mean_h": d[f"{m}_toff"].abs().mean(),
                     "share_toff_ge2h": float((d[f"{m}_toff"].abs() >= 2).mean())})
    # Fensterebene (Mittel ueber die Tage des Fensters), dann gepaart
    pw = d.groupby("t0").agg({**{f"{m}_emax": "mean" for m in MODELS}, **{f"{m}_emin": "mean" for m in MODELS},
                              **{f"{m}_toff": lambda x: x.abs().mean() for m in MODELS}})
    pwa = d.groupby("t0").agg({f"{m}_emax": lambda x: x.abs().mean() for m in MODELS})
    for m in ["T", "A", "AO", "RO", "R"]:
        for metric, src in [("Tmax_abs_error", pwa), ("Tmax_bias", pw), ("Tmin_bias", pw)]:
            key = f"{m}_emax" if "Tmax" in metric else f"{m}_emin"; keyo = key.replace(m, "Oraku", 1)
            r = paired(src[key] - src[keyo]); diffs.append({"season": se, "comparison": f"{m} - Oraku", "metric": metric, **r})
        r = paired(pw[f"{m}_toff"] - pw["Oraku_toff"]); diffs.append({"season": se, "comparison": f"{m} - Oraku", "metric": "abs_time_offset_h", **r})
tab2 = pd.DataFrame(rows); tab2.to_csv(OUT / "tab2_daily_extremes.csv", index=False)
tab2d = pd.DataFrame(diffs); tab2d.to_csv(OUT / "tab2b_daily_extremes_paired.csv", index=False)
P(tab2[tab2.model.isin(MAIN)][["season", "model", "n_days", "Tmax_bias", "Tmax_MAE", "Tmin_bias", "Tmin_MAE", "toff_abs_mean_h", "share_toff_ge2h"]].round(3).to_string(index=False))
P(tab2d[tab2d.comparison.isin(["T - Oraku", "AO - Oraku"]) & (tab2d.season == "ALLE")][["comparison", "metric", "n", "delta", "ci_lo", "ci_hi", "p_wilcoxon", "p_sign"]].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
P("\n[3] Tab. III – illustrative Schwellen (20/22 degC, nur Juni/Juli-Fenster) und grosse Fehler  [deskriptiv]")
rows = []
sm = h[h.dt.dt.month.isin([6, 7])]
for thr in THRESHOLDS:
    obs = sm.truth >= thr
    for m, c in MODELS.items():
        pr = sm[c] >= thr
        rows.append({"threshold_degC": thr, "model": m, "n_hours": len(sm), "n_obs_above": int(obs.sum()),
                     "hit_rate": float((pr & obs).sum() / max(obs.sum(), 1)), "false_alarm_rate": float((pr & ~obs).sum() / max(pr.sum(), 1)),
                     "n_missed": int((~pr & obs).sum()), "n_false": int((pr & ~obs).sum())})
tab3 = pd.DataFrame(rows); tab3.to_csv(OUT / "tab3_thresholds_illustrative.csv", index=False)
P(tab3[tab3.model.isin(MAIN)].round(3).to_string(index=False))
big = tab1[tab1.season == "ALLE"][["model", "share_gt05", "share_gt10", "max_abs"]]
P("  grosse Fehler (alle Stunden): " + "; ".join(f"{r.model}: >0.5 {r.share_gt05:.1%}, >1.0 {r.share_gt10:.1%}, max {r.max_abs:.2f}" for r in big.itertuples()))

# ---------------------------------------------------------------------------
P("\n[4] Tab. IV – Thun-Zwischenprognose (Stufe 1) je Saison")
tab4 = s.groupby("season").agg(n=("t0", "size"), stage1_thun_MAE=("stage1_thun_MAE", "mean")).reindex(["Winter", "Fruehling", "Sommer", "Herbst"]).reset_index()
tab4.loc[len(tab4)] = ["ALLE", len(s), s.stage1_thun_MAE.mean()]
tab4.to_csv(OUT / "tab4_thun_stage1.csv", index=False); P(tab4.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
P("\n[5] Tab. V – Oraku-Version und Abflussquelle je Zeitraum  [deskriptiv/explorativ]")
w = pd.read_csv(FROZEN / "windows_136_candidates.csv", parse_dates=["t0_local"])
w = w[w.status == "ok"].rename(columns={"t0_local": "t0"})
sv = s.merge(w[["t0", "oraku_model"]], on="t0")
rows = []
for mdl, g in sv.groupby("oraku_model", sort=False):
    rows.append({"oraku_model": mdl, "n": len(g), "from": g.t0.min().date(), "to": g.t0.max().date(),
                 "Oraku_MAE": g.Oraku_MAE.mean(), "T_MAE": g.T_MAE.mean(), "AO_MAE": g.AO_MAE.mean(),
                 "flow_bafu": int((g.flow_source == "bafu").sum()), "flow_persist": int((g.flow_source == "persist").sum())})
tab5 = pd.DataFrame(rows).sort_values("from"); tab5.to_csv(OUT / "tab5_versions_flow.csv", index=False); P(tab5.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
P("\n[6] Z4 – Fehler bei dynamischen vs. ruhigen Fenstern (|Delta T ueber 48 h| > 1,5 degC)  [explorativ]")
rng48 = h.groupby("t0").truth.agg(lambda x: x.max() - x.min())
s["range48"] = s.t0.map(rng48); s["dynamic"] = s.range48 > 1.5
rows = []
for dyn, g in s.groupby("dynamic"):
    for m in MODELS:
        rows.append({"dynamic": bool(dyn), "n": len(g), "model": m, "MAE": g[f"{m}_MAE"].mean()})
    for m in ["T", "AO"]:
        r = paired(g[f"{m}_MAE"] - g.Oraku_MAE)
        rows.append({"dynamic": bool(dyn), "n": len(g), "model": f"{m} - Oraku", "MAE": r["delta"], **{k: v for k, v in r.items() if k != "n"}})
tab6 = pd.DataFrame(rows); tab6.to_csv(OUT / "tab6_dynamic_vs_calm.csv", index=False)
P(tab6.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
P("\n[7] Fehler nach Tagesstunde (fuer Abb. F) und nach Prognosestunde (Abb. G)")
hod = h.assign(**{m: h[c] - h.truth for m, c in MODELS.items()}).groupby(["season", "hod"])[list(MODELS)].mean().reset_index()
hod.to_csv(OUT / "bias_by_hour_season.csv", index=False)
hor = pd.DataFrame({m: (h[c] - h.truth).abs().groupby(h.h).mean() for m, c in MODELS.items()}).reset_index()
hor.to_csv(OUT / "mae_by_horizon.csv", index=False)
P("  gespeichert: bias_by_hour_season.csv, mae_by_horizon.csv")

# ---------------------------------------------------------------------------
P("\n[8] Beispielfenster je Saison (Median nach AO-MAE) und Kontrastfenster")
ex = {}
for se in ["Winter", "Fruehling", "Sommer", "Herbst"]:
    g = s[s.season == se].sort_values("AO_MAE").reset_index(drop=True)
    ex[se] = str(g.loc[len(g) // 2, "t0"].date())
worst_oraku = str(s.loc[(s.Oraku_MAE - s.T_MAE).idxmax(), "t0"].date())
ob = s[s.T_MAE > s.Oraku_MAE].assign(d=lambda x: x.T_MAE - x.Oraku_MAE).sort_values("d").reset_index(drop=True)
oraku_better_median = str(ob.loc[len(ob) // 2, "t0"].date())
P(f"  Median-Fenster: {ex}; groesster Oraku-Nachteil: {worst_oraku}; Oraku besser (Median von {len(ob)} Fenstern): {oraku_better_median}")
json.dump({"example_windows": ex, "worst_oraku": worst_oraku, "oraku_better_median": oraku_better_median,
           "n_windows": int(len(s)), "n_hours": int(len(h))}, open(OUT / "metrics.json", "w"), indent=1)
(OUT / "metrics_hydro.log").write_text("\n".join(log))
print("\n-> phase2a/tables/")
