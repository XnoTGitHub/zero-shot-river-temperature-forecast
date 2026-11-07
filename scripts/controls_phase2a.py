# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/phase2a/controls_phase2a.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Gezielte Kontrollen zum Abschluss von Phase 2a (nur eingefrorene Daten, keine Modellläufe).

1. Stabilität der Kennzahl "Zeitpunkt des Tagesmaximums": Amplitude, Mehrfachmaxima,
   Maxima nahe Mitternacht, gewöhnliche vs. zirkuläre Differenz, Sensitivität nach Mindestamplitude.
2. Oraku-Nachmittagsbias nach Modellversion (Version × Saison konfundiert -> nur deskriptiv).
3. Schwellenkennzahlen mit eindeutigen Definitionen: POD = TP/(TP+FN), FAR = FP/(TP+FP), FPR = FP/(FP+TN).
Schreibt nach phase2a/tables/ctrl_*.csv und ctrl_log.txt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from paths import RESULTS as FROZEN, TABLES as TAB  # noqa: E402
TZ = "Europe/Zurich"
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Fruehling", 4: "Fruehling", 5: "Fruehling",
          6: "Sommer", 7: "Sommer", 8: "Sommer", 9: "Herbst", 10: "Herbst", 11: "Herbst"}
MODELS = {"Oraku": "Oraku_pred", "T": "T_pred", "A": "A_pred", "AO": "AO_pred"}
log = []


def P(s=""):
    print(s, flush=True); log.append(s)


def mbb_ci(d, L=5, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed); d = np.asarray(d, float); n = len(d); nb = int(np.ceil(n / L)); means = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, n, size=nb); means[i] = np.concatenate([np.take(d, np.arange(s, s + L) % n) for s in st])[:n].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 6:
        return {"n": len(d), "delta": float(d.mean()) if len(d) else np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_wilcoxon": np.nan}
    lo, hi = mbb_ci(d)
    return {"n": len(d), "delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}


def classify(r):
    """Einstufung einer Teil-/Zusatzanalyse (keine Multiplizitaetskorrektur)."""
    if np.isnan(r["ci_lo"]): return "n zu klein"
    ci_ex = (r["ci_lo"] > 0) or (r["ci_hi"] < 0)
    if ci_ex and r["p_wilcoxon"] < 0.05: return "nominell signifikant (CI und Wilcoxon)"
    if r["p_wilcoxon"] < 0.05: return "nur Wilcoxon (CI des Mittels schliesst 0 ein)"
    if ci_ex: return "nur CI (Wilcoxon n.s.)"
    return "nicht gesichert"


h = pd.read_csv(FROZEN / "all_seasons_prev_runs_dst_hourly.csv", parse_dates=["t0"])
h["dt"] = pd.to_datetime(h["datetime"], utc=True).dt.tz_convert(TZ); h["season"] = h.t0.dt.month.map(SEASON)
h["hod"] = h.dt.dt.hour; h["day"] = h.dt.dt.date
w = pd.read_csv(FROZEN / "windows_136_candidates.csv", parse_dates=["t0_local"]); w = w[w.status == "ok"]
ver = dict(zip(w.t0_local, w.oraku_model))
h["version"] = h.t0.map(ver)

# =============================================================================
P("=" * 78); P("1  Stabilitaet 'Zeitpunkt des Tagesmaximums'"); P("=" * 78)
dc = h.groupby(["t0", "day"]).size(); complete = dc[dc >= 23].index
hc = h.set_index(["t0", "day"]).loc[complete].reset_index()
g = hc.groupby(["t0", "day"])
rows = []
for (t0, day), d in g:
    r = {"t0": t0, "day": day, "season": SEASON[t0.month], "amp": d.truth.max() - d.truth.min(),
         "truth_tmax": int(d.loc[d.truth.idxmax(), "hod"]),
         "truth_ties": int((d.truth >= d.truth.max() - 0.05).sum())}
    r["truth_near_midnight"] = r["truth_tmax"] <= 1 or r["truth_tmax"] >= 22
    for m, c in MODELS.items():
        tm = int(d.loc[d[c].idxmax(), "hod"]); off = tm - r["truth_tmax"]
        circ = ((off + 12) % 24) - 12
        r[f"{m}_tmax"] = tm; r[f"{m}_off"] = off; r[f"{m}_circ"] = circ; r[f"{m}_ties"] = int((d[c] >= d[c].max() - 0.05).sum())
    rows.append(r)
dd = pd.DataFrame(rows); dd.to_csv(TAB / "ctrl_tmax_per_day.csv", index=False)
P(f"vollstaendige Tage: {len(dd)}; Tagesamplitude Messung: Median {dd.amp.median():.2f} degC, < 0,5 degC: {(dd.amp<0.5).sum()} Tage, < 1,0: {(dd.amp<1.0).sum()}")
P(f"Messung: Maximum nahe Mitternacht (h<=1 oder >=22): {dd.truth_near_midnight.sum()} Tage ({dd.truth_near_midnight.mean():.0%}); Mehrfachmaxima (>=2 Stunden innerhalb 0,05 degC des Maximums): {(dd.truth_ties>=2).sum()} Tage")
P("Anteil |gewoehnlich| != |zirkulaer|: " + ", ".join(f"{m}: {(dd[f'{m}_off'].abs()!=dd[f'{m}_circ'].abs()).mean():.1%}" for m in MODELS))
P("\nMittlerer |Zeitversatz| [h], gewoehnlich / zirkulaer, nach Saison:")
for se in ["Winter", "Fruehling", "Sommer", "Herbst", "ALLE"]:
    s_ = dd if se == "ALLE" else dd[dd.season == se]
    P(f"  {se:9s} n={len(s_):3d} " + "  ".join(f"{m}: {s_[f'{m}_off'].abs().mean():.2f}/{s_[f'{m}_circ'].abs().mean():.2f}" for m in ["Oraku", "T", "AO"]))
P("\nSensitivitaet: gepaarte Fensterdifferenz des mittleren |zirkulaeren Zeitversatzes| (T - Oraku, AO - Oraku) nach Mindestamplitude und Saison")
rows = []
for amin in [0.0, 0.5, 1.0, 1.5]:
    sub = dd[dd.amp >= amin]
    for se in ["Winter", "Fruehling", "Sommer", "Herbst", "ALLE"]:
        s_ = sub if se == "ALLE" else sub[sub.season == se]
        pw = s_.groupby("t0").agg({f"{m}_circ": lambda x: x.abs().mean() for m in MODELS})
        for m in ["T", "AO"]:
            r = paired(pw[f"{m}_circ"] - pw["Oraku_circ"]); r.update({"min_amp": amin, "season": se, "comparison": f"{m} - Oraku", "n_days": len(s_), "einstufung": classify(r)})
            rows.append(r)
sens = pd.DataFrame(rows); sens.to_csv(TAB / "ctrl_tmax_sensitivity.csv", index=False)
for se in ["Winter", "Fruehling", "Sommer", "Herbst", "ALLE"]:
    P(f"  {se}:")
    for r in sens[(sens.season == se)].itertuples():
        P(f"    amp>={r.min_amp:.1f} {r.comparison:10s} n_win={r.n:3d} Δ={r.delta:+.2f} h CI=[{r.ci_lo:+.2f};{r.ci_hi:+.2f}] p={r.p_wilcoxon:.3g}  -> {r.einstufung}")

# =============================================================================
P(); P("=" * 78); P("2  Oraku-Nachmittagsbias nach Modellversion (deskriptiv; Version x Saison konfundiert)"); P("=" * 78)
h["e_Oraku"] = h.Oraku_pred - h.truth; h["e_T"] = h.T_pred - h.truth; h["e_AO"] = h.AO_pred - h.truth
aft = h[h.hod.between(12, 17)]
rows = []
for (v, se), d in aft.groupby(["version", "season"]):
    rows.append({"version": v, "season": se, "n_windows": d.t0.nunique(), "afternoon_bias_Oraku": d.e_Oraku.mean(), "afternoon_bias_T": d.e_T.mean(), "afternoon_bias_AO": d.e_AO.mean()})
vb = pd.DataFrame(rows).sort_values(["season", "version"]); vb.to_csv(TAB / "ctrl_afternoon_bias_version.csv", index=False)
P("Mittlerer Fehler 12-17 Uhr (Prognose - Messung) je Version und Saison:")
P(vb.round(3).to_string(index=False))
P("\nInnerhalb des Fruehlings (einzige Saison mit mehreren Versionen und n >= 7):")
fr = vb[vb.season == "Fruehling"]
P(fr.round(3).to_string(index=False))
P("Lesart: Der Nachmittagsbias von Oraku tritt bei LR/dev (Maerz), 1.0 (April/Mai) und 1.3 (Ende Mai) auf; die Groesse variiert mit Version UND Jahreszeit; eine Trennung ist mit diesen Daten nicht moeglich.")
# Monatsweise (Saisoneffekt innerhalb einer Version)
hm = aft.assign(month=aft.dt.dt.month).groupby(["version", "month"]).agg(n_windows=("t0", "nunique"), bias_Oraku=("e_Oraku", "mean"), bias_T=("e_T", "mean")).reset_index()
hm.to_csv(TAB / "ctrl_afternoon_bias_version_month.csv", index=False)
P("\nNach Version und Monat (zeigt Saisoneffekt innerhalb LR/dev und innerhalb 1.3):")
P(hm.round(3).to_string(index=False))

# =============================================================================
P(); P("=" * 78); P("3  Schwellenkennzahlen mit eindeutigen Definitionen (illustrativ, Juni/Juli)"); P("=" * 78)
sm = h[h.dt.dt.month.isin([6, 7])]
rows = []
for thr in [20.0, 22.0]:
    obs = sm.truth >= thr
    for m, c in MODELS.items():
        pr = sm[c] >= thr
        TP = int((pr & obs).sum()); FP = int((pr & ~obs).sum()); FN = int((~pr & obs).sum()); TN = int((~pr & ~obs).sum())
        rows.append({"threshold_degC": thr, "model": m, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                     "POD_hit_rate": TP / (TP + FN), "FAR_false_alarm_ratio": FP / (TP + FP) if TP + FP else np.nan,
                     "FPR_false_positive_rate": FP / (FP + TN) if FP + TN else np.nan, "bias_score": (TP + FP) / (TP + FN)})
thr = pd.DataFrame(rows); thr.to_csv(TAB / "ctrl_thresholds_defined.csv", index=False)
P(thr.round(3).to_string(index=False))
P("POD = TP/(TP+FN); FAR = FP/(TP+FP) (bisher 'Fehlalarmrate' genannt -> umbenannt in Fehlalarmanteil); FPR = FP/(FP+TN); Bias-Score = (TP+FP)/(TP+FN).")

# =============================================================================
P(); P("=" * 78); P("4  Einstufung der saisonalen/ergaenzenden Vergleiche (Tab. I/II)"); P("=" * 78)
t1b = pd.read_csv(TAB / "tab1b_paired_diffs_vs_oraku.csv"); t1b["einstufung"] = t1b.apply(classify, axis=1); t1b.to_csv(TAB / "tab1b_paired_diffs_vs_oraku.csv", index=False)
t2b = pd.read_csv(TAB / "tab2b_daily_extremes_paired.csv"); t2b["einstufung"] = t2b.apply(classify, axis=1); t2b.to_csv(TAB / "tab2b_daily_extremes_paired.csv", index=False)
P("Gesamtzeitraum (Primaertests, Holm-korrigiert -> 'gesichert'): nur Q1 T-Oraku und Q3a AO-Oraku ueber alle 129 Fenster.")
P("Alle saisonalen Vergleiche und Tagesextrema-Vergleiche: nominell, ohne Multiplizitaetskorrektur -> hoechstens 'nominell signifikant'.")
P(t1b[(t1b.metric == "mae") & t1b.comparison.isin(["T - Oraku", "AO - Oraku"])][["season", "comparison", "delta", "ci_lo", "ci_hi", "p_wilcoxon", "einstufung"]].round(3).to_string(index=False))
P(t2b[t2b.comparison.isin(["T - Oraku", "AO - Oraku"]) & t2b.metric.isin(["Tmax_abs_error", "abs_time_offset_h"])][["season", "comparison", "metric", "delta", "ci_lo", "ci_hi", "p_wilcoxon", "einstufung"]].round(3).to_string(index=False))
(TAB / "ctrl_log.txt").write_text("\n".join(log))
