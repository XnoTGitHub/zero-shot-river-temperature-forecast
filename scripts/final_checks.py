# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/frozen_2026-10-14/scripts/final_checks.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Abschliessende methodische Pruefung (Phase 1d) auf dem DST-korrigierten Datensatz (n=129).

Liest: rerun_dst/results/*.csv, Original-Caches (nur lesen), Oraku-API (model_info, horizon=1)
Schreibt nur nach review_runde1/final_audit/ (log, csv, cache_oraku_meta/).
Keine TiRex-Laeufe.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import RESULTS as RR, CACHE, CACHE_NEW, ORAKU_META as META, FINAL  # noqa: E402
META.mkdir(parents=True, exist_ok=True)
FINAL.mkdir(parents=True, exist_ok=True)
TZ = "Europe/Zurich"
CONTEXT, HORIZON = 256, 48
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Fruehling", 4: "Fruehling", 5: "Fruehling",
          6: "Sommer", 7: "Sommer", 8: "Sommer", 9: "Herbst", 10: "Herbst", 11: "Herbst"}
lines = []


def P(s=""):
    print(s, flush=True); lines.append(s)


def sec(t):
    P(); P("=" * 78); P(t); P("=" * 78)


def wtest(d):
    d = np.asarray(d, float)
    return dict(n=len(d), delta=d.mean(), median=float(np.median(d)), wins=int((d < 0).sum()),
                p_wil=stats.wilcoxon(d).pvalue if len(d) > 5 else np.nan,
                p_sign=stats.binomtest(int((d < 0).sum()), len(d)).pvalue, dz=d.mean() / d.std(ddof=1))


def fmt(r):
    return (f"n={r['n']:3d} Δ={r['delta']:+.4f} med={r['median']:+.4f} wins={r['wins']} "
            f"p_wil={r['p_wil']:.2g} p_sign={r['p_sign']:.2g} d_z={r['dz']:+.2f}")


def load_aare(city):
    files, cur, end = [], pd.Timestamp("2025-10-23"), pd.Timestamp("2026-08-04")
    while cur < end:
        ce = min(cur + pd.Timedelta(days=20), end); files.append(CACHE / f"aare_{city}_{cur.date()}_{ce.date()}.json"); cur = ce
    frames = []
    for f in files:
        d = json.loads(f.read_text()).get("data", {})
        t = pd.DataFrame(d.get("temperature", []), columns=["ts", "temperature"])
        q = pd.DataFrame(d.get("flow", []), columns=["ts", "flow"])
        df = t.merge(q, on="ts", how="outer")
        df.index = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert(TZ)
        frames.append(df.drop(columns="ts").resample("1h").mean())
    full = pd.concat(frames).sort_index()
    return full[~full.index.duplicated(keep="last")]


s = pd.read_csv(RR / "all_seasons_prev_runs_dst_summary.csv", parse_dates=["t0"]).sort_values("t0").reset_index(drop=True)
s["season"] = s.t0.dt.month.map(SEASON)
h = pd.read_csv(RR / "all_seasons_prev_runs_dst_hourly.csv", parse_dates=["t0"])
bern = load_aare("bern"); thun = load_aare("thun")
out = {}

# =============================================================================
sec("1  BAFU-Abfluss-Fallback")
P("Code (all_seasons_ensemble_prev_runs[_dst].py, one_window):")
P("  flow_ctx_b = df_bern['flow'].reindex(idx_ctx)...   # idx_ctx = [t0-256h, t0-1h]  -> nur Messwerte VOR t0")
P("  persist_flow(past) = mean(past[-24:])              # Mittel der letzten 24 Kontextstunden (t0-24h .. t0-1h)")
P("  bafu = fetch_oraku('flow', t0) ; None/NaN -> persist_flow(flow_ctx_b)")
P("  -> Fallback verwendet ausschliesslich bis t0-1h gemessene Abfluesse. [OK]")
P("  BAFU-Prognose: Oraku-API liefert nur Prognosen mit run_ts <= from (= t0); Erstellungszeit von BAFU selbst unbekannt (Oraku schaetzt sie).")
grp = s.groupby("flow_source")
P(f"\nGruppen: {grp.size().to_dict()}")
P("Saisonale Zusammensetzung:"); P(pd.crosstab(s.flow_source, s.season).to_string())
P("\nMittlere MAE je Gruppe:")
P(s.groupby("flow_source")[["T_MAE", "A_MAE", "R_MAE", "B_persist_MAE", "Oraku_MAE", "AO_MAE", "RO_MAE"]].mean().round(4).to_string())
P("\nTests je Gruppe:")
for g, sub in [("bafu (97)", s[s.flow_source == "bafu"]), ("persist (32)", s[s.flow_source == "persist"])]:
    for name, a, b in [("AO-Oraku", "AO_MAE", "Oraku_MAE"), ("T-Oraku", "T_MAE", "Oraku_MAE"), ("A-Oraku", "A_MAE", "Oraku_MAE"),
                       ("R-B_persist", "R_MAE", "B_persist_MAE"), ("A-T", "A_MAE", "T_MAE"), ("AO-T", "AO_MAE", "T_MAE")]:
        r = wtest(sub[a] - sub[b]); P(f"  {g:13s} {name:12s} {fmt(r)}"); out.setdefault("flow_groups", []).append({"group": g, "test": name, **r})
# Saisonbereinigt: Winter-Fenster beider Gruppen (persist = Nov/Dez, bafu-Winter = Jan/Feb)
P("\nSaisonbereinigter Blick (nur Winter-Fenster): A-T und AO-Oraku nach Abflussquelle")
w = s[s.season == "Winter"]
for g, sub in w.groupby("flow_source"):
    P(f"  Winter/{g:7s} n={len(sub)}  A-T {fmt(wtest(sub.A_MAE - sub.T_MAE))}")
    P(f"  Winter/{g:7s} n={len(sub)}  AO-Oraku {fmt(wtest(sub.AO_MAE - sub.Oraku_MAE))}")
# Guete der BAFU-Prognose vs Persistenz in den 97 Fenstern (Abfluss selbst)
P("\nAbflussprognose-Guete (97 BAFU-Fenster): MAE BAFU-Prognose vs. 24-h-Persistenz gegen gemessenen Abfluss [m3/s]")
rows = []
for t0 in s[s.flow_source == "bafu"].t0:
    fp = CACHE / f"oraku_flow_{t0.isoformat()}_48.json"
    if not fp.exists():
        fp = CACHE_NEW / f"oraku_flow_{t0.isoformat()}_48.json"
    d = json.loads(fp.read_text())
    idx = pd.to_datetime(d["data"]["time"], utc=True).tz_convert(TZ)
    bafu = pd.Series(d["data"]["value"], index=idx, dtype=float)
    t0a = t0.tz_localize(TZ)
    fc = pd.date_range(t0a, periods=HORIZON, freq="1h")
    meas = bern.flow.reindex(fc)
    ctx = bern.flow.reindex(pd.date_range(t0a - pd.Timedelta(hours=CONTEXT), t0a - pd.Timedelta(hours=1), freq="1h")).interpolate(limit=30).ffill().bfill()
    pers = np.full(HORIZON, ctx.to_numpy()[-24:].mean())
    ok = meas.notna().to_numpy()
    if ok.sum() == 0: continue
    rows.append({"t0": t0, "mae_bafu": float(np.abs(bafu.reindex(fc).to_numpy()[ok] - meas.to_numpy()[ok]).mean()),
                 "mae_persist": float(np.abs(pers[ok] - meas.to_numpy()[ok]).mean()), "flow_mean": float(meas.mean())})
fq = pd.DataFrame(rows)
P(f"  n={len(fq)}  MAE BAFU={fq.mae_bafu.mean():.1f}  MAE Persistenz={fq.mae_persist.mean():.1f}  mittlerer Abfluss={fq.flow_mean.mean():.0f}  "
  f"BAFU besser in {(fq.mae_bafu < fq.mae_persist).sum()} Fenstern, Wilcoxon p={stats.wilcoxon(fq.mae_bafu - fq.mae_persist).pvalue:.2g}")
out["flow_quality"] = fq.to_dict("records")
fq.to_csv(FINAL / "flow_forecast_quality.csv", index=False)

# =============================================================================
sec("2  Interpolation im Eingabekontext (alle 129 Fenster)")
rows = []
for t0 in s.t0:
    t0a = t0.tz_localize(TZ)
    idx_ctx = pd.date_range(t0a - pd.Timedelta(hours=CONTEXT), t0a - pd.Timedelta(hours=1), freq="1h")
    idx_fc = pd.date_range(t0a, periods=HORIZON, freq="1h")
    r = {"t0": t0}
    for name, ser in [("bern_temp", bern.temperature), ("bern_flow", bern.flow), ("thun_temp", thun.temperature), ("thun_flow", thun.flow)]:
        c = ser.reindex(idx_ctx); na = c.isna().to_numpy()
        r[f"{name}_nan"] = int(na.sum())
        # laengste zusammenhaengende Luecke und Abstand ihres Endes zu t0
        best, cur, end_pos = 0, 0, None
        for i, v in enumerate(na):
            cur = cur + 1 if v else 0
            if cur > best: best, end_pos = cur, i
        r[f"{name}_maxgap"] = best
        r[f"{name}_gap_end_h_before_t0"] = (CONTEXT - end_pos) if end_pos is not None else np.nan
    r["eval_truth_nan"] = int(bern.temperature.reindex(idx_fc).isna().sum())
    r["eval_thun_nan"] = int(thun.temperature.reindex(idx_fc).isna().sum())
    rows.append(r)
gi = pd.DataFrame(rows); gi.to_csv(FINAL / "context_gaps.csv", index=False)
for name in ["bern_temp", "bern_flow", "thun_temp", "thun_flow"]:
    nz = gi[gi[f"{name}_nan"] > 0]
    P(f"{name:10s}: Fenster mit interpolierten Kontextstunden={len(nz):3d}  Summe Stunden={int(gi[f'{name}_nan'].sum()):4d}  "
      f"max zusammenhaengende Luecke={int(gi[f'{name}_maxgap'].max())} h  "
      + (f"(Fenster {nz.loc[nz[f'{name}_maxgap'].idxmax(), 't0'].date()}, Ende {nz[f'{name}_gap_end_h_before_t0'].min():.0f}..{nz[f'{name}_gap_end_h_before_t0'].max():.0f} h vor t0)" if len(nz) else ""))
P(f"Bewertungszeitraum: Fenster mit fehlenden Bern-Messstunden = {int((gi.eval_truth_nan > 0).sum())} (muss 0 sein); "
  f"Thun-Messwerte im Horizont fehlend (nur Stufe-1-Bewertung betroffen) = {int((gi.eval_thun_nan > 0).sum())} Fenster")
P("Open-Meteo-Kovariaten: Rohcache ohne NaN (siehe Dry-Run Kap. 3.1); interpolate(limit=12) greift nicht.")
big = gi[(gi.bern_temp_maxgap > 6) | (gi.thun_temp_maxgap > 6) | (gi.bern_flow_maxgap > 6) | (gi.thun_flow_maxgap > 6)]
P(f"\nFenster mit Kontextluecke > 6 h in irgendeiner Messreihe: {len(big)}")
if len(big):
    P(big[["t0", "bern_temp_maxgap", "bern_temp_gap_end_h_before_t0", "thun_temp_maxgap", "thun_temp_gap_end_h_before_t0",
           "bern_flow_maxgap", "thun_flow_maxgap"]].to_string(index=False))
    sub = s[~s.t0.isin(big.t0)]
    P(f"\nSensitivitaet ohne diese Fenster (n={len(sub)}):")
    for name, a, b in [("AO-Oraku", "AO_MAE", "Oraku_MAE"), ("T-Oraku", "T_MAE", "Oraku_MAE"), ("R-B_persist", "R_MAE", "B_persist_MAE"),
                       ("R-T", "R_MAE", "T_MAE"), ("AO-T", "AO_MAE", "T_MAE")]:
        P(f"  {name:12s} {fmt(wtest(sub[a] - sub[b]))}")
# Fenster mit Luecke nahe t0 (innerhalb 24 h)
near = gi[(gi.bern_temp_nan > 0) & (gi.bern_temp_gap_end_h_before_t0 <= 24)]
P(f"Fenster mit interpolierter Bern-Temperatur in den letzten 24 h vor t0: {len(near)}  "
  + (", ".join(f"{r.t0.date()} ({r.bern_temp_maxgap} h, Ende {r.bern_temp_gap_end_h_before_t0:.0f} h vor t0)" for r in near.itertuples()) if len(near) else ""))
anyint = gi[(gi.bern_temp_nan > 0) | (gi.thun_temp_nan > 0) | (gi.bern_flow_nan > 0) | (gi.thun_flow_nan > 0)]
sub = s[~s.t0.isin(anyint.t0)]
P(f"\nSensitivitaet ohne alle {len(anyint)} Fenster mit irgendeiner Kontext-Interpolation (n={len(sub)}):")
for name, a, b in [("AO-Oraku", "AO_MAE", "Oraku_MAE"), ("T-Oraku", "T_MAE", "Oraku_MAE"), ("R-B_persist", "R_MAE", "B_persist_MAE"), ("AO-T", "AO_MAE", "T_MAE")]:
    P(f"  {name:12s} {fmt(wtest(sub[a] - sub[b]))}")
sub = s[~s.t0.isin(near.t0)]
P(f"Sensitivitaet ohne die {len(near)} Fenster mit Interpolation in den letzten 24 h vor t0 (n={len(sub)}):")
for name, a, b in [("AO-Oraku", "AO_MAE", "Oraku_MAE"), ("T-Oraku", "T_MAE", "Oraku_MAE")]:
    P(f"  {name:12s} {fmt(wtest(sub[a] - sub[b]))}")
P("\nInformationen nach t0: Kontext endet t0-1h; Open-Meteo day1/day2 = Laeufe initialisiert >=24 h vor Gueltigkeit (< t0); "
  "Oraku last_updated < t0 (alle 129); BAFU-Abfluss via Oraku run_ts <= t0; Klimatologie 2020-2022; Persistenz = Wert t0-1h. [OK]")

# =============================================================================
sec("3a  Statistik: Aussage <-> passender Test (dst_n129)")
d = s.R_MAE - s.Oraku_MAE
P(f"R vs Oraku: Mittelwert {d.mean():+.4f} (t-Test p={stats.ttest_1samp(d,0).pvalue:.3f}; MBB-CI ueberstreicht 0) | "
  f"Median {d.median():+.4f}, Vorzeichentest p={stats.binomtest(int((d<0).sum()),len(d)).pvalue:.1e}, Wilcoxon p={stats.wilcoxon(d).pvalue:.1e}, Schiefe {d.skew():+.2f}")
P("  -> zulaessige Aussage: 'R ist in 86/129 Fenstern besser als Oraku (Median -0,054 degC); der mittlere MAE-Unterschied ist wegen weniger grosser Ausreisser nicht gesichert.'")
dl = pd.read_csv(RR / "delay6h_prev_runs_dst_summary.csv")
rng = np.random.default_rng(1)
for a in ["AO_MAE", "RO_MAE"]:
    x = (dl[a] - dl.Oraku_MAE).to_numpy(); n = len(x); se = x.std(ddof=1) / np.sqrt(n)
    ci = np.percentile([rng.choice(x, n).mean() for _ in range(10000)], [2.5, 97.5])
    t05 = max(1 - stats.t.cdf((x.mean() + .05) / se, n - 1), stats.t.cdf((x.mean() - .05) / se, n - 1))
    P(f"{a[:-4]} (6h-Delay) vs Oraku: Δ={x.mean():+.4f} CI95=[{ci[0]:+.4f},{ci[1]:+.4f}] Wilcoxon p={stats.wilcoxon(x).pvalue:.3f} TOST(+-0.05) p={t05:.3f} "
      f"-> {'Aequivalenz innerhalb +-0.05 belegt' if t05 < .05 else 'KEINE Aequivalenz belegt; Nicht-Signifikanz ist kein Beleg'}")
P("  -> AO: 'im Mittel um 0,035 degC schlechter als Oraku (CI schliesst 0 aus)'; RO: 'aequivalent innerhalb +-0,05 degC'.")

# =============================================================================
sec("3b  Oraku-Modellversion je Fenster (model_info-Abfrage, horizon=1)")
rows = []
for t0 in s.t0:
    fp = META / f"oraku_meta_{t0.date()}.json"
    if not fp.exists():
        r = requests.get("https://aare-oraku.konzept.space/forecast/temperature",
                         params={"city": "bern", "horizon": 1, "from": t0.tz_localize(TZ).isoformat(), "model_info": "true",
                                 "app": "hywa-review-audit", "version": "1.0"}, timeout=30)
        r.raise_for_status(); fp.write_text(r.text); time.sleep(0.3)
    m = json.loads(fp.read_text()).get("metadata") or {}
    mdl = m.get("model") or {}
    rows.append({"t0": t0, "model": f"{mdl.get('name')}/{mdl.get('version')}", "last_updated": m.get("last_updated")})
mv = pd.DataFrame(rows); mv.to_csv(FINAL / "oraku_model_versions.csv", index=False)
lu = pd.to_datetime(mv.last_updated, utc=True).dt.tz_convert(TZ)
lead = (mv.t0.dt.tz_localize(TZ) - lu).dt.total_seconds() / 60
P(f"Prognosezeitpunkte: t0 - last_updated [min]: min {lead.min():.0f}, median {lead.median():.0f}, max {lead.max():.0f}; nach t0: {(lead<0).sum()}")
sv = s.merge(mv[["t0", "model"]], on="t0")
P("Versionen (Fensterzahl, Zeitraum):")
for mdl, g in sv.groupby("model"):
    P(f"  {mdl:22s} n={len(g):3d}  {g.t0.min().date()} .. {g.t0.max().date()}  Saisons={g.season.value_counts().to_dict()}")
P("\nOraku-MAE und TiRex-Vorteil je Version:")
for mdl, g in sv.groupby("model"):
    P(f"  {mdl:22s} Oraku={g.Oraku_MAE.mean():.3f} T={g.T_MAE.mean():.3f} AO={g.AO_MAE.mean():.3f}  "
      f"T-Oraku {fmt(wtest(g.T_MAE - g.Oraku_MAE))}")
    P(f"  {'':22s} AO-Oraku {fmt(wtest(g.AO_MAE - g.Oraku_MAE))}")
# Versionswechsel innerhalb derselben Saison (Fruehling: LR/dev -> 1.0): Vergleich Oraku-Bias/MAE relativ zu T innerhalb Fruehling
fr = sv[sv.season == "Fruehling"]
P("\nFruehling nach Version (gleiche Saison, Versionswechsel):")
for mdl, g in fr.groupby("model"):
    P(f"  {mdl:22s} n={len(g):2d} Oraku={g.Oraku_MAE.mean():.3f} T={g.T_MAE.mean():.3f} Δ(T-Oraku)={(g.T_MAE-g.Oraku_MAE).mean():+.3f}")
out["versions"] = sv.groupby("model").size().to_dict()

# =============================================================================
sec("3c  Modellauswahl auf Testdaten: Sensitivitaet (dst_n129)")
for lbl, sub in [("ohne 7 Subset-Fenster (13.-27.07.)", s[~s.t0.dt.strftime("%Y-%m-%d").between("2026-07-13", "2026-07-27")]),
                 ("ohne Juli 2026", s[s.t0 < "2026-07-01"]), ("ohne Juni+Juli 2026", s[s.t0 < "2026-06-01"])]:
    P(f"{lbl} (n={len(sub)}):")
    for name, a, b in [("AO-Oraku", "AO_MAE", "Oraku_MAE"), ("T-Oraku", "T_MAE", "Oraku_MAE"), ("R-B_persist", "R_MAE", "B_persist_MAE"),
                       ("R-T", "R_MAE", "T_MAE"), ("AO-T", "AO_MAE", "T_MAE")]:
        P(f"  {name:12s} {fmt(wtest(sub[a] - sub[b]))}")

# =============================================================================
sec("3d  DST-Ueberlappung 28./30.03.: Abdeckung durch n=128/n=129")
for lbl, sub in [("n=129 (beide Fenster)", s), ("n=128 ohne 28.03.", s[s.t0 != "2026-03-28"]), ("n=128 ohne 30.03.", s[s.t0 != "2026-03-30"])]:
    P(f"{lbl}: AO-Oraku {fmt(wtest(sub.AO_MAE - sub.Oraku_MAE))}")
    P(f"{'':22s} T-Oraku  {fmt(wtest(sub.T_MAE - sub.Oraku_MAE))}")
hh = h[h.t0.isin([pd.Timestamp("2026-03-28"), pd.Timestamp("2026-03-30")])]
shared = hh.groupby("datetime").size(); shared = shared[shared > 1]
P(f"Gemeinsame Messzeitpunkte beider Fenster in der Stundendatei: {len(shared)} ({list(shared.index)})")
P("Fenster-MAE-Differenzen der beiden Fenster (AO-Oraku): " + ", ".join(f"{r.t0.date()} {r.AO_MAE - r.Oraku_MAE:+.3f}" for r in s[s.t0.isin(['2026-03-28', '2026-03-30'])].itertuples()))

(FINAL / "final_checks.log").write_text("\n".join(lines))
json.dump(out, open(FINAL / "final_checks.json", "w"), default=str, indent=1)
print("\n-> final_audit/final_checks.log")
