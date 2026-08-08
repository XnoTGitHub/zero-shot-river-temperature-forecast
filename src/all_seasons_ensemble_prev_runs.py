"""Strikt-kausale Neuberechnung des 128-Fenster-Primaertests mit Open-Meteo
Previous Runs API. Adressiert Reviewer-Kritik "Issue-Time ueber Historical-
Forecast API nicht explizit kontrollierbar".

Aenderung gegenueber all_seasons_ensemble.py:
  - Wetterkovariaten (Bern, Thun, Thunersee-Wind) kommen nicht mehr aus
    'historical-forecast-api', sondern aus 'previous-runs-api':
      Kontext (t0-256h .. t0-1h)        -> _previous_day1  (24h vor Gueltigkeit)
      Prognose t0 .. t0+23              -> _previous_day1  (Modelllauf immer < t0)
      Prognose t0+24 .. t0+47           -> _previous_day2  (Modelllauf immer < t0)
  - Alle Werte sind damit nachweisbar aus einem Modelllauf, der *vor* t0
    initialisiert wurde. Konservativ (echte operative Lead-Times waeren
    kuerzer), aber strikt kausal.

Rest der Pipeline (TiRex-Aufrufe, Ensemble-Rezepte, Oraku, Baselines) ist
IDENTISCH zu all_seasons_ensemble.py, damit der Vergleich nur die eine
Aenderung isoliert.

Output:
  src/all_seasons_prev_runs_summary.csv
  src/all_seasons_prev_runs_hourly.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from tirex2 import load_model as load_v2
from tirex2.model.types import TimeseriesType


BASE = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE / "data" / "aare_dashboard_cache"
CACHE_DIR.mkdir(exist_ok=True)

LAT_BERN, LON_BERN = 46.948, 7.447
LAT_THUN_LAKE, LON_THUN_LAKE = 46.72, 7.75
HORIZON = 48
CONTEXT = 256
AXIS_FROM_DEG = 114.0

WINDOW_START = pd.Timestamp("2025-11-04")
WINDOW_END = pd.Timestamp("2026-08-01")
STEP_DAYS = 2


# =============================================================================
# I/O helpers (mit Cache) — identisch zu all_seasons_ensemble.py, ausser Prefix
# =============================================================================
def _get(url, params, cache_key, retries=5):
    fp = CACHE_DIR / f"{cache_key}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60,
                             headers={"User-Agent": "hywa-allseasons-prevruns/1.0"})
            r.raise_for_status()
            data = r.json()
            fp.write_text(json.dumps(data))
            return data
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    [_get retry {attempt+1}/{retries} in {wait}s: {type(e).__name__}]",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"_get failed after {retries} retries: {cache_key}: {last_err}")


def fetch_aare_city_chunked(city, start_date, end_date):
    frames = []
    cur = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    while cur < end_ts:
        chunk_end = min(cur + pd.Timedelta(days=20), end_ts)
        d = _get("https://aareguru.existenz.ch/v2018/history",
                 {"city": city, "start": str(cur.date()),
                  "end": str(chunk_end.date()),
                  "app": "hywa-allseasons", "version": "1.0"},
                 cache_key=f"aare_{city}_{cur.date()}_{chunk_end.date()}")
        dat = d.get("data", {})

        def to_df(k, col):
            return pd.DataFrame(dat.get(k, []), columns=["ts", col]).assign(
                datetime=lambda x: pd.to_datetime(x["ts"], unit="s"))
        df = (to_df("temperature", "temperature")
              .merge(to_df("flow", "flow"), on=["ts", "datetime"], how="outer")
              .sort_values("datetime").set_index("datetime").drop(columns=["ts"]))
        df.index = df.index.tz_localize("UTC").tz_convert("Europe/Zurich").tz_localize(None)
        frames.append(df.resample("1h").mean())
        cur = chunk_end
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    return full


# =============================================================================
# Previous-Runs Fetches — GEAENDERT
# =============================================================================
def fetch_meteo_prev_runs(lat, lon, start, end, key):
    """Liefert DataFrame mit Spalten tt_d1, tt_d2, solar_d1, solar_d2.
    Der Endpunkt akzeptiert dieselbe start_date/end_date-Semantik wie
    historical-forecast-api. Alle _previous_dayN-Werte sind aus Modelllaeufen,
    die dayN Tage vor Gueltigkeitszeitpunkt initialisiert wurden.
    """
    d = _get("https://previous-runs-api.open-meteo.com/v1/forecast",
             {"latitude": lat, "longitude": lon,
              "hourly": ("temperature_2m_previous_day1,"
                         "temperature_2m_previous_day2,"
                         "shortwave_radiation_previous_day1,"
                         "shortwave_radiation_previous_day2"),
              "start_date": str(start), "end_date": str(end),
              "timezone": "Europe/Zurich"},
             cache_key=f"meteo_prevruns_tt_solar_{key}_{start}_{end}")
    h = d["hourly"]
    return pd.DataFrame({
        "tt_d1": h["temperature_2m_previous_day1"],
        "tt_d2": h["temperature_2m_previous_day2"],
        "solar_d1": h["shortwave_radiation_previous_day1"],
        "solar_d2": h["shortwave_radiation_previous_day2"],
    }, index=pd.to_datetime(h["time"]))


def fetch_wind_prev_runs(start, end):
    d = _get("https://previous-runs-api.open-meteo.com/v1/forecast",
             {"latitude": LAT_THUN_LAKE, "longitude": LON_THUN_LAKE,
              "hourly": ("wind_speed_10m_previous_day1,"
                         "wind_speed_10m_previous_day2,"
                         "wind_direction_10m_previous_day1,"
                         "wind_direction_10m_previous_day2"),
              "start_date": str(start), "end_date": str(end),
              "timezone": "Europe/Zurich"},
             cache_key=f"wind_thun_prevruns_{start}_{end}")
    h = d["hourly"]
    return pd.DataFrame({
        "wind_speed_d1": h["wind_speed_10m_previous_day1"],
        "wind_speed_d2": h["wind_speed_10m_previous_day2"],
        "wind_dir_d1":  h["wind_direction_10m_previous_day1"],
        "wind_dir_d2":  h["wind_direction_10m_previous_day2"],
    }, index=pd.to_datetime(h["time"]))


def fetch_oraku(kind, t0, horizon):
    from_iso = t0.tz_localize("Europe/Zurich").isoformat()
    d = _get(f"https://aare-oraku.konzept.space/forecast/{kind}",
             {"city": "bern", "horizon": horizon, "from": from_iso,
              "app": "hywa-allseasons", "version": "1.0"},
             cache_key=f"oraku_{kind}_{t0.isoformat()}_{horizon}")
    times = d.get("data", {}).get("time", [])
    vals = d.get("data", {}).get("value", [])
    if not times:
        return None
    idx = pd.to_datetime(times).tz_convert("Europe/Zurich").tz_localize(None)
    return pd.Series(vals, index=idx, dtype=float)


# =============================================================================
# Causally-consistent assembly: kombiniert d1 (Kontext + h<=24) und d2 (h>24..48)
# =============================================================================
def assemble_causal(df_d1_d2, t0, col_base, idx_full):
    """Baut eine kausal-konsistente Zeitreihe:
      idx < t0             -> _previous_day1
      t0 <= idx < t0+24    -> _previous_day1
      t0+24 <= idx < t0+48 -> _previous_day2
    """
    s_d1 = df_d1_d2[f"{col_base}_d1"].reindex(idx_full)
    s_d2 = df_d1_d2[f"{col_base}_d2"].reindex(idx_full)
    cutoff = t0 + pd.Timedelta(hours=24)
    out = s_d1.copy()
    mask_d2 = (out.index >= cutoff)
    out.loc[mask_d2] = s_d2.loc[mask_d2]
    return out


# =============================================================================
# TiRex-2 (identisch)
# =============================================================================
_TIREX2 = None
def tirex2():
    global _TIREX2
    if _TIREX2 is None:
        print("  loading TiRex-2 ...")
        _TIREX2 = load_v2("NX-AI/TiRex-2", device="cpu")
    return _TIREX2


def forecast_tirex2(temp_ctx, past_channels, future_channels, horizon):
    target = torch.tensor(temp_ctx, dtype=torch.float32)
    past = torch.tensor(np.stack(past_channels, axis=0), dtype=torch.float32)
    fut = torch.tensor(np.stack(future_channels, axis=0), dtype=torch.float32)
    ts = TimeseriesType(target=target, past_covariates=past, future_covariates=fut)
    out = tirex2().forecast(timeseries=[ts], prediction_length=horizon,
                            output_type="numpy")
    return out[0][0, 4]


def wind_alongshore(speed, dir_deg):
    return speed * np.cos(np.deg2rad(dir_deg - AXIS_FROM_DEG))


def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))


def load_climatology():
    df = pd.read_csv(BASE / "data" / "aare_temperature_hourly.csv",
                     parse_dates=["datetime"])
    df = df[df["datetime"] < "2023-01-01"].copy()
    df["doy"] = df["datetime"].dt.dayofyear
    df["hour"] = df["datetime"].dt.hour
    clim = df.groupby(["doy", "hour"])["temperature"].agg(["mean", "std"])
    return clim


def clim_predict(clim, idx):
    keys = list(zip(idx.dayofyear, idx.hour))
    return np.array([clim.loc[k, "mean"] if k in clim.index else np.nan
                     for k in keys])


# =============================================================================
# Ein Fenster — nur Meteo-Bereitstellung geaendert, alles andere identisch
# =============================================================================
def one_window(t0, df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind, clim):
    ctx_start = t0 - pd.Timedelta(hours=CONTEXT)
    fc_end = t0 + pd.Timedelta(hours=HORIZON - 1)
    idx_ctx = pd.date_range(ctx_start, t0 - pd.Timedelta(hours=1), freq="1h")
    idx_fc = pd.date_range(t0, fc_end, freq="1h")
    idx_full = idx_ctx.append(idx_fc)

    truth = df_bern["temperature"].reindex(idx_fc)
    if truth.isna().any():
        return {"t0": t0, "status": "no_truth"}

    temp_ctx_b = df_bern["temperature"].reindex(idx_ctx).interpolate(limit=30)
    flow_ctx_b = df_bern["flow"].reindex(idx_ctx).interpolate(limit=30).ffill().bfill()
    temp_ctx_t = df_thun["temperature"].reindex(idx_ctx).interpolate(limit=30).ffill().bfill()
    flow_ctx_t = df_thun["flow"].reindex(idx_ctx).interpolate(limit=30).ffill().bfill()
    if (temp_ctx_b.isna().any() or flow_ctx_b.isna().any()
            or temp_ctx_t.isna().any() or flow_ctx_t.isna().any()):
        return {"t0": t0, "status": "ctx_nan"}

    # NEU: kausal-konsistent zusammenbauen (d1 fuer <t0+24, d2 fuer >=t0+24)
    tt_b_causal = assemble_causal(s_meteo_bern, t0, "tt", idx_full)
    tt_t_causal = assemble_causal(s_meteo_thun, t0, "tt", idx_full)
    solar_t_causal = assemble_causal(s_meteo_thun, t0, "solar", idx_full)
    ws_causal = assemble_causal(df_wind, t0, "wind_speed", idx_full)
    wd_causal = assemble_causal(df_wind, t0, "wind_dir", idx_full)

    # Interpolation + Randfuellung wie im Original
    tt_b = tt_b_causal.interpolate(limit=12).ffill().bfill()
    tt_t = tt_t_causal.interpolate(limit=12).ffill().bfill()
    solar_t = solar_t_causal.interpolate(limit=12).ffill().bfill()
    ws = ws_causal.interpolate(limit=12).ffill().bfill()
    wd = wd_causal.interpolate(limit=12).ffill().bfill()
    if any(s.isna().any() for s in (tt_b, tt_t, solar_t, ws, wd)):
        return {"t0": t0, "status": "meteo_nan"}

    wind_full = wind_alongshore(ws.to_numpy(), wd.to_numpy())

    # Oraku (unveraendert)
    oraku = fetch_oraku("temperature", t0, HORIZON)
    if oraku is None:
        return {"t0": t0, "status": "no_oraku"}
    oraku = oraku.reindex(idx_fc)
    if oraku.isna().any():
        return {"t0": t0, "status": "oraku_nan"}
    pred_O = oraku.to_numpy()

    def persist_flow(past):
        return pd.Series(np.full(HORIZON, past[-24:].mean()), index=idx_fc)
    try:
        bafu = fetch_oraku("flow", t0, HORIZON)
        if bafu is None:
            bafu = persist_flow(flow_ctx_b.to_numpy())
        else:
            bafu = bafu.reindex(idx_fc)
            if bafu.isna().any():
                bafu = persist_flow(flow_ctx_b.to_numpy())
    except Exception:
        bafu = persist_flow(flow_ctx_b.to_numpy())

    flow_b_past = flow_ctx_b.to_numpy()
    flow_b_full = np.concatenate([flow_b_past, bafu.to_numpy()])
    flow_t_past = flow_ctx_t.to_numpy()
    flow_t_full = np.concatenate([flow_t_past, np.full(HORIZON, flow_t_past[-24:].mean())])

    tt_b_past = tt_b.iloc[:CONTEXT].to_numpy(); tt_b_all = tt_b.to_numpy()
    tt_t_past = tt_t.iloc[:CONTEXT].to_numpy(); tt_t_all = tt_t.to_numpy()
    solar_t_past = solar_t.iloc[:CONTEXT].to_numpy(); solar_t_all = solar_t.to_numpy()
    wind_past = wind_full[:CONTEXT]

    tc_b = temp_ctx_b.to_numpy()
    tc_t = temp_ctx_t.to_numpy()

    # === T: nur tt ===
    pred_T = forecast_tirex2(tc_b, [tt_b_past], [tt_b_all], HORIZON)

    # === A: tt + flow + wind ===
    pred_A = forecast_tirex2(tc_b,
                             [flow_b_past, tt_b_past, wind_past],
                             [flow_b_full, tt_b_all, wind_full], HORIZON)

    # === Stage 1: Thun-Prognose ===
    pred_thun = forecast_tirex2(
        tc_t,
        [flow_t_past, tt_t_past, solar_t_past, wind_past],
        [flow_t_full, tt_t_all, solar_t_all, wind_full], HORIZON)

    # === R: A + Thun-Stage-1 ===
    thun_full_recursive = np.concatenate([tc_t, pred_thun])
    pred_R = forecast_tirex2(
        tc_b,
        [flow_b_past, tt_b_past, wind_past, tc_t],
        [flow_b_full, tt_b_all, wind_full, thun_full_recursive], HORIZON)

    # === B_persist ===
    thun_full_persist = np.concatenate([tc_t, np.full(HORIZON, tc_t[-1])])
    pred_B = forecast_tirex2(
        tc_b,
        [flow_b_past, tt_b_past, wind_past, tc_t],
        [flow_b_full, tt_b_all, wind_full, thun_full_persist], HORIZON)

    pred_AO = 0.5 * (pred_A + pred_O)
    pred_RO = 0.5 * (pred_R + pred_O)

    pred_pers = np.full(HORIZON, temp_ctx_b.iloc[-1])
    pred_clim = clim_predict(clim, idx_fc)
    truth_arr = truth.to_numpy()

    truth_thun = df_thun["temperature"].reindex(idx_fc)
    thun_valid = ~truth_thun.isna().to_numpy()
    stage1_mae = (mae(pred_thun[thun_valid], truth_thun[thun_valid].to_numpy())
                  if thun_valid.sum() > 0 else np.nan)

    return {
        "t0": t0, "status": "ok", "n": HORIZON,
        "T_MAE": mae(pred_T, truth_arr),
        "A_MAE": mae(pred_A, truth_arr),
        "R_MAE": mae(pred_R, truth_arr),
        "B_persist_MAE": mae(pred_B, truth_arr),
        "Oraku_MAE": mae(pred_O, truth_arr),
        "AO_MAE": mae(pred_AO, truth_arr),
        "RO_MAE": mae(pred_RO, truth_arr),
        "persistence_MAE": mae(pred_pers, truth_arr),
        "climatology_MAE": mae(pred_clim, truth_arr) if not np.any(np.isnan(pred_clim)) else np.nan,
        "stage1_thun_MAE": stage1_mae,
        "truth_mean": float(truth.mean()),
        "hourly": {
            "datetime": idx_fc, "truth": truth_arr,
            "T": pred_T, "A": pred_A, "R": pred_R, "B_persist": pred_B,
            "Oraku": pred_O, "AO": pred_AO, "RO": pred_RO,
        },
    }


def main():
    t0_list = pd.date_range(WINDOW_START, WINDOW_END, freq=f"{STEP_DAYS}D")
    print(f"Kandidaten-Fenster: {len(t0_list)}  "
          f"({WINDOW_START.date()} .. {WINDOW_END.date()}, alle {STEP_DAYS}d)")

    data_start = (WINDOW_START - pd.Timedelta(hours=CONTEXT + 24)).date()
    data_end = (WINDOW_END + pd.Timedelta(hours=HORIZON + 24)).date()

    print(f"\nLade Aareguru-Bern {data_start} .. {data_end} (in 20d-Chunks) ...")
    df_bern = fetch_aare_city_chunked("bern", data_start, data_end)
    print(f"  {len(df_bern)}h")

    print(f"\nLade Aareguru-Thun ...")
    df_thun = fetch_aare_city_chunked("thun", data_start, data_end)
    print(f"  {len(df_thun)}h")

    print(f"\nLade Meteo Bern (Previous Runs) ...")
    s_meteo_bern = fetch_meteo_prev_runs(LAT_BERN, LON_BERN, data_start, data_end, "bern")
    print(f"  {len(s_meteo_bern)}h  NaN d1: {int(s_meteo_bern['tt_d1'].isna().sum())}, "
          f"NaN d2: {int(s_meteo_bern['tt_d2'].isna().sum())}")

    print(f"\nLade Meteo Thunersee (Previous Runs) ...")
    s_meteo_thun = fetch_meteo_prev_runs(LAT_THUN_LAKE, LON_THUN_LAKE, data_start, data_end, "thunlake")
    print(f"  {len(s_meteo_thun)}h  NaN d1: {int(s_meteo_thun['tt_d1'].isna().sum())}, "
          f"NaN d2: {int(s_meteo_thun['tt_d2'].isna().sum())}")

    print(f"\nLade Wind Thunersee (Previous Runs) ...")
    df_wind = fetch_wind_prev_runs(data_start, data_end)
    print(f"  {len(df_wind)}h  NaN dir_d1: {int(df_wind['wind_dir_d1'].isna().sum())}, "
          f"NaN dir_d2: {int(df_wind['wind_dir_d2'].isna().sum())}")

    print(f"\nBaue Klimatologie aus 2020-2022 ...")
    clim = load_climatology()

    summary_rows = []
    hourly_rows = []
    status_counts = {}
    t_start = time.time()
    for i, t0 in enumerate(t0_list):
        r = one_window(t0, df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind, clim)
        st = r["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        if st != "ok":
            print(f"  [{i+1:3d}/{len(t0_list)}] {t0.date()}   skip: {st}")
            continue
        elapsed = time.time() - t_start
        rate = (i + 1) / max(elapsed, 1e-3)
        eta = (len(t0_list) - i - 1) / max(rate, 1e-3)
        print(f"  [{i+1:3d}/{len(t0_list)}] {t0.date()}  "
              f"A={r['A_MAE']:.3f} R={r['R_MAE']:.3f} Oraku={r['Oraku_MAE']:.3f} "
              f"AO={r['AO_MAE']:.3f} RO={r['RO_MAE']:.3f}  "
              f"[{rate:.2f} f/s, ETA {eta/60:.1f} min]")
        keep = {k: v for k, v in r.items() if k not in ("status", "hourly")}
        summary_rows.append(keep)
        hh = r["hourly"]
        for j, dt in enumerate(hh["datetime"]):
            hourly_rows.append({
                "t0": t0, "h": j + 1, "datetime": dt,
                "truth": hh["truth"][j],
                "T_pred": hh["T"][j], "A_pred": hh["A"][j],
                "R_pred": hh["R"][j], "B_persist_pred": hh["B_persist"][j],
                "Oraku_pred": hh["Oraku"][j],
                "AO_pred": hh["AO"][j], "RO_pred": hh["RO"][j],
            })

    print("\n" + "=" * 60)
    print("STATUS-VERTEILUNG")
    for k, v in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {v}")

    if not summary_rows:
        print("Keine erfolgreichen Fenster.")
        return

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(BASE / "src/all_seasons_prev_runs_summary.csv", index=False)
    df_hourly = pd.DataFrame(hourly_rows)
    df_hourly.to_csv(BASE / "src/all_seasons_prev_runs_hourly.csv", index=False)

    print("\n" + "=" * 60)
    print(f"ERFOLGREICHE FENSTER: {len(df_sum)}")
    print("Mittlere MAEs (deg C):")
    for c in ["T_MAE","A_MAE","R_MAE","B_persist_MAE","Oraku_MAE",
              "AO_MAE","RO_MAE","persistence_MAE","climatology_MAE","stage1_thun_MAE"]:
        if c in df_sum.columns:
            print(f"  {c:20s}  mean = {df_sum[c].mean():.4f}   "
                  f"median = {df_sum[c].median():.4f}   "
                  f"n_valid = {df_sum[c].notna().sum()}")
    for a, b in [("AO_MAE", "Oraku_MAE"), ("RO_MAE", "Oraku_MAE"),
                 ("R_MAE", "Oraku_MAE"), ("AO_MAE", "RO_MAE"),
                 ("R_MAE", "A_MAE"), ("A_MAE", "Oraku_MAE"),
                 ("T_MAE", "Oraku_MAE")]:
        d = df_sum[a] - df_sum[b]
        print(f"  {a} vs {b}: wins={int((d<0).sum())}/{len(df_sum)}   "
              f"mean diff = {d.mean():+.4f}")


if __name__ == "__main__":
    main()
