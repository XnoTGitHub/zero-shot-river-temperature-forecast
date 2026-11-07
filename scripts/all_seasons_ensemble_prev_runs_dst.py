# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/frozen_2026-10-14/scripts/all_seasons_ensemble_prev_runs_dst.py;
# angepasst sind nur Pfade (scripts/paths.py) und die feste TiRex-2-Revision mit Pruefsummenkontrolle. Diff: docs/CODE_CHANGES.diff
"""DST-korrigierter Neulauf des 128/129-Fenster-Primaertests (Previous Runs API).

Kopie von paper_hywa/src/all_seasons_ensemble_prev_runs.py mit genau diesen
Aenderungen (alles andere identisch):
  1. Aareguru-Loader behaelt den tz-aware Index (Europe/Zurich), resample DST-korrekt.
  2. Open-Meteo-Caches werden ueber dst_relabel UTC-rekonstruiert und tz-aware
     geliefert (fester utc_offset_seconds der API wird korrigiert).
  3. Oraku-Antworten bleiben tz-aware.
  4. one_window: t0 wird tz-aware (00:00 Lokalzeit); Kontext (256 h) und
     Horizont (48 h) sind damit echte verstrichene Stunden.
  5. Cache-Politik: Originalcache nur lesen; neue Abrufe ausschliesslich nach
     review_runde1/rerun_dst/cache_new/. Ausgaben nach rerun_dst/results/.
  6. CLI: --only t0[,t0...] (Regressionstest), --tag (Dateisuffix).

Keine Originaldatei wird veraendert.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from tirex2 import load_model as load_v2
from tirex2.model.types import TimeseriesType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dst_relabel import relabel_open_meteo  # noqa: E402

from paths import CACHE as CACHE_DIR, CACHE_NEW, RESULTS, CLIMATOLOGY, verify_tirex2_checkpoint  # noqa: E402
CACHE_NEW.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

TZ = "Europe/Zurich"
LAT_BERN, LON_BERN = 46.948, 7.447
LAT_THUN_LAKE, LON_THUN_LAKE = 46.72, 7.75
HORIZON = 48
CONTEXT = 256
AXIS_FROM_DEG = 114.0

WINDOW_START = pd.Timestamp("2025-11-04")
WINDOW_END = pd.Timestamp("2026-08-01")
STEP_DAYS = 2


# =============================================================================
# I/O helpers — Originalcache read-only, neue Abrufe nach CACHE_NEW
# =============================================================================
def _get(url, params, cache_key, retries=5):
    fp_old = CACHE_DIR / f"{cache_key}.json"
    fp_new = CACHE_NEW / f"{cache_key}.json"
    if fp_old.exists():
        return json.loads(fp_old.read_text())
    if fp_new.exists():
        return json.loads(fp_new.read_text())
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60,
                             headers={"User-Agent": "hywa-allseasons-prevruns-dst/1.0"})
            r.raise_for_status()
            data = r.json()
            fp_new.write_text(json.dumps(data))
            print(f"    [NEUER ABRUF -> cache_new/{cache_key}.json]", flush=True)
            return data
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    [_get retry {attempt+1}/{retries} in {wait}s: {type(e).__name__}]", flush=True)
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
        # AENDERUNG 1: tz-aware behalten (kein tz_localize(None))
        df.index = df.index.tz_localize("UTC").tz_convert(TZ)
        frames.append(df.resample("1h").mean())
        cur = chunk_end
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    return full


def _prevruns_json_path(cache_key):
    fp_old = CACHE_DIR / f"{cache_key}.json"
    if fp_old.exists():
        return fp_old
    fp_new = CACHE_NEW / f"{cache_key}.json"
    if fp_new.exists():
        return fp_new
    raise FileNotFoundError(f"Open-Meteo-Cache fehlt (kein Neuabruf vorgesehen): {cache_key}")


def fetch_meteo_prev_runs(lat, lon, start, end, key):
    # AENDERUNG 2: UTC-rekonstruiert, tz-aware (dst_relabel)
    fp = _prevruns_json_path(f"meteo_prevruns_tt_solar_{key}_{start}_{end}")
    res = relabel_open_meteo(fp)
    la = res["local_aware"]
    return pd.DataFrame({
        "tt_d1": la["temperature_2m_previous_day1"].to_numpy(),
        "tt_d2": la["temperature_2m_previous_day2"].to_numpy(),
        "solar_d1": la["shortwave_radiation_previous_day1"].to_numpy(),
        "solar_d2": la["shortwave_radiation_previous_day2"].to_numpy(),
    }, index=la.index)


def fetch_wind_prev_runs(start, end):
    fp = _prevruns_json_path(f"wind_thun_prevruns_{start}_{end}")
    res = relabel_open_meteo(fp)
    la = res["local_aware"]
    return pd.DataFrame({
        "wind_speed_d1": la["wind_speed_10m_previous_day1"].to_numpy(),
        "wind_speed_d2": la["wind_speed_10m_previous_day2"].to_numpy(),
        "wind_dir_d1": la["wind_direction_10m_previous_day1"].to_numpy(),
        "wind_dir_d2": la["wind_direction_10m_previous_day2"].to_numpy(),
    }, index=la.index)


def fetch_oraku(kind, t0_naive, horizon):
    from_iso = t0_naive.tz_localize(TZ).isoformat()
    d = _get(f"https://aare-oraku.konzept.space/forecast/{kind}",
             {"city": "bern", "horizon": horizon, "from": from_iso,
              "app": "hywa-allseasons", "version": "1.0"},
             cache_key=f"oraku_{kind}_{t0_naive.isoformat()}_{horizon}")
    times = d.get("data", {}).get("time", [])
    vals = d.get("data", {}).get("value", [])
    if not times:
        return None
    # AENDERUNG 3: tz-aware behalten
    idx = pd.to_datetime(times, utc=True).tz_convert(TZ)
    return pd.Series(vals, index=idx, dtype=float)


def assemble_causal(df_d1_d2, t0, col_base, idx_full):
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
        print("  loading TiRex-2 ...", flush=True)
        _TIREX2 = load_v2(verify_tirex2_checkpoint(), device="cpu")  # Revision 05e5b26 (paths.py)
    return _TIREX2


def forecast_tirex2(temp_ctx, past_channels, future_channels, horizon):
    target = torch.tensor(temp_ctx, dtype=torch.float32)
    past = torch.tensor(np.stack(past_channels, axis=0), dtype=torch.float32)
    fut = torch.tensor(np.stack(future_channels, axis=0), dtype=torch.float32)
    ts = TimeseriesType(target=target, past_covariates=past, future_covariates=fut)
    out = tirex2().forecast(timeseries=[ts], prediction_length=horizon, output_type="numpy")
    return out[0][0, 4]


def wind_alongshore(speed, dir_deg):
    return speed * np.cos(np.deg2rad(dir_deg - AXIS_FROM_DEG))


def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))


def load_climatology():
    df = pd.read_csv(CLIMATOLOGY, parse_dates=["datetime"])
    df = df[df["datetime"] < "2023-01-01"].copy()
    df["doy"] = df["datetime"].dt.dayofyear
    df["hour"] = df["datetime"].dt.hour
    return df.groupby(["doy", "hour"])["temperature"].agg(["mean", "std"])


def clim_predict(clim, idx):
    keys = list(zip(idx.dayofyear, idx.hour))
    return np.array([clim.loc[k, "mean"] if k in clim.index else np.nan for k in keys])


# =============================================================================
# Ein Fenster — AENDERUNG 4: t0 tz-aware, sonst identisch
# =============================================================================
def one_window(t0_naive, df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind, clim):
    t0 = t0_naive.tz_localize(TZ)
    ctx_start = t0 - pd.Timedelta(hours=CONTEXT)
    fc_end = t0 + pd.Timedelta(hours=HORIZON - 1)
    idx_ctx = pd.date_range(ctx_start, t0 - pd.Timedelta(hours=1), freq="1h")
    idx_fc = pd.date_range(t0, fc_end, freq="1h")
    idx_full = idx_ctx.append(idx_fc)
    assert len(idx_ctx) == CONTEXT and len(idx_fc) == HORIZON

    truth = df_bern["temperature"].reindex(idx_fc)
    if truth.isna().any():
        return {"t0": t0_naive, "status": "no_truth"}

    temp_ctx_b = df_bern["temperature"].reindex(idx_ctx).interpolate(limit=30)
    flow_ctx_b = df_bern["flow"].reindex(idx_ctx).interpolate(limit=30).ffill().bfill()
    temp_ctx_t = df_thun["temperature"].reindex(idx_ctx).interpolate(limit=30).ffill().bfill()
    flow_ctx_t = df_thun["flow"].reindex(idx_ctx).interpolate(limit=30).ffill().bfill()
    if (temp_ctx_b.isna().any() or flow_ctx_b.isna().any()
            or temp_ctx_t.isna().any() or flow_ctx_t.isna().any()):
        return {"t0": t0_naive, "status": "ctx_nan"}

    tt_b_causal = assemble_causal(s_meteo_bern, t0, "tt", idx_full)
    tt_t_causal = assemble_causal(s_meteo_thun, t0, "tt", idx_full)
    solar_t_causal = assemble_causal(s_meteo_thun, t0, "solar", idx_full)
    ws_causal = assemble_causal(df_wind, t0, "wind_speed", idx_full)
    wd_causal = assemble_causal(df_wind, t0, "wind_dir", idx_full)

    tt_b = tt_b_causal.interpolate(limit=12).ffill().bfill()
    tt_t = tt_t_causal.interpolate(limit=12).ffill().bfill()
    solar_t = solar_t_causal.interpolate(limit=12).ffill().bfill()
    ws = ws_causal.interpolate(limit=12).ffill().bfill()
    wd = wd_causal.interpolate(limit=12).ffill().bfill()
    if any(s.isna().any() for s in (tt_b, tt_t, solar_t, ws, wd)):
        return {"t0": t0_naive, "status": "meteo_nan"}

    wind_full = wind_alongshore(ws.to_numpy(), wd.to_numpy())

    oraku = fetch_oraku("temperature", t0_naive, HORIZON)
    if oraku is None:
        return {"t0": t0_naive, "status": "no_oraku"}
    oraku = oraku.reindex(idx_fc)
    if oraku.isna().any():
        return {"t0": t0_naive, "status": "oraku_nan"}
    pred_O = oraku.to_numpy()

    def persist_flow(past):
        return pd.Series(np.full(HORIZON, past[-24:].mean()), index=idx_fc)
    flow_source = "bafu"
    try:
        bafu = fetch_oraku("flow", t0_naive, HORIZON)
        if bafu is None:
            bafu = persist_flow(flow_ctx_b.to_numpy()); flow_source = "persist"
        else:
            bafu = bafu.reindex(idx_fc)
            if bafu.isna().any():
                bafu = persist_flow(flow_ctx_b.to_numpy()); flow_source = "persist"
    except Exception:
        bafu = persist_flow(flow_ctx_b.to_numpy()); flow_source = "persist"

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

    pred_T = forecast_tirex2(tc_b, [tt_b_past], [tt_b_all], HORIZON)
    pred_A = forecast_tirex2(tc_b, [flow_b_past, tt_b_past, wind_past],
                             [flow_b_full, tt_b_all, wind_full], HORIZON)
    pred_thun = forecast_tirex2(tc_t, [flow_t_past, tt_t_past, solar_t_past, wind_past],
                                [flow_t_full, tt_t_all, solar_t_all, wind_full], HORIZON)
    thun_full_recursive = np.concatenate([tc_t, pred_thun])
    pred_R = forecast_tirex2(tc_b, [flow_b_past, tt_b_past, wind_past, tc_t],
                             [flow_b_full, tt_b_all, wind_full, thun_full_recursive], HORIZON)
    thun_full_persist = np.concatenate([tc_t, np.full(HORIZON, tc_t[-1])])
    pred_B = forecast_tirex2(tc_b, [flow_b_past, tt_b_past, wind_past, tc_t],
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
        "t0": t0_naive, "status": "ok", "n": HORIZON,
        "T_MAE": mae(pred_T, truth_arr), "A_MAE": mae(pred_A, truth_arr),
        "R_MAE": mae(pred_R, truth_arr), "B_persist_MAE": mae(pred_B, truth_arr),
        "Oraku_MAE": mae(pred_O, truth_arr), "AO_MAE": mae(pred_AO, truth_arr),
        "RO_MAE": mae(pred_RO, truth_arr), "persistence_MAE": mae(pred_pers, truth_arr),
        "climatology_MAE": mae(pred_clim, truth_arr) if not np.any(np.isnan(pred_clim)) else np.nan,
        "stage1_thun_MAE": stage1_mae, "truth_mean": float(truth.mean()),
        "flow_source": flow_source,
        "fc_start": idx_fc[0].isoformat(), "fc_end": idx_fc[-1].isoformat(),
        "hourly": {"datetime": idx_fc, "truth": truth_arr, "T": pred_T, "A": pred_A, "R": pred_R,
                   "B_persist": pred_B, "Oraku": pred_O, "AO": pred_AO, "RO": pred_RO,
                   "thun_pred": pred_thun},
    }


def load_inputs():
    data_start = (WINDOW_START - pd.Timedelta(hours=CONTEXT + 24)).date()
    data_end = (WINDOW_END + pd.Timedelta(hours=HORIZON + 24)).date()
    print(f"Lade Aareguru Bern/Thun {data_start}..{data_end} (tz-aware) ...", flush=True)
    df_bern = fetch_aare_city_chunked("bern", data_start, data_end)
    df_thun = fetch_aare_city_chunked("thun", data_start, data_end)
    print("Lade Open-Meteo Previous Runs (DST-korrigiert, tz-aware) ...", flush=True)
    s_meteo_bern = fetch_meteo_prev_runs(LAT_BERN, LON_BERN, data_start, data_end, "bern")
    s_meteo_thun = fetch_meteo_prev_runs(LAT_THUN_LAKE, LON_THUN_LAKE, data_start, data_end, "thunlake")
    df_wind = fetch_wind_prev_runs(data_start, data_end)
    clim = load_climatology()
    return df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind, clim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="Komma-Liste von t0 (YYYY-MM-DD) fuer Teillauf")
    ap.add_argument("--tag", default="dst", help="Suffix der Ausgabedateien")
    args = ap.parse_args()

    t0_list = pd.date_range(WINDOW_START, WINDOW_END, freq=f"{STEP_DAYS}D")
    if args.only:
        wanted = {pd.Timestamp(x.strip()) for x in args.only.split(",") if x.strip()}
        t0_list = pd.DatetimeIndex([t for t in t0_list if t in wanted])
    print(f"Kandidaten-Fenster: {len(t0_list)}", flush=True)

    df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind, clim = load_inputs()

    summary_rows, hourly_rows, status_counts = [], [], {}
    t_start = time.time()
    for i, t0 in enumerate(t0_list):
        r = one_window(t0, df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind, clim)
        st = r["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        if st != "ok":
            print(f"  [{i+1:3d}/{len(t0_list)}] {t0.date()}   skip: {st}", flush=True)
            continue
        elapsed = time.time() - t_start
        rate = (i + 1) / max(elapsed, 1e-3)
        eta = (len(t0_list) - i - 1) / max(rate, 1e-3)
        print(f"  [{i+1:3d}/{len(t0_list)}] {t0.date()}  T={r['T_MAE']:.3f} A={r['A_MAE']:.3f} "
              f"R={r['R_MAE']:.3f} Oraku={r['Oraku_MAE']:.3f} AO={r['AO_MAE']:.3f} RO={r['RO_MAE']:.3f} "
              f"flow={r['flow_source']} [ETA {eta/60:.1f} min]", flush=True)
        summary_rows.append({k: v for k, v in r.items() if k not in ("status", "hourly")})
        hh = r["hourly"]
        for j, dt in enumerate(hh["datetime"]):
            hourly_rows.append({
                "t0": t0, "h": j + 1, "datetime": dt.isoformat(),
                "datetime_utc": dt.tz_convert("UTC").strftime("%Y-%m-%d %H:%M"),
                "truth": hh["truth"][j], "T_pred": hh["T"][j], "A_pred": hh["A"][j],
                "R_pred": hh["R"][j], "B_persist_pred": hh["B_persist"][j],
                "Oraku_pred": hh["Oraku"][j], "AO_pred": hh["AO"][j], "RO_pred": hh["RO"][j],
                "thun_pred": hh["thun_pred"][j],
            })
    print("\nSTATUS:", status_counts, flush=True)
    if not summary_rows:
        print("Keine erfolgreichen Fenster."); return
    pd.DataFrame(summary_rows).to_csv(RESULTS / f"all_seasons_prev_runs_{args.tag}_summary.csv", index=False)
    pd.DataFrame(hourly_rows).to_csv(RESULTS / f"all_seasons_prev_runs_{args.tag}_hourly.csv", index=False)
    print(f"ERFOLGREICHE FENSTER: {len(summary_rows)} -> results/all_seasons_prev_runs_{args.tag}_*.csv", flush=True)


if __name__ == "__main__":
    main()
