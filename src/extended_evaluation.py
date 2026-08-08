"""Erweiterte Evaluation ueber mehrere Sommer (2023-2026) fuer die eigenen
Modelle (T, A, R, Persistenz, Klimatologie).

Aare-Oraku hat historisch KEINE Prognosen via `from`-Parameter zurueck-
gegeben, daher wird der Oraku-Vergleich fuer Extended Eval NICHT gerechnet.
Der Fokus ist auf der Robustheit unserer eigenen Modelle ueber verschiedene
Sommer.

Fenster: 15 pro Sommer (Juli-Mitte bis Ende) fuer 2023, 2024, 2025 und 2026,
also 60 Fenster total.

Kompatibel mit aare_recursive_thun.py aber ohne Oraku-Aufrufe.
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from tirex2 import load_model as load_v2
from tirex2.model.types import TimeseriesType


LAT_BERN, LON_BERN = 46.948, 7.447
LAT_THUN, LON_THUN = 46.758, 7.628
LAT_THUN_LAKE, LON_THUN_LAKE = 46.72, 7.75
CITY_BERN = "bern"
CITY_THUN = "thun"
HORIZON = 48
CONTEXT = 256
AXIS_FROM_DEG = 114.0

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "aare_dashboard_cache"
CACHE_DIR.mkdir(exist_ok=True)
BASE = Path(__file__).resolve().parent.parent


def _get(url, params, cache_key, retries=5):
    fp = CACHE_DIR / f"{cache_key}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=90,
                             headers={"User-Agent": "aare-ext-eval/1.0"})
            r.raise_for_status()
            d = r.json()
            fp.write_text(json.dumps(d))
            return d
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    [_get] fail #{attempt+1} ({e.__class__.__name__}) sleep {wait}s")
            time.sleep(wait)
    raise last_err


def fetch_aare_range(city, start, end):
    d = _get("https://aareguru.existenz.ch/v2018/history",
             {"city": city, "start": str(start), "end": str(end),
              "app": "hywa_ext", "version": "1.0"},
             cache_key=f"aare_{city}_{start}_{end}")
    dat = d["data"]
    def to_df(k, c):
        return pd.DataFrame(dat.get(k, []), columns=["ts", c]).assign(
            datetime=lambda x: pd.to_datetime(x["ts"], unit="s"))
    df = (to_df("temperature", "temperature")
          .merge(to_df("flow", "flow"), on=["ts", "datetime"], how="outer")
          .sort_values("datetime").set_index("datetime").drop(columns=["ts"]))
    df.index = df.index.tz_localize("UTC").tz_convert("Europe/Zurich").tz_localize(None)
    return df.resample("1h").mean()


def fetch_meteo_hist(lat, lon, start, end, tag):
    d = _get("https://historical-forecast-api.open-meteo.com/v1/forecast",
             {"latitude": lat, "longitude": lon,
              "hourly": "temperature_2m,shortwave_radiation",
              "start_date": str(start), "end_date": str(end),
              "timezone": "Europe/Zurich"},
             cache_key=f"meteo_hist_{tag}_{start}_{end}")
    h = d["hourly"]
    return pd.DataFrame({"tt": h["temperature_2m"],
                         "solar": h["shortwave_radiation"]},
                        index=pd.to_datetime(h["time"]))


def fetch_wind_thunersee(start, end):
    d = _get("https://historical-forecast-api.open-meteo.com/v1/forecast",
             {"latitude": LAT_THUN_LAKE, "longitude": LON_THUN_LAKE,
              "hourly": "wind_speed_10m,wind_direction_10m",
              "start_date": str(start), "end_date": str(end),
              "timezone": "Europe/Zurich"},
             cache_key=f"wind_thun_{start}_{end}")
    h = d["hourly"]
    return pd.DataFrame({"wind_speed": h["wind_speed_10m"],
                         "wind_dir": h["wind_direction_10m"]},
                        index=pd.to_datetime(h["time"]))


def fetch_meteo_forecast(lat, lon, start, end, tag):
    """Fuer bafu_flow gibt es keine tiefe Historie. Wir schaetzen den zukuenftigen
    Flow als konstanten Wert aus den letzten Beobachtungen."""
    return None


def wind_alongshore(speed, dir_deg):
    return speed * np.cos(np.deg2rad(dir_deg - AXIS_FROM_DEG))


_TIREX2 = None
def tirex2():
    global _TIREX2
    if _TIREX2 is None:
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


def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))


def load_climatology():
    """Long-term hourly climatology from 2020-2022 (drei Trainings-Jahre)."""
    df = pd.read_csv(BASE / "data" / "aare_temperature_hourly.csv", parse_dates=["datetime"])
    df = df[df["datetime"] < "2023-01-01"].copy()
    df["doy"] = df["datetime"].dt.dayofyear
    df["hour"] = df["datetime"].dt.hour
    clim = df.groupby(["doy", "hour"])["temperature"].agg(["mean", "std", "count"])
    return clim


def clim_predict(clim, idx):
    doy = idx.dayofyear
    hour = idx.hour
    keys = list(zip(doy, hour))
    means = np.array([clim.loc[k, "mean"] if k in clim.index else np.nan for k in keys])
    return means


def one_window(t0, horizon, context, df_bern, df_thun, s_meteo_bern,
               s_meteo_thun, df_wind, clim):
    ctx_start = t0 - pd.Timedelta(hours=context)
    fc_end = t0 + pd.Timedelta(hours=horizon - 1)
    idx_ctx = pd.date_range(ctx_start, t0 - pd.Timedelta(hours=1), freq="1h")
    idx_fc = pd.date_range(t0, fc_end, freq="1h")
    idx_full = idx_ctx.append(idx_fc)

    truth_bern = df_bern["temperature"].reindex(idx_fc)
    if truth_bern.isna().any():
        return None
    truth_thun = df_thun["temperature"].reindex(idx_fc)

    temp_bern_ctx = df_bern["temperature"].reindex(idx_ctx).interpolate(limit=6)
    temp_thun_ctx = df_thun["temperature"].reindex(idx_ctx).interpolate(limit=6).ffill().bfill()
    if temp_bern_ctx.isna().any() or temp_thun_ctx.isna().any():
        return None

    flow_bern_ctx = df_bern["flow"].reindex(idx_ctx).interpolate(limit=6).ffill().bfill()
    flow_thun_ctx = df_thun["flow"].reindex(idx_ctx).interpolate(limit=6).ffill().bfill()
    if flow_bern_ctx.isna().any() or flow_thun_ctx.isna().any():
        return None

    tt_bern = s_meteo_bern["tt"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    tt_thun = s_meteo_thun["tt"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    solar_thun = s_meteo_thun["solar"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    if tt_bern.isna().any() or tt_thun.isna().any() or solar_thun.isna().any():
        return None

    ws = df_wind["wind_speed"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    wd = df_wind["wind_dir"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    if ws.isna().any() or wd.isna().any():
        return None
    wind_full = wind_alongshore(ws.to_numpy(), wd.to_numpy())

    # Ohne BAFU-Flow-Prognose: extrapoliere aktuellen Flow als konstant
    flow_bern_past = flow_bern_ctx.to_numpy()
    flow_bern_future = np.full(horizon, flow_bern_past[-24:].mean())
    flow_bern_full = np.concatenate([flow_bern_past, flow_bern_future])

    flow_thun_past = flow_thun_ctx.to_numpy()
    flow_thun_full = np.concatenate([flow_thun_past,
                                     np.full(horizon, flow_thun_past[-24:].mean())])

    tc_bern = temp_bern_ctx.to_numpy()
    tc_thun = temp_thun_ctx.to_numpy()

    tt_thun_past = tt_thun.iloc[:context].to_numpy()
    tt_thun_all = tt_thun.to_numpy()
    solar_thun_past = solar_thun.iloc[:context].to_numpy()
    solar_thun_all = solar_thun.to_numpy()
    tt_bern_past = tt_bern.iloc[:context].to_numpy()
    tt_bern_all = tt_bern.to_numpy()
    wind_past = wind_full[:context]

    # Stage 1: Thun forecast
    pred_thun = forecast_tirex2(
        tc_thun,
        past_channels=[flow_thun_past, tt_thun_past, solar_thun_past, wind_past],
        future_channels=[flow_thun_full, tt_thun_all, solar_thun_all, wind_full],
        horizon=horizon,
    )
    thun_valid = ~truth_thun.isna().to_numpy()
    mae_thun_forecast = (mae(pred_thun[thun_valid], truth_thun[thun_valid].to_numpy())
                        if thun_valid.sum() > 0 else np.nan)

    # A: TiRex-2 mit tt+flow+wind
    pred_A = forecast_tirex2(
        tc_bern,
        past_channels=[flow_bern_past, tt_bern_past, wind_past],
        future_channels=[flow_bern_full, tt_bern_all, wind_full],
        horizon=horizon,
    )

    # T: TiRex-2 mit nur tt
    pred_T = forecast_tirex2(
        tc_bern,
        past_channels=[tt_bern_past],
        future_channels=[tt_bern_all],
        horizon=horizon,
    )

    # R: Rekursive 2-Stage
    thun_temp_past = temp_thun_ctx.to_numpy()
    thun_full_recursive = np.concatenate([thun_temp_past, pred_thun])
    pred_R = forecast_tirex2(
        tc_bern,
        past_channels=[flow_bern_past, tt_bern_past, wind_past, thun_temp_past],
        future_channels=[flow_bern_full, tt_bern_all, wind_full, thun_full_recursive],
        horizon=horizon,
    )

    # Persistenz
    pred_persist = np.full(horizon, temp_bern_ctx.iloc[-1])
    # Klimatologie
    pred_clim = clim_predict(clim, idx_fc)

    truth_arr = truth_bern.to_numpy()
    return {
        "t0": t0, "n": horizon,
        "T_MAE": mae(pred_T, truth_arr),
        "A_MAE": mae(pred_A, truth_arr),
        "R_MAE": mae(pred_R, truth_arr),
        "persistence_MAE": mae(pred_persist, truth_arr),
        "climatology_MAE": mae(pred_clim, truth_arr) if not np.any(np.isnan(pred_clim)) else np.nan,
        "stage1_thun_MAE": mae_thun_forecast,
    }


def generate_windows_for_summer(year, n_windows=15, month=7):
    """Erzeugt n_windows Startzeitpunkte im Sommer eines Jahres, 24h Abstand."""
    starts = []
    for i in range(n_windows):
        t = pd.Timestamp(year=year, month=month, day=1) + pd.Timedelta(days=i)
        starts.append(t.floor("h") + pd.Timedelta(hours=12))
    return starts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--windows-per-year", type=int, default=15)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--context", type=int, default=CONTEXT)
    args = p.parse_args()

    print(f"Jahre: {args.years}   Fenster pro Jahr: {args.windows_per_year}")
    clim = load_climatology()
    print(f"Klimatologie: {len(clim)} (DoY,Hour)-Zellen")

    all_rows = []
    for year in args.years:
        starts = generate_windows_for_summer(year, n_windows=args.windows_per_year, month=7)
        print(f"\n{'='*60}\nJahr {year}: {len(starts)} Fenster, "
              f"{starts[0].date()} bis {starts[-1].date()}\n{'='*60}")

        aare_start = (starts[0] - pd.Timedelta(hours=args.context + 24)).date()
        aare_end = (starts[-1] + pd.Timedelta(hours=args.horizon + 24)).date()

        df_bern = fetch_aare_range(CITY_BERN, aare_start, aare_end)
        df_thun = fetch_aare_range(CITY_THUN, aare_start, aare_end)
        s_meteo_bern = fetch_meteo_hist(LAT_BERN, LON_BERN, aare_start, aare_end, f"bern_{year}")
        s_meteo_thun = fetch_meteo_hist(LAT_THUN, LON_THUN, aare_start, aare_end, f"thun_{year}")
        df_wind = fetch_wind_thunersee(aare_start, aare_end)
        print(f"Daten geladen fuer {year}: Bern {len(df_bern)}h, Thun {len(df_thun)}h")

        for i, t0 in enumerate(starts):
            print(f"  [{i+1}/{len(starts)}] t0={t0}", end="   ", flush=True)
            r = one_window(t0, args.horizon, args.context, df_bern, df_thun,
                           s_meteo_bern, s_meteo_thun, df_wind, clim)
            if r is None:
                print("uebersprungen")
                continue
            print(f"T={r['T_MAE']:.3f}  A={r['A_MAE']:.3f}  R={r['R_MAE']:.3f}  "
                  f"pers={r['persistence_MAE']:.3f}  clim={r['climatology_MAE']:.3f}  "
                  f"(stage1={r['stage1_thun_MAE']:.3f})")
            r["year"] = year
            all_rows.append(r)
            time.sleep(0.05)

    df = pd.DataFrame(all_rows)
    out = BASE / "src/extended_evaluation.csv"
    df.to_csv(out, index=False)
    print(f"\n{'='*60}\nErgebnis-CSV: {out}\n{'='*60}")

    for year in args.years + ["ALL"]:
        d = df if year == "ALL" else df[df["year"] == year]
        if len(d) == 0:
            continue
        print(f"\nMittlerer MAE  {year}   (n={len(d)})")
        for col in ["T_MAE", "A_MAE", "R_MAE", "persistence_MAE", "climatology_MAE"]:
            v = d[col].mean()
            print(f"  {col:20s}  {v:.4f}")


if __name__ == "__main__":
    main()
