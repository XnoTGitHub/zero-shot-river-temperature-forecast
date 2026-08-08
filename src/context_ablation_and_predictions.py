"""Kontext-Laengen-Ablation UND Speicherung der vollen Prognose-Arrays.

Fuer die 15 Juli-2026-Fenster werden TiRex-2 A und T mit T in {128, 256, 512}
gerechnet und alle Prognose-Werte sowie Wahrheiten gespeichert, so dass
sowohl die Kontext-Ablation-Tabelle als auch der Scatterplot Vorhersage vs.
Wahrheit erzeugt werden koennen.

Zusaetzlich werden Oraku-Vorhersagen aus dem Cache geladen (T=256 default),
und AO-Ensemble berechnet.

Output:
  context_ablation.csv  - MAE pro (t0, model, context)
  predictions_wide.csv  - Fenster-detaillierte Vorhersagen fuer Scatter
"""
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
LAT_THUN_LAKE, LON_THUN_LAKE = 46.72, 7.75
HORIZON = 48
AXIS_FROM_DEG = 114.0

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "aare_dashboard_cache"
BASE = Path(__file__).resolve().parent.parent


def _get(url, params, key, retries=3):
    fp = CACHE_DIR / f"{key}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60,
                             headers={"User-Agent": "hywa-ctx/1.0"})
            r.raise_for_status()
            d = r.json()
            fp.write_text(json.dumps(d))
            return d
        except Exception as e:
            time.sleep(2 ** i)
    raise RuntimeError(f"failed after retries: {key}")


def fetch_aare_range(city, start, end):
    """Fetch in 20-day chunks so Aareguru stays in hourly-resolution mode."""
    chunks = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur < end_ts:
        nxt = min(cur + pd.Timedelta(days=20), end_ts)
        d = _get("https://aareguru.existenz.ch/v2018/history",
                 {"city": city, "start": str(cur.date()), "end": str(nxt.date()),
                  "app": "hywa_ctx", "version": "1.0"},
                 key=f"aare_chunk_{city}_{cur.date()}_{nxt.date()}")
        dat = d["data"]
        def to_df(k, c):
            return pd.DataFrame(dat.get(k, []), columns=["ts", c]).assign(
                datetime=lambda x: pd.to_datetime(x["ts"], unit="s"))
        df = (to_df("temperature", "temperature")
              .merge(to_df("flow", "flow"), on=["ts", "datetime"], how="outer")
              .sort_values("datetime").set_index("datetime").drop(columns=["ts"]))
        chunks.append(df)
        cur = nxt + pd.Timedelta(days=1)
    full = pd.concat(chunks).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    full.index = full.index.tz_localize("UTC").tz_convert("Europe/Zurich").tz_localize(None)
    return full.resample("1h").mean()


def fetch_meteo(lat, lon, start, end, tag):
    d = _get("https://historical-forecast-api.open-meteo.com/v1/forecast",
             {"latitude": lat, "longitude": lon,
              "hourly": "temperature_2m",
              "start_date": str(start), "end_date": str(end),
              "timezone": "Europe/Zurich"},
             key=f"meteo_ctx_{tag}_{start}_{end}")
    return pd.Series(d["hourly"]["temperature_2m"],
                     index=pd.to_datetime(d["hourly"]["time"]), name="tt")


def fetch_wind(start, end):
    d = _get("https://historical-forecast-api.open-meteo.com/v1/forecast",
             {"latitude": LAT_THUN_LAKE, "longitude": LON_THUN_LAKE,
              "hourly": "wind_speed_10m,wind_direction_10m",
              "start_date": str(start), "end_date": str(end),
              "timezone": "Europe/Zurich"},
             key=f"wind_ctx_{start}_{end}")
    return pd.DataFrame({"wind_speed": d["hourly"]["wind_speed_10m"],
                         "wind_dir": d["hourly"]["wind_direction_10m"]},
                        index=pd.to_datetime(d["hourly"]["time"]))


def _fetch_oraku(from_ts, horizon=48, kind="temperature"):
    from_iso = from_ts.tz_localize("Europe/Zurich").isoformat()
    d = _get(f"https://aare-oraku.konzept.space/forecast/{kind}",
             {"city": "bern", "horizon": horizon, "from": from_iso,
              "app": "hywa_ctx", "version": "1.0"},
             key=f"oraku_{kind}_{from_ts.isoformat()}_{horizon}")
    ts = pd.to_datetime(d["data"]["time"]).tz_convert("Europe/Zurich").tz_localize(None)
    return pd.Series(d["data"]["value"], index=ts, name=kind)


_TIREX2 = None
def tirex2():
    global _TIREX2
    if _TIREX2 is None:
        _TIREX2 = load_v2("NX-AI/TiRex-2", device="cpu")
    return _TIREX2


def forecast(temp_ctx, past_channels, future_channels, horizon):
    tgt = torch.tensor(temp_ctx, dtype=torch.float32)
    past = torch.tensor(np.stack(past_channels, 0), dtype=torch.float32)
    fut = torch.tensor(np.stack(future_channels, 0), dtype=torch.float32)
    ts = TimeseriesType(target=tgt, past_covariates=past, future_covariates=fut)
    out = tirex2().forecast(timeseries=[ts], prediction_length=horizon,
                            output_type="numpy")
    return out[0][0, 4]


def wind_alongshore(speed, dir_deg):
    return speed * np.cos(np.deg2rad(dir_deg - AXIS_FROM_DEG))


def one_context_run(t0, context, df_bern, s_meteo, df_wind):
    ctx_start = t0 - pd.Timedelta(hours=context)
    fc_end = t0 + pd.Timedelta(hours=HORIZON - 1)
    idx_ctx = pd.date_range(ctx_start, t0 - pd.Timedelta(hours=1), freq="1h")
    idx_fc = pd.date_range(t0, fc_end, freq="1h")
    idx_full = idx_ctx.append(idx_fc)

    truth = df_bern["temperature"].reindex(idx_fc)
    if truth.isna().any():
        print(f"skip(truth-NaN={int(truth.isna().sum())})", end=" ")
        return None
    temp_ctx = df_bern["temperature"].reindex(idx_ctx).interpolate(limit=6)
    flow_ctx = df_bern["flow"].reindex(idx_ctx).interpolate(limit=6).ffill().bfill()
    tt_full = s_meteo.reindex(idx_full).interpolate(limit=12).ffill().bfill()
    ws = df_wind["wind_speed"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    wd = df_wind["wind_dir"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    if temp_ctx.isna().any():
        print(f"skip(tempctx-NaN={int(temp_ctx.isna().sum())})", end=" ")
        return None
    if flow_ctx.isna().any():
        print(f"skip(flowctx-NaN={int(flow_ctx.isna().sum())})", end=" ")
        return None
    if tt_full.isna().any() or ws.isna().any() or wd.isna().any():
        print("skip(meteo-NaN)", end=" ")
        return None
    wind_full = wind_alongshore(ws.to_numpy(), wd.to_numpy())

    # BAFU-Flow-Vorhersage (aus Oraku), fallback: extrapolieren
    try:
        bafu = _fetch_oraku(t0, HORIZON, kind="flow").reindex(idx_fc).interpolate(limit=6)
        if bafu.isna().any():
            bafu = pd.Series(np.full(HORIZON, flow_ctx.to_numpy()[-24:].mean()),
                             index=idx_fc)
    except Exception as e:
        print(f"[bafu:{e.__class__.__name__}]", end=" ")
        bafu = pd.Series(np.full(HORIZON, flow_ctx.to_numpy()[-24:].mean()),
                         index=idx_fc)

    flow_past = flow_ctx.to_numpy()
    flow_full = np.concatenate([flow_past, bafu.to_numpy()])
    tt_past = tt_full.iloc[:context].to_numpy()
    tt_all = tt_full.to_numpy()

    # T: nur tt
    pred_T = forecast(
        temp_ctx.to_numpy(),
        past_channels=[tt_past],
        future_channels=[tt_all],
        horizon=HORIZON,
    )
    # A: tt + flow + wind
    pred_A = forecast(
        temp_ctx.to_numpy(),
        past_channels=[flow_past, tt_past, wind_full[:context]],
        future_channels=[flow_full, tt_all, wind_full],
        horizon=HORIZON,
    )
    return {"truth": truth.to_numpy(), "T": pred_T, "A": pred_A,
            "t0": t0, "idx_fc": idx_fc}


def main():
    # Fenster wie in aare_recursive_thun.csv (Juli 2026)
    rt = pd.read_csv(BASE / "aare_recursive_thun.csv", parse_dates=["t0"])
    starts = rt["t0"].tolist()
    print(f"n_fenster = {len(starts)}")

    aare_start = (starts[0] - pd.Timedelta(hours=600)).date()
    aare_end = (starts[-1] + pd.Timedelta(hours=HORIZON + 24)).date()

    print("Lade Daten...")
    df_bern = fetch_aare_range("bern", aare_start, aare_end)
    s_meteo = fetch_meteo(LAT_BERN, LON_BERN, aare_start, aare_end, "bern_ctx")
    df_wind = fetch_wind(aare_start, aare_end)
    print(f"  Bern {len(df_bern)}h, meteo {len(s_meteo)}h, wind {len(df_wind)}h")

    contexts = [128, 256, 512]
    ablation_rows = []
    pred_rows = []
    for ci, ctx in enumerate(contexts):
        print(f"\n{'='*50}\nKontext T = {ctx}\n{'='*50}")
        for i, t0 in enumerate(starts):
            print(f"  [{i+1}/{len(starts)}] t0={t0}", end="  ")
            r = one_context_run(t0, ctx, df_bern, s_meteo, df_wind)
            if r is None:
                print("skip")
                continue
            mae_T = float(np.mean(np.abs(r["T"] - r["truth"])))
            mae_A = float(np.mean(np.abs(r["A"] - r["truth"])))
            print(f"T={mae_T:.3f}  A={mae_A:.3f}")
            ablation_rows.append({"t0": t0, "context": ctx,
                                  "T_MAE": mae_T, "A_MAE": mae_A})
            if ctx == 256:
                # Speichere volle Prognosen fuer Scatterplot
                for j, (dt_j, truth_j, tj, aj) in enumerate(
                        zip(r["idx_fc"], r["truth"], r["T"], r["A"])):
                    pred_rows.append({
                        "t0": t0, "h": j + 1, "datetime": dt_j,
                        "truth": truth_j, "T_pred": tj, "A_pred": aj})
            time.sleep(0.05)

    df_abl = pd.DataFrame(ablation_rows)
    df_abl.to_csv(BASE / "src/context_ablation.csv", index=False)
    df_pred = pd.DataFrame(pred_rows)
    df_pred.to_csv(BASE / "src/predictions_wide.csv", index=False)

    print("\n\nKontext-Ablation Zusammenfassung:")
    print(df_abl.groupby("context")[["T_MAE", "A_MAE"]].mean().round(4))


if __name__ == "__main__":
    main()
