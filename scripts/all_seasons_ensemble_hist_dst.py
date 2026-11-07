# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/frozen_2026-10-14/scripts/all_seasons_ensemble_hist_dst.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""Historical-Forecast-Vergleich auf DST-korrigierter, tz-aware Basis.

Kopie von paper_hywa/src/all_seasons_ensemble.py (Kovariaten aus der
historical-forecast-API, jeweils aktuellster Lauf) mit denselben vier
tz-aware-Aenderungen wie all_seasons_ensemble_prev_runs_dst.py. Alle Loader,
Modelle und Oraku-Zugriffe werden von dort importiert; nur die Meteo-Loader
lesen die meteo_hist_*/wind_thun_*-Caches (Offset 7200 -> UTC-Rekonstruktion).
Ausgabe: results/all_seasons_hist_dst_summary.csv / _hourly.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dst_relabel import relabel_open_meteo  # noqa: E402
from all_seasons_ensemble_prev_runs_dst import (  # noqa: E402
    TZ, fetch_aare_city_chunked, fetch_oraku, forecast_tirex2, wind_alongshore, mae,
    load_climatology, clim_predict, _prevruns_json_path,
    HORIZON, CONTEXT, WINDOW_START, WINDOW_END, STEP_DAYS, RESULTS, LAT_BERN, LON_BERN, LAT_THUN_LAKE, LON_THUN_LAKE,
)


def fetch_meteo_hist(lat, lon, start, end, key):
    fp = _prevruns_json_path(f"meteo_hist_tt_solar_{key}_{start}_{end}")
    la = relabel_open_meteo(fp)["local_aware"]
    return pd.DataFrame({"tt": la["temperature_2m"].to_numpy(),
                         "solar": la["shortwave_radiation"].to_numpy()}, index=la.index)


def fetch_wind_hist(start, end):
    fp = _prevruns_json_path(f"wind_thun_{start}_{end}")
    la = relabel_open_meteo(fp)["local_aware"]
    return pd.DataFrame({"wind_speed": la["wind_speed_10m"].to_numpy(),
                         "wind_dir": la["wind_direction_10m"].to_numpy()}, index=la.index)


def one_window(t0_naive, df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind, clim):
    t0 = t0_naive.tz_localize(TZ)
    idx_ctx = pd.date_range(t0 - pd.Timedelta(hours=CONTEXT), t0 - pd.Timedelta(hours=1), freq="1h")
    idx_fc = pd.date_range(t0, t0 + pd.Timedelta(hours=HORIZON - 1), freq="1h")
    idx_full = idx_ctx.append(idx_fc)

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

    tt_b = s_meteo_bern["tt"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    tt_t = s_meteo_thun["tt"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    solar_t = s_meteo_thun["solar"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    ws = df_wind["wind_speed"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
    wd = df_wind["wind_dir"].reindex(idx_full).interpolate(limit=12).ffill().bfill()
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
    try:
        bafu = fetch_oraku("flow", t0_naive, HORIZON)
        if bafu is None:
            bafu = persist_flow(flow_ctx_b.to_numpy())
        else:
            bafu = bafu.reindex(idx_fc)
            if bafu.isna().any():
                bafu = persist_flow(flow_ctx_b.to_numpy())
    except Exception:
        bafu = persist_flow(flow_ctx_b.to_numpy())

    flow_b_past = flow_ctx_b.to_numpy(); flow_b_full = np.concatenate([flow_b_past, bafu.to_numpy()])
    flow_t_past = flow_ctx_t.to_numpy()
    flow_t_full = np.concatenate([flow_t_past, np.full(HORIZON, flow_t_past[-24:].mean())])
    tt_b_past = tt_b.iloc[:CONTEXT].to_numpy(); tt_b_all = tt_b.to_numpy()
    tt_t_past = tt_t.iloc[:CONTEXT].to_numpy(); tt_t_all = tt_t.to_numpy()
    solar_t_past = solar_t.iloc[:CONTEXT].to_numpy(); solar_t_all = solar_t.to_numpy()
    wind_past = wind_full[:CONTEXT]
    tc_b = temp_ctx_b.to_numpy(); tc_t = temp_ctx_t.to_numpy()

    pred_T = forecast_tirex2(tc_b, [tt_b_past], [tt_b_all], HORIZON)
    pred_A = forecast_tirex2(tc_b, [flow_b_past, tt_b_past, wind_past], [flow_b_full, tt_b_all, wind_full], HORIZON)
    pred_thun = forecast_tirex2(tc_t, [flow_t_past, tt_t_past, solar_t_past, wind_past],
                                [flow_t_full, tt_t_all, solar_t_all, wind_full], HORIZON)
    thun_full_recursive = np.concatenate([tc_t, pred_thun])
    pred_R = forecast_tirex2(tc_b, [flow_b_past, tt_b_past, wind_past, tc_t],
                             [flow_b_full, tt_b_all, wind_full, thun_full_recursive], HORIZON)
    pred_AO = 0.5 * (pred_A + pred_O); pred_RO = 0.5 * (pred_R + pred_O)
    truth_arr = truth.to_numpy()
    return {"t0": t0_naive, "status": "ok", "n": HORIZON,
            "T_MAE": mae(pred_T, truth_arr), "A_MAE": mae(pred_A, truth_arr), "R_MAE": mae(pred_R, truth_arr),
            "Oraku_MAE": mae(pred_O, truth_arr), "AO_MAE": mae(pred_AO, truth_arr), "RO_MAE": mae(pred_RO, truth_arr),
            "hourly": {"datetime": idx_fc, "truth": truth_arr, "T": pred_T, "A": pred_A, "R": pred_R,
                       "Oraku": pred_O, "AO": pred_AO, "RO": pred_RO}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="dst"); args = ap.parse_args()
    t0_list = pd.date_range(WINDOW_START, WINDOW_END, freq=f"{STEP_DAYS}D")
    data_start = (WINDOW_START - pd.Timedelta(hours=CONTEXT + 24)).date()
    data_end = (WINDOW_END + pd.Timedelta(hours=HORIZON + 24)).date()
    df_bern = fetch_aare_city_chunked("bern", data_start, data_end)
    df_thun = fetch_aare_city_chunked("thun", data_start, data_end)
    s_meteo_bern = fetch_meteo_hist(LAT_BERN, LON_BERN, data_start, data_end, "bern")
    s_meteo_thun = fetch_meteo_hist(LAT_THUN_LAKE, LON_THUN_LAKE, data_start, data_end, "thunlake")
    df_wind = fetch_wind_hist(data_start, data_end)
    clim = load_climatology()
    rows, hourly, t_start, n_done = [], [], time.time(), 0
    for i, t0 in enumerate(t0_list):
        r = one_window(t0, df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind, clim)
        if r["status"] != "ok":
            print(f"  [{i+1:3d}/{len(t0_list)}] {t0.date()}  skip: {r['status']}", flush=True); continue
        n_done += 1
        eta = (len(t0_list) - i - 1) / max(n_done / max(time.time() - t_start, 1e-3), 1e-3)
        print(f"  [{i+1:3d}/{len(t0_list)}] {t0.date()}  T={r['T_MAE']:.3f} A={r['A_MAE']:.3f} R={r['R_MAE']:.3f} "
              f"Ora={r['Oraku_MAE']:.3f} AO={r['AO_MAE']:.3f} [ETA {eta/60:.1f} min]", flush=True)
        rows.append({k: v for k, v in r.items() if k not in ("status", "hourly")})
        hh = r["hourly"]
        for j, dt in enumerate(hh["datetime"]):
            hourly.append({"t0": t0, "h": j + 1, "datetime": dt.isoformat(), "truth": hh["truth"][j],
                           "T_pred": hh["T"][j], "A_pred": hh["A"][j], "R_pred": hh["R"][j],
                           "Oraku_pred": hh["Oraku"][j], "AO_pred": hh["AO"][j], "RO_pred": hh["RO"][j]})
    pd.DataFrame(rows).to_csv(RESULTS / f"all_seasons_hist_{args.tag}_summary.csv", index=False)
    pd.DataFrame(hourly).to_csv(RESULTS / f"all_seasons_hist_{args.tag}_hourly.csv", index=False)
    print(f"\nErfolg: {len(rows)}/{len(t0_list)} Fenster", flush=True)


if __name__ == "__main__":
    main()
