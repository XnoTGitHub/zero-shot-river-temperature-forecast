# [Repositorium v1.1.0] Inhalt unveraendert aus review_runde1/frozen_2026-10-14/scripts/delay6h_prev_runs_dst.py;
# angepasst sind nur Pfade (scripts/paths.py). Diff: docs/CODE_CHANGES.diff
"""6-h-Delay-Sensitivitaet auf DST-korrigierter, tz-aware Basis.

Kopie von paper_hywa/src/delay6h_prev_runs.py; importiert alle Loader/Modelle
aus all_seasons_ensemble_prev_runs_dst.py (tz-aware). Logik der Verschiebung
(shift_future) unveraendert. Ausgabe: results/delay6h_prev_runs_dst_summary.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from all_seasons_ensemble_prev_runs_dst import (  # noqa: E402
    TZ, fetch_oraku, assemble_causal, forecast_tirex2, wind_alongshore, mae,
    load_inputs, HORIZON, CONTEXT, WINDOW_START, WINDOW_END, STEP_DAYS, RESULTS,
)

DELAY_HOURS = 6


def shift_future(future_arr, past_arr, delay):
    out = future_arr.copy()
    for h in range(HORIZON):
        idx_shift = h - delay
        if idx_shift < 0:
            out[h] = past_arr[CONTEXT + idx_shift]
        else:
            out[h] = future_arr[idx_shift]
    return out


def one_window_delayed(t0_naive, df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind):
    t0 = t0_naive.tz_localize(TZ)
    ctx_start = t0 - pd.Timedelta(hours=CONTEXT)
    fc_end = t0 + pd.Timedelta(hours=HORIZON - 1)
    idx_ctx = pd.date_range(ctx_start, t0 - pd.Timedelta(hours=1), freq="1h")
    idx_fc = pd.date_range(t0, fc_end, freq="1h")
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

    tt_b = assemble_causal(s_meteo_bern, t0, "tt", idx_full).interpolate(limit=12).ffill().bfill()
    tt_t = assemble_causal(s_meteo_thun, t0, "tt", idx_full).interpolate(limit=12).ffill().bfill()
    solar_t = assemble_causal(s_meteo_thun, t0, "solar", idx_full).interpolate(limit=12).ffill().bfill()
    ws = assemble_causal(df_wind, t0, "wind_speed", idx_full).interpolate(limit=12).ffill().bfill()
    wd = assemble_causal(df_wind, t0, "wind_dir", idx_full).interpolate(limit=12).ffill().bfill()
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

    tt_b_arr = tt_b.to_numpy(); tt_t_arr = tt_t.to_numpy()
    solar_t_arr = solar_t.to_numpy(); wind_arr = wind_full.copy()
    tt_b_past = tt_b_arr[:CONTEXT]; tt_b_fut = tt_b_arr[CONTEXT:]
    tt_t_past = tt_t_arr[:CONTEXT]; tt_t_fut = tt_t_arr[CONTEXT:]
    solar_t_past = solar_t_arr[:CONTEXT]; solar_t_fut = solar_t_arr[CONTEXT:]
    wind_past = wind_arr[:CONTEXT]; wind_fut = wind_arr[CONTEXT:]

    tt_b_fut = shift_future(tt_b_fut, tt_b_arr, DELAY_HOURS)
    tt_t_fut = shift_future(tt_t_fut, tt_t_arr, DELAY_HOURS)
    solar_t_fut = shift_future(solar_t_fut, solar_t_arr, DELAY_HOURS)
    wind_fut = shift_future(wind_fut, wind_arr, DELAY_HOURS)
    bafu_fut = shift_future(bafu.to_numpy(),
                            np.concatenate([flow_ctx_b.to_numpy(), bafu.to_numpy()]), DELAY_HOURS)

    tt_b_all = np.concatenate([tt_b_past, tt_b_fut]); tt_t_all = np.concatenate([tt_t_past, tt_t_fut])
    solar_t_all = np.concatenate([solar_t_past, solar_t_fut]); wind_all = np.concatenate([wind_past, wind_fut])
    flow_b_past = flow_ctx_b.to_numpy(); flow_b_full = np.concatenate([flow_b_past, bafu_fut])
    flow_t_past = flow_ctx_t.to_numpy()
    flow_t_full = np.concatenate([flow_t_past, np.full(HORIZON, flow_t_past[-24:].mean())])
    tc_b = temp_ctx_b.to_numpy(); tc_t = temp_ctx_t.to_numpy()

    pred_T = forecast_tirex2(tc_b, [tt_b_past], [tt_b_all], HORIZON)
    pred_A = forecast_tirex2(tc_b, [flow_b_past, tt_b_past, wind_past],
                             [flow_b_full, tt_b_all, wind_all], HORIZON)
    pred_thun = forecast_tirex2(tc_t, [flow_t_past, tt_t_past, solar_t_past, wind_past],
                                [flow_t_full, tt_t_all, solar_t_all, wind_all], HORIZON)
    thun_full_recursive = np.concatenate([tc_t, pred_thun])
    pred_R = forecast_tirex2(tc_b, [flow_b_past, tt_b_past, wind_past, tc_t],
                             [flow_b_full, tt_b_all, wind_all, thun_full_recursive], HORIZON)
    truth_arr = truth.to_numpy()
    pred_AO = 0.5 * (pred_A + pred_O); pred_RO = 0.5 * (pred_R + pred_O)
    return {"t0": t0_naive, "status": "ok", "n": HORIZON,
            "T_MAE": mae(pred_T, truth_arr), "A_MAE": mae(pred_A, truth_arr), "R_MAE": mae(pred_R, truth_arr),
            "Oraku_MAE": mae(pred_O, truth_arr), "AO_MAE": mae(pred_AO, truth_arr), "RO_MAE": mae(pred_RO, truth_arr)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="dst"); args = ap.parse_args()
    t0_list = pd.date_range(WINDOW_START, WINDOW_END, freq=f"{STEP_DAYS}D")
    df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind, _ = load_inputs()
    rows, t_start, n_done = [], time.time(), 0
    for i, t0 in enumerate(t0_list):
        r = one_window_delayed(t0, df_bern, df_thun, s_meteo_bern, s_meteo_thun, df_wind)
        if r["status"] != "ok":
            print(f"  [{i+1:3d}/{len(t0_list)}] {t0.date()}  skip: {r['status']}", flush=True); continue
        n_done += 1
        eta = (len(t0_list) - i - 1) / max(n_done / max(time.time() - t_start, 1e-3), 1e-3)
        print(f"  [{i+1:3d}/{len(t0_list)}] {t0.date()}  T={r['T_MAE']:.3f} A={r['A_MAE']:.3f} R={r['R_MAE']:.3f} "
              f"Ora={r['Oraku_MAE']:.3f} AO={r['AO_MAE']:.3f} RO={r['RO_MAE']:.3f} [ETA {eta/60:.1f} min]", flush=True)
        rows.append({k: v for k, v in r.items() if k != "status"})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"delay6h_prev_runs_{args.tag}_summary.csv", index=False)
    print(f"\nErfolg: {len(df)}/{len(t0_list)} Fenster", flush=True)


if __name__ == "__main__":
    main()
